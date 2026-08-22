"""GO/NO-GO GATE: can Sentinel-2 reliably recover Hi-MAG lakes >=10 ha?

Scope under test: Himalaya sub-regions, lakes >=10 ha, melt season, 2019-2026.

Metrics per lake-year, against the Hi-MAG 2017 polygon:
  IoU        intersection over union of water masks
  precision  detected-water pixels that are inside the truth polygon
  recall     truth pixels that were detected as water
  area_err   (detected - truth) / truth

Two honest caveats baked into how this should be read:

1. Hi-MAG is 2017; imagery is 2019+. Genuine lake change counts against IoU,
   so these numbers are a FLOOR on segmentation accuracy, not a clean score.
2. Rings were decimated to <=46 vertices in transfer, which slightly softens
   the truth boundary and costs a little IoU on the larger lakes.

Results stream to disk so a partial run is still usable.

Usage:
    cd server
    python -m scripts.validate_ge10ha
    python -m scripts.validate_ge10ha --years 2019 2025 --buffer 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import ee

from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE = DATA / "himag_himalaya_ge10ha.json"
OUT = DATA / "_intermediate" / "validate_ge10ha.json"

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_THRESHOLD = 0.60
SCALE = 10


def stats_for(geom, year: int, buffer_m: int, threshold: float):
    region = geom.buffer(buffer_m).bounds()
    coll = (
        ee.ImageCollection(S2)
        .filterBounds(region)
        .filterDate(f"{year}-06-01", f"{year}-10-31")
        .linkCollection(ee.ImageCollection(CS_PLUS), ["cs"])
    )
    composite = coll.map(
        lambda im: im.updateMask(im.select("cs").gte(CS_THRESHOLD))
    ).median()
    water = composite.normalizedDifference(["B3", "B11"]).gt(threshold).unmask(0)
    truth = ee.Image(0).paint(ee.FeatureCollection([ee.Feature(geom)]), 1).unmask(0)
    px = ee.Image.pixelArea()

    def area(mask):
        return px.updateMask(mask).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region,
            scale=SCALE, maxPixels=1e9, tileScale=4,
        ).get("area")

    return ee.Dictionary({
        "year": year,
        "scenes": coll.size(),
        "inter": area(water.And(truth)),
        "union": area(water.Or(truth)),
        "det": area(water),
        "tru": area(truth),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2022, 2025])
    ap.add_argument("--buffer", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    initialize_or_exit()
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    proj = ee.Projection(doc["crs_wkt"])
    lakes = doc["lakes"][: args.limit] if args.limit else doc["lakes"]

    print(f"lakes={len(lakes)} years={args.years} buffer={args.buffer}m "
          f"MNDWI>{args.threshold}", flush=True)

    rows, failed = [], 0
    for k, L in enumerate(lakes, 1):
        geom = ee.Geometry.Polygon(
            [L["ring"]], proj=proj, geodesic=False
        ).transform("EPSG:4326", 1)
        try:
            res = ee.List(
                [stats_for(geom, y, args.buffer, args.threshold) for y in args.years]
            ).getInfo()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [{k}/{len(lakes)}] {L['id']} FAILED {type(exc).__name__}", flush=True)
            continue
        rec = {"id": L["id"], "himag_ha": L["area_m2"] / 1e4,
               "subr": L["subr"], "type": L["type"], "years": {}}
        for r in res:
            inter = r["inter"] or 0.0
            union = r["union"] or 0.0
            det = r["det"] or 0.0
            tru = r["tru"] or 0.0
            rec["years"][str(r["year"])] = {
                "iou": round(inter / union, 4) if union else 0.0,
                "precision": round(inter / det, 4) if det else 0.0,
                "recall": round(inter / tru, 4) if tru else 0.0,
                "area_err_pct": round((det - tru) / tru * 100, 1) if tru else None,
                "detected_ha": round(det / 1e4, 2),
                "scenes": r["scenes"],
            }
        rows.append(rec)
        if k % 10 == 0:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1),
                           encoding="utf-8")
            print(f"  [{k}/{len(lakes)}] ...", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=1), encoding="utf-8")

    def pull(metric):
        return [y[metric] for r in rows for y in r["years"].values() if y[metric] is not None]

    iou, prec, rec_ = pull("iou"), pull("precision"), pull("recall")
    err = [abs(x) for x in pull("area_err_pct")]
    print(f"\n=== {len(rows)} lakes, {len(iou)} lake-years ({failed} failed) ===")
    for name, vals in (("IoU", iou), ("precision", prec), ("recall", rec_)):
        vs = sorted(vals)
        print(f"  {name:<10} mean {statistics.mean(vs):.3f}  median {vs[len(vs)//2]:.3f}  "
              f"p10 {vs[int(.1*len(vs))]:.3f}  p90 {vs[int(.9*len(vs))]:.3f}")
    es = sorted(err)
    print(f"  |area err| median {es[len(es)//2]:.1f}%  p90 {es[int(.9*len(es))]:.1f}%")
    for t in (0.7, 0.8):
        n = sum(1 for x in iou if x >= t)
        print(f"  IoU >= {t}: {n}/{len(iou)} ({100*n/len(iou):.0f}%)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
