"""FEASIBILITY TEST: can we segment glacial lakes from Sentinel-2?

Hi-MAG gives hand-refined 30 m lake polygons for 2008-2017. The proposed work
extends that series with 10 m Sentinel-2. Before committing, this measures
whether a simple MNDWI segmentation actually reproduces the Hi-MAG outlines.

For each sample lake and each test year it computes:
  * IoU  - detected water mask vs the Hi-MAG 2017 polygon
  * area - detected water area vs the Hi-MAG 2017 area
  * n    - clear Sentinel-2 scenes contributing to the composite

IMPORTANT on interpreting IoU: Hi-MAG is 2017, the imagery is 2019+. Real lake
change is mixed into the disagreement, so IoU is a floor on accuracy, not a
clean segmentation score. A lake that genuinely grew will score lower even if
the segmentation is perfect - which is exactly the signal the project is after.

Usage:
    cd server
    python -m scripts.test_lake_segmentation
    python -m scripts.test_lake_segmentation --years 2019 2022 2025 --threshold 0.0
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import ee

from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
SAMPLE = DATA / "himag_sample_2017.json"

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_BAND = "cs"
CS_THRESHOLD = 0.60
SCALE = 10
MELT_START, MELT_END = "-06-01", "-10-31"
BUFFER_M = 400  # analysis window around each lake


def lake_stats(geom, year: int, mndwi_threshold: float) -> dict:
    """Segment water in the melt-season composite and compare to the polygon."""
    region = geom.buffer(BUFFER_M).bounds()
    coll = (
        ee.ImageCollection(S2)
        .filterBounds(region)
        .filterDate(f"{year}{MELT_START}", f"{year}{MELT_END}")
        .linkCollection(ee.ImageCollection(CS_PLUS), [CS_BAND])
    )
    n = coll.size()

    def mask(img):
        return img.updateMask(img.select(CS_BAND).gte(CS_THRESHOLD))

    composite = coll.map(mask).median()
    mndwi = composite.normalizedDifference(["B3", "B11"]).rename("mndwi")
    water = mndwi.gt(mndwi_threshold).unmask(0)

    truth = ee.Image(0).paint(ee.FeatureCollection([ee.Feature(geom)]), 1).unmask(0)
    px = ee.Image.pixelArea()

    def area_of(mask_img):
        return px.updateMask(mask_img).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=SCALE,
            maxPixels=1e9, tileScale=4,
        ).get("area")

    inter = area_of(water.And(truth))
    union = area_of(water.Or(truth))
    det = area_of(water)
    tru = area_of(truth)
    return ee.Dictionary({
        "year": year, "scenes": n,
        "inter_m2": inter, "union_m2": union,
        "detected_m2": det, "truth_m2": tru,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2021, 2023, 2025])
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    initialize_or_exit()
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    proj = ee.Projection(doc["crs_wkt"])
    lakes = doc["lakes"][: args.limit] if args.limit else doc["lakes"]

    print(f"lakes={len(lakes)}  years={args.years}  MNDWI>{args.threshold}\n")
    hdr = f"{'lake':<18}{'ha(HiMAG)':>10}"
    for y in args.years:
        hdr += f"{y:>13}"
    print(hdr)
    print(f"{'':<18}{'':>10}" + "".join(f"{'IoU  dArea%':>13}" for _ in args.years))

    all_iou, rows = [], []
    for L in lakes:
        geom = ee.Geometry.Polygon(
            [L["ring"]], proj=proj, geodesic=False
        ).transform("EPSG:4326", 1)
        results = ee.List([lake_stats(geom, y, args.threshold) for y in args.years]).getInfo()
        line = f"{L['id']:<18}{L['area_m2']/1e4:>10.1f}"
        rec = {"id": L["id"], "himag_ha": L["area_m2"] / 1e4, "years": {}}
        for r in results:
            u, i = r["union_m2"] or 0, r["inter_m2"] or 0
            det, tru = r["detected_m2"] or 0, r["truth_m2"] or 0
            iou = (i / u) if u else 0.0
            d_area = ((det - tru) / tru * 100) if tru else float("nan")
            line += f"{iou:>7.2f}{d_area:>+6.0f}"
            rec["years"][r["year"]] = {
                "iou": round(iou, 3), "detected_ha": round(det / 1e4, 2),
                "truth_ha": round(tru / 1e4, 2), "scenes": r["scenes"],
                "area_delta_pct": round(d_area, 1),
            }
            all_iou.append(iou)
        print(line)
        rows.append(rec)

    print(f"\nIoU across {len(all_iou)} lake-years:")
    print(f"  mean   {statistics.mean(all_iou):.3f}")
    print(f"  median {statistics.median(all_iou):.3f}")
    print(f"  min    {min(all_iou):.3f}   max {max(all_iou):.3f}")
    good = sum(1 for x in all_iou if x >= 0.7)
    print(f"  >=0.70 {good}/{len(all_iou)} ({100*good/len(all_iou):.0f}%)")
    poor = sum(1 for x in all_iou if x < 0.5)
    print(f"  <0.50  {poor}/{len(all_iou)} ({100*poor/len(all_iou):.0f}%)")

    out = DATA / "_intermediate" / "segmentation_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threshold": args.threshold, "results": rows}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
