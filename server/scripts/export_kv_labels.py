"""Rasterise the KV reference inventories onto the existing patch grid.

We cannot compare our model to the reference pixelwise unless both sit on the
same grid, and reproducing GEE's `centre.buffer(half_m).bounds()` offline is
ambiguous. So we let GEE do it: same geometry, same region, same scale, with the
reference polygons painted as extra bands.

Alignment is not assumed - it is CHECKED. Every export also re-paints the Hi-MAG
polygon, and `--verify` compares that against the `y` already stored in the
patch. If those two masks do not agree, the grids differ and every number
downstream would be quietly wrong.

B2 leads the band list exactly as in export_patches.py, so the output grid is
fixed by the same S2 projection as the original export.

Usage:
    cd server
    python -m scripts.export_kv_labels --year 2022
    python -m scripts.export_kv_labels --verify
"""

from __future__ import annotations

import argparse
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import requests

from scripts.export_patches import SCALE, SIZE, fit
from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
OUTDIR = DATA / "patches" / "kv_labels"
EPOCH_FILES = {
    "kv16": DATA / "kv_rings_2016_2017.json",
    "kv22": DATA / "kv_rings_2022_2024.json",
}
_local = threading.local()


def paint(polys: list) -> ee.Image:
    """Rasterise a list of lon/lat polygons (with holes) as a 0/1 band."""
    if not polys:
        return ee.Image(0).toFloat()
    feats = [ee.Feature(ee.Geometry.Polygon(p, proj="EPSG:4326", geodesic=False))
             for p in polys]
    return ee.Image(0).paint(ee.FeatureCollection(feats), 1).toFloat()


def export_one(lake: dict, wkt: str, year: int, rings: dict) -> tuple[str, str]:
    dest = OUTDIR / f"{lake['id']}.npz"
    if dest.exists():
        return lake["id"], "cached"

    proj = ee.Projection(wkt)
    geom = ee.Geometry.Polygon(
        [lake["ring"]], proj=proj, geodesic=False
    ).transform("EPSG:4326", 1)
    half_m = SIZE * SCALE / 2
    region = geom.centroid(1).buffer(half_m).bounds()

    # B2 first: it carries the S2 projection that defines the output grid, so
    # the result lands on exactly the grid export_patches.py produced.
    b2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geom)
          .filterDate(f"{year}-06-01", f"{year}-10-31")
          .select(["B2"]).median().toFloat())
    himag = ee.Image(0).paint(
        ee.FeatureCollection([ee.Feature(geom)]), 1).rename("label").toFloat()

    img = b2.rename("B2").addBands([
        himag,
        paint(rings["kv16"].get(lake["id"], [])).rename("label_kv16"),
        paint(rings["kv22"].get(lake["id"], [])).rename("label_kv22"),
    ])

    url = img.getDownloadURL({
        "region": region, "scale": SCALE, "format": "NPY",
        "bands": ["B2", "label", "label_kv16", "label_kv22"],
    })
    sess = getattr(_local, "sess", None)
    if sess is None:
        sess = _local.sess = requests.Session()
    resp = sess.get(url, timeout=600)
    resp.raise_for_status()
    rec = np.load(io.BytesIO(resp.content))

    np.savez_compressed(
        dest,
        y_himag=(fit(rec["label"], SIZE) > 0.5).astype(np.uint8),
        y_kv16=(fit(rec["label_kv16"], SIZE) > 0.5).astype(np.uint8),
        y_kv22=(fit(rec["label_kv22"], SIZE) > 0.5).astype(np.uint8),
        id=lake["id"],
    )
    return lake["id"], "ok"


def verify() -> None:
    """Confirm the re-painted Hi-MAG mask matches the stored patch label."""
    import glob
    ious, bad = [], []
    for f in sorted(glob.glob(str(OUTDIR / "*.npz"))):
        new = np.load(f, allow_pickle=True)
        lid = str(new["id"])
        old_f = DATA / "patches" / "all_2022" / f"{lid}.npz"
        if not old_f.exists():
            continue
        old = np.load(old_f, allow_pickle=True)["y"].astype(bool)
        a = new["y_himag"].astype(bool)
        u = (a | old).sum()
        iou = float((a & old).sum()) / u if u else 1.0
        ious.append(iou)
        if iou < 0.95:
            bad.append((lid, round(iou, 3)))
    if not ious:
        raise SystemExit("no overlapping patches to verify")
    ious.sort()
    print(f"\ngrid alignment check ({len(ious)} lakes)")
    print(f"  median IoU vs stored label : {ious[len(ious)//2]:.4f}")
    print(f"  p10                        : {ious[len(ious)//10]:.4f}")
    print(f"  below 0.95                 : {len(bad)} ({100*len(bad)/len(ious):.1f}%)")
    if bad[:5]:
        print(f"  worst: {bad[:5]}")
    verdict = "ALIGNED" if ious[len(ious) // 2] > 0.98 else "MISALIGNED - do not trust downstream"
    print(f"  verdict                    : {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true", help="only run the grid check")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    initialize_or_exit()
    doc = json.loads((DATA / "himag_all.json").read_text(encoding="utf-8"))
    rings = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in EPOCH_FILES.items()}
    lakes = doc["lakes"][: args.limit] if args.limit else doc["lakes"]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"painting KV labels for {len(lakes)} lakes -> {OUTDIR}", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(export_one, L, doc["crs_wkt"], args.year, rings): L["id"]
                for L in lakes}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 8:
                    print(f"  FAIL {futs[fut]}: {type(exc).__name__}: {str(exc)[:70]}",
                          flush=True)
            if (done + failed) % 50 == 0:
                print(f"  {done + failed}/{len(lakes)} (ok={done} fail={failed})", flush=True)
    print(f"done: {done} ok, {failed} failed")
    if done:
        verify()


if __name__ == "__main__":
    main()
