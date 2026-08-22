"""Export Sentinel-2 + DEM training patches from Earth Engine.

One patch per lake: a fixed-size window centred on the Hi-MAG lake centroid,
stacking optical bands, MNDWI, and terrain, plus the rasterised Hi-MAG polygon
as the label.

Terrain is the point of the exercise. The MNDWI baseline fails by extending
contiguously past the shoreline, and the leading hypothesis is shadowed steep
terrain that MNDWI cannot separate from water. Slope is the most direct
counter-signal available: a lake surface is flat, a shadowed valley wall is
not. Elevation and hillshade give the model further context.

Label caveat: Hi-MAG polygons are 2017 while the imagery here is 2019, so a
lake that genuinely changed contributes label noise. This is accepted (2017
Sentinel-2 coverage over HMA is too sparse to use) but it caps achievable
accuracy and must be stated alongside any score.

Patches are written as one .npz per lake and the script is resumable - it
skips lakes already exported.

Usage:
    cd server
    python -m scripts.export_patches --split train
    python -m scripts.export_patches --split val --workers 8
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

from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
PATCH_ROOT = DATA / "patches"

S2 = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_THRESHOLD = 0.60
DEM_ASSET = "NASA/NASADEM_HGT/001"  # GLO30 is deprecated in the catalog

OPTICAL = ["B2", "B3", "B4", "B8", "B11", "B12"]
FEATURES = OPTICAL + ["mndwi", "elev", "slope", "hillshade"]
ALL_BANDS = FEATURES + ["label"]

SIZE = 128          # pixels (fixed mode)
SCALE = 10          # m/px  -> 1.28 km window
HALF_M = SIZE * SCALE / 2

# Adaptive windowing. A fixed 128 px window truncates large lakes: 206 lake-years
# were flagged `saturated` (>60% of the patch predicted water), making those areas
# lower bounds rather than measurements. The U-Net is fully convolutional, so it
# runs on any size that is a multiple of 8 - no tiling or stitching needed, and
# therefore no seam artifacts.
ADAPT_MARGIN = 3.0   # window spans ~3x the lake's linear extent
ADAPT_MIN = 128
ADAPT_MAX = 512      # 5.12 km; covers the largest lake in the set (636 ha)


def window_px(area_m2: float) -> int:
    """Patch size for a lake of this area, rounded up to a multiple of 32."""
    import math
    linear_m = math.sqrt(max(area_m2, 1.0))
    want = linear_m * ADAPT_MARGIN / SCALE
    size = int(math.ceil(want / 32.0) * 32)
    return max(ADAPT_MIN, min(ADAPT_MAX, size))

_local = threading.local()


# Seasonal compositing.
#
# The median reports a "typical" melt-season day, which is biased low for lakes
# that are partly ice-covered early in the season. The literature uses "summer
# maxima" to recover true extent - but that means maximum WATER EXTENT, not
# maximum reflectance.
#
# A naive percentile-of-reflectance composite is actively wrong here: it raises
# every band, so water gets brighter and LESS separable. Measured on p90:
# MNDWI over a lake fell from +0.54 to +0.38.
#
# The correct construction is a quality mosaic - for each pixel, take the date
# when it was wettest (highest NDWI). That yields seasonal maximum extent without
# max()'s sensitivity to a single bright cloud-edge frame.
COMPOSITES = ("median", "maxwater", "p75water", "p90water")


def _with_ndwi(im: ee.Image) -> ee.Image:
    return im.addBands(im.normalizedDifference(["B3", "B8"]).rename("ndwi"))


def reduce_season(coll: ee.ImageCollection, how: str) -> ee.Image:
    """Collapse a melt-season collection to one image."""
    if how == "median":
        return coll.median()
    if how == "maxwater":
        # per-pixel, take the observation with the highest NDWI = wettest date
        return coll.map(_with_ndwi).qualityMosaic("ndwi").select(OPTICAL)
    if how.endswith("water") and how.startswith("p"):
        # percentile of the WATER INDEX, not of reflectance: keeps only the
        # wetter observations, then medians those. Less extreme than maxwater.
        pct = int(how[1:-5])
        withn = coll.map(_with_ndwi)
        thresh = withn.select("ndwi").reduce(ee.Reducer.percentile([pct]))
        keep = withn.map(lambda im: im.updateMask(
            im.select("ndwi").gte(thresh)))
        return keep.select(OPTICAL).median()
    raise ValueError(f"unknown composite {how}")


def kv_label_fc(polys: list):
    """FeatureCollection of lon/lat reference polygons (outer ring + holes)."""
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon(p, proj="EPSG:4326", geodesic=False))
        for p in polys
    ])


def stack_for(geom, year: int, composite: str = "median", label_fc=None) -> ee.Image:
    coll = (
        ee.ImageCollection(S2)
        .filterBounds(geom)
        .filterDate(f"{year}-06-01", f"{year}-10-31")
        .linkCollection(ee.ImageCollection(CS_PLUS), ["cs"])
    )
    masked = coll.map(lambda im: im.updateMask(im.select("cs").gte(CS_THRESHOLD))
                      .select(OPTICAL))
    comp = reduce_season(masked, composite)

    optical = comp.select(OPTICAL).divide(10000).toFloat()
    mndwi = comp.normalizedDifference(["B3", "B11"]).rename("mndwi").toFloat()

    dem = ee.Image(DEM_ASSET).select("elevation")
    slope = ee.Terrain.slope(dem).divide(90).rename("slope").toFloat()
    hill = ee.Terrain.hillshade(dem, 150, 55).divide(255).rename("hillshade").toFloat()
    elev = dem.divide(8000).rename("elev").toFloat()

    # Default label is the Hi-MAG 2017 outline. `label_fc` overrides it with a
    # reference inventory whose epoch matches the imagery - training labels must
    # describe the scene the model is actually looking at, or every shoreline
    # disagreement is charged to the model instead of to the calendar.
    label = (
        ee.Image(0)
        .paint(label_fc if label_fc is not None
               else ee.FeatureCollection([ee.Feature(geom)]), 1)
        .rename("label")
        .toFloat()
    )
    return optical.addBands([mndwi, elev, slope, hill, label])


def fit(arr: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop or zero-pad a (H, W) array to exactly size x size."""
    h, w = arr.shape
    out = np.zeros((size, size), dtype=np.float32)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    crop = arr[sh:sh + size, sw:sw + size]
    dh, dw = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    # nan_to_num defaults map +/-inf to +/-3.4e38, which silently poisons
    # training. Masked pixels (no valid observation) must become 0.
    clean = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)
    out[dh:dh + crop.shape[0], dw:dw + crop.shape[1]] = np.clip(clean, -10.0, 10.0)
    return out


def load_fit_sizes() -> dict[str, int]:
    """Per-lake window sizes from scripts.plan_windows, keyed by lake id."""
    path = DATA / "window_sizes.json"
    if not path.exists():
        raise SystemExit("data/window_sizes.json missing - run "
                         "`python -m scripts.plan_windows` first")
    return json.loads(path.read_text(encoding="utf-8"))["sizes"]


LABEL_SOURCES = {
    "himag": None,                                  # Hi-MAG 2017, 30 m Landsat
    "kv16": DATA / "kv_rings_2016_2017.json",       # KV 2016-17, 10 m Sentinel-2
    "kv22": DATA / "kv_rings_2022_2024.json",       # KV 2022-24, 10 m Sentinel-2
}


def load_label_rings(source: str) -> dict | None:
    if source == "himag":
        return None
    path = LABEL_SOURCES[source]
    if not path.exists():
        raise SystemExit(f"{path.name} missing - run `python -m scripts.prepare_kv_labels`")
    return json.loads(path.read_text(encoding="utf-8"))


def export_one(lake: dict, wkt: str, year: int, outdir: Path,
               adaptive: bool = False, composite: str = "median",
               fit_sizes: dict[str, int] | None = None,
               label_rings: dict | None = None) -> tuple[str, str]:
    dest = outdir / f"{lake['id']}.npz"
    if dest.exists():
        return lake["id"], "cached"
    proj = ee.Projection(wkt)
    geom = ee.Geometry.Polygon(
        [lake["ring"]], proj=proj, geodesic=False
    ).transform("EPSG:4326", 1)
    if fit_sizes is not None:
        # window fitted to the lake's real extent - see scripts.plan_windows
        size = fit_sizes[lake["id"]]
    elif adaptive:
        size = window_px(lake["area_m2"])
    else:
        size = SIZE
    half_m = size * SCALE / 2
    centre = geom.centroid(1)
    region = centre.buffer(half_m).bounds()

    label_fc = None
    if label_rings is not None:
        polys = label_rings.get(lake["id"]) or []
        if not polys:
            # no reference outline here - an all-zero label would teach the model
            # that a real lake is land, which is worse than omitting the sample
            return lake["id"], "no_label"
        label_fc = kv_label_fc(polys)

    url = stack_for(geom, year, composite, label_fc).getDownloadURL(
        {"region": region, "scale": SCALE, "format": "NPY", "bands": ALL_BANDS}
    )
    sess = getattr(_local, "sess", None)
    if sess is None:
        sess = _local.sess = requests.Session()
    resp = sess.get(url, timeout=600)
    resp.raise_for_status()
    rec = np.load(io.BytesIO(resp.content))

    feats = np.stack([fit(rec[b], size) for b in FEATURES], axis=0)
    label = (fit(rec["label"], size) > 0.5).astype(np.uint8)
    np.savez_compressed(
        dest, x=feats.astype(np.float32), y=label,
        id=lake["id"], area_m2=lake["area_m2"], subr=lake["subr"], year=year,
        window_px=size,
    )
    return lake["id"], f"ok water={int(label.sum())}px"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "eval", "all"])
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--composite", default="median", choices=COMPOSITES)
    ap.add_argument("--adaptive", action="store_true",
                    help="size each window to the lake instead of a fixed 128 px")
    ap.add_argument("--labels", default="himag", choices=sorted(LABEL_SOURCES),
                    help="which inventory supplies the training label")
    ap.add_argument("--fit", action="store_true",
                    help="use the per-lake window plan from scripts.plan_windows "
                         "(sized to real extent, not sqrt(area))")
    args = ap.parse_args()

    initialize_or_exit()
    src = {
        "train": DATA / "himag_train.json",
        "val": DATA / "himag_val.json",
        "eval": DATA / "himag_himalaya_ge10ha.json",
        "all": DATA / "himag_all.json",
    }[args.split]
    doc = json.loads(src.read_text(encoding="utf-8"))
    lakes = doc["lakes"][: args.limit] if args.limit else doc["lakes"]
    fit_sizes = load_fit_sizes() if args.fit else None
    label_rings = load_label_rings(args.labels)
    suffix = "_fit" if args.fit else ("_adapt" if args.adaptive else "")
    if args.labels != "himag":
        suffix += f"_{args.labels}"
    if args.composite != "median":
        suffix += f"_{args.composite}"
    outdir = PATCH_ROOT / f"{args.split}_{args.year}{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"exporting {len(lakes)} patches -> {outdir}", flush=True)
    done = failed = nolabel = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(export_one, L, doc["crs_wkt"], args.year, outdir,
                        args.adaptive, args.composite, fit_sizes,
                        label_rings): L["id"]
            for L in lakes
        }
        for fut in as_completed(futs):
            lake_id = futs[fut]
            try:
                _, status = fut.result()
                if status == "no_label":
                    nolabel += 1
                else:
                    done += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  FAIL {lake_id}: {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            if (done + failed) % 25 == 0:
                print(f"  {done + failed}/{len(lakes)} (ok={done} fail={failed})", flush=True)
    print(f"done: {done} exported, {nolabel} skipped (no reference outline), "
          f"{failed} failed -> {outdir}")


if __name__ == "__main__":
    main()
