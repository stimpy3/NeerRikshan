"""Run the trained segmentation model over every lake and year -> area time series.

This is the step that turns a validated detector into an actual result: for
each Himalayan glacial lake >=10 ha, the water area in each melt season from
2019 onwards.

Quality gating matters more here than in evaluation. During evaluation every
patch had a Hi-MAG polygon to compare against, so a bad patch showed up as a
bad score. In production there is no label, so a cloud-ruined composite would
silently produce a fake area and then a fake "change". Each lake-year is
therefore flagged:

  ok         usable
  nodata     too much of the patch has no valid observation
  saturated  predicted water fills the patch - the window is too small for
             this lake, so its area is a lower bound, not a measurement

Flagged years are kept in the output rather than dropped, so downstream code
can decide, and so the flag counts themselves are inspectable.

Usage:
    cd server
    python -m scripts.run_inference --model noterr --years 2019 2020 2021 2022 2023 2024 2025

    # yearly update: score only the new season and merge it into the existing
    # series, so the server never needs the earlier years' patches on disk
    python -m scripts.run_inference --model ndwi_fit_kv22 --fit --years 2026 --append
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch
import re as _re

from scripts.train_unet import UNet, derive

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"
PIXEL_M2 = 100.0  # 10 m x 10 m

# Spectral gate on NIR (B8). Snow and water are indistinguishable under MNDWI -
# it is algebraically NDSI - but they separate cleanly in NIR: measured 0.016-0.024
# over water vs 0.35-0.47 over surrounding snow. Applying this gate removed snow
# false positives entirely (6.7% -> 0.0% of frames with mean NIR > 0.15).
#
# 0.15 rather than the MODIS 0.11 standard: MODIS tunes that value to keep water
# OUT of a snow product; this is the reverse problem. A threshold sweep on the
# held-out set put the optimum at 0.15 - best IoU (0.797), most lakes >=0.7 (76%),
# zero snow errors, and it recovers half the recall lost at 0.11 (0.856 vs 0.821).
NIR_BAND = 3
NIR_MAX_WATER = 0.15

NODATA_LIMIT = 0.25      # >25% of patch invalid -> untrustworthy
SATURATION_LIMIT = 0.60  # >60% of patch predicted water -> window too small
# Fraction of the window BORDER predicted as water. This catches truncation that
# `patch_frac` misses: measured against an independent 10 m inventory, every lake
# over 100 ha touched the border while only 57% of its area sat inside the frame,
# yet IoU within the frame was 0.976. Saturation is a blunt proxy; edge contact is
# the direct signal.
EDGE_LIMIT = 0.02


def target_component(pred: np.ndarray) -> np.ndarray:
    """The connected blob containing the lake at the window centre.

    A fitted window spans 1.5x the lake's longest axis - up to 6.4 km - so it
    deliberately contains neighbouring lakes. The model is trained to segment
    them (the label paints every reference polygon in the frame), but the AREA
    we report must be this lake alone. Counting all water in the window inflated
    the aggregate to 1.31x the reference, and individual lakes to over 2x.

    A lake split into lobes by a moraine is under-counted here; that is the
    conservative direction and is preferred to absorbing a neighbour's growth.
    """
    if not pred.any():
        return pred
    lab, n = ndimage.label(pred)
    if n <= 1:
        return pred
    h, w = pred.shape
    cy, cx = h // 2, w // 2
    target = lab[cy, cx]
    if target == 0:
        ys, xs = np.nonzero(pred)
        k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        target = lab[ys[k], xs[k]]
    return lab == target


def target_touches_edge(pred: np.ndarray) -> float:
    """Border fraction belonging to the CENTRED lake, not to any water.

    A window fitted to a lake deliberately contains its neighbours, so "some
    water touches the border" is normal and says nothing about truncation. Only
    the component containing the target lake matters - if that one reaches the
    edge, its area is a lower bound.

    Scoring any border water flagged 23% of lake-years and eliminated 158 lakes
    outright, none of which were actually truncated.
    """
    if not pred.any():
        return 0.0
    lab, n = ndimage.label(pred)
    if n == 0:
        return 0.0
    h, w = pred.shape
    cy, cx = h // 2, w // 2
    target = lab[cy, cx]
    if target == 0:
        # nothing at the exact centre: take the component nearest the centre
        ys, xs = np.nonzero(pred)
        k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        target = lab[ys[k], xs[k]]
    mask = lab == target
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    return float(border.mean())


def load_model(tag: str):
    blob = torch.load(MODELS / f"unet_{tag}.pt", map_location="cpu", weights_only=False)
    net = UNet(len(blob["idx"]), blob["width"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["idx"]


def lonlat(lake_id: str):
    """Hi-MAG ids encode the centroid: GL075630E35228N -> 75.630E, 35.228N."""
    m = _re.match(r"GL(\d{6})([EW])(\d{5})([NS])", lake_id)
    if not m:
        return None, None
    lon = int(m.group(1)) / 1000.0 * (1 if m.group(2) == "E" else -1)
    lat = int(m.group(3)) / 1000.0 * (1 if m.group(4) == "N" else -1)
    return lon, lat


def quality(x: np.ndarray) -> float:
    """Fraction of pixels with no usable observation (all-zero across optical)."""
    return float((np.abs(x[:6]).sum(axis=0) == 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ndwi")
    ap.add_argument("--split", default="all")
    ap.add_argument("--years", type=int, nargs="+",
                    default=[2019, 2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--threshold", type=float, default=0.5)
    # Default OFF as of Stage 1: with NDWI in the inputs the model learned the
    # snow/water distinction itself, and the gate now costs 0.065 IoU and triples
    # catastrophic failures (9 -> 28). It was a patch for a problem now fixed
    # properly upstream.
    ap.add_argument("--nir-filter", action="store_true", default=False,
                    help="apply the NIR spectral gate (obsolete since Stage 1)")
    ap.add_argument("--no-nir-filter", dest="nir_filter", action="store_false")
    ap.add_argument("--nir-max", type=float, default=NIR_MAX_WATER)
    ap.add_argument("--fit", action="store_true",
                    help="read *_fit patch dirs (windows sized to lake extent)")
    ap.add_argument("--append", action="store_true",
                    help="merge --years into the existing lake_timeseries.json "
                         "instead of replacing it")
    args = ap.parse_args()

    out = DATA / "lake_timeseries.json"
    net, idx = load_model(args.model)
    series: dict[str, dict] = {}
    counts = {"ok": 0, "nodata": 0, "saturated": 0, "truncated": 0}
    years_done: list[int] = []

    if args.append:
        if not out.exists():
            raise SystemExit(f"--append needs an existing {out.name}; run without it first")
        prev = json.loads(out.read_text(encoding="utf-8"))
        # Areas from two different models or thresholds are not one series - the
        # trend would fit the switch, not the lake.
        for key in ("model", "threshold", "nir_filter"):
            if prev.get(key) != getattr(args, key):
                raise SystemExit(f"--append: {key}={getattr(args, key)!r} but "
                                 f"{out.name} was built with {prev.get(key)!r}")
        series = {lake["id"]: lake for lake in prev["lakes"]}
        # re-running a year replaces it rather than double-counting it
        years_done = [y for y in prev["years"] if y not in args.years]
        for lake in series.values():
            for y in args.years:
                lake["years"].pop(str(y), None)

    for year in args.years:
        suffix = "_fit" if args.fit else ""
        files = sorted(glob.glob(str(PATCHES / f"{args.split}_{year}{suffix}" / "*.npz")))
        if not files:
            print(f"  {year}: no patches, skipped", flush=True)
            continue
        for n, f in enumerate(files, 1):
            rec = np.load(f, allow_pickle=True)
            lake_id = str(rec["id"])
            x = rec["x"].astype(np.float32)
            with torch.no_grad():
                feats = derive(x)[idx]
                prob = torch.sigmoid(net(torch.from_numpy(feats)[None]))[0, 0].numpy()
            pred = prob > args.threshold
            if args.nir_filter:
                # drop predicted-water pixels too bright in NIR to be water
                pred = pred & (x[NIR_BAND] <= args.nir_max)
            # segment everything, but report only the centred lake
            lake = target_component(pred)
            frac = float(lake.mean())
            nod = quality(x)
            edge = target_touches_edge(pred)
            neighbours = float(pred.sum() - lake.sum()) * PIXEL_M2 / 1e4

            flag = "ok"
            if nod > NODATA_LIMIT:
                flag = "nodata"
            elif frac > SATURATION_LIMIT:
                flag = "saturated"
            elif edge > EDGE_LIMIT:
                # the TARGET lake runs off the window: area is a lower bound
                flag = "truncated"

            e = series.setdefault(lake_id, {
                "id": lake_id,
                "himag_2017_ha": float(rec["area_m2"]) / 1e4,
                "subr": str(rec["subr"]),
                "lon": lonlat(lake_id)[0],
                "lat": lonlat(lake_id)[1],
                "years": {},
            })
            e["years"][str(year)] = {
                "area_ha": round(float(lake.sum()) * PIXEL_M2 / 1e4, 3),
                "neighbour_ha": round(neighbours, 2),
                "flag": flag,
                "nodata_frac": round(nod, 3),
                "patch_frac": round(frac, 3),
                "edge_frac": round(edge, 3),
            }
            if n % 200 == 0:
                print(f"  {year}: {n}/{len(files)}", flush=True)
        years_done.append(year)
        print(f"  {year}: done ({len(files)} lakes)", flush=True)

    # counted over the whole merged series, so an appended file reports the
    # same totals a from-scratch run would
    for lake in series.values():
        for v in lake["years"].values():
            counts[v["flag"]] += 1

    out.write_text(json.dumps({
        "model": args.model, "threshold": args.threshold,
        "nir_filter": args.nir_filter, "nir_max": args.nir_max,
        "years": sorted(years_done), "flag_counts": counts,
        "lakes": list(series.values()),
    }, indent=1), encoding="utf-8")

    total = sum(counts.values())
    print(f"\n{len(series)} lakes, {total} lake-years")
    for k, v in counts.items():
        print(f"  {k:<10} {v:>6} ({100*v/total:.1f}%)" if total else k)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
