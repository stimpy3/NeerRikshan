"""Stage 3 test: does multi-scale training make adaptive windows work?

Runs a 2x2 - {model} x {window geometry} - on the same val lakes:

                       fixed 128 px        adaptive (native size)
  ndwi        current champion            known failure (-0.347 IoU on >=50 ha)
  ndwi_ms     must not regress            the thing being tested

Two questions, and both must pass:
  1. Does `ndwi_ms` recover performance on adaptive windows?
  2. Does multi-scale training cost anything on the fixed windows we ship today?

Saturation (>60% of window predicted water) is reported alongside, since that is
what adaptive windowing exists to fix - a model can score well on IoU while still
truncating large lakes.

Usage:
    cd server
    python -m scripts.eval_windows --models ndwi ndwi_ms
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics as st
from pathlib import Path

import numpy as np
import torch

from scripts.train_unet import UNet, derive

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"
SATURATED = 0.60


def load(tag: str):
    blob = torch.load(MODELS / f"unet_{tag}.pt", map_location="cpu", weights_only=False)
    net = UNet(len(blob["idx"]), blob["width"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["idx"]


def run_dir(net, idx, directory: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(PATCHES / directory / "*.npz"))):
        rec = np.load(f, allow_pickle=True)
        raw = rec["x"].astype(np.float32)
        truth = rec["y"].astype(bool)
        # pad to a multiple of 8 so the 3 downsample levels line up
        c, h, w = raw.shape
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            raw = np.pad(raw, ((0, 0), (0, ph), (0, pw)))
            truth = np.pad(truth, ((0, ph), (0, pw)))
        with torch.no_grad():
            logit = net(torch.from_numpy(derive(raw)[idx])[None])
            pred = torch.sigmoid(logit)[0, 0].numpy() > 0.5
        i = float((pred & truth).sum())
        u = float((pred | truth).sum())
        out[os.path.basename(f)] = {
            "iou": i / u if u else 0.0,
            "frac": float(pred.mean()),
            "area_ha": float(rec["area_m2"]) / 1e4,
        }
    return out


def summarise(name: str, d: dict) -> None:
    ious = [v["iou"] for v in d.values()]
    sat = sum(1 for v in d.values() if v["frac"] > SATURATED)
    big = [v["iou"] for v in d.values() if v["area_ha"] >= 50]
    print(f"  {name:<26}{st.mean(ious):>8.4f}{st.pstdev(ious):>9.3f}"
          f"{(st.mean(big) if big else float('nan')):>12.4f}{sat:>12}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ndwi", "ndwi_ms"])
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()

    results = {}
    print(f"\n{'config':<28}{'IoU':>8}{'std':>9}{'IoU >=50ha':>12}{'saturated':>12}")
    print("-" * 69)
    for tag in args.models:
        if not (MODELS / f"unet_{tag}.pt").exists():
            print(f"  (skip {tag} - not trained)")
            continue
        net, idx = load(tag)
        for geom, d in (("fixed", f"val_{args.year}"), ("adaptive", f"val_{args.year}_adapt")):
            if not glob.glob(str(PATCHES / d / "*.npz")):
                continue
            r = run_dir(net, idx, d)
            results[(tag, geom)] = r
            summarise(f"{tag} on {geom}", r)

    print("\n--- paired deltas (same lake, same geometry) ---")
    base = ("ndwi", "fixed")
    if base in results:
        for key, r in results.items():
            if key == base:
                continue
            common = set(r) & set(results[base])
            d = [r[k]["iou"] - results[base][k]["iou"] for k in common]
            m, sd = st.mean(d), st.pstdev(d)
            verdict = "REAL" if abs(m) > sd / 2 else "tie"
            print(f"  {key[0]} on {key[1]:<10} vs ndwi/fixed: "
                  f"{m:>+8.4f} +/-{sd:<6.3f} improved {100*sum(1 for z in d if z>0)/len(d):>3.0f}%  {verdict}")


if __name__ == "__main__":
    main()
