"""The 2x2: {model} x {window geometry}, scored on val.

                       fixed 128 px          fitted to lake extent
  ndwi          what we ship today       known failure (-0.35 IoU on >=50 ha)
  ndwi_fit      must not regress         the thing being tested

Both questions matter. A model that wins on fitted windows but loses on fixed
ones has traded one bias for another, and we would not know which lakes moved.

Also reports EDGE CONTACT - the fraction of the window border predicted as water.
That is the direct truncation signal: under fixed windows every lake over 100 ha
touched the border while still scoring IoU 0.976 inside the frame, so IoU alone
cannot tell you whether the measurement is complete.

Writes data/_intermediate/window_2x2.json for the experiment driver's gate.

Usage:
    cd server
    python -m scripts.eval_geometry --models ndwi ndwi_fit
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from scripts.train_unet import UNet, derive

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"
BINS = [(0, 25), (25, 50), (50, 1e9)]


def load(tag: str):
    blob = torch.load(MODELS / f"unet_{tag}.pt", map_location="cpu", weights_only=False)
    net = UNet(len(blob["idx"]), blob["width"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["idx"]


def run_dir(net, idx, directory: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(directory / "*.npz"))):
        rec = np.load(f, allow_pickle=True)
        x = rec["x"].astype(np.float32)
        truth = rec["y"].astype(bool)
        c, h, w = x.shape
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
            truth = np.pad(truth, ((0, ph), (0, pw)))
        with torch.no_grad():
            pred = torch.sigmoid(
                net(torch.from_numpy(derive(x)[idx])[None]))[0, 0].numpy() > 0.5
        i = float((pred & truth).sum())
        u = float((pred | truth).sum())
        border = np.concatenate([pred[0], pred[-1], pred[:, 0], pred[:, -1]])
        out[str(rec["id"])] = {
            "iou": i / u if u else 0.0,
            "recall": i / truth.sum() if truth.sum() else 0.0,
            "edge": float(border.mean()),
            "area_ha": float(rec["area_m2"]) / 1e4,
            "pred_ha": float(pred.sum()) / 100.0,
        }
    return out


def summarise(d: dict) -> dict:
    v = list(d.values())
    return {
        "n": len(v),
        "iou": round(st.mean(r["iou"] for r in v), 4),
        "iou_std": round(st.pstdev([r["iou"] for r in v]), 4),
        "recall": round(st.mean(r["recall"] for r in v), 4),
        "edge_gt_2pct": sum(1 for r in v if r["edge"] > 0.02),
        "iou_ge50ha": round(
            st.mean([r["iou"] for r in v if r["area_ha"] >= 50] or [float("nan")]), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ndwi", "ndwi_fit"])
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--dirs", nargs="+", default=None,
                    help="patch dirs to score on, as name=dirname pairs. Every "
                         "model is scored on every dir, so the comparison is "
                         "against ONE target rather than across label sets.")
    args = ap.parse_args()

    if args.dirs:
        geoms = {}
        for spec in args.dirs:
            name, _, d = spec.partition("=")
            geoms[name] = PATCHES / (d or name)
    else:
        geoms = {"fixed": PATCHES / f"val_{args.year}",
                 "fitted": PATCHES / f"val_{args.year}_fit"}
    results, raw = {}, {}

    hdr = f"{'model':<12}{'geometry':<10}{'n':>5}{'IoU':>9}{'std':>8}{'recall':>9}{'IoU>=50ha':>12}{'edge>2%':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag in args.models:
        if not (MODELS / f"unet_{tag}.pt").exists():
            print(f"  (skip {tag} - not trained)")
            continue
        net, idx = load(tag)
        results[tag] = {}
        for gname, gdir in geoms.items():
            if not glob.glob(str(gdir / "*.npz")):
                continue
            d = run_dir(net, idx, gdir)
            raw[(tag, gname)] = d
            s = summarise(d)
            results[tag][gname] = s
            print(f"{tag:<12}{gname:<10}{s['n']:>5}{s['iou']:>9.4f}{s['iou_std']:>8.3f}"
                  f"{s['recall']:>9.4f}{s['iou_ge50ha']:>12.4f}{s['edge_gt_2pct']:>9}")

    # paired deltas against the shipping configuration
    base = (args.models[0], next(iter(geoms)))
    if base in raw:
        print(f"\npaired per-lake delta vs ndwi/fixed (same lake, both models)")
        for key, d in raw.items():
            if key == base:
                continue
            common = set(d) & set(raw[base])
            deltas = [d[k]["iou"] - raw[base][k]["iou"] for k in common]
            m, sd = st.mean(deltas), st.pstdev(deltas)
            verdict = "REAL" if abs(m) > sd / 2 else "tie"
            print(f"  {key[0]:<12}{key[1]:<9}{m:>+9.4f} +/-{sd:<6.3f}"
                  f"  improved {100*sum(1 for z in deltas if z>0)/len(deltas):>3.0f}%  {verdict}")
            results.setdefault(key[0], {}).setdefault(key[1], {}).update(
                {"paired_delta": round(m, 4), "paired_std": round(sd, 4)})

    # area recovered on big lakes is the point of the exercise
    big = [(k, r) for k, r in raw.items()]
    print(f"\narea measured on lakes >=50 ha (mean ha)")
    for key, d in big:
        v = [r["pred_ha"] for r in d.values() if r["area_ha"] >= 50]
        if v:
            print(f"  {key[0]:<12}{key[1]:<9}{st.mean(v):>8.1f} ha  (n={len(v)})")

    dest = DATA / "_intermediate" / "window_2x2.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
