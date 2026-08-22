"""Head-to-head: U-Net vs the MNDWI baseline, on the held-out evaluation set.

Same 100 lakes, same 300 lake-years (2019/2022/2025), same patches, same
metrics. The baseline is recomputed on the identical patches rather than
reusing the earlier Earth Engine numbers, so the comparison is apples to
apples - the earlier baseline used a different analysis window and is not
directly comparable.

These 100 lakes never appear in training or validation (verified disjoint).

Reported per model: IoU, precision, recall, and absolute area error, both
pooled over all pixels and as a per-lake distribution. Also breaks results
down by lake size and by year, since the failure modes differ.

Caveat that applies to every number here: Hi-MAG labels are 2017 while imagery
is 2019-2025, so genuine lake change counts as error. Scores are a floor.

Usage:
    cd server
    python -m scripts.evaluate_models --years 2019 2022 2025
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from scripts.train_unet import BAND_SETS, FEATURES, UNet

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"
MNDWI_IDX = FEATURES.index("mndwi")


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    inter = float((pred & truth).sum())
    union = float((pred | truth).sum())
    p_sum, t_sum = float(pred.sum()), float(truth.sum())
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / p_sum if p_sum else 0.0,
        "recall": inter / t_sum if t_sum else 0.0,
        "area_err_pct": (p_sum - t_sum) / t_sum * 100 if t_sum else None,
        "inter": inter, "union": union, "pred": p_sum, "truth": t_sum,
    }


def summarise(rows: list[dict], name: str) -> dict:
    ti = sum(r["inter"] for r in rows)
    tu = sum(r["union"] for r in rows)
    tp = ti
    fp = sum(r["pred"] for r in rows) - ti
    fn = sum(r["truth"] for r in rows) - ti
    ious = sorted(r["iou"] for r in rows)
    errs = sorted(abs(r["area_err_pct"]) for r in rows if r["area_err_pct"] is not None)
    med = lambda v: v[len(v) // 2] if v else 0.0
    return {
        "model": name, "n": len(rows),
        "pooled_iou": ti / tu if tu else 0.0,
        "pooled_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "pooled_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "median_iou": med(ious),
        "mean_iou": statistics.mean(ious) if ious else 0.0,
        "p10_iou": ious[int(0.1 * len(ious))] if ious else 0.0,
        "p90_iou": ious[int(0.9 * len(ious))] if ious else 0.0,
        "iou_ge_0.7": sum(1 for x in ious if x >= 0.7),
        "iou_ge_0.8": sum(1 for x in ious if x >= 0.8),
        "median_abs_area_err": med(errs),
    }


def show(s: dict) -> None:
    print(f"  {s['model']:<22} IoU med {s['median_iou']:.3f} mean {s['mean_iou']:.3f} "
          f"| P {s['pooled_precision']:.3f} R {s['pooled_recall']:.3f} "
          f"| >=0.7 {s['iou_ge_0.7']}/{s['n']} ({100*s['iou_ge_0.7']/s['n']:.0f}%) "
          f"| |dArea| {s['median_abs_area_err']:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2022, 2025])
    ap.add_argument("--models", nargs="+", default=["all", "noterr"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    nets = {}
    for tag in args.models:
        ckpt = MODELS / f"unet_{tag}.pt"
        if not ckpt.exists():
            print(f"  (skipping '{tag}' - {ckpt.name} not found)")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        net = UNet(len(blob["idx"]), blob["width"])
        net.load_state_dict(blob["state"])
        net.eval()
        nets[tag] = (net, blob["idx"])
    if not nets:
        raise SystemExit("no trained models found - run train_unet first")

    per_model: dict[str, list[dict]] = {f"unet_{k}": [] for k in nets}
    per_model["mndwi_baseline"] = []
    detail = []

    for year in args.years:
        files = sorted(glob.glob(str(PATCHES / f"eval_{year}" / "*.npz")))
        if not files:
            print(f"  (no eval patches for {year})")
            continue
        for f in files:
            rec = np.load(f, allow_pickle=True)
            x, truth = rec["x"].astype(np.float32), rec["y"].astype(bool)
            row = {"id": str(rec["id"]), "year": year,
                   "area_ha": float(rec["area_m2"]) / 1e4, "subr": str(rec["subr"])}

            base = metrics(x[MNDWI_IDX] > 0.0, truth)
            per_model["mndwi_baseline"].append(base)
            row["mndwi"] = {k: base[k] for k in ("iou", "precision", "recall")}

            for tag, (net, idx) in nets.items():
                with torch.no_grad():
                    logit = net(torch.from_numpy(x[idx])[None])
                    pred = (torch.sigmoid(logit)[0, 0].numpy() > args.threshold)
                m = metrics(pred, truth)
                per_model[f"unet_{tag}"].append(m)
                row[f"unet_{tag}"] = {k: m[k] for k in ("iou", "precision", "recall")}
            detail.append(row)

    print(f"\n=== held-out eval: {len(detail)} lake-years ===")
    summaries = []
    for name, rows in per_model.items():
        if rows:
            s = summarise(rows, name)
            summaries.append(s)
            show(s)

    base_s = next((s for s in summaries if s["model"] == "mndwi_baseline"), None)
    if base_s:
        print("\n  delta vs MNDWI baseline:")
        for s in summaries:
            if s["model"] == "mndwi_baseline":
                continue
            print(f"    {s['model']:<20} IoU {s['median_iou']-base_s['median_iou']:+.3f}  "
                  f"precision {s['pooled_precision']-base_s['pooled_precision']:+.3f}  "
                  f"recall {s['pooled_recall']-base_s['pooled_recall']:+.3f}")

    # breakdowns
    if detail:
        key = f"unet_{list(nets)[0]}"
        print("\n  by lake size (median IoU):")
        for lo, hi, lbl in [(0, 20, "10-20 ha"), (20, 50, "20-50 ha"), (50, 1e9, ">50 ha")]:
            sub = [d for d in detail if lo <= d["area_ha"] < hi]
            if not sub:
                continue
            b = statistics.median(d["mndwi"]["iou"] for d in sub)
            u = statistics.median(d[key]["iou"] for d in sub)
            print(f"    {lbl:<10} n={len(sub):<4} MNDWI {b:.3f}  ->  U-Net {u:.3f}  ({u-b:+.3f})")
        print("\n  by year (median IoU):")
        for y in args.years:
            sub = [d for d in detail if d["year"] == y]
            if not sub:
                continue
            b = statistics.median(d["mndwi"]["iou"] for d in sub)
            u = statistics.median(d[key]["iou"] for d in sub)
            print(f"    {y}      n={len(sub):<4} MNDWI {b:.3f}  ->  U-Net {u:.3f}  ({u-b:+.3f})")

    out = DATA / "_intermediate" / "model_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summaries": summaries, "detail": detail}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
