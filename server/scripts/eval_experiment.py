"""Score one or more models on the VALIDATION split, with paired statistics.

The 100-lake eval set is frozen and must not be touched between experiments -
repeatedly testing on it would tune the model to it and make the final number
meaningless. Every experiment in the ladder is judged here, on val (217 lakes).

Why paired deltas rather than two independent medians: lakes differ enormously
in difficulty, and that variance swamps the effect of a model change. Comparing
the SAME lake under two models cancels it out, so a 0.01 improvement that holds
on 80% of lakes is detectable where a median comparison would call it noise.

Reported per model:
  IoU / precision / recall   mean +/- std, median, p10, p90
  |area error|               median
  snow-confusion rate        % of lakes with mean NIR > 0.15 inside prediction
  catastrophic failures      IoU < 0.5

Reported per comparison (vs --ref):
  mean paired delta +/- std, and the share of lakes that improved

Usage:
    cd server
    python -m scripts.eval_experiment --models noterr ndwi --ref noterr
    python -m scripts.eval_experiment --models ndwi --nir-gate 0.15
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from scripts.train_unet import BAND_SETS, UNet, derive

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"
RESULTS = DATA / "_intermediate" / "experiments.json"
NIR_BAND = 3
SNOW_NIR = 0.15


def load(tag: str):
    blob = torch.load(MODELS / f"unet_{tag}.pt", map_location="cpu", weights_only=False)
    net = UNet(len(blob["idx"]), blob["width"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["idx"]


def spread(v: list[float]) -> dict:
    s = sorted(v)
    return {
        "mean": round(st.mean(s), 4),
        "std": round(st.pstdev(s), 4),
        "median": round(s[len(s) // 2], 4),
        "p10": round(s[int(0.10 * len(s))], 4),
        "p90": round(s[int(0.90 * len(s))], 4),
    }


def score(tag: str, year: int, nir_gate: float | None) -> dict:
    net, idx = load(tag)
    files = sorted(glob.glob(str(PATCHES / f"val_{year}" / "*.npz")))
    if not files:
        raise SystemExit(f"no val patches for {year}")

    per_lake, snow, ious = {}, [], []
    for f in files:
        rec = np.load(f, allow_pickle=True)
        raw = rec["x"].astype(np.float32)
        x = derive(raw)[idx]
        truth = rec["y"].astype(bool)
        with torch.no_grad():
            pred = torch.sigmoid(net(torch.from_numpy(x)[None]))[0, 0].numpy() > 0.5
        if nir_gate is not None:
            pred = pred & (raw[NIR_BAND] <= nir_gate)

        i = float((pred & truth).sum())
        u = float((pred | truth).sum())
        p, t = float(pred.sum()), float(truth.sum())
        iou = i / u if u else 0.0
        per_lake[str(rec["id"])] = {
            "iou": iou,
            "precision": i / p if p else 0.0,
            "recall": i / t if t else 0.0,
            "area_err": abs(p - t) / t * 100 if t else 0.0,
        }
        ious.append(iou)
        if p >= 30:
            snow.append(float(raw[NIR_BAND][pred].mean()))

    vals = per_lake.values()
    return {
        "model": tag,
        "nir_gate": nir_gate,
        "n": len(per_lake),
        "iou": spread([v["iou"] for v in vals]),
        "precision": spread([v["precision"] for v in vals]),
        "recall": spread([v["recall"] for v in vals]),
        "median_abs_area_err": round(st.median(v["area_err"] for v in vals), 1),
        "snow_rate_pct": round(100 * sum(1 for z in snow if z > SNOW_NIR) / max(len(snow), 1), 2),
        "median_nir_in_pred": round(st.median(snow), 4) if snow else None,
        "failures_below_0.5": sum(1 for v in ious if v < 0.5),
        "iou_ge_0.7_pct": round(100 * sum(1 for v in ious if v >= 0.7) / len(ious), 1),
        "_per_lake": per_lake,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="baseline model for paired deltas")
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--nir-gate", type=float, default=None,
                    help="apply NIR<=X spectral gate (omit for none)")
    args = ap.parse_args()

    rows = []
    for tag in args.models:
        if not (MODELS / f"unet_{tag}.pt").exists():
            print(f"  (skip {tag} - not trained)")
            continue
        rows.append(score(tag, args.year, args.nir_gate))

    gate = "none" if args.nir_gate is None else f"<={args.nir_gate}"
    print(f"\n=== VAL ({rows[0]['n']} lakes), year {args.year}, NIR gate {gate} ===\n")
    hdr = f"{'model':<10}{'IoU mean+/-std':>18}{'median':>8}{'p10':>7}{'prec':>7}{'recall':>8}{'snow%':>7}{'<0.5':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<10}"
              f"{r['iou']['mean']:>11.3f} +/-{r['iou']['std']:<5.3f}"
              f"{r['iou']['median']:>8.3f}{r['iou']['p10']:>7.3f}"
              f"{r['precision']['mean']:>7.3f}{r['recall']['mean']:>8.3f}"
              f"{r['snow_rate_pct']:>6.1f}%{r['failures_below_0.5']:>6}")

    if args.ref and any(r["model"] == args.ref for r in rows):
        ref = next(r for r in rows if r["model"] == args.ref)
        print(f"\n--- paired per-lake delta vs '{args.ref}' ---")
        print(f"{'model':<10}{'dIoU mean+/-std':>20}{'improved':>11}{'verdict':>12}")
        for r in rows:
            if r["model"] == args.ref:
                continue
            common = set(r["_per_lake"]) & set(ref["_per_lake"])
            d = [r["_per_lake"][k]["iou"] - ref["_per_lake"][k]["iou"] for k in common]
            m, sd = st.mean(d), st.pstdev(d)
            better = 100 * sum(1 for z in d if z > 0) / len(d)
            # a change is only real if the mean shift is large relative to its
            # own spread across lakes
            verdict = "REAL" if abs(m) > sd / 2 else "tie"
            print(f"{r['model']:<10}{m:>+13.4f} +/-{sd:<5.3f}{better:>10.0f}%{verdict:>12}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "_per_lake"} for r in rows]
    RESULTS.write_text(json.dumps(slim, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
