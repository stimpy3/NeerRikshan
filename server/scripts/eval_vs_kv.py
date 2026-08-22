"""Evaluate the NDWI model against the independent KV 10 m inventories.

This is the first time the model is scored against ground truth it was never
trained on, produced by other people from a different pipeline. Everything up to
now was measured against Hi-MAG - the same labels the model was fitted to, and
which we have since shown run ~8% small.

Three reference masks, deliberately kept separate:

  y (Hi-MAG 2017, 30 m)  what we trained on   - the familiar number
  y_kv16 (2016-17, 10 m) same epoch, finer    - isolates LABEL resolution
  y_kv22 (2022-24, 10 m) our imagery's epoch  - the honest external score

Comparing the first two against the SAME prediction is the clean experiment: the
prediction is identical, so any difference in score is a property of the labels,
not the model.

Sizes are binned on the reference area, not ours, so a truncated lake is still
counted in the bin it actually belongs to.

Usage:
    cd server
    python -m scripts.eval_vs_kv --model ndwi --year 2022
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
KV = DATA / "patches" / "kv_labels"
MODELS = DATA / "models"
PX_HA = (10 * 10) / 1e4          # one 10 m pixel in hectares
BINS = [(0, 15), (15, 25), (25, 50), (50, 100), (100, 1e9)]


def load(tag: str):
    blob = torch.load(MODELS / f"unet_{tag}.pt", map_location="cpu", weights_only=False)
    net = UNet(len(blob["idx"]), blob["width"])
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob["idx"]


def scores(pred: np.ndarray, truth: np.ndarray) -> dict | None:
    """IoU / precision / recall for one lake. None if the reference is empty."""
    t = truth.sum()
    if t == 0:
        return None
    i = float((pred & truth).sum())
    u = float((pred | truth).sum())
    p = float(pred.sum())
    return {
        "iou": i / u if u else 0.0,
        "precision": i / p if p else 0.0,
        "recall": i / t,
        "pred_ha": p * PX_HA,
        "ref_ha": float(t) * PX_HA,
        "ratio": (p / t) if t else 0.0,
    }


def table(name: str, rows: list[dict]) -> dict:
    if not rows:
        print(f"\n{name}: no lakes")
        return {}
    agg_p = sum(r["pred_ha"] for r in rows)
    agg_r = sum(r["ref_ha"] for r in rows)
    s = {
        "n": len(rows),
        "iou_mean": round(st.mean(r["iou"] for r in rows), 4),
        "iou_std": round(st.pstdev([r["iou"] for r in rows]), 4),
        "iou_median": round(st.median(r["iou"] for r in rows), 4),
        "precision": round(st.mean(r["precision"] for r in rows), 4),
        "recall": round(st.mean(r["recall"] for r in rows), 4),
        "median_ratio": round(st.median(r["ratio"] for r in rows), 4),
        "aggregate_ratio": round(agg_p / agg_r, 4) if agg_r else None,
        "iou_below_0.5": sum(1 for r in rows if r["iou"] < 0.5),
    }
    print(f"\n{name}")
    print(f"  n={s['n']}   IoU {s['iou_mean']:.4f} +/-{s['iou_std']:.3f}"
          f"   median {s['iou_median']:.4f}")
    print(f"  precision {s['precision']:.4f}   recall {s['recall']:.4f}"
          f"   IoU<0.5: {s['iou_below_0.5']}")
    print(f"  area ratio  median {s['median_ratio']:.3f}"
          f"   aggregate {s['aggregate_ratio']:.3f}")
    return s


def by_size(name: str, rows: list[dict]) -> list[dict]:
    print(f"\n{name} - split by REFERENCE lake size")
    print(f"  {'size (ha)':<12}{'n':>5}{'IoU':>9}{'prec':>8}{'recall':>9}"
          f"{'med ratio':>11}{'agg ratio':>11}")
    print("  " + "-" * 63)
    out = []
    for lo, hi in BINS:
        g = [r for r in rows if lo <= r["ref_ha"] < hi]
        if not g:
            continue
        lab = f"{lo}-{hi:g}" if hi < 1e9 else f"{lo}+"
        ap = sum(r["pred_ha"] for r in g)
        ar = sum(r["ref_ha"] for r in g)
        print(f"  {lab:<12}{len(g):>5}{st.mean(r['iou'] for r in g):>9.4f}"
              f"{st.mean(r['precision'] for r in g):>8.4f}"
              f"{st.mean(r['recall'] for r in g):>9.4f}"
              f"{st.median(r['ratio'] for r in g):>11.3f}{ap/ar:>11.3f}")
        out.append({"bin": lab, "n": len(g),
                    "iou": round(st.mean(r["iou"] for r in g), 4),
                    "precision": round(st.mean(r["precision"] for r in g), 4),
                    "recall": round(st.mean(r["recall"] for r in g), 4),
                    "median_ratio": round(st.median(r["ratio"] for r in g), 3),
                    "aggregate_ratio": round(ap / ar, 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ndwi")
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    net, idx = load(args.model)
    files = sorted(glob.glob(str(KV / "*.npz")))
    if not files:
        raise SystemExit("no KV label patches - run scripts.export_kv_labels first")

    ref_sets = {"himag17": "y_himag", "kv16": "y_kv16", "kv22": "y_kv22"}
    collected: dict[str, list[dict]] = {k: [] for k in ref_sets}
    missing = 0

    for f in files:
        kv = np.load(f, allow_pickle=True)
        lid = str(kv["id"])
        patch = DATA / "patches" / f"all_{args.year}" / f"{lid}.npz"
        if not patch.exists():
            missing += 1
            continue
        rec = np.load(patch, allow_pickle=True)
        x = derive(rec["x"].astype(np.float32))[idx]
        with torch.no_grad():
            pred = torch.sigmoid(net(torch.from_numpy(x)[None]))[0, 0].numpy() > 0.5

        for tag, key in ref_sets.items():
            s = scores(pred, kv[key].astype(bool))
            if s:
                s["id"] = lid
                collected[tag].append(s)

    print(f"\n=== model '{args.model}' on {args.year} imagery, {len(files)} lakes "
          f"({missing} without a patch) ===")

    results = {"model": args.model, "year": args.year, "overall": {}, "by_size": {}}
    titles = {
        "himag17": "vs Hi-MAG 2017 (30 m Landsat)  [what we trained on]",
        "kv16":    "vs KV 2016-17 (10 m Sentinel-2)  [same epoch, finer labels]",
        "kv22":    "vs KV 2022-24 (10 m Sentinel-2)  [EXTERNAL, matched epoch]",
    }
    for tag in ("himag17", "kv16", "kv22"):
        results["overall"][tag] = table(titles[tag], collected[tag])

    for tag in ("himag17", "kv22"):
        results["by_size"][tag] = by_size(titles[tag], collected[tag])

    # paired: same prediction, two label sets - isolates label quality
    common = {r["id"]: r for r in collected["himag17"]}
    pairs = [(common[r["id"]]["iou"], r["iou"])
             for r in collected["kv16"] if r["id"] in common]
    if pairs:
        d = [b - a for a, b in pairs]
        m, sd = st.mean(d), st.pstdev(d)
        print(f"\npaired dIoU, SAME prediction: KV 2016-17 labels vs Hi-MAG labels")
        print(f"  {m:+.4f} +/-{sd:.3f} over {len(d)} lakes, "
              f"improved {100*sum(1 for z in d if z>0)/len(d):.0f}%")
        print("  (positive = the model agrees better with the 10 m labels, i.e. "
              "part of its\n   'error' against Hi-MAG was Hi-MAG's)")
        results["paired_label_delta"] = {"mean": round(m, 4), "std": round(sd, 4),
                                         "n": len(d)}

    out = DATA / "_intermediate" / "eval_vs_kv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
