"""Unattended driver: train a frozen model on matched imagery+labels, then apply it.

Design (revised after checking imagery availability):

  TRAIN     2022 Sentinel-2 imagery  +  KV 2022-24 10 m labels
            Labels describe the scene the model is actually looking at, so a
            shoreline disagreement is the model's error rather than the calendar's.
            2022 is also the MIDPOINT of 2019-2025, so the frozen model
            extrapolates equally little toward either end of the series.

  FREEZE    no further fitting

  APPLY     run the frozen model on 2019-2025 imagery, one segmentation per year,
            and compare areas across years. Later imagery is used to APPLY the
            learned segmentation, not to score it against a stale outline.

Why not 2016-17, which would have kept KV 2022-24 as an untouched validator:
Sentinel-2 L2A begins 2017-01-15 and Himalayan coverage that year is ~3 scenes
per lake before cloud masking, with 18% of lakes having none at all. Measured,
not assumed - see the availability probe in EXPERIMENTS.md.

Windows are sized per lake (scripts.plan_windows) so the whole lake is inside the
frame. Under the fixed 128 px window every lake over 100 ha ran off the edge with
only 57% of its area inside, which is why their measured growth was exactly zero.

NOTHING IN THE CURRENT PRODUCTION SET IS OVERWRITTEN - lake_timeseries.json and
lake_change.json are backed up and restored; new results land in *_fit.json.

Usage:
    cd server
    python -m scripts.run_window_experiment
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PY = sys.executable
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
TRAIN_YEAR = 2022
LABELS = "kv22"
MODEL = f"ndwi_fit_{LABELS}"
TARGET = f"val_{TRAIN_YEAR}_fit_{LABELS}"
LOG = DATA / "_intermediate" / "window_experiment.log"


def run(label: str, *args: str, required: bool = True) -> bool:
    cmd = [PY, "-m", *args]
    line = f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {label}\n  {' '.join(cmd[2:])}\n{'='*70}\n"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        t = time.time()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        fh.write(f"\n[exit {proc.returncode} after {(time.time()-t)/60:.1f} min]\n")
    ok = proc.returncode == 0
    print(f"  -> {'ok' if ok else 'FAILED'} ({(time.time()-t)/60:.1f} min)", flush=True)
    if required and not ok:
        raise SystemExit(f"step failed: {label} (see {LOG})")
    return ok


def seed_fit_dirs(year: int) -> int:
    """Reuse existing 128 px patches - 53% of lakes need no new download.

    A lake whose fitted window is 128 px gets the same region and scale as the
    original fixed export, so the file is identical in meaning and can be copied.
    """
    sizes = json.loads((DATA / "window_sizes.json").read_text(encoding="utf-8"))["sizes"]
    src = DATA / "patches" / f"all_{year}"
    dst = DATA / "patches" / f"all_{year}_fit"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for lake_id, px in sizes.items():
        if px != 128:
            continue
        s, d = src / f"{lake_id}.npz", dst / f"{lake_id}.npz"
        if not d.exists() and s.exists():
            shutil.copy2(s, d)
            copied += 1
    return copied


def gate() -> tuple[bool, str]:
    """Both models scored on ONE target: the KV-labelled fitted val set."""
    path = DATA / "_intermediate" / "window_2x2.json"
    if not path.exists():
        return False, "no evaluation written"
    r = json.loads(path.read_text(encoding="utf-8"))
    try:
        old = r["ndwi"]["target"]
        new = r[MODEL]["target"]
    except KeyError as exc:
        return False, f"evaluation missing {exc}"
    gain = new["iou"] - old["iou"]
    msg = (f"IoU {old['iou']:.4f} -> {new['iou']:.4f} ({gain:+.4f}); "
           f">=50 ha {old['iou_ge50ha']:.4f} -> {new['iou_ge50ha']:.4f}; "
           f"edge-touching {old['edge_gt_2pct']} -> {new['edge_gt_2pct']}")
    if gain <= 0:
        return False, "no improvement against matched-epoch labels. " + msg
    return True, msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--skip-production", action="store_true")
    args = ap.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"frozen-model experiment  started {time.ctime()}\n"
                   f"train: {TRAIN_YEAR} imagery + {LABELS} labels, fitted windows\n",
                   encoding="utf-8")
    t0 = time.time()

    # 1. training data: imagery and labels from the SAME epoch
    for split in ("val", "train"):
        run(f"export {split} {TRAIN_YEAR} + {LABELS} labels", "scripts.export_patches",
            "--split", split, "--year", str(TRAIN_YEAR), "--fit",
            "--labels", LABELS, "--workers", "6")

    # 2. train on native-scale crops of the fitted windows
    run(f"train {MODEL}", "scripts.train_unet", "--bands", "ndwi", "--fit",
        "--year", str(TRAIN_YEAR),      # train_unet defaults to 2019
        "--labels", LABELS, "--crop", str(args.crop), "--epochs", str(args.epochs))

    # 3. score old and new on the SAME target - matched-epoch 10 m labels
    run("evaluate on matched-epoch labels", "scripts.eval_geometry",
        "--models", "ndwi", MODEL, "--dirs", f"target={TARGET}")

    ok, msg = gate()
    print(f"\nGATE: {'PASS' if ok else 'STOP'} - {msg}", flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\nGATE: {'PASS' if ok else 'STOP'} - {msg}\n")
    if not ok or args.skip_production:
        print(f"stopping before production. total {(time.time()-t0)/60:.0f} min")
        return

    # 4. the frozen model on the held-out 100, reported once
    run(f"export eval-100 {TRAIN_YEAR}", "scripts.export_patches",
        "--split", "eval", "--year", str(TRAIN_YEAR), "--fit",
        "--labels", LABELS, "--workers", "6", required=False)
    run("score frozen model on held-out 100", "scripts.eval_geometry",
        "--models", MODEL, "--dirs", f"heldout=eval_{TRAIN_YEAR}_fit_{LABELS}",
        required=False)

    # 5. apply the frozen model across the series
    for year in YEARS:
        n = seed_fit_dirs(year)
        print(f"  {year}: reused {n} existing 128 px patches", flush=True)
        run(f"export all/{year} fitted", "scripts.export_patches",
            "--split", "all", "--year", str(year), "--fit", "--workers", "6")

    backups = {}
    for name in ("lake_timeseries.json", "lake_change.json"):
        src = DATA / name
        if src.exists():
            bak = DATA / name.replace(".json", "_prefit.bak.json")
            shutil.copy2(src, bak)
            backups[src] = bak
    try:
        run("inference 2019-2025", "scripts.run_inference", "--model", MODEL, "--fit")
        run("change detection", "scripts.detect_change", "--min-years", "4")
        run("compare change vs KV epochs", "scripts.compare_change", required=False)
        for name in ("lake_timeseries.json", "lake_change.json"):
            src = DATA / name
            if src.exists():
                shutil.copy2(src, DATA / name.replace(".json", "_fit.json"))
    finally:
        for src, bak in backups.items():
            shutil.copy2(bak, src)

    print(f"\nDONE in {(time.time()-t0)/60:.0f} min")
    print("new results : lake_timeseries_fit.json / lake_change_fit.json")
    print("current production untouched - nothing promoted automatically")


if __name__ == "__main__":
    main()
