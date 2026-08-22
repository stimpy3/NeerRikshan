"""Render per-lake, per-year RGB thumbnails so change can be seen, not just plotted.

A trend line says a lake grew. It does not let anyone check whether the model
was looking at water, cloud, or shadow. These thumbnails make each year's
evidence inspectable.

No new Earth Engine calls: every exported patch already carries B4/B3/B2, so
the true-colour image is reconstructed locally from data on disk.

Two variants per lake-year:
    <id>_<year>.png       true colour
    <id>_<year>_ov.png    true colour with the model's water outline in red

The overlay matters more than it looks - it turns "trust the number" into
"see what the model actually segmented", which is the difference between a
demo and something someone can audit.

Usage:
    cd server
    python -m scripts.render_thumbnails --top 60
    python -m scripts.render_thumbnails --all --size 320
"""

from __future__ import annotations

import argparse
import glob
import re
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.run_inference import load_model
from scripts.train_unet import derive

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
THUMBS = DATA / "thumbnails"

# Percentile stretch: glacial scenes mix near-black water with bright snow, so
# a fixed 0-1 scale renders the lake as an unreadable dark blob.
LO, HI = 2, 98


def to_rgb(x: np.ndarray, size: int) -> Image.Image:
    rgb = np.stack([x[2], x[1], x[0]], axis=-1)  # B4, B3, B2
    valid = rgb[np.isfinite(rgb).all(axis=-1) & (rgb.sum(axis=-1) > 0)]
    if valid.size:
        lo, hi = np.percentile(valid, LO), np.percentile(valid, HI)
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((rgb - lo) / (hi - lo), 0, 1)
    img = Image.fromarray((out * 255).astype(np.uint8))
    return img.resize((size, size), Image.NEAREST)


def outline(mask: np.ndarray) -> np.ndarray:
    """1-px boundary of a binary mask, without pulling in scipy."""
    m = mask.astype(bool)
    edge = np.zeros_like(m)
    edge[:-1, :] |= m[:-1, :] ^ m[1:, :]
    edge[:, :-1] |= m[:, :-1] ^ m[:, 1:]
    return edge & m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ndwi_fit_kv22")
    ap.add_argument("--fit", action="store_true",
                    help="read *_fit patch dirs (windows sized to lake extent)")
    ap.add_argument("--split", default="all")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--top", type=int, default=60,
                    help="render only the N highest-priority lakes")
    ap.add_argument("--all", action="store_true", help="render every lake")
    args = ap.parse_args()

    change = json.loads((DATA / "lake_change.json").read_text(encoding="utf-8"))
    rows = change["lakes"]
    for r in rows:
        s = r.get("slope_ha_per_yr") or 0.0
        r["_p"] = s * (r.get("r2") or 0.0) if s > 0 else 0.0
    if args.all:
        wanted = {r["id"] for r in rows}
    else:
        wanted = {r["id"] for r in sorted(rows, key=lambda r: -r["_p"])[: args.top]}

    net, idx = load_model(args.model)
    THUMBS.mkdir(parents=True, exist_ok=True)
    made = skipped = 0

    pattern = f"{args.split}_*_fit" if args.fit else f"{args.split}_[0-9][0-9][0-9][0-9]"
    for d in sorted(glob.glob(str(PATCHES / pattern))):
        # dir names are all_2019 or all_2019_fit - take the 4-digit year, not
        # the last token, or every fitted year lands in one file called "_fit"
        m = re.search(r"_(\d{4})", Path(d).name)
        if not m:
            continue
        year = m.group(1)
        for f in sorted(glob.glob(str(Path(d) / "*.npz"))):
            lake_id = Path(f).stem
            if lake_id not in wanted:
                continue
            dest = THUMBS / f"{lake_id}_{year}.png"
            dest_ov = THUMBS / f"{lake_id}_{year}_ov.png"
            if dest.exists() and dest_ov.exists():
                skipped += 1
                continue
            rec = np.load(f, allow_pickle=True)
            x = rec["x"].astype(np.float32)
            img = to_rgb(x, args.size)
            img.save(dest, optimize=True)

            with torch.no_grad():
                prob = torch.sigmoid(
            net(torch.from_numpy(derive(x)[idx])[None]))[0, 0].numpy()
            edge = outline(prob > 0.5)
            edge_img = Image.fromarray((edge * 255).astype(np.uint8)).resize(
                (args.size, args.size), Image.NEAREST)
            ov = np.array(img).copy()
            e = np.array(edge_img) > 127
            ov[e] = [255, 60, 40]
            Image.fromarray(ov).save(dest_ov, optimize=True)
            made += 1
        print(f"  {year}: {made} rendered", flush=True)

    print(f"\n{made} thumbnails rendered, {skipped} already existed -> {THUMBS}")


if __name__ == "__main__":
    main()
