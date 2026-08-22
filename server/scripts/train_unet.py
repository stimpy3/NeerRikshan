"""Train a small U-Net to segment glacial lakes, with terrain as input.

Why this exists: the MNDWI baseline scores median IoU ~0.51 on the held-out
evaluation set, failing by extending contiguously past the shoreline. MNDWI is
(B3-B11)/(B3+B11), which is algebraically identical to NDSI, the standard snow
index - so it cannot separate snow and ice from water at all. In glacial
terrain snow and ice sit directly against every lake shore, which is exactly
the observed failure.

Two signals the baseline cannot use, and this model can:
  * NIR (B8) - snow reflects strongly, water absorbs nearly everything
  * slope    - a lake surface is flat, a shadowed valley wall is not

The --bands flag supports an ablation so the terrain contribution can be
isolated rather than assumed:
    all      optical + MNDWI + elevation/slope/hillshade
    noterr   optical + MNDWI only
    optical  optical bands only

Usage:
    cd server
    python -m scripts.train_unet --bands all --epochs 40
    python -m scripts.train_unet --bands noterr --epochs 40
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DATA = Path(__file__).resolve().parent.parent / "data"
PATCHES = DATA / "patches"
MODELS = DATA / "models"

FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "mndwi", "elev", "slope", "hillshade"]

# Derived indices, computed on the fly from bands already in every patch so no
# re-export is needed. B3=green(1) B4=red(2) B8=NIR(3) B11=SWIR(4).
#
#   NDWI = (green - NIR) / (green + NIR)
#   NDVI = (NIR - red)   / (NIR + red)
#
# NOTE: NDSI is deliberately absent - NDSI = (green - SWIR)/(green + SWIR), which
# is byte-for-byte the existing `mndwi` band. Adding it would duplicate a column,
# not add information. Measured snow/water separation on 126 patches:
#   MNDWI 0.154 (snow scores HIGHER than water - the shortcut the model took)
#   NDWI  0.462 (the index that actually separates them)
#   NDVI  0.208
DERIVED = ["ndwi", "ndvi"]


def derive(x: np.ndarray) -> np.ndarray:
    """Append NDWI and NDVI channels to a (C, H, W) patch."""
    eps = 1e-6
    green, red, nir = x[1], x[2], x[3]
    ndwi = (green - nir) / (green + nir + eps)
    ndvi = (nir - red) / (nir + red + eps)
    return np.concatenate([x, ndwi[None], ndvi[None]], axis=0)


# Indices are into the DERIVED-extended stack: 0-9 raw, 10=ndwi, 11=ndvi.
BAND_SETS = {
    "optical": list(range(6)),                    # A: optical only
    "noterr": list(range(7)),                     # B: optical + MNDWI
    "ndwi": list(range(6)) + [10, 11],            # C: optical + NDWI + NDVI
    "allidx": list(range(7)) + [10, 11],          # D: optical + MNDWI + NDWI + NDVI
    "all": list(range(10)),                       # legacy: optical + MNDWI + terrain
}


CROP = 128  # all training samples are emitted at this size so they can batch


def rescale(x: np.ndarray, y: np.ndarray, zoom: float, out: int):
    """Downsample by `zoom`, then centre-crop/pad to `out` pixels.

    zoom > 1 shrinks the lake within the frame. Training across a range of zooms
    is what makes the model scale-robust: a fully convolutional net will RUN at
    any input size, but it still learns the lake-to-frame ratio it was shown, and
    inferring at a different ratio collapsed IoU by 0.35 on large lakes.
    """
    import torch.nn.functional as FF
    t = torch.from_numpy(x)[None]
    l = torch.from_numpy(y)[None][None]
    if abs(zoom - 1.0) > 1e-3:
        h = max(out, int(round(x.shape[1] / zoom)))
        t = FF.interpolate(t, size=(h, h), mode="bilinear", align_corners=False)
        l = FF.interpolate(l, size=(h, h), mode="nearest")
    t, l = t[0].numpy(), l[0, 0].numpy()

    c, h, w = t.shape
    xo = np.zeros((c, out, out), dtype=np.float32)
    yo = np.zeros((out, out), dtype=np.float32)
    sh, sw = max(0, (h - out) // 2), max(0, (w - out) // 2)
    tc, lc = t[:, sh:sh + out, sw:sw + out], l[sh:sh + out, sw:sw + out]
    dh, dw = (out - tc.shape[1]) // 2, (out - tc.shape[2]) // 2
    xo[:, dh:dh + tc.shape[1], dw:dw + tc.shape[2]] = tc
    yo[dh:dh + lc.shape[0], dw:dw + lc.shape[1]] = lc
    return xo, yo


def crop_native(x: np.ndarray, y: np.ndarray, out: int, rng,
                water_bias: float = 0.7, centre: bool = False):
    """Random crop at the NATIVE 10 m scale - no resampling, no zoom.

    This is the augmentation `rescale` could not provide. `rescale` only ever
    shrinks the lake within the frame (zoom 1.0-2.5), but windows fitted to the
    lake do the opposite: the lake fills ~2/3 of the frame. Training only ever
    saw small-lake-in-big-frame, so inferring on a fitted window was out of
    distribution and collapsed IoU by 0.35 on large lakes.

    Cropping a fitted patch at native scale produces exactly the regimes that
    were missing - crops from a large lake's interior are entirely water, which
    a fixed 128 px window centred on a small lake never shows.
    """
    c, h, w = x.shape
    if h < out or w < out:                      # pad small patches up to `out`
        H, W = max(h, out), max(w, out)
        xo = np.zeros((c, H, W), dtype=np.float32)
        yo = np.zeros((H, W), dtype=np.float32)
        xo[:, :h, :w], yo[:h, :w] = x, y
        x, y, h, w = xo, yo, H, W

    if centre:
        top, left = (h - out) // 2, (w - out) // 2
    else:
        ys, xs = np.nonzero(y > 0.5)
        if len(ys) and rng.random() < water_bias:
            # keep most crops on the lake, else the big windows are mostly land
            k = int(rng.integers(len(ys)))
            j = out // 4
            top = int(np.clip(ys[k] - out // 2 + rng.integers(-j, j + 1), 0, h - out))
            left = int(np.clip(xs[k] - out // 2 + rng.integers(-j, j + 1), 0, w - out))
        else:
            top = int(rng.integers(0, h - out + 1))
            left = int(rng.integers(0, w - out + 1))
    return (x[:, top:top + out, left:left + out].copy(),
            y[top:top + out, left:left + out].copy())


class Patches(Dataset):
    def __init__(self, split: str, year: int, idx: list[int], augment: bool,
                 adaptive: bool = False, multiscale: bool = False,
                 fit: bool = False, crop: int = CROP, labels: str = "himag"):
        suffix = "_fit" if fit else ("_adapt" if adaptive else "")
        if labels != "himag":
            suffix += f"_{labels}"
        self.files = sorted(glob.glob(str(PATCHES / f"{split}_{year}{suffix}" / "*.npz")))
        if not self.files:
            raise SystemExit(f"no patches in {PATCHES / f'{split}_{year}{suffix}'} - run export_patches first")
        self.idx, self.augment = idx, augment
        self.multiscale = multiscale
        self.fit, self.crop = fit, crop
        self.rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        rec = np.load(self.files[i], allow_pickle=True)
        x = derive(rec["x"].astype(np.float32))[self.idx]
        y = rec["y"].astype(np.float32)
        if self.fit:
            x, y = crop_native(x, y, self.crop, self.rng, centre=not self.augment)
        elif self.multiscale:
            zoom = float(np.random.uniform(1.0, 2.5)) if self.augment else 1.0
            x, y = rescale(x, y, zoom, CROP)
        elif x.shape[1] != CROP:
            x, y = rescale(x, y, 1.0, CROP)
        y = y[None]
        if self.augment:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1].copy(), y[:, :, ::-1].copy()
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :].copy(), y[:, ::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x, y = np.rot90(x, k, (1, 2)).copy(), np.rot90(y, k, (1, 2)).copy()
        return torch.from_numpy(x), torch.from_numpy(y)


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin: int, width: int = 24):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = block(cin, w), block(w, w * 2), block(w * 2, w * 4)
        self.bott = block(w * 4, w * 8)
        self.u3 = nn.ConvTranspose2d(w * 8, w * 4, 2, 2)
        self.d3 = block(w * 8, w * 4)
        self.u2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2)
        self.d2 = block(w * 4, w * 2)
        self.u1 = nn.ConvTranspose2d(w * 2, w, 2, 2)
        self.d1 = block(w * 2, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def dice_term(p, y):
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1.0) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1.0))


def focal_term(logits, y, gamma: float = 2.0, alpha: float = 0.25):
    """Down-weights easy pixels so the loss concentrates on hard ones.

    Relevant here because the remaining errors are concentrated at shorelines and
    snow edges - pixels the model is already nearly right about dominate a plain
    BCE and drown out the ones that actually matter.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * y + (1 - p) * (1 - y)          # prob assigned to the true class
    w = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt).pow(gamma) * bce).mean()


def boundary_term(p, y):
    """Penalises disagreement near the label edge.

    Targets the measured failure: 72% of false positives sit within 30 m of the
    true shoreline. A morphological gradient marks the boundary band, and error
    inside that band is weighted up.
    """
    pad = F.max_pool2d(y, 3, 1, 1)
    edge = (pad - (-F.max_pool2d(-y, 3, 1, 1)))   # dilation - erosion
    denom = edge.sum() + 1.0
    return ((p - y).abs() * edge).sum() / denom


def loss_fn(logits, y, mode: str = "bce_dice"):
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    dice = dice_term(p, y).mean()
    if mode == "bce_dice":
        return bce + dice
    if mode == "focal":
        return bce + dice + focal_term(logits, y)
    if mode == "boundary":
        return bce + dice + boundary_term(p, y)
    raise ValueError(f"unknown loss mode {mode}")


@torch.no_grad()
def evaluate(model, loader, thr: float = 0.5):
    model.eval()
    ti = tu = tp_ = fp_ = fn_ = 0.0
    for x, y in loader:
        p = (torch.sigmoid(model(x)) > thr).float()
        ti += (p * y).sum().item()
        tu += ((p + y) > 0).float().sum().item()
        tp_ += (p * y).sum().item()
        fp_ += (p * (1 - y)).sum().item()
        fn_ += ((1 - p) * y).sum().item()
    return {
        "iou": ti / tu if tu else 0.0,
        "precision": tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0,
        "recall": tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="all", choices=sorted(BAND_SETS))
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--adaptive", action="store_true",
                    help="read *_adapt patch dirs (lake-sized windows)")
    ap.add_argument("--multiscale", action="store_true",
                    help="random zoom augmentation so the model is scale-robust")
    ap.add_argument("--loss", default="bce_dice",
                    choices=["bce_dice", "focal", "boundary"])
    ap.add_argument("--fit", action="store_true",
                    help="read *_fit patch dirs (windows sized to real lake "
                         "extent) and train on native-scale random crops")
    ap.add_argument("--crop", type=int, default=CROP,
                    help="training crop size in px when --fit (must divide by 8)")
    ap.add_argument("--labels", default="himag", choices=["himag", "kv16", "kv22"],
                    help="which inventory supplied the labels in the patch dirs")
    args = ap.parse_args()
    if args.crop % 8:
        raise SystemExit("--crop must be a multiple of 8 (3 downsample levels)")

    torch.manual_seed(0)
    np.random.seed(0)
    idx = BAND_SETS[args.bands]
    names = FEATURES + DERIVED
    print(f"bands={args.bands} -> {[names[i] for i in idx]}")

    tr = DataLoader(Patches("train", args.year, idx, True, args.adaptive,
                            args.multiscale, args.fit, args.crop, args.labels),
                    batch_size=args.batch, shuffle=True, num_workers=0)
    va = DataLoader(Patches("val", args.year, idx, False, args.adaptive,
                            args.multiscale, args.fit, args.crop, args.labels),
                    batch_size=args.batch, num_workers=0)
    print(f"train={len(tr.dataset)} val={len(va.dataset)} patches")

    model = UNet(len(idx), args.width)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"params={n_par/1e6:.2f}M")

    tag = args.bands if args.loss == "bce_dice" else f"{args.bands}_{args.loss}"
    if args.multiscale:
        tag += "_ms"
    if args.fit:
        tag += "_fit"
    if args.labels != "himag":
        tag += f"_{args.labels}"
    MODELS.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, t0 = 0.0, time.time()
        for x, y in tr:
            opt.zero_grad()
            l = loss_fn(model(x), y, args.loss)
            l.backward()
            opt.step()
            tot += l.item()
        sched.step()
        m = evaluate(model, va)
        hist.append({"epoch": ep, "loss": tot / len(tr), **m})
        flag = ""
        if m["iou"] > best:
            best = m["iou"]
            torch.save({"state": model.state_dict(), "bands": args.bands,
                        "idx": idx, "width": args.width}, MODELS / f"unet_{tag}.pt")
            flag = "  *best"
        print(f"ep{ep:>3} loss {tot/len(tr):.4f}  val IoU {m['iou']:.4f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f}  {time.time()-t0:.0f}s{flag}",
              flush=True)

    (MODELS / f"history_{tag}.json").write_text(
        json.dumps({"args": vars(args), "history": hist}, indent=1), encoding="utf-8")
    print(f"\nbest val IoU {best:.4f} -> {MODELS / f'unet_{tag}.pt'}")


if __name__ == "__main__":
    main()
