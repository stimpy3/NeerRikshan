"""Decide a patch window size per lake, sized to the lake's actual extent.

The fixed 128 px (1.28 km) window truncates large lakes: 100% of lakes over
100 ha touch the window edge with only 57% of their area inside, so their growth
cannot be measured at all. Segmentation on those lakes is excellent
(IoU 0.976) - the defect is the frame, not the model.

Sizing on sqrt(area) is not enough. Glacial lakes are elongated: actual extent
runs a median 1.71x, and up to 3.61x, the square-equivalent side. Area-based
sizing still truncated 11 lakes, including a 13.9 ha lake 1780 m long. So the
window is sized from the polygon's real bounding extent instead.

Extent is taken as the max over Hi-MAG 2017 AND KV 2022-24 outlines, so the
window fits the lake at its LARGEST known size and leaves room to grow - a
window fitted to the 2017 outline would re-introduce truncation by 2025.

Writes data/window_sizes.json: {lake_id: px}. Both export and inference read
this file, so the two cannot silently disagree about geometry.

Usage:
    cd server
    python -m scripts.plan_windows
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SCALE = 10          # m/px
STEP = 32           # windows are multiples of 32 so 3 downsamples divide evenly
MIN_PX = 128
MAX_PX = 640        # 6.4 km - covers the largest lake in the set (5585 m across)
MARGIN = 1.5        # window spans 1.5x the lake's longest axis


def himag_extent_m(lake: dict) -> float:
    xs = [p[0] for p in lake["ring"]]
    ys = [p[1] for p in lake["ring"]]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def kv_extent_m(polys: list) -> float:
    ext = 0.0
    for poly in polys:
        ring = poly[0]
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        mid = sum(lats) / len(lats)
        ext = max(ext,
                  (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(mid)),
                  (max(lats) - min(lats)) * 110_540)
    return ext


def size_for(extent_m: float, margin: float = MARGIN) -> int:
    want = extent_m * margin / SCALE
    return max(MIN_PX, min(MAX_PX, int(math.ceil(want / STEP) * STEP)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()

    doc = json.loads((DATA / "himag_all.json").read_text(encoding="utf-8"))
    kv_path = DATA / "kv_rings_2022_2024.json"
    kv = json.loads(kv_path.read_text(encoding="utf-8")) if kv_path.exists() else {}
    if not kv:
        print("WARNING: no KV rings found - sizing on Hi-MAG 2017 only, which "
              "leaves no room for growth since 2017")

    sizes, extents, truncated = {}, {}, []
    for L in doc["lakes"]:
        e = max(himag_extent_m(L), kv_extent_m(kv.get(L["id"], [])))
        px = size_for(e, args.margin)
        sizes[L["id"]] = px
        extents[L["id"]] = round(e, 1)
        if e > px * SCALE:
            truncated.append((L["id"], round(e), px * SCALE))

    dest = DATA / "window_sizes.json"
    dest.write_text(json.dumps({
        "scale_m": SCALE, "margin": args.margin,
        "min_px": MIN_PX, "max_px": MAX_PX,
        "note": "extent = max(Hi-MAG 2017, KV 2022-24) longest axis, metres",
        "extent_m": extents, "sizes": sizes,
    }), encoding="utf-8")

    c = Counter(sizes.values())
    print(f"{'window':<12}{'covers':>9}{'lakes':>8}")
    print("-" * 29)
    for px in sorted(c):
        print(f"{px}x{px:<7}{px*SCALE/1000:>7.2f}km{c[px]:>8}")
    reuse = c.get(MIN_PX, 0)
    print(f"\n{len(sizes)} lakes")
    print(f"  still {MIN_PX}px (existing patches reusable): {reuse} ({100*reuse/len(sizes):.0f}%)")
    print(f"  need a new export                          : {len(sizes)-reuse}")
    print(f"  still truncated                            : {len(truncated)}")
    if truncated:
        print(f"    {truncated[:5]}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
