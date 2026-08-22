"""Compare the change our model measures against the change the KV epochs show.

Our trend runs 2019 -> 2025 (6 yr); KV runs 2016-17 -> 2022-24 (~6.5 yr between
epoch midpoints). The spans are close enough to compare rates, so both are
converted to %/yr rather than compared as raw totals.

Only KV-vs-KV is a valid reference change: both epochs come from one pipeline.
Hi-MAG is never differenced against KV - different detection criteria would show
up as growth that never happened.

The size split is the point of this script. If the fixed 1.28 km window truncates
large lakes, their growth is mechanically clipped, and our rate should fall below
the reference's specifically in the big bins while matching in the small ones.
That is a directional prediction, and it is testable here.

Usage:
    cd server
    python -m scripts.compare_change
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from scripts.compare_labels import EPOCHS, load_reference, match

DATA = Path(__file__).resolve().parent.parent / "data"
OUR_YEARS = 2025 - 2019            # 6
KV_YEARS = 6.5                     # 2016.5 -> 2023.0, epoch midpoints
BINS = [(0, 15), (15, 25), (25, 50), (50, 100), (100, 1e9)]


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=750.0)
    ap.add_argument("--min-r2", type=float, default=0.0,
                    help="restrict to lakes whose trend beats this R^2")
    args = ap.parse_args()

    ch = json.loads((DATA / "lake_change.json").read_text(encoding="utf-8"))
    lakes = [L for L in ch["lakes"] if not L.get("suspect")]
    print(f"our lakes (clean): {len(lakes)}  model '{ch['model']}'")

    m16 = match(lakes, load_reference(EPOCHS["kv_2016_2017"]), args.tol)
    m22 = match(lakes, load_reference(EPOCHS["kv_2022_2024"]), args.tol)
    common = set(m16) & set(m22)
    by = {L["id"]: L for L in lakes}

    rows = []
    for k in common:
        a, b = m16[k]["area_ha"], m22[k]["area_ha"]
        L = by[k]
        if a <= 0 or L["area_first_ha"] <= 0:
            continue
        if L.get("r2", 0) < args.min_r2:
            continue
        rows.append({
            "id": k,
            "ref_ha": b,
            "ours_pct_yr": (L["pct_change"] / OUR_YEARS),
            "kv_pct_yr": (100 * (b - a) / a) / KV_YEARS,
            "ours_abs_ha": L["abs_change_ha"] / OUR_YEARS,
            "kv_abs_ha": (b - a) / KV_YEARS,
        })

    print(f"lakes in both inventories: {len(rows)}"
          f"{f' (R^2 >= {args.min_r2})' if args.min_r2 else ''}\n")

    o = [r["ours_pct_yr"] for r in rows]
    k = [r["kv_pct_yr"] for r in rows]
    agree = sum(1 for r in rows if (r["ours_pct_yr"] > 0) == (r["kv_pct_yr"] > 0))
    print("OVERALL rate of change (%/yr)")
    print(f"  ours    median {st.median(o):+.3f}   mean {st.mean(o):+.3f}")
    print(f"  KV      median {st.median(k):+.3f}   mean {st.mean(k):+.3f}")
    print(f"  ratio of medians          {st.median(o)/st.median(k):.2f}"
          if st.median(k) else "")
    print(f"  sign agreement            {100*agree/len(rows):.0f}%")
    print(f"  correlation (Pearson r)   {pearson(o, k):.3f}")

    ao = sum(r["ours_abs_ha"] for r in rows)
    ak = sum(r["kv_abs_ha"] for r in rows)
    print(f"  aggregate ha/yr           ours {ao:+.1f}   KV {ak:+.1f}"
          f"   ({100*ao/ak:.0f}% of reference)" if ak else "")

    print("\nBY REFERENCE LAKE SIZE - the truncation test")
    print(f"  {'size (ha)':<12}{'n':>5}{'ours %/yr':>12}{'KV %/yr':>11}"
          f"{'ours/KV':>10}{'sign agree':>12}")
    print("  " + "-" * 62)
    out_bins = []
    for lo, hi in BINS:
        g = [r for r in rows if lo <= r["ref_ha"] < hi]
        if len(g) < 3:
            continue
        lab = f"{lo}-{hi:g}" if hi < 1e9 else f"{lo}+"
        mo, mk = st.median(r["ours_pct_yr"] for r in g), st.median(r["kv_pct_yr"] for r in g)
        sa = 100 * sum(1 for r in g if (r["ours_pct_yr"] > 0) == (r["kv_pct_yr"] > 0)) / len(g)
        ratio = (mo / mk) if mk else float("nan")
        print(f"  {lab:<12}{len(g):>5}{mo:>+12.3f}{mk:>+11.3f}{ratio:>10.2f}{sa:>11.0f}%")
        out_bins.append({"bin": lab, "n": len(g), "ours_pct_yr": round(mo, 3),
                         "kv_pct_yr": round(mk, 3), "ratio": round(ratio, 2),
                         "sign_agree_pct": round(sa)})

    big = [r for r in rows if r["ref_ha"] >= 50]
    small = [r for r in rows if r["ref_ha"] < 50]
    if big and small:
        print(f"\n  >=50 ha  ({len(big)} lakes): ours {st.median(r['ours_pct_yr'] for r in big):+.3f}"
              f"  vs KV {st.median(r['kv_pct_yr'] for r in big):+.3f} %/yr")
        print(f"  < 50 ha  ({len(small)} lakes): ours {st.median(r['ours_pct_yr'] for r in small):+.3f}"
              f"  vs KV {st.median(r['kv_pct_yr'] for r in small):+.3f} %/yr")

    res = {"n": len(rows), "ours_median_pct_yr": round(st.median(o), 3),
           "kv_median_pct_yr": round(st.median(k), 3),
           "sign_agreement_pct": round(100 * agree / len(rows)),
           "pearson_r": round(pearson(o, k), 3),
           "aggregate_ha_per_yr": {"ours": round(ao, 1), "kv": round(ak, 1)},
           "by_size": out_bins}
    dest = DATA / "_intermediate" / "change_vs_kv.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
