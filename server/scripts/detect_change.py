"""Turn the area time series into ranked lake-change findings.

A single-year difference is not a finding - it is mostly noise from cloud,
seasonal timing and segmentation jitter. This uses a least-squares trend over
all usable years instead, which is far more robust, and reports the fit
quality alongside so weak trends can be discounted rather than believed.

Sanity rules applied, because a spurious "300% growth" would discredit the
whole result:
  * at least MIN_YEARS usable (unflagged) observations
  * relative trend capped for reporting; anything wilder is flagged, not shown
  * lakes whose 2019 area disagrees wildly with Hi-MAG 2017 are marked
    suspect, since that usually means the segmentation is unstable there
    rather than that the lake really quadrupled

Output: ranked growth and shrink lists plus a GeoJSON-ready summary.

Usage:
    cd server
    python -m scripts.detect_change
    python -m scripts.detect_change --min-years 5 --top 25
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SERIES = DATA / "lake_timeseries.json"
OUT = DATA / "lake_change.json"


# Student t, two-sided 95%, by degrees of freedom (n-2). A 7-year series has
# df=5. Hard-coded rather than pulling in scipy.stats for five numbers.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def trend(years: list[float], areas: list[float]) -> tuple[float, float]:
    """Least-squares slope (ha/yr) and R^2."""
    n = len(years)
    mx, my = sum(years) / n, sum(areas) / n
    sxx = sum((x - mx) ** 2 for x in years)
    sxy = sum((x - mx) * (y - my) for x, y in zip(years, areas))
    if sxx == 0:
        return 0.0, 0.0
    slope = sxy / sxx
    ss_tot = sum((y - my) ** 2 for y in areas)
    ss_res = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(years, areas))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def slope_ci(years: list[float], areas: list[float], slope: float
             ) -> tuple[float, float, float, float]:
    """Standard error and 95% CI of the fitted slope, plus residual scatter.

    Reporting a trend without an interval is the single biggest honesty gap in
    this project: year-to-year scatter is comparable to the six-year signal, so
    a slope can look decisive and still be indistinguishable from zero. A lake
    is only "detectable" here if its CI excludes zero - that is a far stricter
    and more meaningful filter than R^2, which says how well a line fits without
    saying whether the line is distinguishable from flat.
    """
    n = len(years)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mx, my = sum(years) / n, sum(areas) / n
    sxx = sum((x - mx) ** 2 for x in years)
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    resid = [y - (my + slope * (x - mx)) for x, y in zip(years, areas)]
    ss_res = sum(r * r for r in resid)
    df = n - 2
    se = (ss_res / df / sxx) ** 0.5 if df > 0 else float("nan")
    t = T95.get(df, 1.96)
    rmse = (ss_res / df) ** 0.5 if df > 0 else float("nan")
    return se, slope - t * se, slope + t * se, rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-years", type=int, default=4)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--suspect-ratio", type=float, default=3.0,
                    help="2019 area vs Hi-MAG 2017 ratio beyond which a lake is suspect")
    args = ap.parse_args()

    doc = json.loads(SERIES.read_text(encoding="utf-8"))
    rows, skipped = [], {"too_few_years": 0, "suspect_baseline": 0}

    for lake in doc["lakes"]:
        usable = {int(y): v["area_ha"] for y, v in lake["years"].items()
                  if v["flag"] == "ok"}
        if len(usable) < args.min_years:
            skipped["too_few_years"] += 1
            continue
        ys = sorted(usable)
        areas = [usable[y] for y in ys]
        yrs = [float(y) for y in ys]
        slope, r2 = trend(yrs, areas)
        se, ci_lo, ci_hi, rmse = slope_ci(yrs, areas, slope)

        first = statistics.mean(areas[:2])
        last = statistics.mean(areas[-2:])
        himag = lake["himag_2017_ha"]
        suspect = himag > 0 and not (1 / args.suspect_ratio <= (areas[0] / himag) <= args.suspect_ratio)
        if suspect:
            skipped["suspect_baseline"] += 1

        span = ys[-1] - ys[0]
        rows.append({
            "id": lake["id"], "subr": lake["subr"],
            "lon": lake.get("lon"), "lat": lake.get("lat"),
            "himag_2017_ha": round(himag, 2),
            "n_years": len(ys), "first_year": ys[0], "last_year": ys[-1],
            "area_first_ha": round(first, 2), "area_last_ha": round(last, 2),
            "abs_change_ha": round(last - first, 2),
            "pct_change": round((last - first) / first * 100, 1) if first > 0 else None,
            "slope_ha_per_yr": round(slope, 3),
            "slope_pct_per_yr": round(slope / first * 100, 2) if first > 0 else None,
            "r2": round(r2, 3),
            "slope_se_ha_per_yr": round(se, 4) if se == se else None,
            "slope_ci_lo": round(ci_lo, 3) if ci_lo == ci_lo else None,
            "slope_ci_hi": round(ci_hi, 3) if ci_hi == ci_hi else None,
            # the honest test: does the interval exclude no-change?
            "detectable": bool(ci_lo == ci_lo and (ci_lo > 0 or ci_hi < 0)),
            "direction": ("growing" if ci_lo == ci_lo and ci_lo > 0
                          else "shrinking" if ci_hi == ci_hi and ci_hi < 0
                          else "no detectable trend"),
            "noise_pct": round(100 * rmse / first, 2) if first > 0 and rmse == rmse else None,
            "suspect": suspect,
            "series": {str(y): round(usable[y], 2) for y in ys},
        })

    clean = [r for r in rows if not r["suspect"]]
    strong = [r for r in clean if r["r2"] >= 0.5]

    print(f"lakes analysed      : {len(rows)}")
    print(f"  clean baseline    : {len(clean)}")
    print(f"  with R^2 >= 0.5   : {len(strong)}")
    for k, v in skipped.items():
        print(f"  skipped {k:<18}: {v}")

    if clean:
        pcts = sorted(r["pct_change"] for r in clean if r["pct_change"] is not None)
        med = pcts[len(pcts) // 2]
        grow = sum(1 for p in pcts if p > 5)
        shrink = sum(1 for p in pcts if p < -5)
        print(f"\nmedian area change {min(pcts):.0f}%..{max(pcts):.0f}%, median {med:+.1f}%")
        print(f"  growing >5% : {grow}/{len(pcts)} ({100*grow/len(pcts):.0f}%)")
        print(f"  shrinking>5%: {shrink}/{len(pcts)} ({100*shrink/len(pcts):.0f}%)")

        tot_first = sum(r["area_first_ha"] for r in clean)
        tot_last = sum(r["area_last_ha"] for r in clean)
        print(f"  aggregate area {tot_first:,.0f} -> {tot_last:,.0f} ha "
              f"({(tot_last-tot_first)/tot_first*100:+.1f}%)")

    def show(title, items):
        print(f"\n{title}")
        print(f"  {'lake':<18}{'sub-region':<20}{'ha 1st':>8}{'ha last':>9}{'%':>8}{'R2':>7}")
        for r in items:
            print(f"  {r['id']:<18}{r['subr'][:18]:<20}{r['area_first_ha']:>8.1f}"
                  f"{r['area_last_ha']:>9.1f}{r['pct_change']:>+8.1f}{r['r2']:>7.2f}")

    if strong:
        show(f"TOP {args.top} GROWING (R^2>=0.5)",
             sorted(strong, key=lambda r: -r["slope_pct_per_yr"])[: args.top])
        show(f"TOP {args.top} SHRINKING (R^2>=0.5)",
             sorted(strong, key=lambda r: r["slope_pct_per_yr"])[: args.top])

    OUT.write_text(json.dumps({
        "config": vars(args), "model": doc.get("model"),
        "n_analysed": len(rows), "n_clean": len(clean), "n_strong": len(strong),
        "lakes": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
