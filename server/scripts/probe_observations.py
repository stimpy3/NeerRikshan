"""FEASIBILITY PROBE - run this before writing any pipeline code.

Question it answers: do we have enough clear Sentinel-2 looks at an individual
bheri, in an individual calendar month, to build a per-pond monthly climatology?

The binding constraint on this project is not compute - it is clear
observations. Kolkata's monsoon (Jun-Sep) is heavily clouded, and L2A surface
reflectance coverage over India is patchy before ~2019. Both of those are
assumptions until measured. This script measures them.

It deliberately does NOT derive the real pond units (that comes later, and
depends on a tuned MNDWI threshold). It samples pond-sized regions from JRC
Global Surface Water instead, which is independent of anything we have yet to
tune, so the observation counts are not circular.

Output: a per-calendar-month verdict of usable / pool-with-neighbours / skip,
plus the raw per-observation table cached for local re-analysis.

Usage:
    cd server
    python -m scripts.probe_observations
    python -m scripts.probe_observations --ponds 30 --start-year 2017
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ee
import requests

from scripts.gee_auth import initialize_or_exit

# --- Constants. Every one of these belongs in the manifest if it ships. -------

WDPA_ID = 900772  # East Calcutta Wetlands (WDPA SITE_ID; point-only record)
# Approximate EKW extent. Probe-only - see aoi() for why this is acceptable here.
PROBE_BBOX = [88.38, 22.40, 88.58, 22.60]  # [west, south, east, north]
S2 = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
GSW = "JRC/GSW1_4/GlobalSurfaceWater"

CS_BAND = "cs"
CS_THRESHOLD = 0.60  # per-pixel Cloud Score+ cutoff for "clear"
POND_CLEAR_FRACTION = 0.70  # >=70% of a pond's pixels clear => usable observation
SCALE = 10  # Sentinel-2 native resolution

# Sampling regions: JRC occurrence >= this % counts as reliably-water historically.
GSW_OCCURRENCE = 70
MIN_POND_HA = 2.0  # big enough that a 10 m grid gives a meaningful pixel count

# Decision gates (see plan section 2.2a).
GATE_USABLE = 5  # >= this many year-samples in a calendar month => usable
GATE_POOL = 3  # >= this => usable only when pooled with month +/- 1

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "_intermediate"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def aoi() -> ee.Geometry:
    """Area of interest for the probe.

    NOTE: WDPA stores East Calcutta Wetlands (SITE_ID 900772) as a POINT, not a
    polygon - verified against WCMC/WDPA/current/{polygons,points}. There is no
    digitized legal boundary in WDPA, and OSM has no site-level relation either.
    Sourcing the real Ramsar boundary is an open task before the main pipeline.

    It does not block this probe. The probe measures cloud/observation
    availability, which does not vary meaningfully across a 20 km box, and it
    samples ponds from JRC GSW inside the AOI - the persistent water bodies in
    this box are the EKW bheris. A generous rectangle is therefore sufficient
    and is honest about being an approximation.
    """
    polys = ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(
        ee.Filter.eq("SITE_ID", WDPA_ID)
    )
    if polys.size().getInfo() > 0:
        print("[probe] using WDPA polygon for SITE_ID", WDPA_ID)
        return polys.geometry()

    print(
        "[probe] WDPA has no polygon for SITE_ID "
        f"{WDPA_ID} (point-only record).\n"
        "        Falling back to the approximate EKW rectangle - fine for "
        "observation counting,\n"
        "        NOT a substitute for the legal boundary in the real pipeline."
    )
    return ee.Geometry.Rectangle(PROBE_BBOX)


def sample_ponds(aoi: ee.Geometry, n: int) -> ee.FeatureCollection:
    """Pond-sized sampling regions from JRC GSW, largest first.

    These are NOT the project's real units. They exist only so that
    "observations per pond" is measured over realistic pond footprints.
    """
    water = ee.Image(GSW).select("occurrence").gte(GSW_OCCURRENCE).selfMask()
    vectors = water.reduceToVectors(
        geometry=aoi,
        scale=30,  # GSW native; fine for sampling regions
        geometryType="polygon",
        eightConnected=False,
        maxPixels=1e9,
    )
    with_area = vectors.map(
        lambda f: f.set("ha", f.geometry().area(1).divide(1e4))
    ).filter(ee.Filter.gte("ha", MIN_POND_HA))

    ranked = with_area.sort("ha", False).limit(n)
    # Stable integer label so the CSV can be grouped locally.
    indexed = ranked.toList(n)
    return ee.FeatureCollection(
        [
            ee.Feature(ee.Feature(indexed.get(i)).geometry(), {"pond": i})
            for i in range(n)
        ]
    )


def clear_fraction_rows(
    ponds: ee.FeatureCollection, aoi: ee.Geometry, year: int
) -> list[dict]:
    """One row per (image, pond): what fraction of that pond was clear."""
    coll = (
        ee.ImageCollection(S2)
        .filterBounds(aoi)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .linkCollection(ee.ImageCollection(CLOUD_SCORE_PLUS), [CS_BAND])
    )

    def per_image(img: ee.Image) -> ee.FeatureCollection:
        img = ee.Image(img)
        clear = img.select(CS_BAND).gte(CS_THRESHOLD).rename("clear")
        date = img.date().format("YYYY-MM-dd")
        stats = clear.reduceRegions(
            collection=ponds,
            reducer=ee.Reducer.mean(),
            scale=SCALE,
            tileScale=4,
        )
        return stats.map(
            lambda f: ee.Feature(
                None, {"pond": f.get("pond"), "date": date, "clear": f.get("mean")}
            )
        )

    rows = ee.FeatureCollection(coll.map(per_image)).flatten()
    url = rows.getDownloadURL(filetype="CSV", selectors=["pond", "date", "clear"])
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def analyse(rows: list[dict]) -> dict:
    """All statistics are computed locally so they are cheap to re-run."""
    # (pond, year, month) -> count of usable observations
    usable: dict[tuple[int, int, int], int] = defaultdict(int)
    images_per_year: dict[int, set[str]] = defaultdict(set)

    for row in rows:
        raw = (row.get("clear") or "").strip()
        date = (row.get("date") or "").strip()
        if not date:
            continue
        year, month = int(date[:4]), int(date[5:7])
        images_per_year[year].add(date)
        if not raw:
            continue  # image did not cover this pond
        if float(raw) >= POND_CLEAR_FRACTION:
            usable[(int(row["pond"]), year, month)] += 1

    ponds = sorted({k[0] for k in usable}) or [0]
    years = sorted(images_per_year)

    # For each calendar month: per pond, how many distinct years had >=1 usable look.
    per_month: dict[int, dict] = {}
    for month in range(1, 13):
        samples = [
            sum(1 for y in years if usable.get((p, y, month), 0) > 0) for p in ponds
        ]
        samples.sort()
        median = samples[len(samples) // 2] if samples else 0
        if median >= GATE_USABLE:
            verdict = "usable"
        elif median >= GATE_POOL:
            verdict = "pool +/-1"
        else:
            verdict = "skip"
        per_month[month] = {
            "median_year_samples": median,
            "min_year_samples": samples[0] if samples else 0,
            "max_year_samples": samples[-1] if samples else 0,
            "verdict": verdict,
        }

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "constants": {
            "cs_threshold": CS_THRESHOLD,
            "pond_clear_fraction": POND_CLEAR_FRACTION,
            "gsw_occurrence": GSW_OCCURRENCE,
            "min_pond_ha": MIN_POND_HA,
            "gate_usable": GATE_USABLE,
            "gate_pool": GATE_POOL,
        },
        "n_ponds": len(ponds),
        "images_per_year": {y: len(v) for y, v in sorted(images_per_year.items())},
        "per_calendar_month": per_month,
    }


def report(result: dict) -> None:
    print("\n=== Sentinel-2 L2A scene availability by year ===")
    print("(tests the assumption that usable L2A over India starts ~2019)")
    for year, count in result["images_per_year"].items():
        print(f"  {year}: {count:4d} scenes")

    print(
        f"\n=== Usable year-samples per calendar month "
        f"(median over {result['n_ponds']} ponds) ==="
    )
    pct = int(POND_CLEAR_FRACTION * 100)
    print(
        f"  a 'year-sample' = a year with >=1 observation where {pct}% "
        "of the pond was clear\n"
    )
    print(f"  {'month':<6}{'median':>8}{'min':>6}{'max':>6}   verdict")
    for month in range(1, 13):
        m = result["per_calendar_month"][month]
        print(
            f"  {MONTHS[month - 1]:<6}{m['median_year_samples']:>8}"
            f"{m['min_year_samples']:>6}{m['max_year_samples']:>6}   {m['verdict']}"
        )

    verdicts = {m: result["per_calendar_month"][m]["verdict"] for m in range(1, 13)}
    usable = [MONTHS[m - 1] for m, v in verdicts.items() if v == "usable"]
    pooled = [MONTHS[m - 1] for m, v in verdicts.items() if v == "pool +/-1"]
    skip = [MONTHS[m - 1] for m, v in verdicts.items() if v == "skip"]

    print("\n=== VERDICT ===")
    print(f"  usable as-is : {', '.join(usable) if usable else '(none)'}")
    print(f"  pool +/-1    : {', '.join(pooled) if pooled else '(none)'}")
    print(f"  skip         : {', '.join(skip) if skip else '(none)'}")

    if len(usable) >= 6:
        print(
            "\n  => Per-month climatology is viable. Proceed with the plan as written;\n"
            "     emit no anomaly for the 'skip' months rather than a weak number."
        )
    elif len(usable) + len(pooled) >= 6:
        print(
            "\n  => Per-month bins are thin. Pool month +/-1 where flagged, and consider\n"
            "     harmonic regression (annual sin/cos fit) as the primary mechanism."
        )
    else:
        print(
            "\n  => Per-month climatology is NOT viable on this archive.\n"
            "     Switch to harmonic regression over all observations at once\n"
            "     (plan 2.2a fallback). Do not force the per-month approach."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="EKW observation feasibility probe")
    parser.add_argument("--ponds", type=int, default=20)
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--force", action="store_true", help="ignore cached rows")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "probe_rows.csv"

    if cache.exists() and not args.force:
        print(f"[probe] reusing cached observations: {cache}")
        print("        (pass --force to re-query Earth Engine)")
        with cache.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        initialize_or_exit()
        print("[probe] resolving area of interest...")
        area = aoi()
        print(f"[probe] sampling {args.ponds} pond-sized regions from JRC GSW...")
        ponds = sample_ponds(area, args.ponds)

        rows = []
        for year in range(args.start_year, args.end_year + 1):
            print(f"[probe] querying {year}...", end=" ", flush=True)
            try:
                year_rows = clear_fraction_rows(ponds, area, year)
            except Exception as exc:  # noqa: BLE001
                print(f"failed ({exc})")
                continue
            print(f"{len(year_rows)} rows")
            rows.extend(year_rows)

        if not rows:
            print("\n[probe] no rows returned - nothing to analyse.", file=sys.stderr)
            raise SystemExit(1)

        with cache.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pond", "date", "clear"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[probe] cached {len(rows)} rows to {cache}")

    result = analyse(rows)
    report(result)

    out = OUT_DIR / "probe_result.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[probe] wrote {out}")


if __name__ == "__main__":
    main()
