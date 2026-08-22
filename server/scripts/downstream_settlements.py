"""Find the populated places along each lake's already-traced flood path.

`trace_downstream.py` gives a total head-count for the corridor, which answers
"how many" but not "where" - and "where" is what someone deciding whether to
drive up a valley actually needs.

This reads the paths already stored in lake_exposure.json (no re-tracing) and
samples GHSL population in discs along each one, keeping the clusters that stand
out. One reduceRegions call per lake rather than one per sample point.

The result is deliberately coarse: GHSL is a 100 m population grid, so a
"settlement" here is a cluster of inhabited cells, not a named town. Names are
left to the basemap, which already labels them.

Usage:
    cd server
    python -m scripts.downstream_settlements
    python -m scripts.downstream_settlements --limit 5 --verify
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import ee

from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
POP_ASSET = "JRC/GHSL/P2023A/GHS_POP/2020"
SAMPLE_KM = 5.0        # spacing of sample discs along the path
DISC_M = 2500          # radius of each disc
MIN_PEOPLE = 300       # below this a cluster is not worth a marker
MAX_MARKERS = 8        # keep the map readable


def resample(path: list, spacing_km: float) -> list:
    """Thin a dense vertex list down to points every `spacing_km`."""
    out, acc = [path[0]], 0.0
    for a, b in zip(path, path[1:]):
        d = math.hypot((b[0] - a[0]) * 111.32 * math.cos(math.radians(a[1])),
                       (b[1] - a[1]) * 110.54)
        acc += d
        if acc >= spacing_km:
            out.append(b)
            acc = 0.0
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def settlements_for(path: list) -> list[dict]:
    """Population inside a disc at each sample point, strongest first."""
    pts = resample(path, SAMPLE_KM)
    if len(pts) < 2:
        return []
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]).buffer(DISC_M), {"i": i})
        for i, (lon, lat) in enumerate(pts)
    ])
    reduced = ee.Image(POP_ASSET).reduceRegions(
        collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()

    rows = []
    for f in reduced.get("features", []):
        pop = f["properties"].get("sum") or 0
        i = f["properties"]["i"]
        if pop >= MIN_PEOPLE:
            lon, lat = pts[i]
            rows.append({"lon": round(lon, 4), "lat": round(lat, 4),
                         "pop": int(round(pop)),
                         "km": round(i * SAMPLE_KM, 1)})
    rows.sort(key=lambda r: -r["pop"])
    return rows[:MAX_MARKERS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    initialize_or_exit()
    dest = DATA / "lake_exposure.json"
    doc = json.loads(dest.read_text(encoding="utf-8"))
    lakes = doc["lakes"]
    ids = list(lakes)[: args.limit] if args.limit else list(lakes)
    print(f"sampling settlements along {len(ids)} traced paths\n")

    done = empty = 0
    for n, lid in enumerate(ids, 1):
        path = lakes[lid].get("path") or []
        if len(path) < 2:
            continue
        try:
            found = settlements_for(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {lid}: {type(exc).__name__}: {str(exc)[:60]}", flush=True)
            continue
        lakes[lid]["settlements"] = found
        done += 1
        if not found:
            empty += 1
        if args.verify or n <= 5:
            top = found[0] if found else None
            print(f"  {lid}  {len(found)} clusters"
                  + (f"  largest {top['pop']:,} people at {top['km']:.0f} km" if top else "  (none)"))
        if n % 20 == 0:
            print(f"  ... {n}/{len(ids)}", flush=True)

    dest.write_text(json.dumps(doc), encoding="utf-8")
    print(f"\n{done} lakes sampled, {empty} with no cluster above {MIN_PEOPLE} people")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
