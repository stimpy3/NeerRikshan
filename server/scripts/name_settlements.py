"""Attach real place names to the settlement clusters on each flood path.

`downstream_settlements.py` finds *where* people are (GHSL is a 100 m population
grid, so a cluster is inhabited cells, not a named town). This resolves those
clusters to actual place names so the detail page can say "Chungthang" rather
than "~11,000 people at 65 km".

Uses the GeoNames cities500 dump - a plain tab-separated file of every populated
place with 500+ inhabitants worldwide, downloadable without an account. Matching
is done offline against a local copy, so there is no rate limit and no API key,
unlike Nominatim reverse geocoding which would need ~750 throttled requests.

A cluster is named only if a GeoNames place sits within --radius km of it.
Clusters with no match keep their coordinates and stay unnamed rather than being
given the nearest far-away city, which would be a lie about where the people are.

Usage:
    cd server
    python -m scripts.name_settlements
    python -m scripts.name_settlements --radius 12 --verify
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
DUMP = DATA / "_intermediate" / "cities500.zip"
DUMP_URL = "https://download.geonames.org/export/dump/cities500.zip"

# Himalayan arc plus the plains the floods drain into - no point indexing Brazil.
BBOX = (68.0, 20.0, 100.0, 40.0)
CELL = 0.25  # ~25 km lookup grid


def ensure_dump() -> Path:
    if DUMP.exists():
        return DUMP
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {DUMP_URL} ...", flush=True)
    r = requests.get(DUMP_URL, timeout=300)
    r.raise_for_status()
    DUMP.write_bytes(r.content)
    return DUMP


def load_places() -> dict:
    """Grid-index GeoNames places inside the region of interest."""
    grid: dict[tuple[int, int], list] = {}
    kept = 0
    with zipfile.ZipFile(ensure_dump()) as z, z.open("cities500.txt") as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            p = raw.split("\t")
            if len(p) < 15:
                continue
            try:
                lat, lon = float(p[4]), float(p[5])
                pop = int(p[14] or 0)
            except ValueError:
                continue
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            # A few GeoNames rows carry a comma-joined alias in the name field
            # (e.g. "Kalimpong, Крукети"). Keep the part before the comma.
            name = p[1].split(",")[0].strip() or p[2].split(",")[0].strip()
            if not name:
                continue
            rec = {"name": name, "lat": lat, "lon": lon, "pop": pop, "cc": p[8]}
            grid.setdefault((int(lon / CELL), int(lat / CELL)), []).append(rec)
            kept += 1
    print(f"indexed {kept:,} populated places in the region")
    return grid


def km(lon1, lat1, lon2, lat2) -> float:
    return math.hypot((lon2 - lon1) * 111.32 * math.cos(math.radians(lat1)),
                      (lat2 - lat1) * 110.54)


def nearest(grid: dict, lon: float, lat: float, radius_km: float):
    """Closest named place within radius; prefers the larger of two close ones."""
    span = int(radius_km / (CELL * 100)) + 1
    cx, cy = int(lon / CELL), int(lat / CELL)
    best = None
    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):
            for rec in grid.get((cx + dx, cy + dy), []):
                d = km(lon, lat, rec["lon"], rec["lat"])
                if d > radius_km:
                    continue
                # a town 3 km away beats a hamlet 2 km away
                score = d - 0.4 * math.log10(max(rec["pop"], 1))
                if best is None or score < best[0]:
                    best = (score, d, rec)
    return best


def main() -> None:
    # Place names carry Devanagari, Tibetan and diacritics; the default Windows
    # console codec is cp1252 and would crash on the first one.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=10.0,
                    help="max km between a population cluster and a named place")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    grid = load_places()
    dest = DATA / "lake_exposure.json"
    doc = json.loads(dest.read_text(encoding="utf-8"))

    named = unnamed = 0
    for lid, rec in doc["lakes"].items():
        for s in rec.get("settlements") or []:
            hit = nearest(grid, s["lon"], s["lat"], args.radius)
            if hit:
                _score, d, place = hit
                s["name"] = place["name"]
                s["country"] = place["cc"]
                s["name_km"] = round(d, 1)
                named += 1
            else:
                s.pop("name", None)
                unnamed += 1
        if args.verify and rec.get("settlements"):
            top = rec["settlements"][0]
            print(f"  {lid}  {top.get('name', '(unnamed)')}"
                  f"  {top['pop']:,} people at {top['km']:.0f} km")

    dest.write_text(json.dumps(doc), encoding="utf-8")
    total = named + unnamed
    print(f"\nnamed {named}/{total} clusters ({100*named/max(total,1):.0f}%)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
