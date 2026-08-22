"""Trace the downstream flow path from each lake and measure what lies in it.

A growth rate is not a risk. "This lake gained 7.2 ha/yr" becomes actionable only
when paired with what sits below it, because a fast-growing lake draining an
empty valley does not need a field survey and a slow one above a town does.

Method:
  1. Walk HydroSHEDS D8 flow direction (WWF/HydroSHEDS/03DIR, 3 arc-sec ~90 m)
     downstream from the lake centroid, up to --max-km.
  2. Buffer that path into a flood corridor.
  3. Sum exposure inside the corridor from datasets verified present in GEE:
        JRC/GHSL/P2023A/GHS_POP/2020        population count
        JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020  settlement class
        WRI/GPPD/power_plants               hydropower and other plants

D8 encoding is a power-of-two compass: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE.
0 marks an outlet or sink and 255 is no data; both terminate the walk.

SELF-CHECK: a correct downstream trace must lose elevation. Every path is scored
against the HydroSHEDS DEM and the fraction of descending steps reported. A trace
that climbs is wrong, and this catches an inverted or mis-registered direction
raster immediately rather than after the numbers reach a map.

Tiles are cached on disk, so lakes sharing a valley cost one download between them.

Usage:
    cd server
    python -m scripts.trace_downstream --limit 5 --verify
    python -m scripts.trace_downstream --max-km 100
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path

import ee
import numpy as np
import requests

from scripts.gee_auth import initialize_or_exit

DATA = Path(__file__).resolve().parent.parent / "data"
TILES = DATA / "_intermediate" / "hydro_tiles"
DIR_ASSET = "WWF/HydroSHEDS/03DIR"
DEM_ASSET = "WWF/HydroSHEDS/03VFDEM"
POP_ASSET = "JRC/GHSL/P2023A/GHS_POP/2020"
SMOD_ASSET = "JRC/GHSL/P2023A/GHS_SMOD_V2-0/2020"
PLANTS = "WRI/GPPD/power_plants"

TILE_DEG = 1.0                 # tile size for the cached direction raster
ARCSEC = 3.0 / 3600.0          # 3 arc-second cells
CORRIDOR_M = 1000              # flood corridor half-width

# D8 -> (dcol, drow) in raster order (row increases southward)
D8 = {1: (1, 0), 2: (1, 1), 4: (0, 1), 8: (-1, 1),
      16: (-1, 0), 32: (-1, -1), 64: (0, -1), 128: (1, -1)}


def tile_key(lon: float, lat: float) -> tuple[int, int]:
    return math.floor(lon / TILE_DEG), math.floor(lat / TILE_DEG)


def fetch_tile(kx: int, ky: int, band_asset: str, tag: str) -> np.ndarray:
    """Download and cache one tile of a HydroSHEDS raster."""
    TILES.mkdir(parents=True, exist_ok=True)
    dest = TILES / f"{tag}_{kx}_{ky}.npy"
    if dest.exists():
        return np.load(dest)
    lon0, lat0 = kx * TILE_DEG, ky * TILE_DEG
    region = ee.Geometry.Rectangle(
        [lon0, lat0, lon0 + TILE_DEG, lat0 + TILE_DEG], None, False)
    url = ee.Image(band_asset).getDownloadURL({
        "region": region, "scale": ARCSEC * 111320, "format": "NPY"})
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
    arr = np.load(io.BytesIO(r.content))
    arr = arr[arr.dtype.names[0]] if arr.dtype.names else arr
    np.save(dest, arr)
    return arr


class Raster:
    """Tile-cached point sampler for a global HydroSHEDS grid."""

    def __init__(self, asset: str, tag: str):
        self.asset, self.tag, self.cache = asset, tag, {}

    def get(self, lon: float, lat: float):
        k = tile_key(lon, lat)
        if k not in self.cache:
            try:
                self.cache[k] = fetch_tile(k[0], k[1], self.asset, self.tag)
            except Exception:
                self.cache[k] = None
        arr = self.cache[k]
        if arr is None:
            return None
        h, w = arr.shape
        lon0, lat0 = k[0] * TILE_DEG, k[1] * TILE_DEG
        col = int((lon - lon0) / TILE_DEG * w)
        row = int((lat0 + TILE_DEG - lat) / TILE_DEG * h)   # north-up raster
        if not (0 <= col < w and 0 <= row < h):
            return None
        return arr[row, col], (col, row), (h, w), (lon0, lat0)


def trace(lon: float, lat: float, dirs: Raster, max_km: float) -> list[tuple[float, float]]:
    """Walk D8 downstream, returning the path as lon/lat vertices."""
    path = [(lon, lat)]
    travelled = 0.0
    cur_lon, cur_lat = lon, lat
    for _ in range(20000):
        s = dirs.get(cur_lon, cur_lat)
        if s is None:
            break
        val = int(s[0])
        move = D8.get(val)
        if move is None:            # 0 = outlet/sink, 255 = nodata
            break
        dcol, drow = move
        step_lon = dcol * ARCSEC
        step_lat = -drow * ARCSEC   # row increases southward
        nlon, nlat = cur_lon + step_lon, cur_lat + step_lat
        d = math.hypot(step_lon * 111.32 * math.cos(math.radians(cur_lat)),
                       step_lat * 110.54)
        travelled += d
        cur_lon, cur_lat = nlon, nlat
        path.append((round(cur_lon, 5), round(cur_lat, 5)))
        if travelled >= max_km:
            break
    return path


def descending_fraction(path: list[tuple[float, float]], dem: Raster) -> tuple[float, float]:
    """Share of steps that lose elevation, and total drop (m). The self-check."""
    # Sample coarsely. At 90 m in a steep gorge the vertex can land on a valley
    # wall rather than the channel, so per-step monotonicity fails on paths that
    # are actually correct. Every 10th vertex (~1 km) averages that out while
    # still catching a genuinely inverted direction raster.
    els = []
    for lon, lat in path[::10]:
        s = dem.get(lon, lat)
        if s is not None:
            els.append(float(s[0]))
    if len(els) < 3:
        return float("nan"), float("nan")
    down = sum(1 for a, b in zip(els, els[1:]) if b <= a)
    return down / (len(els) - 1), els[0] - els[-1]


def exposure(path: list[tuple[float, float]]) -> dict:
    """Population, settlement class and power plants inside the flood corridor."""
    if len(path) < 2:
        return {}
    line = ee.Geometry.LineString([list(p) for p in path], None, False)
    corridor = line.buffer(CORRIDOR_M)
    pop = ee.Image(POP_ASSET).reduceRegion(
        ee.Reducer.sum(), corridor, 100, maxPixels=1e9)
    smod = ee.Image(SMOD_ASSET).reduceRegion(
        ee.Reducer.max(), corridor, 1000, maxPixels=1e9)
    plants = ee.FeatureCollection(PLANTS).filterBounds(corridor)
    got = ee.Dictionary({
        "pop": pop.get("population_count"),
        "smod": smod.get("smod_code"),
        "plants": plants.size(),
        "plant_mw": plants.aggregate_sum("capacitymw"),
    }).getInfo()
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-km", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="print the elevation self-check per lake")
    ap.add_argument("--watchlist-only", action="store_true", default=True)
    args = ap.parse_args()

    initialize_or_exit()
    ch = json.loads((DATA / "lake_change.json").read_text(encoding="utf-8"))
    lakes = [x for x in ch["lakes"]
             if not x.get("suspect") and x.get("detectable")
             and x.get("direction") == "growing"]
    lakes.sort(key=lambda r: -(r.get("slope_ci_lo") or 0))
    if args.limit:
        lakes = lakes[: args.limit]
    print(f"tracing {len(lakes)} confidently growing lakes, up to {args.max_km:g} km\n")

    dirs, dem = Raster(DIR_ASSET, "dir"), Raster(DEM_ASSET, "dem")
    out, bad = {}, 0
    for n, L in enumerate(lakes, 1):
        p = trace(L["lon"], L["lat"], dirs, args.max_km)
        if len(p) < 3:
            bad += 1
            continue
        frac, drop = descending_fraction(p, dem)
        km = 0.0
        for a, b in zip(p, p[1:]):
            km += math.hypot((b[0]-a[0]) * 111.32 * math.cos(math.radians(a[1])),
                             (b[1]-a[1]) * 110.54)
        rec = {"path": p[::3], "path_km": round(km, 1),
               "descending_frac": round(frac, 3) if frac == frac else None,
               "drop_m": round(drop) if drop == drop else None}
        try:
            rec.update(exposure(p))
        except Exception as exc:
            rec["exposure_error"] = f"{type(exc).__name__}: {str(exc)[:60]}"
        out[L["id"]] = rec
        if args.verify or n <= 5:
            print(f"  {L['id']}  {km:6.1f} km  drop {rec['drop_m']:>5} m  "
                  f"descending {100*(frac if frac==frac else 0):3.0f}%  "
                  f"pop {rec.get('pop') or 0:>10,.0f}  plants {rec.get('plants', 0)}")
        if n % 10 == 0:
            print(f"  ... {n}/{len(lakes)}", flush=True)

    dest = DATA / "lake_exposure.json"
    dest.write_text(json.dumps({"max_km": args.max_km,
                                "corridor_m": CORRIDOR_M,
                                "lakes": out}), encoding="utf-8")
    ok = [r for r in out.values() if (r.get("descending_frac") or 0) >= 0.9]
    print(f"\ntraced {len(out)} lakes ({bad} failed)")
    print(f"self-check: {len(ok)}/{len(out)} paths descend on >=90% of steps")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
