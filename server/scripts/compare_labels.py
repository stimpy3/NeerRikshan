"""Compare our Hi-MAG 2017 labels and our model against an independent inventory.

Reference: Kumar & Vijay (2026), "Inventory of Glacial Lakes in High Mountain Asia
for the Periods 2016-2017 and 2022-2024", Zenodo 17948783, CC-BY-4.0.
Sentinel-2 median composites at 10 m - versus Hi-MAG's Landsat at 30 m.

Three questions, deliberately separated:

  A  Hi-MAG 2017  vs  KV 2016-2017     same epoch, 30 m vs 10 m.
     Any systematic gap here is a LABEL property, not a model error - it bounds
     how much of our measured segmentation error was ever ours to fix.

  B  our model 2022  vs  KV 2022-2024  external validation.
     The first check of our output against ground truth we did not train on,
     built by other people with a different pipeline.

  C  KV 2016-17  vs  KV 2022-2024      an independent change estimate.
     Both epochs come from ONE consistent pipeline, so differencing them is
     legitimate - unlike differencing Hi-MAG against KV, which would confound
     real change with methodology. Cross-checks our +1.1%.

Matching is mutual-nearest on centroid within --tol metres. Because inventories
disagree about whether a complex lake is one polygon or several, we report both
the nearest single match and the sum of all reference polygons whose centroid
falls inside the lake's own footprint radius; a large gap between the two means
the reference split the lake.

Usage:
    cd server
    python -m scripts.compare_labels
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from scripts.shapefile_reader import read_dbf

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = DATA / "labels_kv2026"
EPOCHS = {
    "kv_2016_2017": "Glacial_lakes_2016_2017_median",
    "kv_2022_2024": "Glacial_lakes_2022_2024_median",
}
# Himalaya arc, generous - our lakes span roughly lon 73-95, lat 27-36
BBOX = (72.0, 26.0, 96.0, 37.0)


def load_reference(stem: str) -> list[dict]:
    rows = read_dbf(LABELS / f"{stem}.dbf")
    out = []
    for r in rows:
        try:
            lat, lon = float(r["Latitude"]), float(r["Longitude"])
            area = float(r["Area"])
        except (ValueError, KeyError):
            continue
        if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
            out.append({"lon": lon, "lat": lat, "area_ha": area / 1e4,
                        "elev": r.get("Lake_Elev")})
    return out


def metres(dlon: float, dlat: float, lat: float) -> float:
    return math.hypot(dlon * 111_320 * math.cos(math.radians(lat)), dlat * 110_540)


class Grid:
    """Coarse lon/lat bucket index - avoids an O(n*m) scan over 20k polygons."""

    CELL = 0.05  # ~5 km

    def __init__(self, pts: list[dict]):
        self.cells: dict[tuple[int, int], list[int]] = {}
        self.pts = pts
        for i, p in enumerate(pts):
            self.cells.setdefault(self._key(p["lon"], p["lat"]), []).append(i)

    def _key(self, lon: float, lat: float) -> tuple[int, int]:
        return int(lon / self.CELL), int(lat / self.CELL)

    def near(self, lon: float, lat: float, radius_m: float) -> list[int]:
        span = int(radius_m / (self.CELL * 100_000)) + 1
        cx, cy = self._key(lon, lat)
        hits = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                hits += self.cells.get((cx + dx, cy + dy), [])
        return hits


def match(lakes: list[dict], ref: list[dict], tol_m: float) -> dict:
    """Mutual-nearest match; also sum reference polygons inside the footprint."""
    grid = Grid(ref)
    best: dict[int, tuple[float, str]] = {}
    prop: dict[str, tuple[int, float]] = {}

    for L in lakes:
        lon, lat = L["lon"], L["lat"]
        cands = grid.near(lon, lat, tol_m)
        if not cands:
            continue
        pick, pd = None, tol_m + 1
        for i in cands:
            d = metres(ref[i]["lon"] - lon, ref[i]["lat"] - lat, lat)
            if d < pd:
                pick, pd = i, d
        if pick is None:
            continue
        prop[L["id"]] = (pick, pd)
        if pick not in best or pd < best[pick][0]:
            best[pick] = (pd, L["id"])

    out = {}
    for lid, (i, d) in prop.items():
        if best[i][1] != lid:          # another lake claimed it more strongly
            continue
        L = next(x for x in lakes if x["id"] == lid)
        # footprint radius of an equal-area circle, floored so tiny lakes still
        # get a sensible search window
        r = max(math.sqrt(max(L["himag_2017_ha"], 1.0) * 1e4 / math.pi), 150.0)
        tot = sum(ref[j]["area_ha"] for j in grid.near(L["lon"], L["lat"], r)
                  if metres(ref[j]["lon"] - L["lon"], ref[j]["lat"] - L["lat"],
                            L["lat"]) <= r)
        out[lid] = {"area_ha": ref[i]["area_ha"], "dist_m": d,
                    "sum_ha": tot, "n_parts": sum(
                        1 for j in grid.near(L["lon"], L["lat"], r)
                        if metres(ref[j]["lon"] - L["lon"],
                                  ref[j]["lat"] - L["lat"], L["lat"]) <= r)}
    return out


def report(name: str, pairs: list[tuple[float, float]], unit: str = "ha") -> dict:
    """pairs = (ours, reference). Reports ratio and signed percentage gap."""
    ratios = [a / b for a, b in pairs if b > 0]
    diffs = [100 * (a - b) / b for a, b in pairs if b > 0]
    ours, ref = sum(a for a, _ in pairs), sum(b for _, b in pairs)
    s = {
        "n": len(ratios),
        "median_ratio": round(st.median(ratios), 3),
        "mean_ratio": round(st.mean(ratios), 3),
        "median_pct_gap": round(st.median(diffs), 1),
        "aggregate_ours": round(ours, 1),
        "aggregate_ref": round(ref, 1),
        "aggregate_pct_gap": round(100 * (ours - ref) / ref, 1) if ref else None,
        "within_10pct": round(100 * sum(1 for d in diffs if abs(d) <= 10) / len(diffs), 1),
        "within_25pct": round(100 * sum(1 for d in diffs if abs(d) <= 25) / len(diffs), 1),
    }
    print(f"\n{name}")
    print(f"  matched lakes        {s['n']}")
    print(f"  median ratio         {s['median_ratio']:.3f}   (1.000 = perfect)")
    print(f"  median gap           {s['median_pct_gap']:+.1f}%")
    print(f"  aggregate            {s['aggregate_ours']:.0f} vs {s['aggregate_ref']:.0f} {unit}"
          f"  ({s['aggregate_pct_gap']:+.1f}%)")
    print(f"  within 10% / 25%     {s['within_10pct']:.0f}% / {s['within_25pct']:.0f}%")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=750.0, help="match radius, metres")
    ap.add_argument("--year", type=int, default=2022)
    args = ap.parse_args()

    ts = json.loads((DATA / "lake_timeseries.json").read_text(encoding="utf-8"))
    lakes = ts["lakes"]
    print(f"our lakes: {len(lakes)}  (model '{ts['model']}')")

    refs, matches = {}, {}
    for tag, stem in EPOCHS.items():
        refs[tag] = load_reference(stem)
        matches[tag] = match(lakes, refs[tag], args.tol)
        print(f"{tag}: {len(refs[tag])} polygons in bbox, "
              f"{len(matches[tag])} matched to our lakes")

    by_id = {L["id"]: L for L in lakes}
    results = {}

    # --- A: label vs label, same epoch -------------------------------------
    m = matches["kv_2016_2017"]
    pairs = [(by_id[k]["himag_2017_ha"], v["area_ha"]) for k, v in m.items()]
    results["A_himag2017_vs_kv2016"] = report(
        "A  Hi-MAG 2017 (30 m Landsat)  vs  KV 2016-17 (10 m Sentinel-2)", pairs)

    split = [v for v in m.values() if v["n_parts"] > 1]
    print(f"  reference splits lake into >1 polygon: {len(split)}/{len(m)} "
          f"({100*len(split)/max(len(m),1):.0f}%)")

    # --- B: our model vs an inventory we did not train on -------------------
    m = matches["kv_2022_2024"]
    pairs, flagged = [], 0
    for k, v in m.items():
        yr = by_id[k]["years"].get(str(args.year))
        if not yr or yr["flag"] != "ok":
            flagged += 1
            continue
        pairs.append((yr["area_ha"], v["area_ha"]))
    results["B_model2022_vs_kv2022"] = report(
        f"B  our U-Net {args.year} (10 m)  vs  KV 2022-24 (10 m)  [EXTERNAL]", pairs)
    print(f"  excluded (not flagged ok): {flagged}")

    # --- C: independent change estimate, one consistent pipeline ------------
    common = set(matches["kv_2016_2017"]) & set(matches["kv_2022_2024"])
    a = sum(matches["kv_2016_2017"][k]["area_ha"] for k in common)
    b = sum(matches["kv_2022_2024"][k]["area_ha"] for k in common)
    per = [100 * (matches["kv_2022_2024"][k]["area_ha"]
                  - matches["kv_2016_2017"][k]["area_ha"])
           / matches["kv_2016_2017"][k]["area_ha"]
           for k in common if matches["kv_2016_2017"][k]["area_ha"] > 0]
    grow = sum(1 for d in per if d > 5)
    shrink = sum(1 for d in per if d < -5)
    print(f"\nC  KV 2016-17 -> KV 2022-24  [independent change estimate]")
    print(f"  lakes in both        {len(common)}")
    print(f"  aggregate            {a:.0f} -> {b:.0f} ha  ({100*(b-a)/a:+.1f}%)")
    print(f"  median lake change   {st.median(per):+.1f}%")
    print(f"  growing >5%          {grow} ({100*grow/len(per):.0f}%)")
    print(f"  shrinking >5%        {shrink} ({100*shrink/len(per):.0f}%)")
    results["C_kv_change"] = {
        "n": len(common), "agg_first": round(a, 1), "agg_last": round(b, 1),
        "agg_pct": round(100 * (b - a) / a, 2),
        "median_pct": round(st.median(per), 2),
        "growing": grow, "shrinking": shrink,
    }

    out = DATA / "_intermediate" / "label_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
