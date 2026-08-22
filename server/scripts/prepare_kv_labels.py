"""Extract KV inventory rings that fall inside each lake's patch window.

Produces one JSON per epoch mapping our lake id -> list of polygons in lon/lat,
ready to be painted into the same GEE grid the patches were exported on.

We keep EVERY reference polygon intersecting the 1.28 km window, not just the
matched lake. For a pixelwise IoU that is the correct target: any water the
reference says is there is water our model should have found. Taking only the
single matched polygon would score neighbouring lakes inside the window as false
positives.

Ring winding follows the shapefile spec - outer rings clockwise (negative
shoelace), holes counter-clockwise - so islands stay holes instead of being
filled in as water.

Usage:
    cd server
    python -m scripts.prepare_kv_labels
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.shapefile_reader import read_dbf, read_polygons, ring_centroid_area
from scripts.compare_labels import BBOX, EPOCHS, Grid, metres

DATA = Path(__file__).resolve().parent.parent / "data"
LABELS = DATA / "labels_kv2026"

# half-diagonal of the 128 px x 10 m window, plus slack
WINDOW_RADIUS_M = 128 * 10 / 2 * math.sqrt(2) + 100


def group_rings(rings: list[list[tuple[float, float]]]) -> list[list[list[list[float]]]]:
    """Group a shapefile record's rings into polygons with holes."""
    polys: list[list[list[list[float]]]] = []
    for ring in rings:
        if len(ring) < 4:
            continue
        _cx, _cy, area = ring_centroid_area(ring)
        pts = [[round(x, 6), round(y, 6)] for x, y in ring]
        if area < 0 or not polys:      # clockwise -> new outer ring
            polys.append([pts])
        else:                          # counter-clockwise -> hole of the last
            polys[-1].append(pts)
    return polys


def main() -> None:
    lakes = json.loads((DATA / "lake_timeseries.json").read_text(encoding="utf-8"))["lakes"]
    print(f"our lakes: {len(lakes)}")

    for tag, stem in EPOCHS.items():
        rows = read_dbf(LABELS / f"{stem}.dbf")
        pts = []
        for i, r in enumerate(rows):
            try:
                lat, lon = float(r["Latitude"]), float(r["Longitude"])
            except (ValueError, KeyError):
                continue
            if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
                pts.append({"lon": lon, "lat": lat, "rec": i})

        grid = Grid(pts)
        # which shapefile records do we need, and for which lakes
        want: dict[int, list[str]] = {}
        for L in lakes:
            for j in grid.near(L["lon"], L["lat"], WINDOW_RADIUS_M):
                if metres(pts[j]["lon"] - L["lon"], pts[j]["lat"] - L["lat"],
                          L["lat"]) <= WINDOW_RADIUS_M:
                    want.setdefault(pts[j]["rec"], []).append(L["id"])

        print(f"{tag}: {len(pts)} polygons in bbox, {len(want)} inside a patch window")

        out: dict[str, list] = {L["id"]: [] for L in lakes}
        for idx, rings in read_polygons(LABELS / f"{stem}.shp", BBOX):
            if idx not in want:
                continue
            polys = group_rings(rings)
            for lid in want[idx]:
                out[lid].extend(polys)

        n_cov = sum(1 for v in out.values() if v)
        dest = DATA / f"kv_rings_{tag.replace('kv_', '')}.json"
        dest.write_text(json.dumps(out), encoding="utf-8")
        print(f"  lakes with >=1 reference polygon: {n_cov}/{len(lakes)}")
        print(f"  wrote {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
