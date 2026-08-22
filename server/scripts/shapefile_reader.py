"""Minimal ESRI shapefile reader - polygons and .dbf attributes, pure Python.

Written instead of adding geopandas/GDAL, which is a heavy and fragile install on
Windows for what is a genuinely simple binary format. We only need polygon rings
and a couple of attribute columns.

The .shp records carry a per-record bounding box, so a bbox filter can skip the
point payload entirely - that is what makes a 166 MB HMA-wide file cheap to read
when we only want the Himalaya.

Format reference: ESRI Shapefile Technical Description (July 1998).
  .shp  100-byte header, then records: [recnum:i32be][len:i32be][shape...]
        polygon shape = [type:i32le][box:4f64][nparts:i32][npoints:i32]
                        [parts:nparts*i32][points:npoints*2f64]
  .dbf  dBase III: 32-byte header, 32-byte field descriptors ending in 0x0D,
        then fixed-width ASCII records each prefixed by a deletion flag byte.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

POLYGON = 5
POLYGON_Z = 15
POLYGON_M = 25
_POLY_TYPES = {POLYGON, POLYGON_Z, POLYGON_M}


def read_dbf(path: Path) -> list[dict]:
    """Read a dBase III table into a list of dicts (all values left as str)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    n_rec, hdr_len, rec_len = struct.unpack("<IHH", raw[4:12])

    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\0")[0].decode("latin-1")
        ftype = chr(raw[pos + 11])
        flen = raw[pos + 16]
        fields.append((name, ftype, flen))
        pos += 32

    rows = []
    for i in range(n_rec):
        base = hdr_len + i * rec_len
        rec, off = {}, base + 1  # +1 skips the deletion flag
        for name, _ftype, flen in fields:
            rec[name] = raw[off:off + flen].decode("latin-1").strip()
            off += flen
        rows.append(rec)
    return rows


def read_polygons(shp: Path, bbox: tuple[float, float, float, float] | None = None
                  ) -> Iterator[tuple[int, list[list[tuple[float, float]]]]]:
    """Yield (record_index, rings) for polygon records intersecting `bbox`.

    `rings` is a list of vertex lists; ring 0 is the outer ring and any further
    rings are holes (shapefiles encode holes by winding order, which we keep).
    Record index is 0-based and lines up with the .dbf row order.
    """
    with open(shp, "rb") as fh:
        fh.seek(24)
        file_len = struct.unpack(">i", fh.read(4))[0] * 2
        fh.seek(100)
        idx = -1
        while fh.tell() < file_len:
            head = fh.read(8)
            if len(head) < 8:
                break
            _num, content_len = struct.unpack(">ii", head)
            body_len = content_len * 2
            start = fh.tell()
            idx += 1

            shp_type = struct.unpack("<i", fh.read(4))[0]
            if shp_type not in _POLY_TYPES:
                fh.seek(start + body_len)
                continue

            xmin, ymin, xmax, ymax = struct.unpack("<4d", fh.read(32))
            if bbox and (xmax < bbox[0] or xmin > bbox[2]
                         or ymax < bbox[1] or ymin > bbox[3]):
                # skip the vertex payload entirely - this is the fast path
                fh.seek(start + body_len)
                continue

            n_parts, n_points = struct.unpack("<ii", fh.read(8))
            parts = list(struct.unpack(f"<{n_parts}i", fh.read(4 * n_parts)))
            flat = struct.unpack(f"<{2 * n_points}d", fh.read(16 * n_points))

            rings = []
            bounds = parts + [n_points]
            for p in range(n_parts):
                a, b = bounds[p], bounds[p + 1]
                rings.append([(flat[2 * k], flat[2 * k + 1]) for k in range(a, b)])
            yield idx, rings

            fh.seek(start + body_len)


def ring_centroid_area(ring: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Signed shoelace area and centroid of one ring, in the ring's own units."""
    a = cx = cy = 0.0
    n = len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-15:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return sum(xs) / n, sum(ys) / n, 0.0
    return cx / (6 * a), cy / (6 * a), a
