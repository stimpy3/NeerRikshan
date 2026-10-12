"""Where the API's files live: local disk on a laptop, S3 when deployed.

Locally nothing changes - JSON comes from server/data and thumbnails from
server/data/thumbnails. On Lambda there is no persistent disk, so DATA_BUCKET
and SITE_BUCKET are set and the same reads go to S3 instead:

    s3://DATA_BUCKET/data/<name>.json      pipeline results (private)
    s3://SITE_BUCKET/thumbnails/<f>.png    lake images, served by CloudFront

boto3 is imported lazily: it ships with the Lambda runtime but is not an API
dependency, so local runs never need it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
THUMBS_DIR = DATA_DIR / "thumbnails"


@lru_cache(maxsize=1)
def _s3():
    import boto3
    return boto3.client("s3")


def on_s3() -> bool:
    return bool(settings.data_bucket)


def read_json(name: str) -> dict | None:
    """A pipeline result by file name, or None if it does not exist yet."""
    if on_s3():
        s3 = _s3()
        try:
            obj = s3.get_object(Bucket=settings.data_bucket, Key=f"data/{name}")
        except s3.exceptions.NoSuchKey:
            return None
        return json.loads(obj["Body"].read())
    path = DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def thumbnail_names(lake_id: str) -> list[str]:
    """File names of every thumbnail rendered for one lake, sorted."""
    if on_s3():
        prefix = f"thumbnails/{lake_id}_"
        pages = _s3().get_paginator("list_objects_v2").paginate(
            Bucket=settings.site_bucket, Prefix=prefix)
        return sorted(o["Key"].rsplit("/", 1)[-1]
                      for page in pages for o in page.get("Contents", []))
    if not THUMBS_DIR.exists():
        return []
    return sorted(f.name for f in THUMBS_DIR.glob(f"{lake_id}_*.png"))


def thumbnail_path(name: str) -> Path | None:
    """Local file for a thumbnail; None when deployed (CloudFront serves it)."""
    if on_s3():
        return None
    path = THUMBS_DIR / name
    return path if path.exists() else None
