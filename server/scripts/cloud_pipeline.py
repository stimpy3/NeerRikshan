"""Yearly update, as run by the Fargate task that EventBridge starts.

The container starts empty, so this pulls what the pipeline needs from S3,
runs the same scripts used locally, and pushes the results back:

    1. Earth Engine key from the private data bucket -> temp file
    2. current results + model from s3://DATA_BUCKET
    3. export_patches   this season only (twice: 2nd pass retries failures)
    4. run_inference    --append, so earlier years' patches are never needed
    5. detect_change    refit trends and confidence intervals
    6. render_thumbnails
    7. back up the live results in S3, then upload the new ones
    8. bump an env var on the API Lambda so it drops its cached copy

Nothing is uploaded until every step has succeeded, so a failed run leaves the
website on last year's list - there is nothing to roll back.

Environment (set by the task definition):
    DATA_BUCKET, SITE_BUCKET, API_FUNCTION, GEE_PROJECT,
    GEE_SERVICE_ACCOUNT_EMAIL
    YEAR          optional, default current year
    MIN_COVERAGE  optional, default 0.90

Usage (inside the container):
    python -m scripts.cloud_pipeline
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import boto3

SERVER = Path(__file__).resolve().parent.parent
DATA = SERVER / "data"
MODEL = "ndwi_fit_kv22"
RESULTS = ["lake_timeseries.json", "lake_change.json"]
THUMB_TOP = 80


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def run(*args: str) -> None:
    log("$ python -m " + " ".join(args))
    subprocess.run([sys.executable, "-m", *args], cwd=SERVER, check=True)


def main() -> None:
    data_bucket = os.environ["DATA_BUCKET"]
    site_bucket = os.environ["SITE_BUCKET"]
    api_function = os.environ["API_FUNCTION"]
    year = int(os.environ.get("YEAR") or dt.date.today().year)
    min_coverage = float(os.environ.get("MIN_COVERAGE", "0.90"))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    s3 = boto3.client("s3")
    log(f"NeerRikshan yearly update, season {year}, model {MODEL}")

    # 1. Earth Engine login. The key never goes into the image: it sits in the
    # private bucket under secrets/, which only this task's role can read (the
    # API role is limited to data/). gee_auth picks the path up through the
    # environment like it does from .env. SSM Parameter Store would be the
    # usual home, but the account's organization policy denies it.
    key_file = Path(tempfile.gettempdir()) / "gee-key.json"
    s3.download_file(data_bucket, "secrets/gee-key.json", str(key_file))
    os.environ["GEE_SERVICE_ACCOUNT_KEY_PATH"] = str(key_file)

    # 2. inputs
    log("1/6 downloading current results and model")
    (DATA / "models").mkdir(parents=True, exist_ok=True)
    for name in RESULTS:
        s3.download_file(data_bucket, f"data/{name}", str(DATA / name))
    s3.download_file(data_bucket, f"models/unet_{MODEL}.pt",
                     str(DATA / "models" / f"unet_{MODEL}.pt"))

    # 3. imagery
    log(f"2/6 exporting {year} patches from Earth Engine")
    for _ in range(2):
        run("scripts.export_patches", "--split", "all", "--year", str(year), "--fit")
    want = len(json.loads((DATA / "himag_all.json").read_text(encoding="utf-8"))["lakes"])
    got = len(list((DATA / "patches" / f"all_{year}_fit").glob("*.npz")))
    log(f"2/6 {got}/{want} lakes exported")
    if got < min_coverage * want:
        raise SystemExit(f"coverage {got}/{want} below {min_coverage:.0%} - "
                         "refusing to publish an incomplete year")

    # 4-6. same scripts as on a laptop
    log(f"3/6 scoring {year} and appending to the series")
    run("scripts.run_inference", "--model", MODEL, "--fit",
        "--years", str(year), "--append")
    log("4/6 refitting trends")
    run("scripts.detect_change", "--min-years", "4")
    log(f"5/6 rendering {year} thumbnails")
    run("scripts.render_thumbnails", "--model", MODEL, "--fit", "--top", str(THUMB_TOP))

    # 7. publish: thumbnails before JSON, so the API never lists an image
    # that is not there yet
    log("6/6 publishing")
    thumbs = sorted((DATA / "thumbnails").glob(f"*_{year}*.png"))
    for f in thumbs:
        s3.upload_file(str(f), site_bucket, f"thumbnails/{f.name}",
                       ExtraArgs={"ContentType": "image/png"})
    log(f"  {len(thumbs)} thumbnails -> s3://{site_bucket}/thumbnails/")
    for name in RESULTS:
        s3.copy_object(Bucket=data_bucket, Key=f"backups/{stamp}/{name}",
                       CopySource={"Bucket": data_bucket, "Key": f"data/{name}"})
    log(f"  previous results backed up -> s3://{data_bucket}/backups/{stamp}/")
    for name in RESULTS:
        s3.upload_file(str(DATA / name), data_bucket, f"data/{name}",
                       ExtraArgs={"ContentType": "application/json"})
    log(f"  new results -> s3://{data_bucket}/data/")

    # 8. The API caches results in memory per Lambda instance. Changing its
    # configuration retires every warm instance, so the next request reads
    # the new files - the serverless equivalent of restarting the server.
    lam = boto3.client("lambda")
    env = lam.get_function_configuration(FunctionName=api_function) \
        .get("Environment", {}).get("Variables", {})
    env["DATA_VERSION"] = stamp
    lam.update_function_configuration(FunctionName=api_function,
                                      Environment={"Variables": env})
    log(f"API {api_function} refreshed (DATA_VERSION={stamp})")
    log("done")


if __name__ == "__main__":
    main()
