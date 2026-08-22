# Pipeline scripts

Offline scripts that query Google Earth Engine and write artifacts into
`server/data/`. **The API never imports these** — it only reads the JSON they
produce. GEE calls take 30 s to several minutes, so nothing here belongs in a
request path.

## One-time setup

1. Register a Google Cloud project for Earth Engine at
   <https://console.cloud.google.com/earth-engine>. Noncommercial use is free,
   requires no billing, and access is normally enabled immediately.
2. Authenticate once:
   ```bash
   earthengine authenticate
   ```
   Credentials are cached in your user profile, outside this repo.
3. Copy `server/.env.example` to `server/.env` and set `GEE_PROJECT` to your
   Cloud project id.

Service-account auth is supported but not recommended for local work — set
`GEE_SERVICE_ACCOUNT_EMAIL` and `GEE_SERVICE_ACCOUNT_KEY_PATH` only for
non-interactive runs, and keep the key file outside the repo.

> `.env` is resolved relative to `server/`, not your shell's working directory,
> so it loads correctly no matter where you invoke Python from.

## `probe_observations.py` — run this first

Answers the question the rest of the pipeline depends on: **are there enough
clear Sentinel-2 looks at a single pond in a single calendar month to build a
per-pond monthly climatology?**

Compute is not the constraint on this project; clear observations are. Kolkata's
monsoon is heavily clouded and L2A coverage over India is patchy before ~2019.
Both are assumptions until this script measures them.

```bash
cd server
python -m scripts.probe_observations
```

Options: `--ponds N` (default 20), `--start-year`, `--end-year`, `--force`
(re-query instead of reusing the cached CSV).

It samples pond-sized regions from JRC Global Surface Water rather than deriving
the project's real units — those depend on an MNDWI threshold that has not been
tuned yet, so using them here would make the measurement circular.

### Reading the output

Per calendar month, the median number of *year-samples* across sampled ponds — a
year-sample being a year with at least one observation where ≥70% of the pond's
pixels were clear:

| verdict | meaning | action |
|---|---|---|
| `usable` | ≥5 year-samples | build the climatology for this month |
| `pool +/-1` | 3–4 year-samples | pool with the neighbouring months |
| `skip` | <3 year-samples | emit **no** anomaly for this month — say so in the UI rather than publishing a number derived from two looks |

The final verdict block recommends one of three mechanisms: per-month
climatology as planned, pooled months, or **harmonic regression** (fit an annual
sin/cos cycle across all observations at once) if the per-month bins come back
thin across the board.

Expected but unconfirmed: usable Oct–May, unusable Jun–Sep. That would be
acceptable — winter harvest drainage, the false positive this design exists to
suppress, happens in the dry season when imagery is best.

## Outputs

```
server/data/_intermediate/    gitignored; regenerable query results
  probe_rows.csv              raw per-(image, pond) clear fractions
  probe_result.json           the analysed verdict
```

Raw rows are cached so the analysis can be re-run locally in milliseconds
without re-querying Earth Engine. All statistics are computed in plain Python
for exactly that reason.
