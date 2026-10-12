"""Glacial lake monitoring endpoints.

Serves the precomputed outputs of the offline pipeline:
    scripts/run_inference.py  -> data/lake_timeseries.json
    scripts/detect_change.py  -> data/lake_change.json

Nothing here touches Earth Engine or the model - the API is a thin reader over
committed artifacts, so responses are fast and reproducible. Files come from
server/data locally and from S3 when deployed (app.core.storage).

Two data-quality ideas are carried through to the API rather than hidden,
because a change number without them is misleading:
  * `suspect`  - the model's first-year area disagrees badly with Hi-MAG 2017,
                 which usually means unstable segmentation at that lake
  * `r2`       - trend fit quality; a large percentage change with low r2 is
                 noise, not a finding
Clients are expected to surface both.
"""

from __future__ import annotations

import math
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from app.core import storage

router = APIRouter(prefix="/lakes", tags=["lakes"])

CHANGE = "lake_change.json"
SERIES = "lake_timeseries.json"


def _load(name: str) -> dict:
    doc = storage.read_json(name)
    if doc is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{name} not found. Run the pipeline first: "
                "python -m scripts.run_inference && python -m scripts.detect_change"
            ),
        )
    return doc


@lru_cache(maxsize=1)
def _change() -> dict:
    return _load(CHANGE)


@lru_cache(maxsize=1)
def _exposure() -> dict:
    """Downstream population and infrastructure, keyed by lake id.

    Optional: the map and table work without it, they simply cannot rank by
    consequence. Produced by scripts.trace_downstream.
    """
    doc = storage.read_json("lake_exposure.json")
    return doc.get("lakes", {}) if doc else {}


def _exposure_score(r: dict) -> float | None:
    """SCREENING score for where to look first. NOT a probability of failure.

        score = (lower 95% bound on ha/yr)  x  log10(1 + people downstream)

    What this combines is two OBSERVED quantities: how fast the lake is growing,
    and how many people live along the path water would take if it drained. It
    is a way to order 94 lakes for attention, nothing more.

    THIS IS NOT GLOF RISK. Risk in the disaster-management sense requires a
    probability of outburst, and none of the terms that would determine one are
    measured here:

        * moraine-dam geometry, freeboard and internal structure
        * ice-cored dam degradation and permafrost state
        * lake volume and depth (we measure AREA only, from above)
        * slope failure, icefall and avalanche potential into the lake
        * seismic or extreme-precipitation triggers
        * breach hydraulics and downstream flood magnitude

    A lake with a high score is one worth looking at. It is not a lake that is
    about to burst, and a low score is not a safety certificate - a small stable
    lake with a fragile dam above a town outranks anything here, and this
    pipeline cannot see it.

    Infrastructure is reported alongside but deliberately NOT folded in:
    converting megawatts into people-equivalents would be an invented exchange
    rate. Plant count and capacity are surfaced for a human to weigh.
    """
    hazard = r.get("priority") or 0.0
    if hazard <= 0:
        return None
    exp = _exposure().get(r["id"])
    if not exp or exp.get("pop") is None:
        return None
    return round(hazard * math.log10(1 + max(exp["pop"], 0)), 4)


# Screening bands. Thresholds are round numbers chosen against the observed
# distribution (median 5,370 people downstream, median 0.64 ha/yr among lakes
# with a detectable trend) and are stated in the UI rather than hidden, because
# any cut-point here is a judgement call and a reader deserves to disagree with it.
BAND_WEIGHT = {"inspect_first": 4, "monitor": 3, "watch": 2,
               "low_exposure": 1, "unassessed": 0}
BAND_LABEL = {
    "inspect_first": "Inspect first",
    "monitor": "Monitor closely",
    "watch": "Keep watching",
    "low_exposure": "Few people downstream",
    "unassessed": "Downstream not traced",
}


def _band(r: dict) -> str | None:
    """Turn two numbers into the decision the numbers are actually for.

    A reader cannot act on "score 27.58". They can act on "growing fast, with a
    town downstream". Both inputs are observed; the banding is the only
    judgement, and it is deliberately coarse - the output is a reading order for
    94 lakes, not a calculation.
    """
    growth = r.get("priority") or 0.0          # lower 95% bound, ha/yr
    if growth <= 0:
        return None
    pop = r.get("downstream_pop")
    if pop is None:
        return "unassessed"
    if pop >= 10_000 and growth >= 1.0:
        return "inspect_first"
    if pop >= 1_000 and growth >= 0.5:
        return "monitor"
    if pop >= 1_000:
        return "watch"
    return "low_exposure"


def _summary(r: dict) -> str | None:
    """One sentence a non-specialist can act on."""
    growth = r.get("priority") or 0.0
    if growth <= 0:
        return None
    pop = r.get("downstream_pop")
    mw = r.get("downstream_mw") or 0
    bits = [f"Growing at least {growth:.1f} ha/yr (95% confident)"]
    if pop is None:
        bits.append("downstream path not traced")
    elif pop >= 1000:
        bits.append(f"~{pop:,} people within {r.get('flow_path_km', 100):.0f} km downstream")
    elif pop > 0:
        bits.append(f"~{pop:,} people downstream")
    else:
        bits.append("no settlements found downstream")
    if mw:
        bits.append(f"{mw:,} MW of power generation in the path")
    return " · ".join(bits)


@lru_cache(maxsize=1)
def _series_by_id() -> dict:
    doc = _load(SERIES)
    return {lake["id"]: lake for lake in doc["lakes"]}



def _priority(r: dict) -> float:
    """Attention score for GLOF-relevant monitoring.

    Growth is the hazard direction, so shrinking lakes score 0. Beyond that the
    score is deliberately physical rather than clever:

        lower 95% confidence bound on area gained per year (ha/yr)

    Absolute gain, not percentage, because a 40 ha lake gaining 4 ha/yr holds
    far more water than a 12 ha lake gaining the same percentage - and it is
    water volume that arrives downstream.

    The rank uses the lower CONFIDENCE BOUND rather than the fitted slope. Only
    152 of 850 lakes (18%) have a trend distinguishable from zero, so ranking
    the other 82% by a point estimate was ranking noise. A lake earns a place
    only if we are confident it is growing at all, and it is then ordered by the
    growth we can defend rather than the growth we happened to fit. This
    replaces slope x r2, which let a confident-looking fit through without ever
    asking whether the line differed from flat.
    """
    if not r.get("detectable") or r.get("direction") != "growing":
        return 0.0
    lower = r.get("slope_ci_lo")
    if lower is None or lower <= 0:
        return 0.0
    return round(lower, 4)


def _decorate(rows: list[dict]) -> list[dict]:
    """Attach priority + rank. Rank is over growing lakes only."""
    exp = _exposure()
    for r in rows:
        r["priority"] = _priority(r)
        e = exp.get(r["id"])
        if e:
            r["downstream_pop"] = round(e.get("pop") or 0)
            r["downstream_plants"] = e.get("plants") or 0
            r["downstream_mw"] = round(e.get("plant_mw") or 0)
            r["flow_path_km"] = e.get("path_km")
            r["settlement_class"] = e.get("smod")
        r["exposure_score"] = _exposure_score(r)
        band = _band(r)
        r["screening_band"] = band
        r["screening_label"] = BAND_LABEL.get(band) if band else None
        r["screening_summary"] = _summary(r)
        # Sortable key that puts the band first and the score second. Sorting on
        # score alone buries "inspect first" lakes beneath faster-growing ones
        # with nobody downstream - which is the ordering this whole feature
        # exists to correct.
        r["screening_priority"] = (
            BAND_WEIGHT.get(band, 0) * 1000 + min(r.get("exposure_score") or 0, 999)
            if band else 0
        )
    ranked = sorted([r for r in rows if r["priority"] > 0],
                    key=lambda r: -(r.get("screening_priority") or 0))
    for i, r in enumerate(ranked, 1):
        r["priority_rank"] = i
    for r in rows:
        r.setdefault("priority_rank", None)
    return rows


@router.get("")
def list_lakes(
    subregion: str | None = None,
    min_r2: float = Query(0.0, ge=0.0, le=1.0),
    include_suspect: bool = False,
    sort: str = Query("screening_priority", pattern="^(screening_priority|priority|exposure_score|downstream_pop|slope_pct_per_yr|pct_change|abs_change_ha|area_last_ha)$"),
    limit: int = Query(500, ge=1, le=2000),
):
    """Lakes ranked by trend, strongest growth first."""
    rows = _decorate(_change()["lakes"])
    if subregion:
        rows = [r for r in rows if r["subr"] == subregion]
    if not include_suspect:
        rows = [r for r in rows if not r["suspect"]]
    rows = [r for r in rows if (r.get("r2") or 0) >= min_r2]
    rows = sorted(rows, key=lambda r: -(r.get(sort) or 0))
    return rows[:limit]


@router.get("/stats")
def stats():
    """Headline numbers for the overview strip."""
    doc = _change()
    clean = [r for r in doc["lakes"] if not r["suspect"]]
    strong = [r for r in clean if (r.get("r2") or 0) >= 0.5]
    detectable = [r for r in clean if r.get("detectable")]
    growing = [r for r in detectable if r.get("direction") == "growing"]
    shrinking = [r for r in detectable if r.get("direction") == "shrinking"]
    first = sum(r["area_first_ha"] for r in clean)
    last = sum(r["area_last_ha"] for r in clean)
    return {
        "lakes_analysed": doc["n_analysed"],
        "lakes_clean": len(clean),
        "lakes_strong_trend": len(strong),
        "lakes_detectable": len(detectable),
        "lakes_no_trend": len(clean) - len(detectable),
        "inspect_first": sum(1 for r in _decorate(doc["lakes"])
                             if r.get("screening_band") == "inspect_first"),
        "monitor": sum(1 for r in _decorate(doc["lakes"])
                       if r.get("screening_band") == "monitor"),
        "with_infrastructure": sum(1 for r in _decorate(doc["lakes"])
                                   if (r.get("downstream_mw") or 0) > 0),
        "growing": len(growing),
        "shrinking": len(shrinking),
        "total_area_first_ha": round(first, 1),
        "total_area_last_ha": round(last, 1),
        "total_area_change_pct": round((last - first) / first * 100, 2) if first else None,
        "model": doc.get("model"),
    }


@router.get("/geojson")
def geojson(include_suspect: bool = False, min_r2: float = 0.0):
    """Point FeatureCollection for the map, one feature per lake."""
    feats = []
    for r in _decorate(_change()["lakes"]):
        if r.get("lon") is None or r.get("lat") is None:
            continue
        if not include_suspect and r["suspect"]:
            continue
        if (r.get("r2") or 0) < min_r2:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": r["id"], "subr": r["subr"],
                "pct_change": r["pct_change"],
                "priority": r.get("priority"),
                "priority_rank": r.get("priority_rank"),
                "slope_pct_per_yr": r["slope_pct_per_yr"],
                "detectable": r.get("detectable", False),
                "direction": r.get("direction"),
                "exposure_score": r.get("exposure_score"),
                "downstream_pop": r.get("downstream_pop"),
                "downstream_mw": r.get("downstream_mw"),
                "screening_band": r.get("screening_band"),
                "area_last_ha": r["area_last_ha"],
                "r2": r["r2"], "suspect": r["suspect"],
            },
        })
    return {"type": "FeatureCollection", "features": feats}


@router.get("/flowpaths")
def flowpaths(limit: int = Query(120, ge=1, le=500)):
    """Downstream flood corridors as LineStrings, for the map.

    Declared before /{lake_id} so the path parameter does not swallow it.
    """
    exp = _exposure()
    if not exp:
        return {"type": "FeatureCollection", "features": []}
    rows = [r for r in _decorate(_change()["lakes"])
            if r["id"] in exp and (r.get("priority") or 0) > 0]
    rows.sort(key=lambda r: -(r.get("exposure_score") or 0))
    feats = []
    for r in rows[:limit]:
        e = exp[r["id"]]
        path = e.get("path") or []
        if len(path) < 2:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[c[0], c[1]] for c in path]},
            "properties": {
                "id": r["id"],
                "exposure_score": r.get("exposure_score"),
                "downstream_pop": r.get("downstream_pop"),
                "downstream_mw": r.get("downstream_mw"),
                "priority_rank": r.get("priority_rank"),
                "path_km": e.get("path_km"),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


@router.get("/{lake_id}")
def get_lake(lake_id: str):
    """Full record for one lake: trend plus the raw per-year series."""
    row = next((r for r in _decorate(_change()["lakes"]) if r["id"] == lake_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"lake {lake_id} not found")
    raw = _series_by_id().get(lake_id, {})
    exp = _exposure().get(lake_id) or {}
    return {
        **row,
        "raw_years": raw.get("years", {}),
        "himag_2017_ha": raw.get("himag_2017_ha", row.get("himag_2017_ha")),
        # geometry for the downstream panel; absent for untraced lakes
        "flow_path": exp.get("path"),
        "settlements": exp.get("settlements"),
        "drop_m": exp.get("drop_m"),
    }


@router.get("/{lake_id}/frames")
def lake_frames(lake_id: str):
    """Years for which a true-colour thumbnail exists, oldest first.

    Returned as a list rather than a range because rendering is opt-in per
    lake (only the watchlist is rendered by default), so gaps are normal and
    the client must not assume a contiguous series.
    """
    frames = []
    for name in storage.thumbnail_names(lake_id):
        stem = name.removesuffix(".png")
        if stem.endswith("_ov"):
            continue
        year = stem.rsplit("_", 1)[-1]
        if not year.isdigit():
            continue
        frames.append({
            "year": int(year),
            "rgb": f"/lakes/{lake_id}/frame/{year}",
            "overlay": f"/lakes/{lake_id}/frame/{year}?overlay=true",
        })
    return {"lake_id": lake_id, "frames": frames}


@router.get("/{lake_id}/frame/{year}")
def lake_frame(lake_id: str, year: int, overlay: bool = False):
    """One rendered thumbnail. `overlay` draws the model's water outline."""
    name = f"{lake_id}_{year}{'_ov' if overlay else ''}.png"
    if storage.on_s3():
        # deployed: the image sits in the site bucket and CloudFront serves it
        # from the same domain, so send the browser there instead of piping
        # the bytes through Lambda
        return RedirectResponse(f"/thumbnails/{name}", status_code=302)
    path = storage.thumbnail_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"no frame {name}")
    return FileResponse(path, media_type="image/png")
