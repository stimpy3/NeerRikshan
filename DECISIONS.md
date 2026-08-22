# Decisions, Failures & Findings

**Glacial Lake Change Monitoring — Himalaya** · Sentinel-2 · 2019–2025 · lakes ≥10 ha

Engineering log. The wrong turns matter more than the headline number.

---

## The gap

`GLOF risk` `inventories end 2017` `no live series`

- Glacial lakes grow → moraine dam fails → **glacial lake outburst flood**
- South Lhonak, Sikkim, Oct 2023: dozens dead, Teesta III dam destroyed
- Hi-MAG inventory **stops at 2017**. No public per-lake annual series exists
- → extend to 2025, **rank** growth instead of rediscovering it

---

## Headline result

`held-out` `100 lakes` `300 lake-years`

```
                IoU    precision  recall   IoU>=0.7   |area err|
MNDWI baseline  0.318    0.336     0.812     17%        114%
U-Net           0.762    0.848     0.894     72%         18%
```

- Biggest gain where baseline failed worst: **10–20 ha lakes, +0.559 IoU**
- Trained on 2019 only → flat across 2019/2022/2025 (no temporal drift)

---

## 1. Verified data by fetching, not assuming

`no unverified deps`

| | |
|---|---|
| ✅ Hi-MAG lakes | 58.7 MB, CC-BY-4.0, downloaded |
| ✅ geoBoundaries | 48 MB GeoJSON, parsed |
| ❌ Global Tailings Portal | "API" returns `{"sales": 100}`; needs written permission |
| ❌ Kumar & Vijay 2022 inventory | Zenodo: **"Files Restricted"** |

**Rule:** data access needs a *successful fetch* before it enters a plan.
Learned the hard way on a prior project — one unverified line cost a day.

---

## 2. Predictions that were wrong

`measured, not assumed`

| Predicted | Actual |
|---|---|
| L2A archive starts ~2019 | **2017-01-15** |
| Monsoon Jun–Sep unusable | **June usable** (8/10 yrs); only Jul–Aug fail |
| Terrain **shadow** causes failure | **Snow/ice** does |
| DEM will improve model | **It doesn't** |
| ≥10 ha rescues baseline | It doesn't (0.485 vs 0.461) |

Caught by a feasibility probe run **before** building the pipeline.

---

## 3. Core insight: MNDWI *is* NDSI

`snow == water algebraically`

- MNDWI = `(B3−B11)/(B3+B11)` = **NDSI**, the standard snow index
- Cannot separate snow/ice from water. Snow touches every glacial shoreline

Explains every symptom at once:

- excess detection **contiguous** with lake
- **threshold tuning useless** → IoU 0.485 → 0.495 → 0.464 → 0.441
- **recall 0.88, precision 0.34**

Second failure: over dark water both bands → 0, ratio flips sign on noise.
**27% of training patches had negative mean MNDWI over real water.**

**Discriminator (measured, 668 patches):** NIR `0.016–0.024` over water vs
`0.35–0.47` over snow — **~24× separation**, invisible to a band ratio.

---

## 4. Noise floor killed the aggregate

`signal ≈ noise`

```
year-to-year scatter:   2.8% of lake area
7-year trend signal:   ~3.1%
```

- Typical lake's whole 7-yr change = its own noise
- Only **19% (157/830)** reach R² ≥ 0.5
- Six years is too short. Landsat extension is the real fix, not a better model

---

## 4b. The "net shrinkage" anomaly was a model artifact

`contradiction resolved` `eliminated one by one`

Early runs showed the Himalaya **losing** lake area — the opposite of every
published study. Ruled out one cause at a time on val:

| suspect | test | verdict |
|---|---|---|
| stale 2017 labels | IoU vs 2/5/8-yr gap | flat → **not it** |
| composite choice | median vs max-water | median wins → **not it** |
| model quality | IoU 0.788 | **not it** |
| **spectral inputs** | MNDWI → NDWI, drop NIR gate | ✅ **this was it** |

| | before | after |
|---|---|---|
| aggregate change | **−1.3%** | **+1.1%** |
| growing / shrinking | 92 / 140 | **156 / 96** |
| reliable trends | 103 | **157** |
| suspect lakes | 49 | **3** |

Direction now matches the literature. Magnitude still low (+1.1% vs ~+3.5%
implied) — that residual is the **6-year window**, per §4.

**Lesson:** a result contradicting the field is a bug report about your pipeline
until each link is individually eliminated. The wrong index looked like a
finding for weeks.

---

## 4c. Validated against labels we never trained on

`external check` `the labels were part of the error`

Kumar & Vijay (2026), Sentinel-2 **10 m**, HMA — used as a yardstick only, never
as training data. KV polygons painted into the same GEE grid; alignment proven by
re-painting the Hi-MAG mask and matching the stored label at **IoU 1.0000**.

```
                              IoU     precision  recall
Hi-MAG 2017 (30m, trained on) 0.783     0.853     0.909
KV 2022-24 (10m, EXTERNAL)    0.846     0.943     0.888
```

**The model scores higher against labels it has never seen.** Precision
0.853 → 0.943: much of its "false positive" water was real, and Hi-MAG's 30 m
pixels could not resolve it. No train/val/eval gap (0.841 / 0.859 / 0.844).

**Big lakes are its best case, not its worst** — IoU rises monotonically with
size to **0.976** at 100+ ha. The 55% area shortfall on those same lakes is
**truncation**: 100% of them touch the window edge and only 57% fits inside.
Segmentation and framing are separate failures, and only the second is real.

---

## 4d. Exposure screening — and what it must NOT be called

`hazard != risk` `naming discipline`

Downstream flow traced with HydroSHEDS D8 (3 arc-sec), 100 km per lake, then
intersected with GHSL population, GHSL settlement class and WRI power plants.

**Validated on a real event.** Traced from South Lhonak, Sikkim — the Oct 2023
GLOF that destroyed Teesta III:

```
5,200 m -> 274 m over 150 km, exiting at Sevoke where the Teesta leaves the hills
downstream population 49,997   power plants 1 (510 MW)   settlement class 22
```

All 94 traces net-descend (median drop 2,451 m). The per-step descent metric
reads only 0.87 because at 90 m a vertex in a gorge can land on a valley wall —
the metric is noisy, the paths are not.

**Naming was wrong and got corrected.** I shipped this as `risk_score`,
described as "hazard x consequence". That overclaims: risk requires a
**probability of outburst**, and none of its determinants are measured here —
moraine geometry and freeboard, ice-core degradation, lake volume and depth
(we see surface area only), avalanche potential above the lake, triggers, breach
hydraulics. Renamed to **exposure screening**. The honest claim is:

> This lake is changing, and if an outburst occurred, these are the downstream
> areas potentially exposed.

A high score means *worth looking at*. A low score is **not** a safety
certificate — a small stable lake with a fragile dam above a town outranks
everything here, and this pipeline cannot see it.

---

## 5. Negative result: DEM doesn't help

`ablation` `premise disproved`

| model | median IoU | snow errors (NIR>0.15) |
|---|---|---|
| `optical` (6 bands) | **0.783** | **3.7%** |
| `noterr` (+MNDWI) | 0.774 | 6.7% |
| `all` (+DEM) | 0.762 | 8.1% |

Project was scoped "U-Net **+ DEM**". Terrain made it **worse**.
⚠ Single seed — inter-variant ranking not firm; baseline gap far too large to be noise.

---

## 6. Bugs that would have silently corrupted results

`silent corruption`

- **`np.nan_to_num` maps ±inf → ±3.4e38**, not 0. `np.isfinite()` called them
  clean. Caught only via `-inf` band means. 106 patches repaired
- **Map ignored every filter** — fetched its own fixed dataset while table filtered
- **`map.isStyleLoaded()` false indefinitely** — gating on it left map with *zero* layers
- **Colour scale ±4 %/yr** vs real ±1.5 → 80% of lakes rendered flat grey
- **FastAPI 422 `detail` = array of objects** → rendering it crashed React, blank page
- **Stale uvicorn ×3** — edited routes without restart, looked like app bugs

---

## 7. Imagery caught what statistics missed

`ship evidence with the number`

Top-ranked lake `GL090266E28104N`: **+97%, R²=0.82, all years flagged `ok`**

- **2019 frame:** hazy terrain, no visible lake, outline on non-water
- **2025 frame:** lake fills window (`patch_frac` 0.576)

→ magnitude unreliable **in both directions**. Passed every statistical check,
failed visual inspection in seconds.

UI now shows per-year imagery + segmentation outline toggle.

---

## 8. Open problems

`known limits`

1. **Snow confusion ~10% of frames** (NIR > 0.15 in predicted water). Dropping
   MNDWI input nearly halves it
2. **Fixed 1.28 km window** too small for largest lakes → areas are lower bounds.
   **Now quantified against an independent 10 m inventory: lakes >100 ha are
   recovered at only 55% of true area**, declining monotonically with size. The
   `saturated` flag at `patch_frac>0.60` is too lenient — truncation costs ~45%
   by 0.49. This also *suppresses measured growth* on the largest lakes
3. **Magnitude undershoots** (+1.1% vs ~+3.5% implied) — window too short (§4)
4. **Label/imagery mismatch** — labels 2017, imagery 2019–25 → all scores are floors.
   Scoped precisely: this caps the **segmentation evaluation**, not the
   2019–2025 change detection. Trends compare the *same model* across years, so
   a fixed label bias cancels in the difference. IoU 0.788 is a **lower bound**;
   the trends stand on their own. Two further limits sit inside this one:
   Hi-MAG is **30 m Landsat** while our imagery is **10 m Sentinel-2**, and the
   measured shoreline bleed (72% of false positives within 30 m of the edge) is
   **exactly one Landsat pixel** — so part of the "model error" is plausibly
   label quantisation, not the model
5. **Single-seed ablation** (§5)
6. **Small lakes untouched** — median lake 2.91 ha, 36.5% under 2 ha; ≥10 ha is
   only ~17% of inventory

---

## Method notes

`transferable`

- **Gate before building** — feasibility probe caught 2 wrong assumptions for the
  cost of one script
- **Validation that refuses to emit** — pipeline won't write output failing its checks
- **Label-free QC** — mean NIR inside predicted mask flags bad segmentation with
  no ground truth; works in production
- **Read 474 MB with 7 MB** — parquet footers + HTTP range requests before
  committing to downloads
