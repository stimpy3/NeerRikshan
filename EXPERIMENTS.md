# Experiment Log

One change at a time. Each stage records what changed, what moved, and the
verdict — including ties and regressions.

**Rules**

- Every experiment is scored on **val (217 lakes)**. The **eval (100)** set is
  frozen and opened once, at the end.
- A change is **REAL** only if the mean paired per-lake delta exceeds half its
  own standard deviation across lakes. Otherwise it is a **tie** — recorded, not
  stacked.
- Comparisons are **paired** (same lake, two models). Lake difficulty varies far
  more than model quality, so unpaired medians hide real effects and invent fake
  ones.

---

## Setup — split rebalance

**Why.** With 117 val lakes, differences under ~0.02 IoU are indistinguishable
from noise, and several planned experiments were expected to land in that range.

**Change.** Moved 100 lakes train → val.

| split | before | after |
|---|---|---|
| train | 668 | **568** |
| val | 117 | **217** |
| eval | 100 | 100 (frozen, untouched) |

Verified: **zero overlap** between all three splits, 885 distinct lakes total.
Patch files moved to match; no re-export or re-download needed.

**Cost not anticipated in the plan:** the three existing checkpoints
(`optical`, `noterr`, `all`) were trained on lakes that are now validation data.
They are **invalid** and were retrained. Stage 1 therefore needed 4 runs, not 2.

Val sub-region mix: Central 96 / Eastern 93 / Western 28 — mirrors the
population (Western is genuinely the smallest region at 138 of 885).

---

## Stage 0 — Baseline (E0)

U-Net `noterr` (optical + MNDWI), 128 px lake-centred patches, retrained on the
568/217 split. **Val (217 lakes), no NIR gate:**

| metric | value |
|---|---|
| IoU | **0.725 ± 0.184** (median 0.772, p10 0.520) |
| precision | 0.785 |
| recall | 0.902 |
| snow rate (NIR > 0.15) | 10.2% |
| catastrophic failures (IoU < 0.5) | **21 / 217** |

Note this is *lower* than the 0.774 previously reported — that figure came from
the old, leaky split where val lakes had been seen in training. This is the
honest reference.

---

## Stage 1 — Band sets (E3)

**Hypothesis.** The dominant error is snow confusion, and it is an *input*
problem, not a capacity problem. MNDWI = `(B3−B11)/(B3+B11)` is algebraically
**NDSI**, the standard snow index, so it cannot separate snow from water.

**Correction to the original plan.** Adding "NDSI" as a channel is a no-op — it
*is* the existing `mndwi` band. The index that breaks the tie is
**NDWI = `(B3−B8)/(B3+B8)`** (green vs **NIR**, not SWIR).

**Measured separation** (126 patches containing both water and snow):

| index | water | snow | separation |
|---|---|---|---|
| MNDWI | 0.444 | **0.598** | 0.154 |
| **NDWI** | 0.478 | 0.017 | **0.462** |
| NDVI | −0.234 | −0.026 | 0.208 |

Snow scores **higher** than water under MNDWI — that is the shortcut the model
learned. NDWI separates them 3× better.

**Variants** (NDWI/NDVI derived on the fly from bands already in each patch — no
re-export):

| | bands | channels |
|---|---|---|
| A `optical` | B2,B3,B4,B8,B11,B12 | 6 |
| B `noterr` | A + MNDWI | 7 |
| C `ndwi` | A + NDWI + NDVI | 8 |
| D `allidx` | A + MNDWI + NDWI + NDVI | 9 |

### Results — val (217 lakes), no NIR gate

| model | IoU mean ± std | median | p10 | prec | recall | snow% | fails <0.5 |
|---|---|---|---|---|---|---|---|
| B `noterr` | 0.725 ± 0.184 | 0.772 | 0.520 | 0.785 | 0.902 | 10.2% | 21 |
| A `optical` | 0.722 ± 0.219 | 0.804 | 0.486 | 0.845 | 0.805 | 9.1% | 24 |
| **C `ndwi`** | **0.788 ± 0.130** | 0.818 | 0.576 | 0.868 | 0.902 | 11.5% | **9** |
| D `allidx` | 0.790 ± 0.135 | 0.822 | 0.583 | 0.857 | 0.913 | 12.0% | 7 |

### Paired per-lake delta vs `noterr`

| model | ΔIoU mean ± std | lakes improved | verdict |
|---|---|---|---|
| `optical` | −0.0033 ± 0.109 | 54% | **tie** |
| `ndwi` | **+0.0633 ± 0.111** | **80%** | **REAL** |
| `allidx` | **+0.0651 ± 0.114** | 78% | **REAL** |

`allidx` vs `ndwi`: **+0.0018 ± 0.048, 45% improved → tie.**

### Verdict: adopt `ndwi` (C)

- **+0.063 IoU, improving 80% of lakes** — hypothesis confirmed, and large.
- **Catastrophic failures more than halved: 21 → 9.**
- **Variance dropped** (std 0.184 → 0.130) and p10 rose 0.520 → 0.576 — the gain
  is concentrated in the lakes that were worst, not the easy ones.
- **MNDWI adds nothing once NDWI is present** (`allidx` ≈ `ndwi`, tie). Choose the
  simpler 8-channel model.
- **MNDWI alone adds nothing over raw optical** (`noterr` ≈ `optical`, tie) — the
  model could already derive it from the bands; what it lacked was the *right*
  index.

### ⚠ Consequence: the NIR gate is now obsolete

Re-testing the Stage-0-era NIR ≤ 0.15 filter on the new models:

| model | IoU no gate | IoU with gate | Δ | fails <0.5 |
|---|---|---|---|---|
| `noterr` | 0.725 | 0.689 | −0.036 | 21 → 34 |
| `ndwi` | **0.788** | 0.723 | **−0.065** | 9 → 28 |
| `allidx` | 0.790 | 0.726 | −0.064 | 7 → 29 |

**The gate now costs 0.065 IoU and triples catastrophic failures.** It was a
post-hoc patch for a problem the model has now learned to solve properly; with
NDWI in the inputs it only removes legitimate turbid/shallow water.

**Action: disable the NIR gate in `run_inference.py` when switching to `ndwi`.**

> **Caveat on the snow metric.** `snow%` rose slightly (10.2% → 11.5%) even as
> IoU improved sharply and failures halved. The proxy measures mean NIR inside
> the prediction, which rises when the model correctly includes marginal turbid
> water. It was a good diagnostic when the model was failing outright; it is a
> less reliable signal now, and should not override IoU and failure count.

---

## Stage 2 — Loss function (E5, E6)

**E5 Focal** — `BCE + Dice` → `+ Focal`. Targets hard pixels (shorelines, snow
edges).

**E6 Boundary** — `BCE + Dice` → `+ Boundary` (distance-transform). Targets the
measured shoreline bleed: 72% of false positives lie within 30 m of the true edge.

Run separately, never combined until each is understood alone. Both trained on
the winning `ndwi` inputs.

### Results — val (217 lakes)

| model | IoU mean ± std | median | p10 | prec | recall | fails <0.5 |
|---|---|---|---|---|---|---|
| `ndwi` (BCE+Dice) | 0.788 ± 0.130 | 0.818 | 0.576 | 0.868 | 0.902 | 9 |
| `ndwi_focal` | 0.794 ± 0.118 | 0.826 | 0.590 | 0.876 | 0.899 | 9 |
| `ndwi_boundary` | 0.785 ± 0.134 | 0.815 | 0.581 | 0.866 | 0.899 | 10 |

### Paired per-lake delta vs `ndwi`

| model | ΔIoU mean ± std | lakes improved | verdict |
|---|---|---|---|
| `ndwi_focal` | +0.0059 ± 0.040 | **42%** | **tie** |
| `ndwi_boundary` | −0.0035 ± 0.023 | 35% | **tie** |

### Verdict: no change — keep BCE + Dice

Neither loss clears the bar. Focal's mean is positive but it improves **fewer
than half** the lakes (42%) — it helps a handful substantially and mildly hurts
the majority, which is the signature of noise, not a real effect. Boundary is
marginally negative.

**Why this is a useful negative result:** the shoreline bleed is real and
measured (72% of false positives within 30 m of the true edge), yet a
boundary-weighted loss did not fix it. That points at the **label**, not the
objective — the 2017 Hi-MAG boundary genuinely disagrees with 2019 imagery, so
sharpening the model against a stale edge cannot help. This raises the priority
of Stage 8 (label gap) relative to further loss engineering.

---

## Stage 3 — Sliding-window inference (E1)

Replaces per-lake 128 px windows with overlapping tiles, border pixels discarded,
probabilities averaged over overlaps.

Fixes a **measurement** error, not a model error: 206 lake-years are flagged
`saturated` because the lake outgrew its window, making those areas lower bounds
rather than measurements.

### Approach changed on a discovery

The U-Net is **fully convolutional**, so it accepts any input size that is a
multiple of 8 (verified at 128/192/256/384). Sliding-window tiling with
overlap-averaging is therefore unnecessary — the window can simply be sized to
the lake and run in one pass. Same goal, **no stitching and no seam artifacts**,
which is the main thing overlap-averaging exists to hide.

Window scales with lake area (~3× linear extent, rounded to a multiple of 32,
clamped 128–512 px):

| lake | window |
|---|---|
| 10 ha | 128 px (1.28 km) |
| 50 ha | 224 px |
| 100 ha | 320 px |
| 636 ha | 512 px (5.12 km) |

### Result — adaptive inference on a fixed-window-trained model: FAILS

| | fixed 128 px | adaptive |
|---|---|---|
| saturated lakes (>60% of window) | 7 | **1** |
| IoU (mean, 217 lakes) | **0.788** | 0.707 |
| lakes ≥50 ha — IoU | **0.894** | **0.547** |

Paired: **−0.081 ± 0.161, only 7% of lakes improved → REAL regression.**
On lakes ≥50 ha the collapse is severe: **−0.347 IoU**.

**Why.** *Fully convolutional does not mean scale-invariant.* The model was
trained on 128 px windows where the lake occupies a characteristic fraction of
the frame; it learned that scale prior along with the appearance. Enlarge the
window and the lake shrinks relative to the receptive field, landing outside the
training distribution — worst for exactly the large lakes the change was meant
to help.

**The windowing itself works** — saturation dropped 7 → 1, so truncation is
genuinely fixed. The failure is a **train/inference mismatch**, not a flaw in
adaptive windowing.

### Correct test: multi-scale training — ALSO FAILS

Retrained `ndwi` with random-zoom augmentation (1.0–2.5×, emitted at a fixed
128 px so batches still form) so the model sees lakes at many frame fractions.

| config | IoU | std | IoU ≥50 ha | saturated |
|---|---|---|---|---|
| **`ndwi` fixed** | **0.7881** | 0.130 | **0.8937** | 7 |
| `ndwi` adaptive | 0.7070 | 0.169 | 0.5465 | 1 |
| `ndwi_ms` fixed | 0.7620 | 0.221 | 0.8772 | 13 |
| `ndwi_ms` adaptive | 0.7129 | 0.159 | **0.4905** | **0** |

Paired vs `ndwi`/fixed:

| config | ΔIoU | improved | verdict |
|---|---|---|---|
| `ndwi` adaptive | −0.0811 ± 0.161 | 7% | REAL regression |
| `ndwi_ms` fixed | −0.0261 ± 0.168 | 61% | tie (mean down) |
| `ndwi_ms` adaptive | −0.0753 ± 0.176 | 38% | tie (mean down) |

### Verdict: Stage 3 REJECTED — keep fixed 128 px

Multi-scale augmentation **recovered almost nothing** on adaptive windows
(0.707 → 0.713) and **cost performance on the fixed windows we actually ship**
(0.788 → 0.762). On large lakes under adaptive windows it was *worse* than the
un-augmented model (0.546 → 0.491).

Saturation did reach **0**, confirming the geometry fix works mechanically — but
it buys a truncation fix at the price of segmentation quality, which is a bad
trade when truncation affects only ~3% of lake-years.

**Why the augmentation likely failed.** Zoom is implemented by downsampling,
which degrades effective resolution: at 2.5× a 10 ha lake spans ~14 px in the
crop, near the limit of what the network can resolve. So the model was taught
scale-robustness using increasingly blurry evidence, and paid for it on sharp
full-resolution inputs.

### Untried alternatives (for later, not now)

1. **Train and infer at a fixed 256 px** for every lake — one consistent geometry,
   4× the area, no zoom degradation. Costs a full re-export and slower training.
2. **Variable-size training at batch size 1** — exact geometry match, no
   resampling, but roughly 5× slower per epoch.
3. **Accept truncation** and flag the ~3% of saturated lake-years as lower bounds,
   which the pipeline already does.

Option 3 is in force. Option 1 is the cheapest real fix if truncation later
matters more.

---

## Stage 4 — Seasonal composite (E2)

Compare **median / p75 / p90 / max** melt-season composites.

### Implementation error caught before the full export

The first version took the 90th percentile of **reflectance**. That is wrong:
raising every band makes water *brighter* and therefore **less** separable —
measured MNDWI over a lake fell from **+0.54 to +0.38**. "Summer maxima" in the
literature means maximum **water extent**, not maximum reflectance.

Corrected to water-index based:

| composite | method |
|---|---|
| `median` | baseline |
| `maxwater` | quality mosaic — each pixel from the date it was wettest (max NDWI) |
| `p90water` / `p75water` | keep the wetter observations, then median them |

Note `maxwater` is a **per-pixel** mosaic, so a single bad frame affects only the
pixels it covers, not the whole composite — which answers the outlier concern.

### IoU results — and why they cannot decide this

| composite | IoU | precision | recall | area vs label | ΔIoU vs median |
|---|---|---|---|---|---|
| `median` | **0.7881** | 0.868 | 0.902 | 1.039 | — |
| `maxwater` | 0.7679 | 0.812 | 0.939 | 1.126 | −0.0202 ± 0.094 |
| `p75water` | 0.7437 | 0.786 | 0.941 | 1.124 | −0.0439 ± 0.123 |
| `p90water` | 0.6946 | 0.722 | 0.958 | 1.211 | −0.0935 ± 0.150 |

Recall rises while precision falls — the water composites detect more water. Two
confounds make this table unusable for the decision:

1. **The label is the wrong yardstick.** Hi-MAG was built from median-like
   Landsat composites, so a max-extent composite is penalised for doing its job.
2. **The model was trained on median composites**, so running it on `maxwater` is
   a distribution shift — the same error that sank Stage 3.

### Deciding metric: year-to-year stability (label-free)

Residual scatter around each lake's own trend, 217 lakes × 3 years:

| composite | scatter median | p75 | p90 |
|---|---|---|---|
| **`median`** | **0.78%** | 2.53% | 10.31% |
| `maxwater` | 1.69% | 4.26% | 12.97% |

Paired: **+1.150% ± 4.78, only 36% of lakes improved → TIE, directionally worse.**

### Verdict: Stage 4 REJECTED — keep `median`

### ⭐ The elimination this completes

The project detects **+0.8%** area drift over 2019–2025 where the literature
implies **+3.5%**. That shortfall is now confirmed to be:

- ❌ not the **labels** — IoU flat across a 2→8 year gap (Stage 8)
- ❌ not the **composite** — no stability gain from any variant (Stage 4)
- ❌ not the **model** — `ndwi` scores 0.788 IoU, failures 9/217

**Therefore it is the 6-year window.** Signal ≈ 0.58 %/yr against a noise floor
of the same order. The binding constraint is **time-series length**, not anything
in the pipeline — so extending to Landsat (1990s–2018) is the real fix, and
further model work has low expected value.

> Encouraging side note: median scatter measured **0.78%** with `ndwi` here vs
> **2.8%** previously with `noterr` in production. Not directly comparable (3
> years vs 7 — fewer points fit a line more tightly), but the new model may
> genuinely lower the noise floor. The production run will settle it.

---

## FINAL MODEL

`ndwi` (6 optical + NDWI + NDVI) · **median** composite · fixed **128 px** window ·
**NIR gate OFF**

Because `median` won, the existing 6,188 production patches remain valid — no
re-export needed.

### Production re-run — the shrinkage anomaly is RESOLVED

885 lakes × 7 years, final model vs the previous deployed config:

| | before (`noterr` + NIR gate) | after (`ndwi`, no gate) |
|---|---|---|
| reliable trends (R² ≥ 0.5) | 103 | **157** (+52%) |
| growing > 5% | 92 (12%) | **156 (19%)** |
| shrinking > 5% | 140 (18%) | **96 (12%)** |
| aggregate area change | **−1.3%** | **+1.1%** |
| median lake change | −0.3% | **+0.4%** |
| lakes flagged suspect | 49 | **3** |

**The net-shrinkage result that contradicted every published study was a model
artifact.** It reversed to net *growth* once MNDWI was replaced with NDWI and the
NIR gate was removed. Direction now agrees with the literature.

Magnitude still falls short — **+1.1% measured** vs **~+3.5% implied** by
0.58 %/yr over six years — which is consistent with the elimination above: the
remaining gap is the **short time window**, not the pipeline.

Reliable trends rose by half, so the model change also materially improved
trend detectability, confirming the lower noise floor seen in the stability test.

The previous rank-1 lake `GL090266E28104N` — whose +97% was inflated by a broken
2019 frame — now reads a plausible **64.3 → 95.5 ha (+48.5%, R² 0.94)**.

---

## Stage 8 (run early) — Label gap: NOT the ceiling

**Why run this before the composite sweep.** If the 2017 labels were the binding
constraint, hours spent optimising satellite inputs would be wasted. Stage 2 had
hinted at it: a boundary-aware loss failed to fix a *measured* shoreline error.

**Test.** Score `ndwi` against the **fixed 2017 labels** on 2019 / 2022 / 2025
imagery. If disagreement is accumulating real lake change, IoU must decline as the
gap widens. No retraining needed.

| year | IoU | precision | recall | model/label area |
|---|---|---|---|---|
| 2019 | 0.7881 | 0.868 | 0.902 | 1.039 |
| 2022 | 0.7963 | 0.864 | 0.916 | 1.047 |
| 2025 | 0.7896 | 0.856 | 0.918 | 1.048 |

Paired IoU change 2019 → 2025: **+0.0015 ± 0.105** — 40% of lakes got worse,
60% better. **Flat.**

### Verdict: labels are NOT the limiting factor — skip pseudo-labelling

Disagreement is a **constant offset**, not accumulating error. Had it been the
ceiling, IoU would fall as the gap grows from 2 to 8 years. It does not move.

**Consequence:** the ~4% over-detection vs Hi-MAG is best explained by **sensor
resolution** — Hi-MAG is 30 m Landsat, this is 10 m Sentinel-2, and coarse pixels
systematically under-count lake edges. That bias is constant, and since change
detection compares Sentinel-2 to Sentinel-2, **it cancels out of the trends**. It
would only distort absolute-area claims made against Hi-MAG.

### The real anomaly, now isolated

| | |
|---|---|
| measured area drift 2019 → 2025 | **+0.8%** |
| expected from published HMA expansion (0.58%/yr) | **+3.5%** |

We detect roughly **a quarter** of the growth the literature reports — and it is
now confirmed **not** to be a label artifact. That points squarely at the
measurement chain: seasonal compositing (Stage 4) is the prime remaining suspect.

> **Correction.** A smoke test on the first 30 lakes of `all_2019` showed a
> model/label ratio of 1.27, and I attributed it to systematic over-detection.
> That sample was alphabetical, hence entirely Western Himalaya (~73°E). On the
> 217-lake random val sample the ratio is **1.04**. The 1.27 was a regional
> sampling artifact, not a systematic bias.

---

## Prior findings carried in (before this ladder)

| finding | evidence |
|---|---|
| U-Net ≫ MNDWI baseline | IoU 0.318 → 0.762, precision 0.336 → 0.848 (100 held-out lakes) |
| DEM does not help | optical 0.783 vs +DEM 0.762 — **single seed, unconfirmed** |
| NIR ≤ 0.15 gate | snow errors 6.7% → **0.0%**; IoU 0.774 → 0.797; **but recall 0.904 → 0.856** |
| NIR gate hurt trends | reliable trends (R²≥0.5) **136 → 103**; 49 lakes became `suspect` |
| Noise floor | year-to-year scatter **2.8%** vs 7-year signal **~3.1%** |
| Signal vs literature | our median trend **0.52 %/yr** vs published HMA **0.58 %/yr** — close agreement |
| Shadow is NOT the problem | dark false positives sit on flat, well-lit ground (slope 7.9°, hillshade 0.80); 72% within 30 m of the true shoreline |

**Key tension to keep in view:** per-frame accuracy and trend reliability are
*different objectives* and can move in opposite directions. The NIR gate improved
segmentation while reducing usable trends. Stage results are judged on both.


---

## External validation against an independent 10 m inventory

Kumar & Vijay (2026), *Inventory of Glacial Lakes in HMA, 2016-2017 and 2022-2024*
— [Zenodo 17948783](https://doi.org/10.5281/zenodo.17948783), CC-BY-4.0.
Sentinel-2 median composites at **10 m**, versus Hi-MAG's Landsat at **30 m**.
Used as an independent yardstick only — **not** as training labels, and change is
never computed by differencing it against Hi-MAG (different detection criteria
would masquerade as growth).

795 / 837 of our 885 lakes matched by mutual-nearest centroid within 750 m.

### A — our labels are systematically small

| Hi-MAG 2017 (30 m) vs KV 2016-17 (10 m) | |
|---|---|
| median ratio | **0.919** |
| median gap | **−8.1%** |
| within 10% / 25% | 51% / 91% |

Same lakes, same epoch, different sensor. Hi-MAG under-reads by ~8%, so **part of
what we scored as model error was never ours** — a 30 m label cannot place a
shoreline better than 30 m, which is precisely the scale of our measured
shoreline bleed.

### B — external validation, and a size-dependent bias

Median ratio **1.077**, but the aggregate ratio runs the *other* way (−5.8%).
Splitting by size explains it:

| reference size | our ratio | median `patch_frac` |
|---|---|---|
| 0–15 ha | 1.145 | 0.091 |
| 15–25 ha | 1.088 | 0.125 |
| 25–50 ha | 1.050 | 0.211 |
| 50–100 ha | 0.947 | 0.365 |
| **100+ ha** | **0.552** | 0.489 |

Monotonic, and tracking `patch_frac` exactly. **On lakes over 100 ha we recover
only 55% of the true area.** The fixed 1.28 km window is not a minor caveat — it
halves the largest lakes.

The `saturated` flag (`patch_frac > 0.60`, 4.4% of lake-years) is **far too
lenient**: truncation is already costing ~45% at `patch_frac ≈ 0.49`.

### C — an independent change estimate, and our shortfall quantified

Both KV epochs come from one consistent pipeline, so differencing them is valid:

| | KV 2016-17 → 2022-24 | ours 2019 → 2025 |
|---|---|---|
| aggregate change | **+3.1%** | **+1.1%** |
| median lake change | +0.8% | +0.4% |
| growing >5% | 17% | 19% |
| shrinking >5% | **4%** | **12%** |

Three conclusions:

1. **Direction confirmed** by fully independent data — the growth result holds.
2. **We under-measure growth ~3×.** Partly the short window, but now also
   partly *mechanical*: the biggest lakes are clipped, so their growth cannot be
   recorded. Truncation suppresses exactly the lakes that matter most for GLOF
   hazard.
3. **Our shrinkage is mostly noise.** 12% vs their 4% — consistent with the
   noise-floor argument, now externally corroborated rather than argued.


---

## Change: our model vs the KV epochs

Both converted to **%/yr** (ours 2019→2025 = 6 yr; KV epoch midpoints 2016.5→2023.0
= 6.5 yr). Only KV-vs-KV is differenced — Hi-MAG is never differenced against KV.

Note KV's *mean* rate is +12.9 %/yr, an artefact of tiny denominators; medians are
the only usable statistic here.

| | ours | KV | ratio |
|---|---|---|---|
| median rate | +0.033 %/yr | +0.133 %/yr | **0.25** |
| aggregate | +36.4 ha/yr | +157.9 ha/yr | **23%** |
| sign agreement | — | — | 57% |
| Pearson r (per lake) | — | — | **−0.03** |

### The truncation test — a directional prediction, confirmed

If the fixed window clips large lakes, their growth is mechanically suppressed,
so our rate should fall short **specifically in the big bins**:

| reference size | ours %/yr | KV %/yr | ours/KV |
|---|---|---|---|
| 0–15 ha | +0.058 | +0.113 | 0.51 |
| 15–25 ha | +0.083 | +0.110 | 0.76 |
| 25–50 ha | +0.100 | +0.148 | 0.68 |
| **50–100 ha** | **−0.050** | +0.130 | **−0.38** |
| **100+ ha** | **+0.000** | **+0.512** | **0.00** |

Below 50 ha we undershoot but agree in direction. At **≥50 ha we get the sign
wrong**, and at **100+ ha we measure exactly zero change against a reference
+0.51 %/yr** — the largest lakes are pinned against the window and cannot record
growth at all. The reference says the biggest lakes are growing *fastest*; our
pipeline is blindest exactly there.

The pixelwise evaluation below shows this is a **framing** failure, not a
segmentation one — the model is at its *best* on these lakes.

### The uncomfortable part: no per-lake agreement

**Pearson r = −0.03, sign agreement 57%** (coin-flip is 50%). Per-lake rates do
not correlate with the reference at all.

This does not prove our per-lake trends are wrong — KV's change is a two-point
difference between epochs and carries its own per-lake noise, while ours is a
7-point least-squares fit. Two noisy estimates will not correlate even if both
are unbiased. What it does establish is a **bound**: we cannot externally
validate per-lake rankings, and the watchlist is built on exactly that ranking.

Aggregate direction is corroborated. Individual lake ordering is not.


---

## Pixelwise evaluation against the KV inventories

KV polygons painted into the **same GEE grid** the patches were exported on.
Alignment was not assumed: every export re-painted the Hi-MAG polygon and
compared it to the `y` already stored in the patch — **median and p10 IoU
1.0000 across all 885 lakes**. 885/885 exported, 0 failures.

### The model scores better against labels it never saw

| reference | IoU | precision | recall | median ratio |
|---|---|---|---|---|
| Hi-MAG 2017 (30 m) — **trained on** | 0.7832 ±0.137 | 0.8533 | 0.9088 | 1.049 |
| KV 2016-17 (10 m) — same epoch | 0.8320 ±0.170 | 0.9208 | 0.8853 | 0.956 |
| **KV 2022-24 (10 m) — EXTERNAL** | **0.8455** ±0.142 | **0.9431** | 0.8883 | 0.942 |

Paired on the **same predictions**, KV 2016-17 labels vs Hi-MAG labels:
**+0.0379 ±0.130, 79% of lakes improved.** By our own `|mean| > std/2` rule the
magnitude is a **tie**, so this is not claimed as a large effect — but 79%
one-directional is a strong consistency signal, and the prediction is identical
in both columns, so the difference is a property of the *labels*.

Precision moves most: **0.853 → 0.943**. Much of what Hi-MAG scored as false
positive was real water that a 30 m sensor could not resolve. Recall dips
slightly (0.909 → 0.888) — against finer labels we do miss genuine detail.

### By lake size (external reference)

| size | IoU | precision | recall | median ratio | aggregate ratio |
|---|---|---|---|---|---|
| 0–15 ha | 0.7347 | 0.7904 | 0.8861 | 1.012 | **1.276** |
| 15–25 ha | 0.8267 | 0.9350 | 0.8785 | 0.934 | 0.956 |
| 25–50 ha | 0.8556 | 0.9748 | 0.8739 | 0.919 | 0.901 |
| 50–100 ha | 0.9060 | 0.9822 | 0.9183 | 0.958 | 0.937 |
| **100+ ha** | **0.9756** | **0.9972** | 0.9783 | 0.990 | 0.982 |

IoU rises **monotonically** with size. The model is near-perfect on large lakes
and weakest on small ones, where it over-reads aggregate area by **28%**.

### Reconciling this with the 55% area shortfall

These two results look contradictory and are not. Pixelwise IoU is computed
**inside the window**, and the reference is clipped by that same window — so it
*cannot* see truncation by construction.

| reference size | fraction of the lake inside the window | touching window edge |
|---|---|---|
| 0–15 ha | 1.168 | 30% |
| 15–25 ha | 1.160 | 31% |
| 25–50 ha | 1.139 | 57% |
| 50–100 ha | 1.005 | **96%** |
| **100+ ha** | **0.573** | **100%** |

(Ratios above 1 mean the window also contains neighbouring lakes.)

**Every lake over 100 ha touches the window edge, and only 57% of it is inside.**
So: segmentation on large lakes is excellent (IoU 0.9756); the *measurement* is
truncated. **The window is the defect, not the model.** That reframes the fix —
it is a geometry change, not a modelling one.

### Generalisation — no overfitting

| split | IoU vs Hi-MAG | IoU vs KV 2022-24 (external) |
|---|---|---|
| train (540) | 0.7812 | 0.8408 |
| val (207) | 0.7963 | 0.8586 |
| eval (92, frozen) | 0.7658 | 0.8436 |

Val slightly **exceeds** train, so the 885-lake figures are not inflated by
memorisation. The frozen eval set was read here for a final report against an
external reference — no model was selected or tuned on it, and nothing was
retrained.


---

## Fitted windows + matched-epoch labels — the truncation fix

Design change, after the availability probe below forced it:

```
TRAIN   2022 imagery + KV 2022-24 10 m labels, window fitted per lake
FREEZE
APPLY   2019-2025, one segmentation per year, compare areas across years
```

**Why not 2016-17**, which would have kept KV 2022-24 as an untouched validator:
Sentinel-2 L2A begins 2017-01-15, and 2017 Himalayan coverage is a **median of 3
scenes per lake with 18% of lakes having none**, before cloud masking removes
~77%. Measured on 40 lakes, not assumed. 2022 is also the series midpoint, so the
frozen model extrapolates equally little toward either end.

Windows sized from real polygon extent (`scripts/plan_windows.py`), not
sqrt(area): lakes are elongated (median 1.71x the square-equivalent side), and
area-based sizing still truncated 11 lakes. Extent-based sizing truncates **zero**.

### Segmentation

| | val (207) | **held-out 100 (93)** |
|---|---|---|
| IoU | 0.8974 | **0.9055** |
| recall | 0.9656 | 0.9639 |
| IoU >=50 ha | 0.8486 | 0.7871 |

Held-out **exceeds** val — no overfitting. Against the old model on the same
target: **+0.1100 +/-0.153, 90% of lakes improved, REAL**. On lakes >=50 ha,
IoU 0.5764 -> 0.8486.

The old model's 374.7 ha mean on big lakes was not better measurement — truth is
161.3 ha, so it was reading **2.32x truth**, flooding a window it was never
trained for. The new model reads 1.18x.

### The truncation test — answered

| reference size | fixed 128 | **fitted** | KV |
|---|---|---|---|
| 100+ ha | **+0.000 %/yr** | **+0.083** | +0.100 |
| 50-100 ha | −0.050 | +0.017 | +0.130 |
| aggregate vs reference | **23%** | **70%** | — |
| median rate ratio | 0.25 | **0.68** | — |
| R^2 >= 0.5 | 157 | **191** | — |
| `saturated` lake-years | 270 | **0** | — |

**Lakes over 100 ha read exactly +0.000 %/yr because they could not grow inside
the frame.** They now track the reference at 83% of its rate. The 23% figure was
itself computed on a set containing only 32 usable lakes >=100 ha — the fixed
pipeline dropped the rest as `saturated`. The fitted run has 67.

### Two bugs the bigger windows introduced, both caught before publication

1. **Truncation flag counted any border water.** A fitted window deliberately
   contains neighbours, so it fired on 23.1% of lake-years and eliminated 158
   lakes. Now tests only the target lake's connected component:
   ok 73.6% -> **95.7%**, truncated 23.1% -> **1.1%**.
2. **Area counted every predicted pixel in the window.** Negligible at 128 px,
   but at 640 px it inflated the aggregate to 47,554 ha against a reference
   ~34,600 — **1.31x**, with 25 lakes over 2x. Area is now the connected
   component containing the centred lake; 5,300 ha of neighbouring water is
   correctly excluded. Ratio now **1.16x** (median 1.151).

Neither was a model error. Both were accounting errors created by changing the
frame, and both would have produced a confident, wrong headline.

### What did not move

**Pearson r = −0.003** per lake, sign agreement 62%. Aggregate and size-binned
rates now agree with the reference; **individual lake rankings still cannot be
externally validated.** That is the noise floor, and only a longer series fixes
it. The watchlist ranks on exactly this unvalidatable quantity.

Residual: we still read **~16% high** on area overall, and the 50-100 ha bin
tracks the reference at only 0.13 — unexplained, and worth a look before this is
promoted.


---

## Diagnosing the 50-100 ha "anomaly" — there wasn't one

### It was my statistical error

I reported that the 50-100 ha bin tracked the reference at only 0.13. That was a
**ratio of two medians, neither of which is resolvable at n=80**. Bootstrap CIs:

| bin | ours %/yr [95% CI] | KV %/yr [95% CI] | overlap |
|---|---|---|---|
| 0–15 | +0.090 [+0.020, +0.180] | +0.113 [+0.080, +0.171] | yes |
| 15–25 | +0.090 [−0.000, +0.130] | +0.110 [+0.079, +0.160] | yes |
| 25–50 | +0.105 [+0.040, +0.160] | +0.141 [+0.090, +0.205] | yes |
| **50–100** | **+0.010 [−0.070, +0.120]** | **+0.130 [+0.035, +0.222]** | **yes** |
| 100+ | +0.040 [−0.040, +0.120] | +0.100 [+0.031, +0.329] | yes |

**Every bin overlaps.** Lesson: never report a ratio of two point estimates
without checking either is distinguishable from the other. A ratio of noise is
noise, and it looked like a finding.

### But the same table showed something real

Ours is lower than KV in **5 of 5 bins** (sign test p ≈ 0.03) — a *uniform*
~25% shortfall, not a size-dependent one. The truncation fix removed the
size dependence; this is what remains. Two causes, both benign:

**1. Estimator (~20%).** Ours is a 7-point least-squares slope; KV is a 2-point
epoch difference. On identical lakes:

```
ours, 7-point least squares   +0.070 %/yr
ours, 2-point 2019->2025      +0.088 %/yr
KV,   2-point epoch diff      +0.124 %/yr
```

**2. Different periods.** KV spans 2016.5–2023, ours 2019–2025.

### Rejected: the 15% area over-read

A constant shoreline halo would dilute percentage rates. Tested directly —
excess area is **proportional**, not a halo:

| ref size | excess/area | implied halo width |
|---|---|---|
| 0–15 ha | 0.160 | 16 m |
| 25–50 ha | 0.151 | 25 m |
| 100+ ha | 0.145 | **57 m** |

`excess/area` is flat at 0.141–0.160 across a 10x size range while halo width is
not. A proportional bias cancels exactly in a percentage change, so **the area
over-read cannot explain the rate shortfall.** It affects absolute hectares only.

### The real finding: growth decelerated after ~2022

| period | median | aggregate |
|---|---|---|
| 2019 → 2022 | **+0.279 %/yr** | **+1.024 %/yr** |
| 2022 → 2025 | **−0.115 %/yr** | **+0.415 %/yr** |

64% of lakes decelerate. This survives every control we can apply:

- **Not data quality.** On 692 lakes with *zero* nodata in every year:
  +0.953 → +0.393 %/yr, essentially unchanged.
- **Not the training epoch.** The OLD model (2019 imagery, Hi-MAG 2017 30 m
  labels, fixed 128 px windows) gives **+0.279 → −0.180** — an identical
  first-half median from a wholly independent pipeline.
- **Not compositing.** Median usable scenes per lake per year: 33, 32, 32, 32,
  31, 31, 39. No break at 2022.

That explains KV's higher rate honestly: its window is weighted toward the faster
early period, ours toward the slower late one. **Both are correct for what they
measure.**

Note the divergence in the second half: **aggregate stays positive (+0.415)
while the median goes negative (−0.115)** — large lakes keep growing while the
typical lake stalls. Caveat: three-year halves are short, and this should not be
over-claimed on a six-year series.


---

## Uncertainty, and the watchlist is externally validated after all

Every trend now carries a standard error and a 95% CI on the fitted slope
(`scripts/detect_change.py`). A lake counts as **detectable** only if that
interval excludes zero — a far stricter test than R^2, which measures how well a
line fits without asking whether the line differs from flat.

```
clean lakes            850
R^2 >= 0.5             191
DETECTABLE             152  (18%)   -> 94 growing, 58 shrinking
no detectable trend    698  (82%)
median residual noise  1.65% of area
```

Every detectable lake also has R^2 >= 0.5; none with R^2 < 0.5 is detectable. So
R^2 was directionally right but permissive.

### The r = 0 result was an artefact of averaging over noise

Correlation with the independent KV change, by subset:

| subset | n | Pearson r | sign agreement |
|---|---|---|---|
| all clean lakes | 771 | **−0.002** | 62% |
| R^2 >= 0.5 | 166 | +0.248 | 78% |
| detectable | 140 | +0.242 | 79% |
| **watchlist (detectable + growing)** | **85** | **+0.678** | **88%** |

**Correction to earlier entries in this file:** "per-lake rankings cannot be
externally validated" was wrong. It held for the full 850-lake list, where 82% of
entries are noise, and I generalised it to the product. On the lakes the product
actually ranks, agreement with an independent inventory is strong.

The lesson matches the 50-100 ha mistake: a statistic computed over a population
that is mostly noise tells you about the noise, not about the subset you care
about.

### Ranking changed accordingly

Priority was `slope x r2`. It is now the **lower 95% confidence bound on ha/yr** —
a lake earns a place only if we are confident it is growing at all, and is then
ordered by growth we can defend rather than growth we happened to fit. Rank 1 is
`GL087092E27798N`: 233.8 -> 289.1 ha, lower bound **7.2 ha/yr**, R^2 0.981,
residual noise 1.19%.
