# Experiment ledger

Every entry states what was predicted **before** the measurement, what was measured, and
what the result forces next. Failures are kept in the same table as successes: a
documented negative result is what stops the project from re-running the same idea with
new parameters.

Protocol: see `scripts/run_hypotheses.py` and `src/gems/holdout.py`. It was frozen before
any candidate was run. Budgets are **matched** — sparse (as many pixels as the fold has
truth pixels) and dense (3.3 % of valid area, the density real submissions use).

---

## 0. The bar, measured under the frozen protocol

| | Dense budget | Sparse budget |
|---|---|---|
| uniform random noise | 0.1199 | 0.0465 |
| reference field (holds 0.1563 publicly) | 0.1253 | 0.0000 |

`evidence/reference_holdout.json`, `evidence/hypotheses.json`.

**Reading.** At the density a real submission is judged at, the best artefact in the
family is **0.0055 better than noise**. That single number explains the entire history of
this project and it is the reason the next section's candidates are aimed where they are.

---

## 1. Hypotheses tested

| # | Hypothesis | Layers | Predicted before running | Measured (dense / sparse) | Verdict |
|---|---|---|---|---|---|
| H1 | Relay-ramp bridges between different catalogue strands | catalogue geometry | positive; large enough to matter | 0.0000 / 0.0012 | **REJECTED** |
| H2 | Amplitude-invariant tilt-angle edge lineaments | `tc`, ridge-enhanced | the supplied stack already contains a magnetic edge product, so no gain | 0.0013 / 0.0000 | **REJECTED** |
| H2b | The supplied `tmi_hg` band scored directly, as a control for H2 | `tmi_hg` | should match H2 if H2 adds nothing | 0.0000 / 0.0000 | **CONTROL — confirms no gain from re-transforming** |
| H3 | Relay-bridge pixels ranked by the visible catalogue's topology | catalogue geometry | nothing; if this fails, catalogue-topology features are closed | 0.0045 / 0.0012 | **REJECTED** |
| H4 | Joint strain localisation (dilational jog: shear ∧ dilatation ∧ second invariant) | `geod_shearrate`, `geod_dilaterate`, `geod_2ndinv` | low; the channels are smooth regional products | 0.0000 / 0.0000 | **REJECTED** |
| CTRL | Absent a ridge detector, does *any* topographic ranking work? | `det_elev_slope` | negative | 0.0000 / 0.0000 | **REJECTED** |
| BASE | Negative distance to the visible catalogue — the memorisation control the organizers' masking rule exists to neutralise | catalogue geometry | positive if the task is really distance-to-catalogue | 0.0000 on 116,974 px | **REJECTED** |
| H6 | The 0.1563 field over-emits; truncating a re-ranked support raises the score | reference support, ranked by H3 | +0.005 … +0.03, optimum at 40–80 % | best fraction **100 %**, delta **+0.0000** | **FALSIFIED** |

Ranked pre-registered order by expected gain: **H1 > H3 > H4 > H2**. Measured order:
**H3 > H2 > H1 = H4**. The ranking was wrong and that is recorded.

---

## 2. The two results that matter

### N-1 — Coverage dominates detection at competition density

Uniform noise scores 0.1199 at the dense budget and beats every physical detector. The
geometry is simple: the metric credits a prediction anywhere within 300 m of truth, the
catalogue is spread across the map, and a submission emits about 3 % of the valid area. A
detector that concentrates its budget on its strongest anomaly leaves most of the map
uncovered.

**Consequence.** Any local score computed at the dense budget without a null baseline is
uninterpretable. Every candidate in this repository is therefore reported against the
null, and the null is reported at both budgets.

### N-2 — The bottleneck is detection, not emission shaping

The break-even marginal precision for the reference field is 0.0197; its measured marginal
precision is 0.0197 under the old protocol and 0.0315 under the frozen one. The obvious
next idea follows — re-rank the support and truncate at the optimum. **H6 falsified it.**
The DTI curve rises monotonically to the full support, because the ranking has no power to
separate true from false positives *within* the support; truncation removes both in the
same proportion while the 0.8-weighted false-negative term rises.

**Consequence.** Every threshold, floor, thinning radius and emission-count variant in
this family's history has been spent on the wrong variable. Re-tuning them again is
explicitly disallowed by the standing brief, and this result is the reason.

### N-3 — The holdout's truth is the catalogue, so it cannot see a discovery

H1 and H4 place pixels in the empty space *between* catalogue strands; the holdout's truth
*is* the catalogue. Their zeros are therefore structural, not evidential.
`scripts/diagnose_h1_gaps.py` checks this against the alternative explanation:

| Gap window | Bridges | Share |
|---|---|---|
| ≤ 1.2 km (probably a rasterisation split) | 81 | 7.2 % |
| 1.3–1.6 km (shorter than published) | 47 | 4.2 % |
| 1.7–3.2 km (**inside** the published relay-ramp range) | 372 | 33.0 % |
| 3.3–5.0 km (longer than published) | 576 | 51.1 % |

Only 7.2 % are short enough to be rasterisation splits, and 33.0 % fall inside the
published 1.6–3.2 km relay-ramp width range. The detector is finding the right structure;
the test cannot see it. **Two arms are required: this catalogue-recovery arm, and an
off-catalogue arm built from an independent official compilation.**

Context: the supplied labels raster has **3,199 components** with a **median size of 12
pixels (1.2 km)** and a largest of 360 pixels. Most mapped "faults" in it are 1–2 km
segments, which any structural inference over its topology must account for.

---

## 3. Bugs found by testing, and fixed

| Defect | Found by | Fix |
|---|---|---|
| `depth_to_top_from_spectrum`'s docstring documented one slope convention while the arithmetic used another | the round-trip unit test written during Pass 2 | convention stated explicitly in the docstring, both literature forms recorded, and a test added that verifies `-slope/(4π)` recovers a known depth |
| `_nms_thin` accepted a `radius_px` argument and ignored it — a latent trap for any caller expecting an NMS radius to change the result | Pass 2 static review | parameter removed; the function renamed `_thin` with the reason documented |
| `run_hypotheses.py` held nine feature bands plus a per-fold copy in memory, which exceeded 3 GB and was OOM-killed at fold 3 | running it | bands loaded, blanked and released one at a time; peak memory now one or two bands |
| `build_submission.py` read the clock twice, leaving two differently-named files from one run | running it | one timestamp per build, and the previous build's files are removed |
| `metric.tilt_from_potential` reference implementation did not divide by `den` in normalised convolution (fixed during development) | `test_documented_slope_convention_matches_the_arithmetic` neighbour case | corrected before the first real run |
| `validate_submission`'s dict was extended with `sha256`/`bytes` only in the caller, so a direct call raised `KeyError` | running `build_submission.py` | the caller computes them; the gate's own contract is unchanged and tested |
| Five `pyflakes` findings (unused imports, an undefined name from a half-finished refactor) | Pass 2 static review | all cleared; `pyflakes` runs clean over `src/`, `scripts/` and `tests/` |

---

## 4. Statistics that must travel with every number

* **The protocol's null is mandatory.** 0.1199 dense at the dense budget. A candidate
  below it has demonstrated nothing.
* **`TP_w + FN_w = |G|` exactly**, so only weighted true positives and false-positive mass
  matter.
* **Uniform scaling is monotone**, so the optimum is to emit exactly 1.0 wherever a pixel
  is emitted at all. The submission is binary; the problem is *which* pixels.
* Break-even marginal precision: **0.0323 at DTI 0.1563**, **0.0649 at DTI 0.3049**.
* Grid: 3730 × 3292, EPSG:32611, 100 m cells. Valid footprint **5,165,840** cells where
  all 19 bands carry data; the sample submission's own finite footprint is 5,167,373.
  Catalogue: **60,988** positive pixels, **7,111,787** nodata.
* The reference field emits 172,974 pixels; the family's scored submissions emit
  3.0–3.6 % of the valid area.

---

## 5. Irregularities

1. **Duplicated work.** 449 blob groups appear at more than one path. `GEMSDOE4` is 86.4 %,
   `GEMSDOE2` 85.2 % and `5GEMSDOE` 84.9 % byte-identical to `GEMSDOE` across shared work
   files. The 0.1563 raster appears at 15 paths in 4 repositories.
2. **Duplicated predictions.** `8GEMSDOE_Hedge-v2_submission.tif` and
   `5GEMSDOE/candidate_s5_catalogue_hedge.tif` have support Jaccard 1.0000.
3. **The `[0, 1]` rejection is not reproducible.** 31 published GeoTIFFs audited; 0 with an
   out-of-range value, 0 with NaN inside the footprint.
4. **A promised band is absent.** The problem description advertises a top-of-crustal
   magnetic source-depth estimate; the delivered 19-band raster does not contain one,
   verified band by band.
5. **No radiometric channel in the delivered stack.** The GeoDAWN release publishes
   airborne radiometric grids for this survey, so the highest-value input is available but
   absent from the competition data.
6. **A 100 km smoothing radius on the seismicity layers** makes them unusable for 1–5 km
   fault targeting. Recorded so the idea is not proposed again.
