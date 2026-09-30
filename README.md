# 17GEMSDOE — DOE GEMS Prize Challenge

> **Working charter — read this at the start of every session.**
> The goal is to place at the top of the [DOE GEMS Prize Challenge](https://www.drivendata.org/competitions/306/competition-doe-gems/)
> leaderboard with an auditable, scientifically grounded system that produces a valid,
> one-click-downloadable single-band GeoTIFF, and to improve performance without leakage.
>
> **Arena operating values.** *Maximize P(Win)*: weigh trade-offs, assess risk and take
> the evidence-backed path. *Own the Outcome*: own the result end to end; when a problem
> arises and there is a means to act, act without being asked; treat failure as signal.
>
> **Non-negotiables.** Never claim a score, source, experiment or validation result that
> was not actually checked. Every number published here is produced by a script in this
> repository. Every factual claim cites a source that was opened, labelled with how deeply
> it was verified.

**Site:** <https://buffedlizard55-lab.github.io/17GEMSDOE/> — one click downloads the
submission; the name and comment to paste are directly underneath it.

---

## 1. The standing brief (verbatim)

Kept intact so it is read every session rather than remembered.

> Review the repo.
>
> There should be an easy to download submission tif file as described by the prompt.
> Read the entire prompt.
>
> Treat every session as one iteration of a disciplined loop, not a single pass: before
> touching code, state a specific, falsifiable hypothesis about where the catalogue is
> geometrically incomplete — grounded in real structural geology, not intuition, since
> published inventories of Great Basin geothermal systems (Faulds et al.) show roughly a
> third cluster at fault step-overs and another fifth each at terminations and
> intersections, precisely the small, easily-missed connecting structures a regional
> catalogue is most likely to skip — then name the feature or transform meant to catch it
> and predict, in writing, the direction and rough size of the score change before running
> anything. Test that prediction only on a hide-and-recover holdout that withholds whole
> fault segments with a buffer from the model's inputs, mirroring the organizers'
> confirmed scoring behavior (known-fault pixels are pixel-exact masked; nearby-but-not-
> matching predictions are still fully penalized), so the model is judged on recovering
> hidden structure rather than memorizing distance-to-known-fault; nothing touches a
> submission slot until it beats the current holdout best under this exact test. Since the
> metric algebraically reduces to a 0.2/0.8-weighted precision-recall trade, check a
> candidate's marginal precision against that threshold instead of eyeballing the map.
> When a hypothesis fails, log why in the same place as the successes — a documented
> negative result narrows the next hypothesis and is worth writing down for Phase 2's
> expert reviewers — and let that failure redirect the next test rather than triggering a
> re-run of the same idea with new hyperparameters. Nothing gets asserted in the writeup
> without a source you actually opened, and the loop doesn't stop until the holdout, then
> the leaderboard, shows real movement.
>
> Here are the results from our groups submissions, separated by ....: [GEMSDOE1 0.1563;
> 6GEMSDOE 0.0286; GEMSDOE3 0.1193; GEMSDOE2 0.1560; GEMSDOE4 0.0343; 5GEMSDOE 0.1563;
> 7GEMSDOE 0.1461; 8GEMSDOE 0.1563; 9GEMSDOE 0.0107; 10GEMSDOE h16-continuation 0.0461,
> h20-dem10-scarp-thin 0.0921; 11GEMSDOE 0.0202; 12GEMSDOE 0.1294; 15GEMSDOE 0.0782;
> 14GEMSDOE 0.0020]
>
> We need to figure out why we keep scoring 0.1563, are we copying the same work over and
> over again? we need to come up with different ideas, and not just the same idea tried a
> different way. Need to figure out why 5GEMSDOE and GEMSDOE1 have the same score. We
> should not be generating the same score submissions, they should all be unique.
>
> Before implementing, generate 3–5 candidate geological hypotheses we haven't tried yet,
> each naming: the specific layer(s) involved, the physical signature being targeted (e.g.,
> an edge-detection or curvature transform), why it should catch a fault missing from the
> USGS/INGENIOUS catalogue rather than one already in it, and how it differs from anything
> already implemented in this repo. Rank them by expected DTI improvement and
> implementation cost. Validate the top candidate on our spatially-blocked holdout set
> before touching a weekly submission slot — do not spend a submission slot on an idea
> that hasn't beaten the current holdout best. If a candidate can't be validated without
> new external data, name the specific free, official source needed and check it's
> obtainable before proposing the idea as viable.
>
> Work line by line verifying from official verified trusted sources, provide links for
> manual review. There should be no manual input, work on your own to complete tasks. Flag
> any irregularities for review. No hallucinations. Verify no hallucinations. The goal of
> this project is to get a full list that follow our requirements. Verify line by line.
>
> We need to quickly look at the results and our results. We have a good understanding of
> how our hypothesis, methodology, calculations, analysis are done so we should be able to
> figure out a way to score higher on the leaderboard using previous results and scoring
> that we have across the sites listed above. We need to come up with distinct and unique
> strategies to score higher in this competition leaderboard. We need to start doing heavy
> and deep research into the part of the project that matters the most, which is the
> scientific discovery of geothermal vents. We should store all of our information and
> knowledge that we can gather from official verified sources. This will serve as a
> starting point for other projects as well. We need to think outside the box but still be
> grounded in proper scientific research, we are ultimately aiming for a top prize that
> many others are competing for. So it's important to be contrarian but be smart about it.
> We need to find sources of data that others are over looking or areas of the project
> when it comes to geothermal vents. We need to do deep research and critical thinking and
> come up with new hypothesis to test.
>
> 0.3049 is the highest score right now so we need to design a new strategy, research,
> testing, analyzing, and generating submission system than the current website. It should
> be unique, take unique approaches to generating a submission that can score higher than
> .3049.
>
> The site should be able to generate a TIF file that is required for submission. It
> should be as easy as download to click a File to submit into the competition. This needs
> to be in the executive summary or the very beginning of the site. It should be obvious
> when you visit the site. I tried to submit the document that i downloaded from the site
> but it returned this error on the submission form: **"Predicted values must be in range
> [0, 1]"**. Also we need to give it a unique name and a short comment to help you or your
> team tell submissions apart later e.g. clustering with k=25.
>
> Create a executive summary subpage that explains exactly how to make a submission into
> the contest. Work on the next steps from the previous sessions first. Create a github
> page for this repo that has clean ui, user friendly, simple and easy to use. It should
> include all relevant information in an easy to read format with official verified links
> as sources for review.
>
> **The single remaining blocker to training is data placement**: run
> `bash scripts/download_competition_data.sh` on any unrestricted machine into `data/`,
> then `python scripts/prepare_data.py` — after that the full train→inference→validate
> pipeline is ready to run (GPU needed for training; metric/losses/validation all verified
> working here on CPU).
>
> Run this task through multiple passes. Pass 1: implement completely and verify. Pass 2:
> review for bugs, missing requirements, incorrect assumptions and edge cases; fix
> everything found. Pass 3: re-check the entire implementation against the original
> request; improve accuracy, reliability, completeness and code quality. Do not stop after
> the first pass. Go ahead and create a pull request and then merge onto main. Make
> suggestions for what work still needs to be done and any limitations in the way of a
> successful project.

---

## 2. Answers to the two forensic questions

Both were answered from bytes, not from score arithmetic. Reproduce with
`python scripts/forensic_audit.py`.

**"Why do we keep scoring 0.1563?"** Because the same bytes keep being submitted. The
raster with `sha256 7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15`
(570,890 bytes, 172,974 positive pixels) is published at **15 paths across 4 different
repositories**. Beyond that one file, **449 blob groups** are duplicated somewhere in the
family, and by hash comparison `GEMSDOE4` is **86.4 %**, `GEMSDOE2` **85.2 %** and
`5GEMSDOE` **84.9 %** byte-identical to `GEMSDOE` across shared work files. This is not an
inference from similar scores — it is a hash comparison of the source trees.

**"Why do 5GEMSDOE and GEMSDOE1 have the same score?"** They submitted the same file. Two
accounts uploading byte-identical rasters receive the same score by construction. Nothing
about model agreement is needed to explain it. The related case is
`8GEMSDOE_Hedge-v2_submission.tif` and `5GEMSDOE/candidate_s5_catalogue_hedge.tif`, which
are different bytes with a **support Jaccard of 1.0000** — the same map under two names.

---

## 3. The submissions

`docs/downloads/` holds two one-click artifacts. Both pass every format check
against the official grid (float32, EPSG:32611, 3730x3292, values in [0, 1],
NaN only outside the footprint).

| | Validated candidate (new score attempt) | Safe rebuild (reproduces 0.1563) |
|---|---|---|
| File | `17GEMSDOE_D-supervised-expression_86125863_20260930T034153Z.tif` | `17GEMSDOE_A-verified-01563-support_c5ea501e_20260930T021105Z.tif` |
| SHA-256 | `8612586354611df8602a2b0bad8a7916946d1bc0bc32f63e39f1fffbcccced8d` | `c5ea501eba0ab840d26f0c017d01074c4bd9040ec5e2c5ac19bde8655d150e64` |
| Prediction | HistGB on 14 bands + 2 gradient magnitudes, no location features, 0/1 top-k at 3.3 % density (170,401 px) | the 0.1563 field's support, 172,974 px |
| Hide-and-recover | mean dense DTI **0.2795** (folds 0.254-0.296) vs gate 0.1253 | 0.1253 (leaks: trained on withheld segments) |
| Value range | [0.0, 1.0], **no NaN anywhere** | [0.0, 1.0], no NaN anywhere |
| Format gate | 10 checks passed, 0 failed | 10 checks passed, 0 failed |
| Comment to paste | `D-supervised-expression: HistGB on 14 bands+2 gradmag, no location features, 0/1 top-k at 3.3% density (170,401 px), hide-and-recover mean dense DTI 0.2795 vs gate 0.1253, 86125863` | `A-verified 0.1563-support rebuild (all-finite [0,1], 172,974 px, c5ea501e, 20260930)` |

Every artifact name carries a content hash and a UTC stamp, so two different
submissions can never share a name and the leaderboard entry is always traceable
to exact bytes. The NaN-footprint twin of the safe rebuild is published too
(`..._nanfootprint_...tif`), for a form that requires the sample submission's
convention exactly.

The reported upload error — *"Predicted values must be in range [0, 1]"* — is
handled by `src/gems/io.py`, which checks every requirement in one place and
refuses to publish a failing file. The primary artifacts contain **no NaN at
all**, which is the one construction that cannot trigger that message. An audit
of **31 published GeoTIFFs** across six repositories found **0** with an
out-of-range value and **0** with NaN inside the footprint, so the exact file
that produced the rejection is not identifiable from the published artifacts.
That is recorded as an open irregularity rather than guessed at
(`evidence/range_audit.json`).

---

## 4. The measurements (this session, pre-registered)

Every hypothesis below was written down with direction and rough size before
the run; the outcome is recorded whether it passed or failed.

| Test | Prediction | Measured | Verdict |
|---|---|---|---|
| H-XSUR: relay-bridge / edge-lineament on the off-catalogue surrogate | H2 >= 1.5x random, H3 > random (0.10-0.25 vs 0.05-0.10) | H2 0.019, H3 0.014, random 0.372 | **failed** — logged; catalogue-topology detectors find no off-catalogue structure |
| H-BASIN: surrogate truth hides under basin fill | truth in flat, deep-cover terrain | truth slope mean 14.6 vs 7.4; 58 % steepest quartile | **falsified** — it is range structure; logged |
| H-COV: coverage-optimal spreading at fixed budget | >= 1.5x clustered on holdout (0.12 -> 0.18-0.25) | 0.1499 vs 0.1188 (1.26x) holdout; 1.23x surrogate | direction held, size short |
| Union: lattice filler added to the 0.1563 support | union 0.13-0.17, marginal > break-even | 0.1204 vs base 0.1272; marginal 0.016-0.023 < 0.024-0.032 | **failed** — gate refused publication |
| H-SUP: supervised catalogue-expression model | mean dense DTI >= 0.1253 (0.13-0.20) | **0.2795** (folds 0.254-0.296); surrogate 0.1225 (p2: < 0.1854) | **p1 held decisively**; p2 failed |

Supporting instruments built this session:

* **Off-catalogue surrogate arm** (`evidence/xcat_surrogate.json`). Qfaults was
  tested first and **quantitatively refused**: 60,938 of its 60,939
  in-footprint pixels are already within 300 m of a competition label — the
  same lines (Giddens & Faulds 2025 derive their compilation "primarily from
  the Quaternary Fault and Fold Database"). The SGMC structure raster
  (Horton et al. 2017, DOI 10.3133/ds1052) leaves **59,035 px** of structure
  beyond the catalogue's credit halo; that is the surrogate truth. Uniform
  random beats every rule-based detector on it.
* **Terrain census** (`evidence/terrain_census.json`): the surrogate truth is
  steep range structure (58 % steepest slope quartile, 1.3 % flattest), and
  top-k slope emission both overshoots its value range and under-covers
  (49.2 % of truth within R vs 79.1 % for equal-size random).
* **Emission shaping is closed** by three independent negatives: H6 truncation
  is monotone to full support; the family's thinned variants lost leaderboard
  points (12GEMSDOE 0.1294 < 0.1563); the union filler's marginal precision is
  below break-even in every fold.

The metric algebra that ties this together: `DTI = T / (0.2(T+F) + 0.8G)`, and
at equilibrium the score is roughly five times the marginal hit rate. The
0.3049 leader therefore finds hidden faults at about twice our rate (h ~ 6.5 %
vs 3.2 %). No arrangement of pixels closes that gap — only detection skill.

---

## 5. Quick start

```bash
bash scripts/download_competition_data.sh   # alias for fetch_competition_data.sh
python scripts/prepare_data.py              # verify every byte against official SHA-256s
python scripts/forensic_audit.py            # the answers in section 2
python scripts/run_hypotheses.py            # the frozen protocol, rule-based candidates
python scripts/run_xcat_surrogate.py        # off-catalogue surrogate screen
python scripts/run_coverage_emission.py     # matched-budget spreading test
python scripts/run_supervised_holdout.py    # H-SUP: supervised recovery test
python scripts/build_union_candidate.py     # union gate (refuses unless it wins)
python scripts/build_submission.py --source reference   # safe rebuild
python scripts/build_supervised_submission.py           # validated candidate
python scripts/validate_submission.py docs/downloads/<file>.tif   # standalone gate
python scripts/build_site.py                # regenerate the site from evidence/
python -m pytest tests/ -q
```

Requires `numpy`, `scipy`, `rasterio`, `scikit-image`, `scikit-learn`, `pytest`.
Everything runs on CPU (H-SUP trains a fold in ~12 s).

---

## 6. Getting the competition data

**Route A — unrestricted machine.** Download from the official
[data tab](https://www.drivendata.org/competitions/306/competition-doe-gems/data/) (login
required) into `data/`, then run `prepare_data.py`.

**Route B — the GitHub bridge (used here, verified working).** This sandbox can reach only
`api.github.com` and the Python package index. The bytes were previously transported into
the sibling `GEMSDOE` repository under `data/bridge/`, split into sub-100 MB parts;
`scripts/fetch_competition_data.sh` reassembles them (part names follow the bridge's
`manifest.json`) and `prepare_data.py` refuses to proceed unless every SHA-256 matches
the official pin:

| File | Bytes | SHA-256 |
|---|---|---|
| `training_features.tif` | 418,912,844 | `4371c82e3b8339b807bdffcf4ef59a225520fe2988d521be208ae33743123bc5` |
| `labels.tif` | 425,830 | `7ba308ccdc4418b31a178f4f1ef21aaa6e152e4028f2f6f64b01f7eb25ae4093` |
| `sample_submission.tif` | 1,599,597 | `2176d08e485aa2cd2860ce8df539db4faf4d76163b38a4dd8c30a40454d35cbc` |

---

## 7. Ranked next candidates

Full detail, with sources, on the [hypotheses page](docs/hypotheses.html) and in
`evidence/candidate_hypotheses.json`.

1. **Airborne radiometric alteration lineaments (K, eU, eTh, K/eTh) — H-RAD.** The
   delivered 19 bands contain **no radiometric channel** — verified band by band — so
   this is new information. Hydrothermal fluids concentrate K and strip Th along a
   damage zone; gamma rays sample the upper half metre, so the method sees faults with
   **no scarp** — exactly the population the SGMC screen says our signals miss. Free and
   official: [USGS GeoDAWN, doi 10.5066/P93LGLVQ](https://doi.org/10.5066/P93LGLVQ).
   *Cost: high (transport via bridge). Expected gain: highest.*
2. **1 m lidar scarp morphology — H-LIDAR.** The supplied elevation is a 100 m
   detrended pair, where a 2-5 m scarp is sub-pixel; the organizers distribute the
   high-resolution source themselves. *Cost: medium-high.*
3. **Slip- and dilation-tendency gate — H-DIL (new this session).** USGS "slip tendency
   and dilation tendency calculated for Quaternary faults in the Great Basin"
   (official, free, named on [usgs.gov](https://www.usgs.gov/programs/earthquake-hazards/faults));
   stress-optimally oriented splay tips and relay ramps are where the structure breaks
   next. No repository in the family uses mechanics data. *Cost: medium.*
4. **Multi-scale magnetic worm skeleton with strike-matched orientation — H-WORM.**
   Expected to fail; listed because a documented failure closes the line. *Cost: medium.*

Closed or falsified this session: **H-COVER** (coverage-optimal emission — tested,
direction held 1.26x but the union filler fails the marginal-precision threshold;
emission shaping closed), **H-COVER-2** (basin-gated targeting — falsified: the
off-catalogue population is range structure), **seismicity-gap detection** (rejected:
`ieq_n100a15`/`deq_n100a15` are smoothed at 100 km and cannot localise a 1-5 km fault).

---

## 8. What still needs doing

1. **Spend a weekly slot on the validated candidate** (H-SUP, gate cleared 0.2795 vs
   0.1253) from an account that can reach the competition, and record the leaderboard
   response. That number decides the central open question: do hidden faults look like
   catalogue faults?
2. **Add the radiometric channel and retrain** (H-RAD). The only genuinely new
   information available; the SGMC screen says our current signals do not see
   off-catalogue structure.
3. **Add 1 m lidar scarp morphology** (H-LIDAR) at the resolution the catalogue was
   drawn at.
4. **Pursue a discovery-grade validation arm**: Qfaults is refused (same lines), SGMC is
   necessary-not-sufficient; remaining routes are community fault mapping from 1 m lidar
   or expert play-fairway layers.
5. **Keep every submission unique** — enforced now by content-hash naming at build time;
   the family still carries 449 duplicated blob groups that predate this pipeline.

---

## 9. Limitations

Read [docs/limitations.html](docs/limitations.html) before trusting any number here. In
short: no new **leaderboard** score is claimed, because this environment has no
DrivenData authentication — the validated candidate is built, tested and downloadable,
but unproven publicly; the holdout's truth is the supplied catalogue, so it ranks
recovery, not discovery, and the SGMC surrogate screen (which ranks discovery-adjacent
skill) says the supervised model does not beat random on structure unlike the catalogue;
the reference field's holdout score leaks because it was trained on the withheld
segments; 2 CPU cores, 3 GB of RAM and no GPU cap model size; and outbound network
access is limited to GitHub and PyPI, so the highest-value external datasets
(radiometrics, 1 m lidar) are identified, named and source-verified but not fetched.
