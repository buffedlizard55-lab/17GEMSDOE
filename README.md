# 17GEMSDOE — GEMS Prize Challenge

> **Working charter — read this before every project session.** The objective is to build an auditable, scientifically grounded system that produces a valid, one-click-downloadable single-band GeoTIFF for the DOE GEMS Prize Challenge and improves performance without data leakage. Aim to maximize the probability of a genuine win and own the result end to end. Never claim a score, source, experiment, or validation result that was not actually checked.

## Mission and operating requirements

- Develop and test distinct geological hypotheses for previously unmapped faults in the GeoDAWN region; do not mistake variations of one feature recipe for independent strategies.
- Start each iteration by writing a falsifiable hypothesis, naming the feature/transform and the predicted score change *before* running it. Maintain positive **and negative** results in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
- Use a spatially blocked, hide-and-recover holdout: withhold complete fault segments and buffer them out of model inputs. Do not reward memorization of distance to known faults. The competition's official metric uses distance weighting; follow the organizer implementation/specification exactly, not an improvised proxy.
- Compare marginal precision with the score tradeoff implied by the metric; do not select by visual appeal. Do not replace or populate a submission slot unless the candidate beats the current holdout best under the same test.
- Build a reproducible data → training → inference → validation workflow, with tests for probability range, CRS, dimensions, affine transform, nodata, band count, and float32 GeoTIFF format. Put the download button and clear submission instructions at the top of the eventual site. Each submission needs a unique descriptive name and a short note.
- Keep evidence and source links in an auditable knowledge base. Use official or primary sources where possible, verify each factual statement against the cited source, and flag access problems or uncertainty instead of filling gaps.
- The requested outcome is an easy-to-use project site and a dependable current information feed, with a submission GeoTIFF that can be downloaded and uploaded to DrivenData. A score above the current leaderboard best is an aspiration, **not a guaranteed result**.
- Arena operating values: **Maximize P(Win)**: assess tradeoffs and choose evidence-backed actions that improve the odds. **Own the Outcome**: treat failure and success as evidence and fix problems when able.

## State of this checkout (2026-09-29)

At the start of this review, the checkout contained only a minimal README and one initial commit. There was no training code, site, experiment history, `data/` directory, competition raster, holdout, or submission artifact; this review added only documentation. Therefore this repository alone cannot establish why earlier team submissions tied at 0.1563, reproduce those submissions, run a holdout, validate a new feature, or produce a competition-ready TIFF. Scores and experiment labels in the user-provided request are **reported by the user**, not independently verified here. Do not infer identical rasters from equal rounded scores: score equality to four decimals does not prove file identity; compare file hashes and pixel arrays when the files are available.

The DrivenData problem description confirms that the challenge targets geological faults; supplied predictor layers include geophysical and elevation-derived features; labels are from USGS/INGENIOUS; the score is a distance-weighted Tversky index with α=0.2, β=0.8, and 300 m support; and submissions must be single-band float32 GeoTIFF probabilities in [0,1] with the training data's EPSG:32611 CRS, 100 m resolution, and matching bounds. The competition training data is behind login (as noted in the user-provided context); no credentials or dataset were available in this checkout, so no download or modeling was attempted.

## Current blocker / next action

Obtain the permitted competition data into `data/` on an authenticated machine, then inspect its license and metadata before running the organizer/reference preparation workflow. This checkout has no `scripts/download_competition_data.sh`, `scripts/prepare_data.py`, model code, tests, or reference solution files, despite their mention in the supplied task context. Do not claim those scripts ran or are ready. Once source rasters/labels are available, establish a leakage-safe spatial holdout and baseline before testing candidate transforms in the experiment log. Only after a reproducible improvement should a GeoTIFF be generated and surfaced for download.

## Verified primary references

1. [DrivenData — GEMS Prize problem description and submission format](https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/) — accessed for this review. Describes objectives, provided features, labels, metric, and required GeoTIFF format.
2. [DrivenData — competition rules landing page](https://www.drivendata.org/competitions/306/competition-doe-gems/rules/) — says participation is subject to official rules and links to HeroX rules. Review the linked official rules before external-data use or submission.
3. [DrivenData reference-solution repository](https://github.com/drivendataorg/gems-prize-reference-solution) — accessed; repository page describes a reference workflow and requires users to download challenge data. It is not code in this checkout.
4. [USGS GeoDAWN dataset landing page](https://www.usgs.gov/data/geodawn-airborne-magnetic-and-radiometric-surveys-northwestern-great-basin-nevada-and) — official USGS source link referenced by DrivenData; it returned a USGS maintenance page at review time, so details were not independently verified from that page in this pass.
5. [Great Basin Center for Geothermal Energy — INGENIOUS project](https://gbcge.org/current-projects/ingenious/) — accessed; describes the project and links its associated publications/data releases. Use the source publication, not this summary alone, for geological claims.
6. [DrivenData — live public leaderboard](https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/) — accessed 2026-09-29; displayed top public score 0.3168 and entries at 0.1563. Scores are time-sensitive; consult the live page.

**Access irregularity:** The prompt says the public competition data page redirects to login and provides third-party Dropbox links. This checkout has none of those downloaded files. No external Dropbox artifact was independently validated or used. The supplied NREL PDF could not be fetched from this environment. The live public leaderboard page was fetched on 2026-09-29: its displayed top score was **0.3168** (team DARD), not the prompt's earlier 0.3049 snapshot. It displayed **0.1563** for both SDCF9 and smashi34 (plus another participant, extradr19, at 0.1563). The page is dynamic and this is a time-stamped public-board observation, not a claim about private/final scores. The matching scores do not establish identical submissions. See the experiment log for what could and could not be tested.

## Submission checklist (when implementation/data exists)

1. Use one documented experiment/config and locked data split.
2. Validate on whole hidden segments plus the predeclared buffer; compare against a saved baseline and current best.
3. Inspect output values: finite predictions inside valid pixels, every value in [0,1], nodata outside coverage.
4. Check exactly one band, float32, EPSG:32611, 100 m pixels, and exact required bounds/transform against the official sample submission or feature raster.
5. Download/open the TIFF and validate it independently before any DrivenData upload. Give it a unique name and short methodology note.
6. Link the submission instructions and make the validated TIFF the obvious primary download on the site. Never expose an unvalidated or placeholder raster as a submission.
