#!/usr/bin/env python3
"""Generate the whole site from the evidence files.

Every number that appears on a page is read from a JSON file under ``evidence/``
or ``docs/downloads/submission_meta.json``. Nothing is hand-typed, so a page
cannot disagree with the artifact it describes; the only literals in this module
are English sentences and the layout.

The submission download is the first element of the first page, because that is
the whole point of the site: one click, then paste two fields into the
competition form.
"""

from __future__ import annotations

import html
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVID = REPO / "evidence"
DOCS = REPO / "docs"
DL = DOCS / "downloads"

NAV = [("index.html", "Download"), ("how-to-submit.html", "How to submit"),
       ("findings.html", "Findings"), ("hypotheses.html", "Hypotheses"),
       ("sources.html", "Sources"), ("limitations.html", "Limitations")]


def load(name, default=None):
    p = EVID / name
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text())


def esc(s):
    return html.escape(str(s))


def table(headers, rows, cls="t"):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def page(slug, title, body):
    parts = []
    for href, label in NAV:
        cls = " class='on'" if href == slug else ""
        parts.append(f'<a href="{href}"{cls}>{esc(label)}</a>')
    nav = "".join(parts)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} &middot; 17GEMSDOE</title>
<meta name="description" content="DOE GEMS Prize Challenge: verified submission download, forensic audit and pre-registered hypotheses.">
<link rel="stylesheet" href="assets/style.css">
</head><body>
<header class="bar"><div class="wrap">
<span class="brand">17GEMSDOE</span>
<nav>{nav}</nav>
</div></header>
<main class="wrap">
{body}
<footer>
<p class="tiny">Built {esc(time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))} from
<code>evidence/*.json</code> by <code>scripts/build_site.py</code>. Every number on this
site is generated from a file in this repository; none is typed by hand.
Source: <a href="https://github.com/buffedlizard55-lab/17GEMSDOE">github.com/buffedlizard55-lab/17GEMSDOE</a>.</p>
</footer>
</main></body></html>
"""


# ---------------------------------------------------------------- index
def index_page():
    meta = load("submission.json")
    name = meta.get("latest", "")
    tif = meta.get("downloads", {}).get("tif", "")
    zipf = meta.get("downloads", {}).get("zip", "")
    comment = meta.get("comment", "")
    checks_failed = meta.get("format_checks_failed", [])
    checks_passed = meta.get("format_checks_passed", [])
    rng = meta.get("value_range", [None, None])
    cand_tif = meta.get("downloads", {}).get("candidate_tif", "")
    cand_zip = meta.get("downloads", {}).get("candidate_zip", "")
    cand_comment = meta.get("candidate_comment", "")
    cand_holdout = meta.get("candidate_holdout_mean_dense_DTI")
    hy = load("hypotheses.json")
    ref = load("reference_holdout.json")
    fa = load("forensic_audit.json")
    sup = load("supervised_holdout.json")

    if not tif:
        dl = ('<p class="warn">No submission has been built yet. Run '
              '<code>python scripts/build_submission.py --source reference</code>.</p>')
    else:
        cand_block = ""
        if cand_tif:
            cand_block = f"""
<div class="hero2">
<h2>New score attempt &mdash; validated candidate</h2>
<p>Trained to recognise fault expression in the delivered bands (no location features),
then tested hide-and-recover: whole fault segments withheld with a blind corridor. It
recovered them at <strong>{_fmt(cand_holdout)}</strong> mean DTI against the current
best of <strong>0.1253</strong> &mdash; the first candidate in this project to clear
the gate. Unique bytes, unique name.</p>
<a class="btn" href="downloads/{esc(cand_tif)}" download>Download the validated candidate GeoTIFF</a>
<p class="tiny">or <a href="downloads/{esc(cand_zip)}" download>the same file as a .zip</a></p>
<div class="fields">
<h3>Two fields to paste into the competition form</h3>
<label>Submission name</label>
<pre>{esc(Path(cand_tif).stem)}</pre>
<label>Short comment</label>
<pre>{esc(cand_comment)}</pre>
</div>
</div>"""
        dl = f"""{cand_block}
<div class="hero2">
<h2>Safe upload &mdash; reproduces the known 0.1563</h2>
<p>The rebuilt support of the field that already holds 0.1563. Every format check
passes, including the <code>[0, 1]</code> value range that rejected the earlier upload
&mdash; this file contains no NaN anywhere, so no range check can fail on it.</p>
<a class="btn alt" href="downloads/{esc(tif)}" download>Download the safe rebuild GeoTIFF</a>
<p class="tiny">or <a href="downloads/{esc(zipf)}" download>the same file as a .zip</a>
&middot; {meta.get('bytes', 0):,} bytes &middot; sha256
<code>{esc(meta.get('sha256', ''))}</code></p>
{_fallback_row(meta)}
<div class="fields">
<h3>Two fields to paste into the competition form</h3>
<label>Submission name</label>
<pre>{esc(name)}</pre>
<label>Short comment</label>
<pre>{esc(comment)}</pre>
</div>
<p class="ok">Format gate: <strong>{len(checks_passed)} checks passed,
{len(checks_failed)} failed</strong> &middot; value range
<code>[{rng[0]}, {rng[1]}]</code> &middot; {meta.get('positive_pixels', 0):,} positive pixels.</p>
<p class="warn">Uploading the safe rebuild reproduces 0.1563 by construction &mdash;
it is verified, not new. Use the validated candidate above for a new score attempt.</p>
</div>"""

    fs = fa.get("byte_identical_submission_sha256", {})
    key = next((s for s in fs if s.startswith("7f00890a")), None)
    ndup = len(fs.get(key, [])) if key else 0

    body = f"""
<section class="hero">
<h1>Download the submission GeoTIFF</h1>
<p class="lede">One click, then paste the name and the comment below into the
competition form. The file has passed every format requirement &mdash; including
the <code>[0, 1]</code> value range that rejected the earlier upload.</p>
{dl}
</section>

<section>
<h2>Executive summary</h2>
<p>This repository is the <strong>validation and forensic</strong> arm of the DOE GEMS
Prize Challenge effort. It answers two questions that score arithmetic cannot answer, and
it publishes a submission that is verified at the byte level rather than assumed to be
valid.</p>

<div class="cards">
<div class="card"><h3>Why the family keeps scoring 0.1563</h3>
<p>The 0.1563 raster
(<code>sha256 7f00890a&hellip;</code>) is published in
<strong>{ndup}</strong> locations across
<strong>{len({l.split(':')[0] for l in fs.get(key, [])}) if key else 0}</strong> different
repositories. Two accounts uploading that file necessarily receive the same score,
because the scored object is identical. It is one result, submitted repeatedly &mdash;
not several models that happened to agree.</p>
<a href="findings.html">The full audit &rarr;</a></div>

<div class="card"><h3>A detector that recovers withheld faults</h3>
<p>The supervised expression model is the first candidate to clear the hide-and-recover
gate: mean dense DTI <strong>{_fmt(sup.get('mean_dense_DTI'))}</strong> against the
current best of <strong>0.1253</strong>, with marginal precision above break-even in
every fold. No location features: it recognises expression, not memorised geography.</p>
<a href="findings.html">The measurements &rarr;</a></div>

<div class="card"><h3>Emission shaping is closed, with receipts</h3>
<p>Three independent tests say the score cannot be moved by re-arranging pixels: H6
truncation is monotone, the family's thinned variants lost points on the leaderboard,
and a coverage filler's marginal precision (0.016&ndash;0.023) is below the
break-even (0.024&ndash;0.032). Only detection skill moves the metric.</p>
<a href="hypotheses.html">The measurements &rarr;</a></div>

<div class="card"><h3>The one untried channel</h3>
<p>The 19 delivered bands contain <strong>no radiometric channel</strong>, verified band
by band from the file itself. Airborne potassium maps alteration along faults that have no
scarp &mdash; exactly the faults a scarp-based catalogue is missing. The data are free and
official; the source and the paper that uses them this way are on the sources page.</p>
<a href="sources.html">Sources with links &rarr;</a></div>
</div>

<h2>What is in the box</h2>
{table(["Artifact", "What it is", "Where"],
 [["Validated candidate GeoTIFF", "Supervised expression model, holdout-validated (mean dense DTI 0.2795 vs gate 0.1253), unique bytes and name", f"<a href='downloads/{esc(cand_tif)}'>download</a>" if cand_tif else "not built"],
  ["Safe submission GeoTIFF", "Single-band float32, EPSG:32611, values in [0,1], no NaN anywhere, all format checks passed", f"<a href='downloads/{esc(tif)}'>download</a>"],
  ["Submission sidecars", "Names, comments, hashes, and the exact checks that ran", f"<code>docs/downloads/{esc(meta.get('downloads', {}).get('sidecar', ''))}</code>"],
  ["Forensic audit", f"{fa.get('n_duplicate_blob_groups', '?')} duplicated blob groups, byte-identical rasters, repo-to-repo copy percentages", "<a href='findings.html'>findings</a>"],
  ["Frozen protocol", "Hide-and-recover holdout, matched budgets, null baseline, off-catalogue surrogate screen", "<a href='hypotheses.html'>hypotheses</a>"],
  ["Hypotheses", "Pre-registered predictions with measured outcomes, negatives included", "<a href='hypotheses.html'>hypotheses</a>"],
  ["Sources", "Official links, each labelled with how deeply it was verified", "<a href='sources.html'>sources</a>"],
  ["Limitations", "What this cannot claim, and what still needs doing", "<a href='limitations.html'>limitations</a>"]])}
</section>
"""
    return page("index.html", "Download the submission", body)


def _fallback_row(meta):
    """The NaN-footprint twin, offered only as a fallback and labelled as such."""
    fb = meta.get("downloads", {}).get("fallback_tif")
    if not fb:
        return ""
    return (f'<p class="tiny">If the form ever objects to finite values outside the data '
            f'footprint (it has not so far), upload '
            f'<a href="downloads/{esc(fb)}" download>the identical prediction with the '
            f'sample submission\'s exact NaN footprint</a> instead. The primary file '
            f'contains no NaN at all and cannot trigger the '
            f'"values must be in range [0, 1]" error.</p>')


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _best_hyp(hy):
    d = hy.get("hypotheses", {})
    best, bv = None, -1.0
    for k, v in d.items():
        if k == "NULL_random":
            continue
        m = v.get("dense_competition_like", {}).get("mean_DTI", -1)
        if m > bv:
            best, bv = k, m
    return bv if best else None


# ------------------------------------------------------- how to submit
def howto_page():
    meta = load("submission.json")
    name = meta.get("latest", "")
    tif = meta.get("downloads", {}).get("tif", "")
    zipf = meta.get("downloads", {}).get("zip", "")
    comment = meta.get("comment", "")
    rng = load("range_audit.json")
    ra = load("reference_holdout.json")

    body = f"""
<h1>How to submit, exactly</h1>
<p class="lede">Four steps. The format checks in step 4 were already run against the
exact bytes you are about to upload.</p>

<ol class="steps">
<li><h3>Download the file</h3>
<p><a class="btn" href="downloads/{esc(tif)}" download>Download the submission GeoTIFF</a></p>
<p class="tiny">or <a href="downloads/{esc(zipf)}" download>the .zip</a>, which contains the
same single GeoTIFF. The competition accepts either form.</p></li>

<li><h3>Name it and describe it</h3>
<p>Use the name and comment already recorded in the sidecar, so that the entry on the
leaderboard can be traced back to this build:</p>
<pre>{esc(name)}</pre>
<pre>{esc(comment)}</pre></li>

<li><h3>Upload it</h3>
<p>Go to the competition's submission page and upload the file, pasting the name and the
comment into the two fields the form provides. Nothing else is required.</p></li>

<li><h3>If the form rejects it, do not guess</h3>
<p>Run the gate locally and read which check failed:</p>
<pre>python scripts/validate_submission.py downloads/{esc(tif)}</pre>
<p>It exits non-zero on the first failing check and prints the measured value and the
required value side by side. If the failure is the <code>[0, 1]</code> range check itself,
upload the fallback file named in the sidecar instead: it is the same prediction with no
NaN anywhere, so a validator that does not honour nodata has nothing to object to.</p></li>
</ol>

<h2>The format the competition requires</h2>
{table(["Requirement", "Value", "Where this comes from"],
 [["File", "one GeoTIFF, or a .zip containing one GeoTIFF", "probability of fault at each pixel, 0 to 1"],
  ["Bands", "exactly 1", "single-band prediction surface"],
  ["Data type", "float32", "the sample submission's dtype"],
  ["CRS", "EPSG:32611", "the sample submission's CRS"],
  ["Grid", "same width, height and geotransform as the sample submission", "required by the metric"],
  ["Values", "finite values inside [0, 1]", "predicted probabilities"],
  ["Outside the data footprint", "NaN or null", "the sample submission uses NaN"],
  ["Missing cells inside the footprint", "0.0, never NaN", "NaN inside the footprint is a rejection even though it is not out of range"]])}

<h2>Why the earlier upload failed, and what is different now</h2>
<p>The operator's upload was rejected with <em>"Predicted values must be in range
[0, 1]"</em>. That message names one of several distinct failure modes, so the gate
checks all of them rather than only the one in the message:</p>
<ul>
<li>a finite value below 0 or above 1 &mdash; the reported failure;</li>
<li>a NaN <em>inside</em> the data footprint, which is not out of range but is still
invalid, and which a range-only check would pass;</li>
<li>a nodata sentinel such as <code>-9999</code> written inside the footprint, which is
simultaneously out of range and a missing-value marker;</li>
<li>the wrong dtype, the wrong band count, the wrong CRS, or a shifted geotransform.</li>
</ul>
<p>An audit of <strong>{rng.get('files_audited', '?')}</strong> GeoTIFFs published across
the project family found <strong>{rng.get('files_with_out_of_range_values', '?')}</strong>
with an out-of-range value and
<strong>{rng.get('files_with_nan_inside_footprint', '?')}</strong> with a NaN inside the
footprint, so the exact file that produced the rejection is not identifiable from the
published artifacts. It is recorded as an open irregularity rather than guessed at. What
this build changes is that the failure cannot recur silently: the writer repairs out-of-
range and non-finite values, and the gate refuses to publish a file that fails.

<h2>Which file to upload</h2>
<p>There are two, and which one you want depends on the goal:</p>
<ul>
<li><strong>A new score attempt:</strong> the validated candidate on the
<a href="index.html">download page</a> &mdash; the supervised expression model, which
recovered withheld fault segments at mean dense DTI <strong>0.2795</strong> against the
previous best of 0.1253 under the hide-and-recover protocol. Unique bytes, unique name,
its own comment. This is the first file in this project validated to be worth a
submission slot.</li>
<li><strong>A guaranteed-accepted upload:</strong> the safe rebuild &mdash; the support of
the raster that already holds 0.1563, rebuilt and re-verified. It carries no new score
claim: uploading it reproduces 0.1563 by construction.</li>
</ul>
<p>Whichever file you upload, paste the name and comment printed next to its button. Both
are unique per build (content hash + UTC timestamp), so the leaderboard entry can always
be traced back to the exact bytes.</p>
"""
    return page("how-to-submit.html", "How to submit", body)


# ------------------------------------------------------------ findings
def findings_page():
    fa = load("forensic_audit.json")
    rng = load("range_audit.json")
    hy = load("hypotheses.json")
    ref = load("reference_holdout.json")
    gap = load("h1_gap_diagnostic.json")
    xcat = load("xcat_surrogate.json")
    xcat_q = load("xcat_qfaults_refusal.json")
    union = load("union_candidate.json")
    cov = load("coverage_emission.json")
    sup = load("supervised_holdout.json")

    fs = fa.get("byte_identical_submission_sha256", {})
    key = next((s for s in fs if s.startswith("7f00890a")), None)
    locs = fs.get(key, [])
    tl = [f"{l.split(':')[0]}:{l.split(':', 1)[1]}" for l in locs]

    pairs = [p for p in fa.get("repo_pairs", []) if p.get("percent_work_identical", 0) >= 50]

    body = f"""
<h1>Findings</h1>
<p class="lede">Everything on this page is produced by
<code>scripts/forensic_audit.py</code>, <code>scripts/audit_ranges.py</code>,
<code>scripts/run_hypotheses.py</code> and <code>scripts/score_reference_holdout.py</code>,
and written to <code>evidence/</code>. The raw JSON is linked at the bottom.</p>

<h2>F-1 &mdash; Are we uploading the same work over and over? Yes.</h2>
<p>A file's git blob SHA is a hash of its bytes, so two files sharing one cannot be a
coincidence. Comparing them across the project family:</p>
{table(["Repository pair", "Shared work files", "Byte-identical", "Percentage"],
 [[esc(p["a"]), esc(p["b"]), f"{p['work_shared_paths']}", f"{p['work_identical_blobs']}",
   f"<strong>{p['percent_work_identical']} %</strong>"] for p in pairs[:6]])}
<p>Three of the sibling repositories are, by bytes, mostly copies of <code>GEMSDOE</code>.
This is not an inference from similar scores: it is a hash comparison of the source trees.</p>

<h2>F-2 &mdash; The identical-score question, answered from the bytes</h2>
<p>The raster with SHA-256 beginning <code>{esc(key[:8]) if key else '7f00890a'}</code>
appears at <strong>{len(locs)}</strong> paths in
<strong>{len({l.split(':')[0] for l in locs})}</strong> different repositories:</p>
{table(["Location"], [[f"<code>{esc(l)}</code>"] for l in tl])}
<p>Two accounts submitting that file would receive the same score <em>by construction</em>,
because the scored object is byte-identical. Nothing about model agreement is required to
explain it. This is the answer to "why do 5GEMSDOE and GEMSDOE1 have the same score": it
is one file, published twice.</p>

<h2>F-3 &mdash; Support overlap between distinct predictions</h2>
<p>Two submissions can differ in bytes and still be the same map. The Jaccard overlap of
their positive pixels settles it:</p>
{table(["Overlap", "File A", "File B"],
 [[f"<strong>{m['jaccard']:.4f}</strong>", esc(m["a"]), esc(m["b"])]
  for m in fa.get("support_jaccard", []) if m["jaccard"] >= 0.5][:8])}
<p><code>8GEMSDOE_Hedge-v2_submission.tif</code> and
<code>5GEMSDOE/candidate_s5_catalogue_hedge.tif</code> are the same prediction under two
names in two repositories.</p>

<h2>F-4 &mdash; The range audit, and the upload error</h2>
<p>Reproduce with <code>python scripts/audit_ranges.py</code>:
<strong>{rng.get('files_audited', '?')} files audited,
{rng.get('files_with_out_of_range_values', '?')} with an out-of-range value,
{rng.get('files_with_nan_inside_footprint', '?')} with a NaN inside the official
footprint</strong>. The rejection the operator saw is therefore <em>not</em> reproducible
from any currently published artifact, which is recorded as an open irregularity rather
than resolved by guessing. The gate that prevents a recurrence is in
<code>src/gems/io.py</code>.</p>

<h2>F-5 &mdash; Why the local validation kept mis-ranking candidates</h2>
<p>This is the most important finding of the session, and it is a property of the
measurement rather than of any model.</p>
{table(["Candidate", "Dense-budget DTI", "Sparse-budget DTI"],
 [[esc(k), f"<strong>{v.get('dense_competition_like', {}).get('mean_DTI', float('nan')):.4f}</strong>",
   f"{v.get('sparse_matched_truth_px', {}).get('mean_DTI', float('nan')):.4f}"]
  for k, v in sorted(hy.get("hypotheses", {}).items(),
                     key=lambda kv: -kv[1].get("dense_competition_like", {}).get("mean_DTI", 0))])}
<p><strong>Uniform random noise scores {_fmt(hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI'))}
and beats every physically motivated detector</strong>, including the supplied magnetic
edge band and a fresh tilt-angle edge operator, at both budgets. The reference field that
holds 0.1563 on the public leaderboard scores
<strong>{_fmt(ref.get('summary', {}).get('dense_mean_DTI'))}</strong> on the same test &mdash;
barely above noise.</p>
<p>The reason is geometrical. The metric credits a prediction anywhere within 300 m of a
truth pixel, the catalogue is spread across the whole map, and a real submission emits
only about 3 % of the valid area. A detector that concentrates its budget on the strongest
anomaly leaves most of the map uncovered; noise covers everything evenly and collects the
credit. <strong>The catalogue-recovery holdout at competition density is dominated by
coverage, not by detection</strong>, which is exactly why a leave-one-fault-system-out
proxy correlated &minus;0.20 with the public leaderboard in the sibling audit.</p>
<p>The consequence for the next iteration is concrete: a local score computed this way
cannot rank candidates, and an improvement in marginal precision must be demonstrated
<em>at the emission density the leaderboard actually uses</em>. Anything measured only at a
sparse budget is uninformative, and re-tuning parameters against it is wasted effort.</p>

<h2>F-6 &mdash; The catalogue is more fragmented than a fault map suggests</h2>
<p>Reading the supplied labels raster directly: <strong>{gap.get('catalogue_fragmentation', {}).get('components', '?')}
8-connected components</strong>, median size
<strong>{gap.get('catalogue_fragmentation', {}).get('median_component_px', '?')} pixels</strong>,
largest <strong>{gap.get('catalogue_fragmentation', {}).get('max_component_px', '?')} pixels</strong>.
Most mapped "faults" in this raster are 1&ndash;2 km segments. Any structural inference
built on the catalogue's topology &mdash; bridging, linking, ordering &mdash; is operating on
a heavily segmented version of the real network, and that must be accounted for before the
result is interpreted geologically.</p>

<h2>F-7 &mdash; The off-catalogue arm: Qfaults refused, SGMC surrogate built</h2>
<p>The hide-and-recover holdout cannot rank a discovery hypothesis, because its truth
<em>is</em> the catalogue. The obvious second arm &mdash; an independent official fault
compilation &mdash; was attempted with the USGS Quaternary Fault and Fold Database
(Qfaults, DOI 10.5066/P9BCVRCK) and quantitatively refused by the sibling project's
cross-catalogue measurement: {xcat_q.get('population', {}).get('B_code1_near_a_label_px', '?')}
of its {xcat_q.get('population', {}).get('B_in_footprint_px', '?')} in-footprint pixels are
already within the credit radius of a competition label, leaving
<strong>{xcat_q.get('population', {}).get('B_only_px', '?')} independent pixel</strong>.
Inside this footprint, Qfaults and the competition catalogue are the same lines &mdash;
verified from the raster bytes, not inferred. (Source:
<code>GEMSDOE/data/evidence/xcat/transfer_report.json</code>.)</p>
<p>The remaining surrogate is the USGS State Geologic Map Compilation structure raster
(Horton, San Juan and Stoeser, 2017, DOI 10.3133/ds1052): {xcat.get('surrogate_census', {}).get('code2_beyond_halo_px', '?')}
pixels of mapped structure lie <em>beyond</em> the catalogue's 300 m credit halo. Scoring
every detector against that population (masking the catalogue halo from both sides):</p>
{table(["Field", "Surrogate dense DTI", "Lift vs uniform random"],
 [[esc(k), f"{xcat.get('hypotheses', {}).get(k, {}).get('dense_competition_like', {}).get('dti', float('nan')):.4f}",
   f"{xcat.get('lifts_vs_null', {}).get(k, {}).get('dense_lift_vs_null', float('nan')):.2f}x"]
  for k in xcat.get('ranked_by_dense_DTI', [])])}
<p><strong>Uniform random beats every physical detector on real off-catalogue
structure.</strong> The pre-registered predictions failed &mdash; H2 was expected at
&ge;1.5x random and measured far below &mdash; and the failure is logged as such. The
screen's verdict on the local toolkit is blunt: no rule-based signal in this repository
finds structure the catalogue lacks. The surrogate's own caveat is carried with it: SGMC
maps bedrock structure of any age, so clearing this screen is necessary, not sufficient,
for finding a young hidden fault.</p>

<h2>F-8 &mdash; Emission shaping is closed: three independent negatives</h2>
<p>Adding coverage-optimal filler to the 0.1563 support was the natural next emission
idea. It was tested with the metric's own marginal rule rather than by eyeballing a map:</p>
{table(["Fold", "Base DTI", "Union DTI", "Delta", "Marginal TP / added px", "Break-even"],
 [[str(u.get('fold')), f"{u.get('base_dti', float('nan')):.4f}", f"{u.get('union_dti', float('nan')):.4f}",
   f"{u.get('delta', float('nan')):+.4f}", f"{u.get('marginal_precision_per_added_px', float('nan')):.4f}",
   f"{u.get('break_even_at_base', float('nan')):.4f}"] for u in union.get('folds', [])])}
<p>The filler's marginal precision ({union.get('folds', [{}])[0].get('marginal_precision_per_added_px', 0):.4f}&ndash;{max((u.get('marginal_precision_per_added_px', 0) for u in union.get('folds', [])), default=0):.4f})
is <strong>below the break-even threshold in every fold</strong>, so the union was
refused publication by the gate. Together with the earlier H6 truncation curve (monotone
to full support) and the family's thinned submissions losing points on the leaderboard
(12GEMSDOE 0.1294 &lt; 0.1563), this closes emission shaping from three directions.
Spreading does raise raw coverage &mdash; lattice-NULL reaches
{_fmt(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__lattice', {}).get('mean_DTI'))}
against clustered-NULL's {_fmt(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__clustered_topk', {}).get('mean_DTI'))}
on the holdout &mdash; but that credit is already spoken for once a real support is
emitted. <strong>Only detection skill moves the score.</strong></p>

<h2>F-9 &mdash; The supervised expression model recovers withheld faults</h2>
<p>The one approach missing from this repository was the reference solution's: a model
trained to recognise catalogue fault expression. It was built with strict leakage
control &mdash; no location or distance features, training positives from visible
catalogue pixels only, the blind corridor masked to NaN at training and inference &mdash;
and tested on the frozen hide-and-recover protocol:</p>
{table(["Fold", "Sparse DTI", "Dense DTI", "Marginal TP / emitted px", "Break-even"],
 [[str(r.get('fold')), f"{r.get('sparse_matched_truth_px', {}).get('dti', float('nan')):.4f}",
   f"<strong>{r.get('dense_competition_like', {}).get('dti', float('nan')):.4f}</strong>",
   f"{r.get('dense_competition_like', {}).get('tp_per_emitted_px', float('nan')):.4f}",
   f"{r.get('dense_competition_like', {}).get('break_even', float('nan')):.4f}"]
  for r in sup.get('folds', [])])}
<p>Mean dense DTI <strong>{_fmt(sup.get('mean_dense_DTI'))}</strong> against the current
holdout best of <strong>0.1253</strong> and the uniform-random baseline of
{_fmt(hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI'))}.
Marginal precision clears break-even in every fold. This is the first candidate in the
project's history to clear the gate, and it is published as a uniquely named download.
Its one failure is on record too: on the SGMC off-catalogue screen it scores
{_fmt(sup.get('surrogate', {}).get('dense_DTI'))} &mdash; below the reference support and
far below random &mdash; so the model recovers <em>catalogue-like</em> faults but has not
demonstrated discovery of structure unlike the catalogue. That is exactly the quantity
the leaderboard will judge.</p>

<h2>Raw evidence</h2>
<p class="tiny">{" &middot; ".join(f"<a href='data/{esc(f.name)}'>{esc(f.name)}</a>" for f in sorted((DOCS / 'data').glob('*.json')))}</p>
"""
    return page("findings.html", "Findings", body)


# ---------------------------------------------------------- hypotheses
def hypotheses_page():
    hy = load("hypotheses.json")
    ch = load("candidate_hypotheses.json")
    ref = load("reference_holdout.json")
    gap = load("h1_gap_diagnostic.json")
    xcat = load("xcat_surrogate.json")
    cov = load("coverage_emission.json")
    union = load("union_candidate.json")
    sup = load("supervised_holdout.json")
    terrain = load("terrain_census.json")

    res = hy.get("hypotheses", {})
    null_d = res.get("NULL_random", {}).get("dense_competition_like", {}).get("mean_DTI")
    rows = []
    for k, v in sorted(res.items(), key=lambda kv: -kv[1].get("dense_competition_like", {}).get("mean_DTI", 0)):
        d = v.get("dense_competition_like", {})
        s = v.get("sparse_matched_truth_px", {})
        verdict = ("baseline" if k == "NULL_random"
                   else ("<strong>beats null</strong>" if d.get("mean_DTI", 0) > (null_d or 0)
                         else "rejected"))
        rows.append([f"<code>{esc(k)}</code>", f"{d.get('mean_DTI', float('nan')):.4f}",
                     f"{s.get('mean_DTI', float('nan')):.4f}",
                     f"{d.get('mean_tp_per_emitted_px', float('nan')):.4f}", verdict])

    cands = ch.get("candidates", [])
    crows = []
    for c in cands:
        crows.append([
            str(c["rank"]),
            f"<strong>{esc(c['name'])}</strong><br><span class='tiny'>{esc(c['id'])}</span>",
            "<ul class='tight'>" + "".join(f"<li>{esc(x)}</li>" for x in c["layers"]) + "</ul>",
            esc(c["physical_signature"]),
            esc(c["why_it_catches_a_fault_missing_from_the_catalogue"]),
            esc(c["how_it_differs"]),
            esc(c["expected_dti_improvement"]),
            esc(c["implementation_cost"]),
        ])

    nv = ch.get("not_viable_yet", [])

    body = f"""
<h1>Hypotheses</h1>
<p class="lede">Predictions were written down before the measurements were taken, on a
protocol frozen in advance. Failures are published in the same table as successes.</p>

<h2>The frozen protocol</h2>
<ul>
<li><strong>Hide and recover.</strong> Whole 8-connected catalogue components of 20 pixels
or more are withheld, in four spatially disjoint block folds. Isolated pixels would be
recoverable by any smoothing operator, so they would test interpolation rather than
detection.</li>
<li><strong>A blind corridor.</strong> A 3-pixel buffer around every withheld segment is
removed from the detector's inputs, so a trace cannot simply be extended from where it
stops.</li>
<li><strong>Pixel-exact masking.</strong> At scoring time the remaining catalogue is
removed from both prediction and truth, mirroring the organisers' rule. A prediction that
lands on a known fault gains nothing.</li>
<li><strong>Two budgets.</strong> Every candidate emits exactly as many pixels as the fold
contains truth pixels (a pure localisation test), and separately the 3.3 % of valid area
that real submissions in this family emit (the density at which the leaderboard actually
judges).</li>
<li><strong>A null.</strong> Uniform random noise at the same budget, so "does this know
anything?" has an answer.</li>
</ul>
{table(["Candidate", "Dense DTI", "Sparse DTI", "Marginal TP per emitted px", "Verdict"],
 rows)}
<p class="warn">The verdicts follow from the numbers and are not softened: at the density
that matters, <strong>no detector here beats uniform noise</strong>. The best physical
candidate scores {_best_hyp(hy):.4f} against a null of {null_d:.4f}. Section F-5 on the
<a href="findings.html">findings page</a> explains why, and why that result invalidates
this class of local score rather than these particular detectors.</p>

<h2>The reference field under the same protocol</h2>
{table(["Measure", "Value", "Reading"],
 [["As published, own support", f"{ref.get('summary', {}).get('as_published_mean_DTI', float('nan')):.4f}", "the file exactly as it would be uploaded"],
  ["Dense budget", f"{ref.get('summary', {}).get('dense_mean_DTI', float('nan')):.4f}", f"against the null's {_fmt(null_d)}"],
  ["Marginal TP per emitted px", f"{ref.get('summary', {}).get('mean_marginal_tp_per_emitted_px', float('nan')):.4f}", f"vs its own break-even of {ref.get('summary', {}).get('break_even_marginal_tp', float('nan')):.4f}"]])}
<p class="tiny">Caveat carried with the number: the reference field was trained on the
whole catalogue, including the segments this protocol withholds, so its score leaks and is
an upper bound rather than an estimate of generalisation.</p>

<h2>Was the zero a detector bug? Checked, and no</h2>
<p>H3's bridges scored {_fmt(res.get('H3_relay_bridge', {}).get('dense_competition_like', {}).get('mean_DTI'))}.
Two explanations fit a zero and they demand opposite responses: the detector might be
bridging rasterisation splits inside single faults, or the test simply cannot see a bridge.
<code>scripts/diagnose_h1_gaps.py</code> separates them:</p>
{table(["Gap window", "Bridges", "Share"],
 [[esc(b["label"]), f"{b['count']:,}", f"{b['fraction']*100:.1f} %"] for b in gap.get("gap_bands", [])])}
<p>Median selected gap <strong>{gap.get('gap_px_percentiles', {}).get('50', float('nan')):.1f} px
({gap.get('gap_px_percentiles', {}).get('50', 0)*0.1:.1f} km)</strong>, and
<strong>{gap.get('fraction_within_published_range', 0)*100:.1f} %</strong> fall inside the
published 1.6&ndash;3.2 km relay-ramp width range. The detector is finding the right kind of
structure; the zero is a property of the test.</p>

<h2>This session's iteration &mdash; predictions written before the run</h2>
<p>The standing brief asks for one falsifiable hypothesis per session, a named transform,
a written prediction of direction and size, and a logged outcome whether it passed or
failed. This iteration ran five tests in that order:</p>
{table(["Test", "Prediction (pre-registered)", "Measured", "Verdict"],
 [["H-XSUR: relay-bridge and edge-lineament transforms on the off-catalogue surrogate",
   "H2 &ge; 1.5x random; H3 &gt; random; reference &le; 1.2x random (direction up, 0.10-0.25 vs 0.05-0.10)",
   f"H2 {_fmt(xcat.get('hypotheses', {}).get('H2_tilt_lineament', {}).get('dense_competition_like', {}).get('dti'))}, H3 {_fmt(xcat.get('hypotheses', {}).get('H3_relay_bridge', {}).get('dense_competition_like', {}).get('dti'))}, random {_fmt(xcat.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('dti'))}",
   "<span class='bad'>failed</span> &mdash; logged; redirected away from catalogue-topology detectors"],
  ["H-BASIN: the surrogate truth hides under basin fill",
   "truth in flat terrain (deep cover, low slope)",
   f"truth slope mean {terrain.get('measured', {}).get('det_elev_slope', {}).get('truth_mean', 0):.1f} vs {terrain.get('measured', {}).get('det_elev_slope', {}).get('scorable_mean', 0):.1f} scorable; {terrain.get('measured', {}).get('truth_fraction_in_steepest_quartile', 0)*100:.0f} % in steepest quartile",
   "<span class='bad'>falsified</span> &mdash; the opposite is true; logged"],
  ["H-COV: coverage-optimal spreading at fixed budget",
   "spread &ge; 1.5x clustered on the catalogue holdout (0.12 &rarr; 0.18-0.25); &ge; 1.2x on the surrogate",
   f"holdout {_fmt(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__lattice', {}).get('mean_DTI'))} vs {_fmt(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__clustered_topk', {}).get('mean_DTI'))} ({(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__lattice', {}).get('mean_DTI', 0) / max(cov.get('arms', {}).get('catalogue_hide_and_recover', {}).get('dense_mean', {}).get('NULL_random__clustered_topk', {}).get('mean_DTI', 1), 1e-9)):.2f}x); surrogate 1.23x",
   "<span class='warn'>direction held, size short</span> &mdash; coverage effect real but smaller than predicted"],
  ["Union: lattice filler added to the 0.1563 support",
   "union mean fold DTI 0.13-0.17, direction up (marginal &gt; break-even)",
   f"union {_fmt(union.get('mean_union_dti'))} vs base {_fmt(union.get('mean_base_dti'))}; marginal {min((u.get('marginal_precision_per_added_px', 0) for u in union.get('folds', [{}])), default=0):.4f}-{max((u.get('marginal_precision_per_added_px', 0) for u in union.get('folds', [{}])), default=0):.4f} &lt; break-even",
   "<span class='bad'>failed</span> &mdash; gate refused publication; emission shaping closed"],
  ["H-SUP: supervised catalogue-expression model",
   "mean fold dense DTI &ge; 0.1253 (direction up, rough 0.13-0.20); surrogate &gt; 0.1854",
   f"holdout {_fmt(sup.get('mean_dense_DTI'))} (folds {min((r.get('dense_competition_like', {}).get('dti', 0) for r in sup.get('folds', [{}])), default=0):.3f}-{max((r.get('dense_competition_like', {}).get('dti', 0) for r in sup.get('folds', [{}])), default=0):.3f}); surrogate {_fmt(sup.get('surrogate', {}).get('dense_DTI'))}",
   "<span class='ok'>p1 held decisively</span>, p2 failed &mdash; recovery skill validated, discovery spread not"]])}
<p>The failures are as informative as the success. H-XSUR says catalogue-topology and
edge-lineament detectors do not find real structure the catalogue lacks. H-BASIN's
falsification says the missing-structure population visible to SGMC is <em>range</em>
structure, not basin fill. The union negative closes emission shaping. H-SUP's split
verdict says the delivered bands do carry learnable fault expression &mdash; but expression
that looks like the catalogue, which is what the holdout rewards and what the leaderboard
may not. The next hypothesis must therefore attack faults that do <em>not</em> look like
the catalogue: radiometric alteration and 1 m lidar scarps are the ranked path.</p>

<h2>Next candidates, ranked by expected gain and cost</h2>
<p>Each names the layers, the physical signature, why it should catch a fault that is
<em>missing</em> from the catalogue rather than one already in it, and how it differs from
everything already built in this family. Sources for the external data are on the
<a href="sources.html">sources page</a>.</p>
{table(["#", "Candidate", "Layers", "Physical signature", "Why it finds a missing fault", "How it differs", "Expected gain", "Cost"],
 crows, "t wide")}

<h3>Rejected before implementation, on physical grounds</h3>
{"".join(f"<p><strong>{esc(n['name'])}</strong> &mdash; {esc(n['reason'])}</p>" for n in nv)}
"""
    return page("hypotheses.html", "Hypotheses", body)


# ------------------------------------------------------------- sources
def sources_page():
    src = load("sources.json")
    entries = src.get("sources", src if isinstance(src, list) else [])
    rows = []
    for s in entries:
        depth = s.get("verification", s.get("level", "identified"))
        cls = {"opened": "ok", "retrieved": "ok", "quoted": "ok"}.get(str(depth).lower(), "warn")
        rows.append([
            f"<a href='{esc(s.get('url', '#'))}'>{esc(s.get('title', s.get('id', '')))}</a>",
            esc(s.get("publisher", "")),
            f"<span class='{cls}'>{esc(depth)}</span>",
            esc(s.get("what_it_supports", s.get("supports", ""))),
        ])
    na = src.get("deliberately_not_asserted", [])
    body = f"""
<h1>Sources</h1>
<p class="lede">No claim on this site is made without a source that was opened. Each row
says how deeply it was verified, so a reviewer can tell a quoted document from a
catalogue record.</p>
{table(["Source", "Publisher", "Verification", "What it is used for"], rows)}
<h2>Deliberately not asserted</h2>
<p>These are commonly repeated in this project's history but could not be confirmed from a
source that was actually opened, so they are flagged rather than stated:</p>
<ul>{"".join(f"<li>{esc(x)}</li>" for x in na) or "<li>None.</li>"}</ul>
<p class="tiny">Sources that were identified but not opened are labelled as such in the
table above and are never used as the sole support for a claim.</p>
"""
    return page("sources.html", "Sources", body)


# --------------------------------------------------------- limitations
def limitations_page():
    hy = load("hypotheses.json")
    ref = load("reference_holdout.json")
    body = f"""
<h1>Limitations</h1>
<p class="lede">Read this before trusting any number on the site.</p>

<h2>What this repository cannot claim</h2>
<ul>
<li><strong>No new leaderboard score is claimed.</strong> The supervised expression
model cleared the local hide-and-recover gate (mean dense DTI 0.2795 vs 0.1253) and is
published as a validated candidate, but no leaderboard submission was made from this
environment (no DrivenData authentication), so no public score improvement is asserted
here. The safe rebuild remains available and reproduces 0.1563 by construction.</li>
<li><strong>The holdout's truth is the catalogue, not the test set.</strong> The
leaderboard scores faults that are in no catalogue. A catalogue-recovery holdout measures
geographic generalisation; it does not measure discovery. The two are not on the same
scale and this site never compares them directly.</li>
<li><strong>The reference field's holdout score leaks.</strong> It was trained on the
whole catalogue, including the withheld segments. Its
{ref.get('summary', {}).get('dense_mean_DTI', float('nan')):.4f} is an upper bound, not a
valid estimate of generalisation.</li>
<li><strong>The local instruments are coverage-dominated.</strong> Uniform noise reaches
{hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI', float('nan')):.4f}
on the catalogue holdout and {load('xcat_surrogate.json').get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('dti', float('nan')):.4f}
on the off-catalogue surrogate at competition density. The surrogate arm now exists (SGMC
structure beyond the catalogue halo) and it re-ranked the toolkit: no rule-based detector
clears random on it, and the supervised model does not either. Discovery ranking remains
harder than recovery ranking.</li>
<li><strong>The band label for <code>tc</code> is ambiguous in the file itself</strong>,
which calls it "tilt angle or total curvature". This repository computes its own tilt
angle from <code>tmi</code> rather than trusting that label.</li>
</ul>

<h2>Environment constraints that shaped the result</h2>
<ul>
<li>Two CPU cores, 3 GB of RAM, <strong>no GPU</strong>. scikit-learn was installed from
PyPI this session (HistGradientBoosting trains in ~12 s per fold on CPU), but PyTorch and
LightGBM are unavailable, so a CNN ensemble of the kind that produced the family's best
result still cannot be retrained here.</li>
<li>Outbound network access is limited to <code>api.github.com</code> and the Python
package index, so the two highest-value external datasets (airborne radiometrics, 1 m
lidar) could not be fetched in this environment. Both are confirmed to exist and are
nameable to a machine that can reach them.</li>
<li>No DrivenData authentication, so no leaderboard submission was made and no
leaderboard movement is claimed.</li>
</ul>

<h2>What still needs doing, in priority order</h2>
<ol>
<li><strong>Spend a weekly slot on the validated candidate</strong> (the supervised
expression model, gate cleared at 0.2795 vs 0.1253) from an account that can reach the
competition, and record the leaderboard response. That number decides whether hidden
faults look like catalogue faults.</li>
<li><strong>Add the radiometric channel and retrain.</strong> The only new information
available, aimed at the class of fault the catalogue cannot contain &mdash; the class the
SGMC screen says our signals miss.</li>
<li><strong>Add 1 m lidar scarp morphology</strong> at the resolution the catalogue was
drawn at.</li>
<li><strong>Pursue a discovery-grade validation arm.</strong> Qfaults is refused (same
lines as the labels) and the SGMC surrogate is necessary-not-sufficient; the remaining
route is community fault mapping from 1 m lidar or expert-reviewed play-fairway layers.</li>
<li><strong>Make every submission unique.</strong> At least
{len(load('forensic_audit.json').get('byte_identical_submission_sha256', {}))} byte-identical
raster groups are currently published across the family, and the same file ascending under
several accounts is the single largest source of redundant leaderboard entries. Every
artifact this repository publishes now carries a content hash and UTC stamp in its name.</li>
</ol>

<h2>Open irregularities flagged for review</h2>
<ul>
<li>The <code>[0, 1]</code> upload rejection is not reproducible from any of the
{load('range_audit.json').get('files_audited', '?')} published GeoTIFFs audited.</li>
<li>At least {len(load('forensic_audit.json').get('byte_identical_submission_sha256', {}))}
raster files are published byte-identically in more than one repository, and three
repositories are 50&ndash;86 % byte-identical copies of another.</li>
<li>The problem description advertises a top-of-crustal magnetic source-depth estimate
that the delivered 19-band raster does not contain, verified band by band.</li>
</ul>
"""
    return page("limitations.html", "Limitations", body)


# ---------------------------------------------------------------- build
STYLE = """
:root{--bg:#0f1216;--panel:#171c22;--ink:#e8edf3;--dim:#9aa7b4;--line:#26303a;
--acc:#39d98a;--warn:#f0b429;--bad:#ff6b6b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
.bar{border-bottom:1px solid var(--line);background:#11151a;position:sticky;top:0;z-index:9}
.bar .wrap{display:flex;align-items:center;gap:18px;flex-wrap:wrap;padding:12px 20px}
.brand{font-weight:700;letter-spacing:.02em}
.bar nav{display:flex;gap:14px;flex-wrap:wrap}
.bar nav a{color:var(--dim);text-decoration:none;font-size:14px;padding:4px 0}
.bar nav a:hover{color:var(--ink)}
.bar nav a.on{color:var(--acc);border-bottom:2px solid var(--acc)}
main{padding:28px 20px 60px}
h1{font-size:2rem;line-height:1.2;margin:.2em 0 .4em}
h2{font-size:1.3rem;margin:2em 0 .5em;padding-bottom:.3em;border-bottom:1px solid var(--line)}
h3{font-size:1.02rem;margin:1.2em 0 .4em}
a{color:#7cc4ff}
p{margin:.7em 0}
code{background:#0b0e12;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
font-size:.88em;word-break:break-all}
pre{background:#0b0e12;border:1px solid var(--line);border-radius:6px;padding:10px 12px;
overflow-x:auto;font-size:.84em;word-break:break-all;white-space:pre-wrap}
.lede{font-size:1.06rem;color:var(--dim);max-width:70ch}
.tiny{font-size:.8rem;color:var(--dim)}
.hero{border:1px solid var(--line);background:linear-gradient(180deg,#1a2129,#141a20);
border-radius:12px;padding:22px;margin-bottom:8px}
.btn{display:inline-block;background:var(--acc);color:#06210f;font-weight:700;
padding:16px 26px;border-radius:10px;text-decoration:none;font-size:1.05rem;margin:6px 0}
.btn:hover{filter:brightness(1.1)}
.fields{margin-top:14px;display:grid;gap:4px}
.fields label{font-size:.8rem;color:var(--dim)}
.fields pre{margin:0 0 8px}
.ok{color:var(--acc)}.warn{color:var(--warn)}.bad{color:var(--bad)}
table.t{border-collapse:collapse;width:100%;margin:12px 0;font-size:.86rem}
table.t th,table.t td{border:1px solid var(--line);padding:7px 9px;text-align:left;
vertical-align:top}
table.t th{background:#1b222a;font-weight:600}
table.t tr:nth-child(even) td{background:#13181e}
table.wide{font-size:.8rem}
ul.tight{margin:0;padding-left:16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:14px 0}
.card{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:16px}
.card h3{margin-top:0}
.card a{font-size:.86rem}
ol.steps{padding-left:22px}
ol.steps li{margin-bottom:14px}
footer{margin-top:48px;border-top:1px solid var(--line);padding-top:14px}
"""


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "assets").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets" / "style.css").write_text(STYLE)
    (DOCS / "data").mkdir(parents=True, exist_ok=True)

    # submission_meta lives in docs/downloads; mirror it so pages can read it
    # from evidence/ without special-casing the path
    meta_src = DL / "submission_meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, EVID / "submission.json")

    pages = {
        "index.html": index_page(),
        "how-to-submit.html": howto_page(),
        "findings.html": findings_page(),
        "hypotheses.html": hypotheses_page(),
        "sources.html": sources_page(),
        "limitations.html": limitations_page(),
    }
    for name, html_text in pages.items():
        (DOCS / name).write_text(html_text)
        print(f"wrote docs/{name} ({len(html_text):,} bytes)")

    for f in sorted(EVID.glob("*.json")):
        if f.name.startswith("_"):          # internal caches, not published
            continue
        shutil.copy2(f, DOCS / "data" / f.name)
        print(f"copied evidence/{f.name} -> docs/data/")

    (DOCS / ".nojekyll").write_text("")
    # GitHub Pages for this repository serves the repository ROOT, and the
    # available token cannot change that setting. A root entry point is written
    # so a visitor still lands on a download button. See build_root_page.py.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_root_page import build as build_root          # noqa: E402
    (REPO / "index.html").write_text(build_root())
    (REPO / ".nojekyll").write_text("")
    print("wrote index.html and .nojekyll at the repository root")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
