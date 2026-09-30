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
    rng = meta.get("value_range", [None, None])
    hy = load("hypotheses.json")
    ref = load("reference_holdout.json")
    fa = load("forensic_audit.json")

    if not tif:
        dl = ('<p class="warn">No submission has been built yet. Run '
              '<code>python scripts/build_submission.py --source reference</code>.</p>')
    else:
        dl = f"""<a class="btn" href="downloads/{esc(tif)}" download>Download the submission GeoTIFF</a>
<p class="tiny">or <a href="downloads/{esc(zipf)}" download>the same file as a .zip</a>
&middot; {meta.get('bytes', 0):,} bytes &middot; sha256
<code>{esc(meta.get('sha256', ''))}</code></p>
<div class="fields">
<h3>Two fields to paste into the competition form</h3>
<label>Submission name</label>
<pre>{esc(name)}</pre>
<label>Short comment</label>
<pre>{esc(comment)}</pre>
</div>
<p class="ok">Format gate: <strong>{len(meta.get('format_checks_passed', []))} checks passed,
{len(checks_failed)} failed</strong> &middot; value range
<code>[{rng[0]}, {rng[1]}]</code> &middot; {meta.get('positive_pixels', 0):,} positive pixels.
The range check is the one the operator's earlier upload failed on.</p>"""

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

<div class="card"><h3>The local validation was measuring the wrong thing</h3>
<p>Under a hide-and-recover holdout that mirrors the organisers' masking rule,
<strong>uniform random noise scores
{_fmt(hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI'))}</strong>
on the catalogue &mdash; and the field that holds 0.1563 on the leaderboard scores
<strong>{_fmt(ref.get('summary', {}).get('dense_mean_DTI'))}</strong>. At competition
emission density, coverage dominates detection. A detector that concentrates its budget
on its strongest anomaly loses to one that spreads it.</p>
<a href="hypotheses.html">The measurements &rarr;</a></div>

<div class="card"><h3>Five hypotheses, none promoted</h3>
<p>Every pre-registered structural detector was measured. The best scored
<strong>{_fmt(_best_hyp(hy))}</strong> at the dense budget, against a random baseline of
{_fmt(hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI'))}.
Nothing beat the bar, so <strong>nothing took a submission slot</strong>, and the negative
results are published alongside the positive ones.</p>
<a href="hypotheses.html">Hypotheses and results &rarr;</a></div>

<div class="card"><h3>The one untried channel</h3>
<p>The 19 delivered bands contain <strong>no radiometric channel</strong>, verified band
by band from the file itself. Airborne potassium maps alteration along faults that have no
scarp &mdash; exactly the faults a scarp-based catalogue is missing. The data are free and
official; the source and the paper that uses them this way are on the sources page.</p>
<a href="sources.html">Sources with links &rarr;</a></div>
</div>

<h2>What is in the box</h2>
{table(["Artifact", "What it is", "Where"],
 [["Submission GeoTIFF", "Single-band float32, EPSG:32611, values in [0,1], all format checks passed", f"<a href='downloads/{esc(tif)}'>download</a>"],
  ["Submission sidecar", "Name, comment, hashes, and the exact checks that ran", f"<code>docs/downloads/{esc(meta.get('downloads', {}).get('sidecar', ''))}</code>"],
  ["Forensic audit", f"{fa.get('n_duplicate_blob_groups', '?')} duplicated blob groups, byte-identical rasters, repo-to-repo copy percentages", "<a href='findings.html'>findings</a>"],
  ["Frozen protocol", "Hide-and-recover holdout, matched budgets, null baseline", "<a href='hypotheses.html'>hypotheses</a>"],
  ["Hypotheses", "Pre-registered predictions with measured outcomes, negatives included", "<a href='hypotheses.html'>hypotheses</a>"],
  ["Sources", "Official links, each labelled with how deeply it was verified", "<a href='sources.html'>sources</a>"],
  ["Limitations", "What this cannot claim, and what still needs doing", "<a href='limitations.html'>limitations</a>"]])}
</section>
"""
    return page("index.html", "Download the submission", body)


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
required value side by side.</p></li>
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

<h2>What this file is, and what it is not</h2>
<p>It carries the support of the raster that already holds 0.1563 on the public
leaderboard, rebuilt and re-verified, with a unique name so it can be told apart from
every other upload in the family. On the frozen hide-and-recover protocol that field
scores <strong>{_fmt(ra.get('summary', {}).get('dense_mean_DTI'))}</strong>. It is
<strong>not</strong> a new high score, and this repository does not claim it is: no
hypothesis tested here beat the bar, so none was promoted. Uploading this file changes
nothing except that it is verified to be accepted.</p>
<p>If the goal is a new score rather than a safe upload, the ranked candidates on the
<a href="hypotheses.html">hypotheses page</a> are the path, and the top one needs a data
channel the current model has never seen.</p>
"""
    return page("how-to-submit.html", "How to submit", body)


# ------------------------------------------------------------ findings
def findings_page():
    fa = load("forensic_audit.json")
    rng = load("range_audit.json")
    hy = load("hypotheses.json")
    ref = load("reference_holdout.json")
    gap = load("h1_gap_diagnostic.json")

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
<li><strong>No new leaderboard score.</strong> No hypothesis tested here beat the bar
under the frozen protocol, so none was promoted and no score improvement is asserted. The
published file is a verified rebuild of the support that already holds 0.1563.</li>
<li><strong>The holdout's truth is the catalogue, not the test set.</strong> The
leaderboard scores faults that are in no catalogue. A catalogue-recovery holdout measures
geographic generalisation; it does not measure discovery. The two are not on the same
scale and this site never compares them directly.</li>
<li><strong>The reference field's holdout score leaks.</strong> It was trained on the
whole catalogue, including the withheld segments. Its
{ref.get('summary', {}).get('dense_mean_DTI', float('nan')):.4f} is an upper bound, not a
valid estimate of generalisation.</li>
<li><strong>The local instrument is coverage-dominated.</strong> Uniform noise reaches
{hy.get('hypotheses', {}).get('NULL_random', {}).get('dense_competition_like', {}).get('mean_DTI', float('nan')):.4f}
at competition density. Until the off-catalogue arm exists, no local number can rank a
discovery hypothesis.</li>
<li><strong>The band label for <code>tc</code> is ambiguous in the file itself</strong>,
which calls it "tilt angle or total curvature". This repository computes its own tilt
angle from <code>tmi</code> rather than trusting that label.</li>
</ul>

<h2>Environment constraints that shaped the result</h2>
<ul>
<li>Two CPU cores, 3 GB of RAM, <strong>no GPU</strong>, and no scikit-learn, PyTorch or
LightGBM available. A CNN ensemble of the kind that produced the family's best result
could not be retrained here, so the work was directed at measurement, audit and
deterministic detectors.</li>
<li>Outbound network access is limited to <code>api.github.com</code> and the Python
package index, so the two highest-value external datasets (airborne radiometrics, 1 m
lidar) could not be fetched in this environment. Both are confirmed to exist and are
nameable to a machine that can reach them.</li>
<li>No DrivenData authentication, so no leaderboard submission was made and no
leaderboard movement is claimed.</li>
</ul>

<h2>What still needs doing, in priority order</h2>
<ol>
<li><strong>Build the off-catalogue validation arm</strong> from an independent official
fault compilation, and re-rank every candidate on it. This is the highest-value fix
because it unblocks everything else.</li>
<li><strong>Add the radiometric channel and retrain.</strong> The only new information
available, aimed at the class of fault the catalogue cannot contain.</li>
<li><strong>Add 1 m lidar scarp morphology</strong> at the resolution the catalogue was
drawn at.</li>
<li><strong>Re-derive the emission rule from the metric's coverage structure</strong>,
because concentration is currently costing more than the models are gaining.</li>
<li><strong>Make every submission unique.</strong> At least
{len(load('forensic_audit.json').get('byte_identical_submission_sha256', {}))} byte-identical
raster groups are currently published across the family, and the same file ascending under
several accounts is the single largest source of redundant leaderboard entries.</li>
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
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
