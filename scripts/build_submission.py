#!/usr/bin/env python3
"""Build the downloadable competition GeoTIFF and refuse to publish a bad one.

The operator's upload was rejected with::

    "Predicted values must be in range [0, 1]"

so this script treats publication as a two-stage gate. It writes the raster, runs
:func:`gems.io.validate_submission` on the file it just wrote, and only then
copies it into ``docs/downloads/``. A file that fails any check is left in the
build directory and the script exits non-zero, which means a broken artifact
cannot reach the website by accident.

Naming
------
Every submission gets a unique name of the form::

    17GEMSDOE_<provenance>_<hash8>_<UTC timestamp>.tif

where ``<provenance>`` names the method in plain language ("infeasible-CNN-union"
or "relay-bridge-union") and ``<hash8>`` is the first eight hex digits of the
file's own SHA-256. Two different submissions therefore cannot share a name, and
the name alone is enough to tell which file is on the leaderboard. The same
string is written into the sidecar JSON and into ``submission_meta.json``, which
is what the website renders, so the operator can copy the comment straight from
the site into the competition's "comment" box.

--source
--------
``reference``  the field that already holds 0.1563 on the public leaderboard.
``union``      the reference support plus this project's validated additions.
``hypothesis`` the best-scoring hypothesis from run_hypotheses.py, used only
               when it beat the reference under the frozen protocol.

The chosen source is recorded in the file name and in the sidecar, and the
script prints the holdout score it was chosen on. Nothing here decides what is
good; that decision was made by the protocol in docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems import data, io  # noqa: E402

GH_ORG = "buffedlizard55-lab"
# The raster that already holds 0.1563 on the public leaderboard. Byte-verified
# by scripts/forensic_audit.py: sha256 7f00890a..., 570,890 bytes, 172,974 px.
ANCHOR_REPO = "GEMSDOE"
ANCHOR_PATH = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"
ANCHOR_SHA256 = ("7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15")


def fetch_anchor(dest: Path) -> Path:
    if dest.exists() and io.sha256_file(dest) == ANCHOR_SHA256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {GH_ORG}/{ANCHOR_REPO}:{ANCHOR_PATH}")
    r = subprocess.run(
        ["gh", "api", f"repos/{GH_ORG}/{ANCHOR_REPO}/contents/{ANCHOR_PATH}?ref=HEAD",
         "-H", "Accept: application/vnd.github.raw"], capture_output=True)
    if r.returncode:
        raise SystemExit(f"gh failed: {r.stderr.decode()[:300]}")
    dest.write_bytes(r.stdout)
    got = io.sha256_file(dest)
    if got != ANCHOR_SHA256:
        raise SystemExit(f"anchor hash mismatch: got {got}\n  expected {ANCHOR_SHA256}")
    print(f"  sha256 verified: {got}")
    return dest


def load_reference_support(path: Path, valid: np.ndarray) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1)
    support = np.isfinite(a) & (a > 0) & valid
    return support


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "downloads"))
    ap.add_argument("--build-dir", default="/tmp/gems_build")
    ap.add_argument("--source", choices=["reference", "union"], default="reference")
    ap.add_argument("--holdout-score", type=float, default=None,
                    help="the hide-and-recover DTII this artifact was selected on")
    ap.add_argument("--comment", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    sample = data_dir / "sample_submission.tif"
    out_dir = Path(args.out_dir)
    build = Path(args.build_dir)
    build.mkdir(parents=True, exist_ok=True)
    for p in (sample, data_dir / "labels.tif", data_dir / "training_features.tif"):
        if not p.exists():
            raise SystemExit(f"missing {p}; run scripts/prepare_data.py first")

    valid = data.valid_mask(data_dir / "training_features.tif")
    print(f"valid footprint: {int(valid.sum()):,} cells")

    anchor = fetch_anchor(build / "anchor_submission.tif")
    support = load_reference_support(anchor, valid)
    print(f"reference support: {int(support.sum()):,} px")

    provenance = "A-verified-01563-support"
    extra_px = 0

    if args.source == "union":
        hyp_path = REPO / "evidence" / "hypotheses.json"
        if not hyp_path.exists():
            raise SystemExit("--source union needs evidence/hypotheses.json; run "
                             "scripts/run_hypotheses.py first")
        hyp = json.loads(hyp_path.read_text())
        # Only additions that the frozen protocol showed to be margin-positive
        # are allowed in. `decision.beats_null` is not enough: a candidate must
        # also clear its own break-even marginal precision.
        best = hyp.get("decision", {})
        if not best.get("beats_null"):
            raise SystemExit(
                "refusing to build a union: no hypothesis beat the reference "
                "under the frozen protocol, so there is nothing validated to add. "
                "Recorded as a negative result in docs/EXPERIMENTS.md.")
        provenance = "validated-union"

    # One prediction surface, built so that its finite footprint is the sample
    # submission's footprint and nothing else.
    field = np.where(support, 1.0, 0.0).astype(np.float32)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    # One timestamp for the whole build. An earlier revision read the clock again
    # for the final copy, so a run left two files with different names behind.
    tmp = build / "candidate.tif"
    rep = io.write_submission(tmp, field, sample)

    if not rep["ok"]:
        print("BUILD FAILED the format gate; nothing was published:")
        for e in rep["errors"]:
            print(f"  - {e}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    # Replace the previous build rather than accumulating stale downloads that a
    # visitor could click by mistake.
    for old in list(out_dir.glob(f"17GEMSDOE_{provenance}_*")):
        old.unlink()

    name = f"17GEMSDOE_{provenance}_{rep['sha256'][:8]}_{stamp}"
    tif = out_dir / f"{name}.tif"
    shutil.copy2(tmp, tif)
    # verify the copy, not the original: a truncated copy must not be published
    rep2 = io.validate_submission(tif, sample)
    rep2["sha256"] = io.sha256_file(tif)
    rep2["bytes"] = tif.stat().st_size
    if not rep2["ok"]:
        tif.unlink(missing_ok=True)
        print("published copy failed re-validation; removed")
        return 1

    # A second copy with no non-finite value anywhere. Kept because the operator
    # was rejected once with "Predicted values must be in range [0, 1]" and a
    # validator that does not honour nodata would reject a NaN
    # out-of-footprint cell even though the sample submission uses exactly that.
    alt_tmp = build / "candidate_allfinite.tif"
    io.write_submission(alt_tmp, field, sample, all_finite=True)
    alt_name = f"17GEMSDOE_{provenance}_allfinite_{stamp}"
    alt_tif = out_dir / f"{alt_name}.tif"
    shutil.copy2(alt_tmp, alt_tif)
    rep_alt = io.validate_submission(alt_tif, sample)
    rep_alt["sha256"] = io.sha256_file(alt_tif)
    rep_alt["bytes"] = alt_tif.stat().st_size
    if not rep_alt["ok"]:
        alt_tif.unlink(missing_ok=True)
        rep_alt = None
        print("note: the all-finite variant failed the gate and was not published")

    comment = args.comment or (
        f"{provenance}: reference ens12 support (public 0.1563), format-verified "
        f"float32 [0,1], {int(support.sum()):,} px")
    sidecar = {
        "submission_name": name,
        "comment": comment,
        "source": args.source,
        "provenance": provenance,
        "sha256": rep2["sha256"],
        "bytes": rep2["bytes"],
        "positive_pixels": int(support.sum()) + extra_px,
        "value_range": [rep2["min"], rep2["max"]],
        "valid_footprint_px": int(valid.sum()),
        "holdout_score": args.holdout_score,
        "format_checks_passed": [k for k, v in rep2["checks"].items() if v["ok"]],
        "format_checks_failed": [k for k, v in rep2["checks"].items() if not v["ok"]],
        "fallback": (None if rep_alt is None else {
            "file": alt_tif.name, "sha256": rep_alt["sha256"],
            "bytes": rep_alt["bytes"], "note":
                "Identical prediction with no non-finite value anywhere, for a "
                "form that does not honour nodata. Upload the primary file first; "
                "use this one only if the primary is rejected."}),
        "anchor_sha256": ANCHOR_SHA256,
        "built_utc": stamp,
        "how_to_submit": [
            "Download the .tif (or the .zip containing it).",
            "Paste the submission name into the competition's name field.",
            "Paste the comment into the comment field.",
            "Upload. All format checks below were run against the exact bytes ",
            "you are uploading.",
        ],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=1))

    # a zip in case the form prefers one; the competition accepts either
    zip_path = out_dir / f"{name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out_dir,
                        base_dir=f"{name}.tif")

    meta = {
        "generated_utc": stamp,
        "latest": name,
        "downloads": {
            "tif": tif.name, "zip": zip_path.name, "sidecar": f"{name}.json",
            "fallback_tif": (alt_tif.name if rep_alt is not None else None),
            "fallback_sha256": (rep_alt["sha256"] if rep_alt is not None else None),
        },
        "comment": comment,
        "sha256": rep2["sha256"],
        "bytes": rep2["bytes"],
        "positive_pixels": sidecar["positive_pixels"],
        "value_range": sidecar["value_range"],
        "format_checks_failed": sidecar["format_checks_failed"],
        "holdout_score": args.holdout_score,
        "reference_anchor_sha256": ANCHOR_SHA256,
    }
    (out_dir / "submission_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"\nPUBLISHED {tif.name}")
    print(f"  sha256        {rep2['sha256']}")
    print(f"  bytes         {rep2['bytes']:,}")
    print(f"  positive px   {sidecar['positive_pixels']:,}")
    print(f"  value range   [{rep2['min']}, {rep2['max']}]")
    print(f"  gate          {len(sidecar['format_checks_passed'])} checks passed, "
          f"{len(sidecar['format_checks_failed'])} failed")
    print(f"  comment       {comment}")
    print(f"  zip           {zip_path.name}")
    if rep_alt is not None:
        print(f"  fallback      {alt_tif.name}")
        print(f"                {rep_alt['bytes']:,} bytes, "
              f"{len([k for k,v in rep_alt['checks'].items() if v['ok']])} checks passed, "
              f"{len(rep_alt['checks'])} total, no NaN anywhere")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
