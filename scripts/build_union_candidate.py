#!/usr/bin/env python3
"""Build the union candidate: reference 0.1563 support + coverage-optimal
lattice filler -- and only publish it if it beats the current holdout best
under the exact hide-and-recover test.

The hypothesis (pre-registered before this run)
-----------------------------------------------
The reference support's marginal precision on hidden truth is the equilibrium
value of its public score (0.1563 <-> marginal hit rate 0.0323, see
gems.metric.break_even_marginal_tp). The holdout measured a spread random
field's marginal hit rate on withheld catalogue at ~3.4 % (evidence/
coverage_emission.json: lattice-NULL dense DTI 0.1499), just ABOVE that
threshold -- so adding spread filler pixels to the empty parts of the map
should add net credit for catalogue-like truth.

Transform: a jittered lattice of 0/1 filler pixels over the valid footprint,
separated >= 4 px from each other (no two share one R = 3 px credit kernel),
excluding pixels already in the support.

Prediction: union mean fold DTI in [0.13, 0.17] at B = 60,000 filler px, up
from the base field's ~0.125 under the same protocol. Direction: up. The
leaderboard direction is contingent on hidden-truth density: up if hidden
faults are at least catalogue-sparse-spread, down if much sparser -- that
gamble is printed in the artifact's comment block and the safe download
remains the pure rebuild.

Gate (from the standing brief): nothing touches a submission slot unless it
beats the current holdout best (0.1253 dense, reference field) under this
exact test. This script refuses to publish otherwise.

Output: evidence/union_candidate.json and, if the gate is cleared,
docs/downloads/17GEMSDOE_C-union-spread_<hash8>_<stamp>.{tif,zip,json}
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
sys.path.insert(0, str(REPO / "scripts"))

from gems import data, io, metric  # noqa: E402
from gems.holdout import Holdout  # noqa: E402
from run_coverage_emission import lattice_select  # noqa: E402

SEED = 17
HOLDOUT_BEST = 0.1253          # current best dense DTI (reference field)
HOLDOUT_NULL = 0.1188          # reproduced uniform-random baseline
ANCHOR_SHA256 = "7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15"
GH_ORG = "buffedlizard55-lab"
ANCHOR_REPO = "GEMSDOE"
ANCHOR_PATH = "data/evidence/runs/ens12-adopted-floor0.1-w0/submission.tif"


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
        raise SystemExit(f"anchor hash mismatch: {got}")
    return dest


def run(data_dir: Path, budget: int, out_path: Path, publish: bool) -> dict:
    t0 = time.time()
    sample = data_dir / "sample_submission.tif"
    labels = data.load_labels(data_dir / "labels.tif")
    valid = data.valid_mask(data_dir / "training_features.tif")

    anchor = fetch_anchor(Path("/tmp/gems_build/anchor_submission.tif"))
    import rasterio
    with rasterio.open(anchor) as ds:
        a = ds.read(1)
    support = np.isfinite(a) & (a > 0) & valid
    print(f"support: {int(support.sum()):,} px")

    fillable = valid & ~support
    rng = np.random.default_rng(SEED)
    filler = lattice_select(labels.shape, budget, fillable, rng)
    union = support | filler
    print(f"filler: {int(filler.sum()):,} px (requested {budget:,}); "
          f"union: {int(union.sum()):,} px")

    ho = Holdout(labels)
    rows = []
    for fi, fold in enumerate(ho.folds):
        base = metric.dti(support.astype(np.float32), fold.truth, mask=fold.masked)
        uni = metric.dti(union.astype(np.float32), fold.truth, mask=fold.masked)
        fill_only = metric.dti(filler.astype(np.float32), fold.truth, mask=fold.masked)
        rows.append({
            "fold": fi,
            "base_dti": base.dti,
            "union_dti": uni.dti,
            "delta": uni.dti - base.dti,
            "base_emitted_unmasked_px": base.emitted_px,
            "union_emitted_unmasked_px": uni.emitted_px,
            "marginal_tp_w": uni.tp_w - base.tp_w,
            "marginal_fp_w": uni.fp_w - base.fp_w,
            "marginal_precision_per_added_px": ((uni.tp_w - base.tp_w)
                                                / max(uni.emitted_px - base.emitted_px, 1)),
            "filler_alone_tp_w": fill_only.tp_w,
            "filler_alone_precision": fill_only.tp_per_emitted_px,
            "break_even_at_base": metric.break_even_marginal_tp(base.dti),
        })
        print(f"  fold {fi}: base={base.dti:.4f} union={uni.dti:.4f} "
              f"delta={uni.dti - base.dti:+.4f}  "
              f"marginal={(uni.tp_w - base.tp_w) / max(uni.emitted_px - base.emitted_px, 1):.4f} "
              f"(break-even {metric.break_even_marginal_tp(base.dti):.4f})")

    mean_base = float(np.mean([r["base_dti"] for r in rows]))
    mean_union = float(np.mean([r["union_dti"] for r in rows]))
    mean_delta = mean_union - mean_base
    gate = mean_union > HOLDOUT_BEST

    report = {
        "hypothesis": ("spread filler adds net credit at the reference operating "
                       "point because spread-random's holdout marginal hit rate "
                       "(~3.4 %) exceeds the 0.1563 break-even (3.23 %)"),
        "preregistered_prediction": {
            "union_mean_fold_DTI": "[0.13, 0.17]",
            "direction": "up vs base",
            "leaderboard_caveat": ("up if hidden faults are at least "
                                   "catalogue-sparse-spread; down if much sparser"),
        },
        "budget": budget,
        "filler_px": int(filler.sum()),
        "union_px": int(union.sum()),
        "folds": rows,
        "mean_base_dti": mean_base,
        "mean_union_dti": mean_union,
        "mean_delta": mean_delta,
        "holdout_best": HOLDOUT_BEST,
        "holdout_null": HOLDOUT_NULL,
        "gate_cleared": bool(gate),
        "prediction_held": bool(0.13 <= mean_union <= 0.17 and mean_delta > 0),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    if gate and publish:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        build = Path("/tmp/gems_build")
        build.mkdir(exist_ok=True)
        tmp = build / "union_candidate.tif"
        field = union.astype(np.float32)
        rep = io.write_submission(tmp, field, sample, all_finite=True)
        if not rep["ok"]:
            report["published"] = None
            report["publish_error"] = rep["errors"]
        else:
            out_dir = REPO / "docs" / "downloads"
            out_dir.mkdir(parents=True, exist_ok=True)
            name = f"17GEMSDOE_C-union-spread_{rep['sha256'][:8]}_{stamp}"
            tif = out_dir / f"{name}.tif"
            shutil.copy2(tmp, tif)
            rep2 = io.validate_submission(tif, sample)
            rep2["sha256"] = io.sha256_file(tif)
            if not rep2["ok"]:
                tif.unlink()
                report["published"] = None
                report["publish_error"] = rep2["errors"]
            else:
                comment = (f"C-union-spread: 0.1563 support + {int(filler.sum()):,} "
                           f"lattice filler px (holdout {mean_union:.4f} vs "
                           f"base {mean_base:.4f}), {rep2['sha256'][:8]}")
                sidecar = {
                    "submission_name": name,
                    "comment": comment,
                    "support_equals_anchor": False,
                    "base_support_sha256": ANCHOR_SHA256,
                    "filler_px": int(filler.sum()),
                    "union_px": int(union.sum()),
                    "sha256": rep2["sha256"],
                    "holdout_mean_union_dti": mean_union,
                    "holdout_mean_base_dti": mean_base,
                    "holdout_best": HOLDOUT_BEST,
                    "gamble": ("The holdout's truth is the supplied catalogue. "
                               "On the leaderboard the truth is faults absent "
                               "from all catalogues: this candidate wins if "
                               "hidden faults are at least catalogue-sparse-"
                               "spread across the map and loses score if they "
                               "are much sparser. The safe download "
                               "(A-verified rebuild) reproduces 0.1563."),
                }
                (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=1))
                zip_path = out_dir / f"{name}.zip"
                shutil.make_archive(str(zip_path.with_suffix("")), "zip",
                                    root_dir=out_dir, base_dir=f"{name}.tif")
                report["published"] = {"name": name, "sha256": rep2["sha256"],
                                       "comment": comment}
                print(f"\nPUBLISHED {tif.name}\n  comment: {comment}")
    elif publish:
        print(f"\nGATE NOT CLEARED (union {mean_union:.4f} <= best {HOLDOUT_BEST}); "
              "nothing published.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"wrote {out_path}  ({report['runtime_seconds']}s)")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--budget", type=int, default=60_000)
    ap.add_argument("--out", default=str(REPO / "evidence" / "union_candidate.json"))
    ap.add_argument("--no-publish", action="store_true")
    args = ap.parse_args()
    run(Path(args.data_dir), args.budget, Path(args.out), publish=not args.no_publish)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
