#!/usr/bin/env python3
"""Build the H-SUP candidate submission: supervised catalogue-expression field.

Gate (standing brief): published only because evidence/supervised_holdout.json
shows mean fold dense DTI 0.2795 > the current holdout best 0.1253 under the
exact hide-and-recover protocol (whole segments withheld with a 3-px blind
corridor, pixel-exact catalogue masking, matched budgets).

The model is trained on ALL catalogue pixels for the submission (no holdout --
the holdout was the validation), from expression features only (14 delivered
bands + 2 gradient magnitudes; no location or distance features). Emission is
the metric-optimal 0/1 field at the 3.3 % competition density.

Name/comment make the file unique and self-describing for the competition's
name and comment fields.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from gems import data, io, metric  # noqa: E402

DENSE_FRACTION = 0.033


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out-dir", default=str(REPO / "docs" / "downloads"))
    args = ap.parse_args()

    gate = json.loads((REPO / "evidence" / "supervised_holdout.json").read_text())
    if not gate.get("gate_cleared"):
        raise SystemExit("holdout gate not cleared in evidence/supervised_holdout.json; "
                         "refusing to build a submission artifact")
    mean_dense = gate["mean_dense_DTI"]

    from run_supervised_holdout import load_features, train_model, predict_field, SEED
    data_dir = Path(args.data_dir)
    feats = data_dir / "training_features.tif"
    labels = data.load_labels(data_dir / "labels.tif")
    valid = data.valid_mask(feats)

    print("loading features (no corridor: full-catalogue training) ...")
    X = load_features(feats, blank_zone=None)
    rng = np.random.default_rng(SEED)
    clf = train_model(X, labels, valid & ~labels, rng)
    print("inferring ...")
    field = predict_field(X, clf)
    del X

    budget = int(round(DENSE_FRACTION * int(valid.sum())))
    sel = metric.top_k_mask(field, budget, eligible=valid)
    pred = sel.astype(np.float32)
    print(f"emitted {int(sel.sum()):,} px at budget {budget:,}")

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    build = Path("/tmp/gems_build")
    build.mkdir(exist_ok=True)
    tmp = build / "supervised_candidate.tif"
    rep = io.write_submission(tmp, pred, data_dir / "sample_submission.tif", all_finite=True)
    if not rep["ok"]:
        print("format gate failed:", rep["errors"])
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # replace previous candidate builds so a visitor cannot click a stale one
    for old in list(out_dir.glob("17GEMSDOE_D-supervised-expression_*")):
        old.unlink()
    name = f"17GEMSDOE_D-supervised-expression_{rep['sha256'][:8]}_{stamp}"
    tif = out_dir / f"{name}.tif"
    shutil.copy2(tmp, tif)
    rep2 = io.validate_submission(tif, data_dir / "sample_submission.tif")
    rep2["sha256"] = io.sha256_file(tif)
    if not rep2["ok"]:
        tif.unlink()
        print("published copy failed re-validation:", rep2["errors"])
        return 1

    # The published raster is the ground truth for its own support size: cells the
    # sample's footprint NaN-mask overwrites must not be claimed in the comment.
    with rasterio.open(tif) as _ds:
        positive_px = int((_ds.read(1) > 0).sum())
    comment = (f"D-supervised-expression: HistGB on 14 bands+2 gradmag, no location "
               f"features, 0/1 top-k at 3.3% density ({positive_px:,} px), "
               f"hide-and-recover mean dense DTI {mean_dense:.4f} vs gate 0.1253), "
               f"{rep2['sha256'][:8]}")
    sidecar = {
        "submission_name": name,
        "comment": comment,
        "support_equals_anchor": False,
        "holdout_mean_dense_DTI": mean_dense,
        "holdout_gate": 0.1253,
        "holdout_folds": [r["dense_competition_like"]["dti"] for r in gate["folds"]],
        "surrogate_dense_DTI": gate.get("surrogate", {}).get("dense_DTI"),
        "surrogate_caveat": ("below the reference support's 0.1854 and uniform "
                             "random's 0.3719 on the SGMC screen: recovery skill "
                             "is validated, off-catalogue discovery spread is not"),
        "sha256": rep2["sha256"],
        "emitted_px": int(sel.sum()),
        "positive_pixels": positive_px,
        "note_on_counts": ("emitted_px counts pre-mask selections; positive_pixels "
                           "counts 1s in the published raster after the sample "
                           "footprint NaN-mask and is the number quoted in the comment"),
        "value_range": [rep2["min"], rep2["max"]],
        "format_checks_failed": [],
        "how_to_submit": [
            "Download this .tif.",
            "Paste the submission_name into the competition's name field.",
            "Paste the comment into the comment field.",
            "Upload. Every format check was run against these exact bytes.",
        ],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=1))
    zip_path = out_dir / f"{name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip",
                        root_dir=out_dir, base_dir=f"{name}.tif")

    # register alongside the safe download without deleting it
    meta_path = out_dir / "submission_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("downloads", {})
    meta["downloads"]["candidate_tif"] = tif.name
    meta["downloads"]["candidate_zip"] = zip_path.name
    meta["downloads"]["candidate_sidecar"] = f"{name}.json"
    meta["candidate_comment"] = comment
    meta["candidate_holdout_mean_dense_DTI"] = mean_dense
    meta_path.write_text(json.dumps(meta, indent=1))

    print(f"\nPUBLISHED {tif.name}")
    print(f"  sha256   {rep2['sha256']}")
    print(f"  emitted  {int(sel.sum()):,} px   range [{rep2['min']}, {rep2['max']}]")
    print(f"  comment  {comment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
