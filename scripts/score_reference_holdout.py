#!/usr/bin/env python3
"""Score the field that already holds 0.1563 under the same frozen protocol.

Without this number the hypothesis results are uninterpretable: knowing that a
detector scored 0.0013 says nothing until the field that is actually on the
leaderboard has been measured on the same folds, at the same budgets, with the
same masking rule.

Two caveats that are printed with the result and must travel with it:

1.  **The reference field was trained on the whole catalogue, including the
    segments this protocol withholds.** Its score here is therefore an upper
    bound that leaks, not a fair estimate. It is reported because it bounds how
    much headroom the protocol has, not because it is a valid measurement of
    generalisation.
2.  **The holdout's truth is the catalogue**, while the leaderboard's truth is a
    set of faults that are in no catalogue. The two numbers are not on the same
    scale and the script never compares them as if they were.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems import data, metric  # noqa: E402
from gems.holdout import Holdout  # noqa: E402

ANCHOR = Path("/tmp/gems_build/anchor_submission.tif")
DENSE_FRACTION = 0.033


def load_support(path: Path, shape) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1)
    return np.isfinite(a) & (a > 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--anchor", default=str(ANCHOR))
    ap.add_argument("--out", default=str(REPO / "evidence" / "reference_holdout.json"))
    args = ap.parse_args()

    anchor = Path(args.anchor)
    if not anchor.exists():
        raise SystemExit(f"{anchor} not found; run scripts/build_submission.py first "
                         "(it fetches and hash-verifies the anchor)")

    t0 = time.time()
    import hashlib
    sha = hashlib.sha256(anchor.read_bytes()).hexdigest()

    labels = data.load_labels(Path(args.data_dir) / "labels.tif")
    valid = data.valid_mask(Path(args.data_dir) / "training_features.tif")
    support = load_support(anchor, labels.shape) & valid
    ho = Holdout(labels)

    print(f"anchor sha256 {sha}")
    print(f"anchor support {int(support.sum()):,} px on {int(valid.sum()):,} valid cells")

    rows = []
    for fi, fold in enumerate(ho.folds):
        dense = int(round(DENSE_FRACTION * int(valid[~fold.blinded].sum())))
        rec = {"fold": fi, "anchor_support_px": int(support.sum()),
               "fold_truth_px": fold.truth_px, "blinded_px": int(fold.blinded.sum())}
        # (a) the field exactly as published: its own support, no re-ranking
        rec["as_published"] = ho.evaluate(support.astype(np.float32), fi)
        # (b) re-ranked down to the fold's truth pixel count, ranked by distance
        #     to the visible catalogue -- the only ordering the file itself
        #     supports, since its values are all exactly 1.0
        from scipy import ndimage
        d = ndimage.distance_transform_edt(~fold.visible).astype(np.float32)
        rank = np.where(support, -d, -np.inf)
        m = metric.top_k_mask(np.where(np.isfinite(rank), rank, -np.inf),
                              int(fold.truth_px), eligible=valid)
        rec["sparse_matched_truth_px"] = ho.evaluate(m.astype(np.float32), fi)
        m2 = metric.top_k_mask(np.where(np.isfinite(rank), rank, -np.inf), dense,
                               eligible=valid)
        rec["dense_competition_like"] = ho.evaluate(m2.astype(np.float32), fi)
        rows.append(rec)
        print(f"  fold {fi}: as_published={rec['as_published']['dti']:.4f}  "
              f"sparse={rec['sparse_matched_truth_px']['dti']:.4f}  "
              f"dense={rec['dense_competition_like']['dti']:.4f}")
        del d, rank, m, m2
        gc.collect()

    hyp_path = REPO / "evidence" / "hypotheses.json"
    hyp = json.loads(hyp_path.read_text()) if hyp_path.exists() else {}
    null_dense = hyp.get("hypotheses", {}).get("NULL_random", {}).get(
        "dense_competition_like", {}).get("mean_DTI")
    null_sparse = hyp.get("hypotheses", {}).get("NULL_random", {}).get(
        "sparse_matched_truth_px", {}).get("mean_DTI")
    ref_dense = float(np.mean([r["dense_competition_like"]["dti"] for r in rows]))

    report = {
        "anchor_sha256": sha,
        "anchor_path": str(anchor),
        "anchor_support_px": int(support.sum()),
        "caveats": [
            "The reference field was trained on the whole catalogue, including the "
            "segments this protocol withholds: its score is an upper bound that "
            "leaks and is not an estimate of generalisation.",
            "The holdout truth is the supplied catalogue; the leaderboard's truth is "
            "faults absent from every catalogue. The numbers are on different scales "
            "and must never be compared directly.",
        ],
        "folds": rows,
        "summary": {
            "as_published_mean_DTI": float(np.mean([r["as_published"]["dti"] for r in rows])),
            "sparse_mean_DTI": float(np.mean([r["sparse_matched_truth_px"]["dti"] for r in rows])),
            "dense_mean_DTI": ref_dense,
            "null_dense_mean_DTI": null_dense,
            "null_sparse_mean_DTI": null_sparse,
            "reference_beats_null_dense": (None if null_dense is None
                                           else bool(ref_dense > null_dense)),
            "break_even_marginal_tp": float(metric.break_even_marginal_tp(ref_dense)),
            "mean_marginal_tp_per_emitted_px": float(np.mean(
                [r["dense_competition_like"]["tp_per_emitted_px"] for r in rows])),
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=1))
    s = report["summary"]
    print("\n=== reference field under the frozen protocol ===")
    print(f"  as published          {s['as_published_mean_DTI']:.4f}")
    print(f"  sparse matched budget {s['sparse_mean_DTI']:.4f}  (null {null_sparse})")
    print(f"  dense budget          {s['dense_mean_DTI']:.4f}  (null {null_dense})")
    print(f"  marginal precision    {s['mean_marginal_tp_per_emitted_px']:.4f} vs "
          f"break-even {s['break_even_marginal_tp']:.4f}")
    print(f"wrote {out}  ({report['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
