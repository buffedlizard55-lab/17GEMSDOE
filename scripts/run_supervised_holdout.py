#!/usr/bin/env python3
"""H-SUP: supervised catalogue-expression model, hide-and-recover tested.

Why (redirected here by three closed lines of work)
--------------------------------------------------
1. Emission shaping is closed: H6 truncation is monotone to full support, the
   family's nms3/dotted variants moved the leaderboard DOWN, and the union
   filler's marginal precision (0.016-0.023) is below the 0.1563 break-even
   (0.024-0.032) in every fold (evidence/union_candidate.json).
2. The off-catalogue surrogate screen found every rule-based detector below
   uniform random (evidence/xcat_surrogate.json).
3. The metric algebra says the score is 5x the marginal hit rate; the 0.3049
   leader has h ~ 6.5 % vs our 3.2 %. Only DETECTION skill closes that gap.

This is the organizer reference solution's approach (a model trained to
recognize catalogue fault expression), which is absent from this repository:
everything here so far is rule-based.

Leakage control (the part that decides whether the number means anything)
------------------------------------------------------------------------
* No location or distance-to-catalogue features. The model sees only
  geophysical/topographic expression per pixel. A distance feature would
  memorize the catalogue and recover nothing hidden.
* Training positives are VISIBLE catalogue pixels only; the withheld segments
  and their 3-px blind corridor contribute neither positives nor negatives.
* At inference the corridor's features are NaN -- the detector cannot see the
  hidden zone's expression either (HistGradientBoosting handles NaN natively).
* Scoring is the frozen protocol: pixel-exact masking of the remaining
  catalogue, matched budgets (sparse = fold truth px; dense = 3.3 % of valid).

Pre-registered predictions (written before the run)
--------------------------------------------------
1. Mean fold dense DTI >= 0.1253 (the current holdout best, held by the leaky
   reference field). Direction: up. Rough size 0.13-0.20.
2. On the off-catalogue surrogate the model's dense DTI exceeds the reference
   support's 0.1854 -- expression transfer, not catalogue memorisation.
Failure redirects to external data (radiometrics/lidar) with this line closed.

Output: evidence/supervised_holdout.json
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

SEED = 17
DENSE_FRACTION = 0.033
N_POS_CAP = 80_000
N_NEG = 750_000
HOLDOUT_BEST = 0.1253

BANDS = ["tc", "tmi_hg", "tmi_vg", "mag_anom", "rtp", "iso_grav_anom_hg",
         "iso_grav_anom_slope", "det_elev_slope", "det_elev", "geod_2ndinv",
         "geod_shearrate", "geod_dilaterate", "cond_surf", "depth_to_base_surf"]


def load_features(feats_path, blank_zone=None):
    """(N_bands+2, H, W) float32 feature stack: 14 bands + 2 gradient magnitudes.

    Memory is the binding constraint (3 GB machine): the stack is 16 x 49 MB
    float32, loaded once and reused for every fold's sampling and inference.
    blank_zone cells are set to NaN -- the detector may not see the blind
    corridor's expression.
    """
    from scipy import ndimage
    F = len(BANDS) + 2
    probe = data.load_band(feats_path, BANDS[0])
    H, W = probe.shape
    del probe
    # Preallocate and fill in place: np.stack of a 16-item list would hold two
    # full copies (~1.6 GB) on a 3 GB machine.
    out = np.empty((F, H, W), np.float32)
    for i, name in enumerate(BANDS):
        a = data.load_band(feats_path, name).astype(np.float32)
        if blank_zone is not None:
            a[blank_zone] = np.nan
        out[i] = a
        del a
        print(f"  loaded {name}")
    for j, (name, sigma) in enumerate((("tc", 2.0), ("det_elev", 2.0))):
        src = out[BANDS.index(name)]
        filled = np.where(np.isfinite(src), src, 0.0)
        g = ndimage.gaussian_gradient_magnitude(filled, sigma=sigma).astype(np.float32)
        del filled
        if blank_zone is not None:
            g[blank_zone] = np.nan
        out[len(BANDS) + j] = g
        del g
        print(f"  computed gradmag({name}, {sigma})")
    return out   # (F, H, W)


def sample_rows(X, pos_mask, neg_pool_mask, rng):
    """Balanced-ish sample of (features, target) rows."""
    pos = np.flatnonzero(pos_mask.ravel())
    neg = np.flatnonzero(neg_pool_mask.ravel())
    if len(pos) > N_POS_CAP:
        pos = rng.choice(pos, N_POS_CAP, replace=False)
    neg = rng.choice(neg, min(N_NEG, len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)])
    F = X.shape[0]
    rows = np.empty((len(idx), F), np.float32)
    for f in range(F):
        rows[:, f] = X[f].ravel()[idx]
    del idx
    return rows, y


def predict_field(X, model, chunk_rows=250_000):
    H, W = X.shape[1], X.shape[2]
    out = np.empty(H * W, np.float32)
    F = X.shape[0]
    flat = X.reshape(F, -1)
    for start in range(0, H * W, chunk_rows):
        stop = min(start + chunk_rows, H * W)
        rows = np.empty((stop - start, F), np.float32)
        for f in range(F):
            rows[:, f] = flat[f, start:stop]
        out[start:stop] = model.predict_proba(rows)[:, 1]
        del rows
    return out.reshape(H, W)


def train_model(X, pos_mask, neg_pool_mask, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, y = sample_rows(X, pos_mask, neg_pool_mask, rng)
    print(f"  training on {len(y):,} rows ({int(y.sum()):,} positive) ...")
    clf = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_depth=6,
        early_stopping=True, validation_fraction=0.1,
        random_state=SEED, verbose=0)
    t0 = time.time()
    clf.fit(rows, y)
    print(f"  trained in {time.time() - t0:.0f}s")
    return clf


def run(data_dir: Path, out_path: Path, max_folds: int | None,
        skip_surrogate: bool) -> dict:
    t0 = time.time()
    feats = data_dir / "training_features.tif"
    labels = data.load_labels(data_dir / "labels.tif")
    valid = data.valid_mask(feats)
    if not data.verify_band_layout(feats):
        raise SystemExit("band layout does not match the verified layout")

    rng = np.random.default_rng(SEED)
    ho = Holdout(labels)

    # One stack per fold (corridor NaN), reused for sampling and inference.
    report = {
        "hypothesis": ("H-SUP: catalogue-expression model recovers withheld "
                       "segments from geophysics alone (no location features)"),
        "preregistered_predictions": {
            "p1": "mean fold dense DTI >= 0.1253 (current holdout best)",
            "direction": "up",
            "rough_size": "0.13-0.20 dense",
            "p2": "surrogate dense DTI > reference support's 0.1854",
        },
        "features": BANDS + ["gradmag(tc,2)", "gradmag(det_elev,2)"],
        "n_features": len(BANDS) + 2,
        "leakage_controls": [
            "no location/distance features",
            "training positives = visible catalogue only",
            "corridor features NaN at training and inference",
            "frozen scoring protocol with pixel-exact catalogue masking",
        ],
        "folds": [],
    }

    n_folds = len(ho.folds) if max_folds is None else min(max_folds, len(ho.folds))
    for fi in range(n_folds):
        fold = ho.folds[fi]
        print(f"\n=== fold {fi}: {fold.n_components} withheld segments, "
              f"{fold.truth_px:,} truth px ===")
        print("loading features with corridor blinded ...")
        X = load_features(feats, blank_zone=fold.blinded)

        pos = fold.visible.copy()                 # visible catalogue: positives
        neg_pool = valid & ~labels & ~fold.blinded
        clf = train_model(X, pos, neg_pool, rng)

        print("  inferring ...")
        field = predict_field(X, clf)
        del X
        gc.collect()

        dense = int(round(DENSE_FRACTION * int(valid.sum())))
        sparse = int(fold.truth_px)
        row = {"fold": fi, "truth_px": fold.truth_px}
        for regime, bud in (("sparse_matched_truth_px", sparse),
                            ("dense_competition_like", dense)):
            sel = metric.top_k_mask(field, bud, eligible=valid)
            res = metric.dti(sel.astype(np.float32), fold.truth, mask=fold.masked)
            row[regime] = res.as_dict()
            print(f"  {regime}: DTI={res.dti:.4f} emitted={res.emitted_px:,} "
                  f"marginal={res.tp_per_emitted_px:.4f} break_even={res.break_even:.4f}")
        report["folds"].append(row)

    mean_dense = float(np.mean([r["dense_competition_like"]["dti"]
                                for r in report["folds"]]))
    mean_sparse = float(np.mean([r["sparse_matched_truth_px"]["dti"]
                                 for r in report["folds"]]))
    report["mean_dense_DTI"] = mean_dense
    report["mean_sparse_DTI"] = mean_sparse
    report["holdout_best"] = HOLDOUT_BEST
    report["gate_cleared"] = bool(mean_dense > HOLDOUT_BEST)

    if not skip_surrogate:
        proxy_path = REPO / "data" / "external" / "proxy_catalogue.tif"
        if proxy_path.exists():
            from scipy import ndimage
            import rasterio
            print("\n=== surrogate arm: train on full catalogue ===")
            X = load_features(feats, blank_zone=None)
            neg_pool = valid & ~labels
            clf = train_model(X, labels, neg_pool, rng)
            field = predict_field(X, clf)
            del X
            gc.collect()
            with rasterio.open(proxy_path) as ds:
                proxy = ds.read(1)
            halo = ndimage.maximum_filter(labels, footprint=np.ones((7, 7), bool))
            truth = (proxy == 2) & ~halo & valid
            elig = valid & ~halo
            dense = int(round(DENSE_FRACTION * int(elig.sum())))
            sel = metric.top_k_mask(field, dense, eligible=elig)
            res = metric.dti(sel.astype(np.float32), truth, mask=halo)
            report["surrogate"] = {"dense_DTI": res.dti, "emitted_px": res.emitted_px,
                                   "tp_per_emitted_px": res.tp_per_emitted_px,
                                   "reference_support_dense_DTI": 0.1854,
                                   "null_dense_DTI": 0.3719}
            print(f"  surrogate dense DTI={res.dti:.4f}")

    report["predictions_held"] = {
        "p1_holdout_best": bool(mean_dense > HOLDOUT_BEST),
        "p2_surrogate_beats_reference": bool(
            report.get("surrogate", {}).get("dense_DTI", 0) > 0.1854),
    }
    report["decision"] = {
        "verdict": (
            "Supervised expression model clears the holdout gate; it is the "
            "first locally validated candidate and earns union-candidate "
            "status under the standing brief."
            if mean_dense > HOLDOUT_BEST else
            "Supervised expression model did not clear the holdout gate. "
            "Documented negative result: the 14 delivered bands at 100 m carry "
            "no learnable expression of hidden faults beyond what uniform "
            "coverage provides. The remaining path is external data "
            "(radiometrics, 1 m lidar) or a fundamentally new feature."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}  ({report['decision']['runtime_seconds']}s)")
    print("predictions held:", report["predictions_held"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "supervised_holdout.json"))
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--skip-surrogate", action="store_true")
    args = ap.parse_args()
    run(Path(args.data_dir), Path(args.out), args.max_folds, args.skip_surrogate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
