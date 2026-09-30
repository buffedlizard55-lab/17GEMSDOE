#!/usr/bin/env python3
"""Train a new model with multi-scale edge features and emit a probability submission.

Key differences from the previous D-supervised approach:
1. Adds gradient magnitude and |Laplacian| features on edge-sensitive bands
2. Emits soft probabilities (not binary 0/1) -- the metric uses p(x) directly
3. Matches the sample submission NaN footprint exactly
4. Uses dilated fault labels (3px buffer) as positive training examples

Memory-efficient: loads all 19 bands once (~933MB), computes derived features
on the fly, predicts in row chunks.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
sys.path.insert(0, str(REPO / "src"))

# Edge-sensitive bands (0-based) to compute gradient/laplacian features for
EDGE_BANDS = [0, 1, 2, 5, 8, 11, 12, 13, 17, 18]  # 10 bands
SIGMAS = (1.0, 2.0)
# Total derived: 10 bands * 2 sigmas * 2 ops (gradmag, |lap|) = 40
# Total features: 19 + 40 = 59


def load_all_bands():
    """Load all 19 bands as (19, H, W) float32 array."""
    with rasterio.open(DATA / "training_features.tif") as ds:
        feats = ds.read().astype(np.float32)
    feats[feats < -1e38] = np.nan
    return feats  # (19, H, W)


def grad_mag_2d(band_2d, sigma):
    """Gradient magnitude of a 2D array."""
    b = np.nan_to_num(band_2d, nan=0.0, posinf=0.0, neginf=0.0)
    gx = ndimage.gaussian_filter(b, sigma, order=(0, 1))
    gy = ndimage.gaussian_filter(b, sigma, order=(1, 0))
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def abs_lap_2d(band_2d, sigma):
    """Absolute Laplacian of a 2D array."""
    b = np.nan_to_num(band_2d, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(ndimage.gaussian_laplace(b, sigma=sigma)).astype(np.float32)


def main():
    from gems.io import validate_submission, sha256_file
    
    t_start = time.time()
    
    print("=== Loading data ===")
    feats = load_all_bands()  # (19, H, W)
    H, W = feats.shape[1:]
    print(f"  Features: {feats.shape}, {feats.nbytes/1e6:.0f}MB")

    with rasterio.open(DATA / "labels.tif") as ds:
        labels = ds.read(1)
    fault = labels == 1
    print(f"  Fault pixels: {fault.sum():,}")

    with rasterio.open(DATA / "sample_submission.tif") as ds:
        sample = ds.read(1)
        sample_profile = ds.profile.copy()
    valid = np.isfinite(sample)
    print(f"  Valid footprint: {valid.sum():,}")

    # Dilated fault labels
    print("  Dilated fault labels (3px buffer)...")
    dilated = ndimage.maximum_filter(fault, size=7)
    print(f"  Dilated: {dilated.sum():,}")

    # Feature validity mask
    feat_valid = np.all(np.isfinite(feats), axis=0) & valid
    print(f"  Feature-valid: {feat_valid.sum():,}")

    n_derived = len(EDGE_BANDS) * len(SIGMAS) * 2
    n_total = 19 + n_derived
    print(f"  Features: 19 original + {n_derived} derived = {n_total}")

    # Sample training pixels
    print("\n=== Sampling training pixels ===")
    rng = np.random.default_rng(42)

    pos_coords = np.argwhere(dilated & feat_valid)
    n_pos = len(pos_coords)
    print(f"  Positive (dilated): {n_pos:,}")

    neg_mask = ~dilated & feat_valid
    neg_all = np.argwhere(neg_mask)
    n_neg = min(n_pos * 2, len(neg_all))
    neg_idx = rng.choice(len(neg_all), n_neg, replace=False)
    neg_coords = neg_all[neg_idx]
    del neg_all, neg_mask
    gc.collect()
    print(f"  Negative (sampled): {n_neg:,}")

    all_coords = np.concatenate([pos_coords, neg_coords])
    del pos_coords, neg_coords
    n_samples = len(all_coords)

    # Build feature matrix
    print("\n=== Building feature matrix ===")
    X = np.zeros((n_samples, n_total), dtype=np.float32)
    
    # Original 19 bands
    rows, cols = all_coords[:, 0], all_coords[:, 1]
    for bi in range(19):
        X[:, bi] = feats[bi, rows, cols]
    
    # Derived features: compute per edge band
    feat_idx = 19
    for bi in EDGE_BANDS:
        band = feats[bi]  # (H, W)
        for s in SIGMAS:
            gm = grad_mag_2d(band, s)
            X[:, feat_idx] = gm[rows, cols]
            feat_idx += 1
            del gm
        for s in SIGMAS:
            lap = abs_lap_2d(band, s)
            X[:, feat_idx] = lap[rows, cols]
            feat_idx += 1
            del lap
        print(f"  Edge band {bi+1}: done")
    gc.collect()

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.concatenate([np.ones(n_pos, dtype=np.float32),
                        np.zeros(n_neg, dtype=np.float32)])
    del all_coords, rows, cols
    print(f"  X: {X.shape}, {X.nbytes/1e6:.1f}MB")

    # Train
    print("\n=== Training ===")
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=0.5,
        max_bins=255,
        random_state=42,
        verbose=1,
    )
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"  Training accuracy: {train_acc:.4f}")
    del X, y
    gc.collect()

    # Predict in chunks
    print("\n=== Predicting (chunked) ===")
    proba = np.full((H, W), np.nan, dtype=np.float32)
    
    # Only predict in valid area
    chunk_size = 300
    for r0 in range(0, H, chunk_size):
        r1 = min(r0 + chunk_size, H)
        chunk_h = r1 - r0
        
        # Check if any valid pixels in this chunk
        chunk_valid = valid[r0:r1]
        if not chunk_valid.any():
            continue
        
        # Build feature matrix for this chunk
        n_pix = chunk_h * W
        X_c = np.zeros((n_pix, n_total), dtype=np.float32)
        
        # Original bands
        for bi in range(19):
            X_c[:, bi] = feats[bi, r0:r1].ravel()
        
        # Derived
        feat_idx = 19
        for bi in EDGE_BANDS:
            band_c = feats[bi, r0:r1]
            for s in SIGMAS:
                gm = grad_mag_2d(band_c, s)
                X_c[:, feat_idx] = gm.ravel()
                feat_idx += 1
                del gm
            for s in SIGMAS:
                lap = abs_lap_2d(band_c, s)
                X_c[:, feat_idx] = lap.ravel()
                feat_idx += 1
                del lap
            del band_c
        
        X_c = np.nan_to_num(X_c, nan=0.0, posinf=0.0, neginf=0.0)
        p = clf.predict_proba(X_c)[:, 1].reshape(chunk_h, W)
        del X_c
        
        # Only write valid pixels
        chunk_mask = valid[r0:r1]
        proba[r0:r1] = np.where(chunk_mask, p, np.nan)
        del p
        gc.collect()
        
        if r0 % 600 == 0:
            elapsed = time.time() - t_start
            print(f"  Row {r0}/{H} ({elapsed:.0f}s)")

    del feats
    gc.collect()

    # Statistics
    v = proba[valid]
    print(f"\n=== Probability field ===")
    print(f"  Range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"  Mean: {v.mean():.4f}")
    print(f"  >0.5: {(v > 0.5).sum():,}")
    print(f"  >0.1: {(v > 0.1).sum():,}")
    print(f"  >0.01: {(v > 0.01).sum():,}")

    # Save
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"17GEMSDOE_E-proba-multiscale_{stamp}"
    
    out_dir = REPO / "docs" / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"{name}.tif"

    profile = sample_profile.copy()
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=float("nan"),
                   compress="lzw")

    with rasterio.open(tif_path, "w", **profile) as ds:
        ds.write(proba, 1)

    # Validate
    rep = validate_submission(tif_path, DATA / "sample_submission.tif")
    print(f"\n=== Validation ===")
    for k, v in rep["checks"].items():
        status = "PASS" if v["ok"] else "FAIL"
        print(f"  [{status}] {k}: {v['detail']}")

    if not rep["ok"]:
        print("FAILED format gate!")
        return 1

    sha = sha256_file(tif_path)

    # Zip
    zip_path = out_dir / f"{name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip",
                        root_dir=out_dir, base_dir=f"{name}.tif")

    # Sidecar
    comment = (f"E-proba-multiscale: HistGB {n_total}feats, dilated-labels(3px), "
               f"soft-proba, acc={train_acc:.4f}, {sha[:8]}")
    
    sidecar = {
        "submission_name": name,
        "comment": comment,
        "sha256": sha,
        "bytes": int(tif_path.stat().st_size),
        "positive_pixels_gt50": int((proba[valid] > 0.5).sum()),
        "positive_pixels_gt10": int((proba[valid] > 0.1).sum()),
        "value_range": [float(rep["min"]), float(rep["max"])],
        "training_accuracy": train_acc,
        "n_features": n_total,
        "key_differences": [
            "soft probability output",
            "multi-scale gradient + |Laplacian| features",
            "dilated labels (3px buffer) matching 300m kernel",
            "NaN outside footprint matching sample",
        ],
        "built_utc": stamp,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=1))

    meta_path = out_dir / "submission_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["downloads"] = meta.get("downloads", {})
    meta["downloads"]["E_tif"] = tif_path.name
    meta["downloads"]["E_zip"] = zip_path.name
    meta["E_comment"] = comment
    meta["E_sha256"] = sha
    meta_path.write_text(json.dumps(meta, indent=1))

    elapsed = time.time() - t_start
    print(f"\n=== Published in {elapsed:.0f}s ===")
    print(f"  {tif_path.name}")
    print(f"  SHA: {sha}")
    print(f"  Size: {tif_path.stat().st_size:,} bytes")
    print(f"  Comment: {comment}")
    print(f"  Form name: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
