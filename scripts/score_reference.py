#!/usr/bin/env python3
"""Fetch and score the byte-verified 0.1563 reference field under our holdout.

This measures the BAR.  Running hypotheses against an assumed bar is how a team
talks itself into a submission; running them against a measured bar is how it
decides.  The reference is the exact raster that scored 0.1563 on the public
leaderboard (sha256 7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15,
570890 bytes, 172974 positive pixels), re-hashed here before use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems.holdout import Holdout  # noqa: E402
from gems.io import sha256_file  # noqa: E402
from gems.metric import break_even_marginal_tp  # noqa: E402

REFERENCE = {
    "repo": "buffedlizard55-lab/5GEMSDOE",
    "path": "data/evidence/leaderboard_anchor/gemsdoe-ens12-adopted-7f00890a.tif",
    "sha256": "7f00890a62878d612fb5eef67a9a364a2df819433dde74b6762ce4fc0fc4fe15",
    "bytes": 570890,
    "positive_px": 172974,
    "public_score": 0.1563,
    "family": ("CNN ensemble (11 ResNet34 U-Net/UNet++/DeepLabV3+ folds) -> "
               "floor 0.1 -> distance-R thinning -> binary"),
}


def fetch_reference(dest: Path) -> Path:
    if dest.exists() and sha256_file(dest) == REFERENCE["sha256"]:
        print(f"reference present and hash-verified: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching reference from {REFERENCE['repo']}:{REFERENCE['path']}")
    cmd = ["gh", "api",
           f"repos/{REFERENCE['repo']}/contents/{REFERENCE['path']}?ref=HEAD",
           "-H", "Accept: application/vnd.github.raw"]
    with open(dest, "wb") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE)
    if r.returncode:
        raise SystemExit(f"gh api failed: {r.stderr.decode()[:300]}")
    got = sha256_file(dest)
    if got != REFERENCE["sha256"]:
        raise SystemExit(f"reference hash mismatch\n got {got}\n want {REFERENCE['sha256']}")
    print(f"reference hash verified: {got}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "reference_holdout.json"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data_dir)
    ref_path = fetch_reference(data / "reference" / "reference_01563.tif")

    with rasterio.open(data / "labels.tif") as ds:
        cat = ds.read(1) == 1
    with rasterio.open(data / "sample_submission.tif") as ds:
        footprint = np.isfinite(ds.read(1))
    with rasterio.open(ref_path) as ds:
        ref = np.nan_to_num(ds.read(1).astype(np.float64), nan=0.0)

    pos = ref > 0
    print(f"reference positive px = {int(pos.sum())} (expected {REFERENCE['positive_px']})")
    assert int(pos.sum()) == REFERENCE["positive_px"], "reference field is not the known 0.1563 raster"

    ho = Holdout.build(cat, n_folds=args.folds, buffer_px=args.buffer, seed=0)
    rows = []
    for f in ho.folds:
        pred = ref.copy()
        pred[~footprint] = 0.0
        r = ho.evaluate(pred, f["fold"], mask_mode="pixel_exact")
        rows.append(r)
        print(f"  fold {f['fold']}: DTI={r['DTI']:.4f} emitted={r['emitted_unmasked_px']:>7d} "
              f"TP_w={r['TP_w']:>9.1f} marginal={r['marginal_hit_rate']:.4f} "
              f"recall_w={r['recall_w']:.4f}")

    sc = np.array([r["DTI"] for r in rows])
    out = {
        "reference": REFERENCE,
        "protocol": {"folds": args.folds, "buffer_px": args.buffer,
                     "mask_mode": "pixel_exact (mirrors organizers)"},
        "rows": rows,
        "mean_DTI": float(sc.mean()), "std_DTI": float(sc.std(ddof=0)),
        "mean_marginal_hit_rate": float(np.mean([r["marginal_hit_rate"] for r in rows])),
        "break_even_marginal_hit_rate": float(break_even_marginal_tp(0.1563)),
        "reading": ("This is what a public score of 0.1563 looks like on our "
                    "hide-and-recover holdout. Any candidate must beat this "
                    "number, on the same folds, before it may take a slot."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nREFERENCE mean holdout DTI = {sc.mean():.4f} "
          f"(std {sc.std(ddof=0):.4f})  mean marginal = {out['mean_marginal_hit_rate']:.4f} "
          f"vs tau={out['break_even_marginal_hit_rate']:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
