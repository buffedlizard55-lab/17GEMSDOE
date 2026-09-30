#!/usr/bin/env python3
"""H6 - margin-optimal budget: is the 0.1563 field over-emitting?

Pre-registered hypothesis (recorded before running)
---------------------------------------------------
The verified algebra gives a break-even marginal hit rate

    tau(DTI) = 0.2*DTI / (1 - 0.2*DTI)          tau(0.1563) = 0.03227.

The reference field emits 172,974 pixels and its measured marginal hit rate on
the hide-and-recover holdout is 0.0197 -- **below** tau.  If that is right, then
the field's *last* pixels cost more than they earn and a strictly ascending
re-ranking of the same pixels, truncated at the optimum, must raise the score.
This is a statement about the metric, so it is falsifiable in one run.

Ranking must be label-free, otherwise the test is circular.  We rank the
reference support by the amplitude-invariant tilt-angle edge score (H3), which
uses only the supplied magnetic bands -- never the labels -- and sweep the
emitted fraction.  Prediction: the DTI-vs-budget curve peaks below 100 % of the
reference support; predicted best fraction 40-80 %, predicted gain +0.005 to
+0.03 over the full-support value, with the usual caveat that the sign of the
holdout delta need not survive to the leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems import edges  # noqa: E402
from gems.emit import top_k_mask  # noqa: E402
from gems.holdout import Holdout  # noqa: E402
from gems.metric import break_even_marginal_tp  # noqa: E402

FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "h6_budget.json"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=3)
    args = ap.parse_args()

    data = Path(args.data_dir)
    with rasterio.open(data / "labels.tif") as ds:
        cat = ds.read(1) == 1
    with rasterio.open(data / "sample_submission.tif") as ds:
        footprint = np.isfinite(ds.read(1))
    with rasterio.open(data / "reference" / "reference_01563.tif") as ds:
        ref = np.nan_to_num(ds.read(1).astype(np.float64), nan=0.0) > 0

    band_index = {}
    with rasterio.open(data / "training_features.tif") as ds:
        for i in range(1, ds.count + 1):
            band_index[ds.tags(i).get("band_name", f"band{i}")] = i
        tc = ds.read(band_index["tc"]).astype(np.float32)
        tmi = ds.read(band_index["tmi"]).astype(np.float32)
        rtp = ds.read(band_index["rtp"]).astype(np.float32)

    score = edges.edge_score_field(tc, tmi=tmi, rtp=rtp)
    score = np.where(footprint, score, np.nan)

    ho = Holdout.build(cat, n_folds=args.folds, buffer_px=args.buffer, seed=0)
    results = {"hypothesis": "H6 margin-optimal budget on the reference support, "
                             "ranked by the label-free tilt-angle edge score",
               "break_even_at_0.1563": float(break_even_marginal_tp(0.1563)),
               "curve": []}

    for frac in FRACTIONS:
        rows = []
        for f in ho.folds:
            eligible = footprint & ~f["remaining"] & ref
            n_avail = int(eligible.sum())
            k = int(round(frac * n_avail))
            mask = top_k_mask(score, k, eligible=eligible)
            pred = np.where(mask, 1.0, 0.0)
            pred[~footprint] = 0.0
            r = ho.evaluate(pred, f["fold"], mask_mode="pixel_exact")
            rows.append(r)
        sc = np.array([r["DTI"] for r in rows])
        mh = float(np.mean([r["marginal_hit_rate"] for r in rows]))
        px = float(np.mean([r["emitted_unmasked_px"] for r in rows]))
        results["curve"].append({"fraction": frac, "mean_DTI": float(sc.mean()),
                                 "std_DTI": float(sc.std(ddof=0)),
                                 "mean_emitted_px": px, "mean_marginal_hit_rate": mh,
                                 "beats_break_even": bool(mh > break_even_marginal_tp(0.1563)),
                                 "rows": rows})
        print(f"  frac={frac:4.2f}  DTI={sc.mean():.4f} +- {sc.std(ddof=0):.4f}  "
              f"px={px:>9.0f}  marginal={mh:.4f}  "
              f"{'BEATS tau' if mh > break_even_marginal_tp(0.1563) else ''}")

    curve = results["curve"]
    best = max(curve, key=lambda c: c["mean_DTI"])
    full = next(c for c in curve if c["fraction"] == 1.00)
    results["best"] = {"fraction": best["fraction"], "mean_DTI": best["mean_DTI"]}
    results["full_support"] = {"fraction": 1.0, "mean_DTI": full["mean_DTI"]}
    results["delta_best_minus_full"] = best["mean_DTI"] - full["mean_DTI"]
    results["verdict"] = (
        f"best fraction {best['fraction']:.2f} at DTI {best['mean_DTI']:.4f} vs "
        f"full-support {full['mean_DTI']:.4f} -> delta {results['delta_best_minus_full']:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("\n" + results["verdict"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
