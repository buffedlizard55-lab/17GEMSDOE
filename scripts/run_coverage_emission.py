#!/usr/bin/env python3
"""H-COV: coverage-optimal emission at matched budget -- does spreading beat
concentration under this metric, and by how much?

Background (measured, see evidence/xcat_surrogate.json and this run)
--------------------------------------------------------------------
The metric credits each truth pixel's BEST prediction within R = 3 px and
charges every emitted pixel (1 - sigma(x)) of false-positive weight. Two
consequences were observed in the surrogate arm:

* equal-size uniform-random sets cover thin-line truth far better than the
  top-k of a concentrated score field (79.1 % vs 49.2 % of truth within R);
* a ranked field overshoots the truth's own value range (slope truth p50
  13.7 vs emitted p50 30.9).

The group's 12GEMSDOE "nms3" submission thinned the reference support and
scored 0.1294 < 0.1563 -- but thinning REDUCES the budget, and the metric likes
pixels. The controlled question this script answers holds the budget EXACTLY
fixed and changes only the spatial arrangement.

Pre-registered predictions (written before the run)
--------------------------------------------------
1. Spread-NULL (jittered lattice, min-distance 4 px) at the SAME budget as
   clustered-NULL raises dense DTI by >= 1.5x on the catalogue holdout and
   >= 1.2x on the surrogate arm. Direction: up. Rough size: catalogue holdout
   0.12 -> 0.18-0.25; surrogate 0.37 -> 0.40-0.50.
2. Spread applied to a physical field (H2, CTRL_slope) at the same budget also
   raises its DTI -- the coverage deficit, not the ranking, is the binding
   constraint.
3. If (1) holds but a spread physical field still loses to spread-NULL, the
   ranking carries no signal and the bottleneck stays detection.

Failure redirects: if spread does NOT raise DTI at fixed budget, the coverage
story is wrong and the emission line of work closes permanently.

Output: evidence/coverage_emission.json
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

from gems import data, detect, metric  # noqa: E402
from gems.holdout import Holdout  # noqa: E402

SEED = 17
DENSE_FRACTION = 0.033
# Acceptance suppresses a (2*MIN_DIST_PX+1)^2 box, so accepted pixels sit at
# Chebyshev separation >= MIN_DIST_PX+1 = 4 px > R = 3 px: no two emitted
# pixels can share one truth pixel's credit kernel. MIN_DIST_PX=4 (separation
# 5) packed only ~130k of a 170k budget -- the matched-budget comparison would
# have been a budget comparison in disguise.
MIN_DIST_PX = 3


def spread_select(score: np.ndarray, budget: int, eligible: np.ndarray,
                  min_dist_px: int = MIN_DIST_PX) -> np.ndarray:
    """Greedy non-maximum suppression: highest score first, suppress accepted
    pixels' neighbourhoods. O(n log n) sort plus one small mask update per
    accepted pixel; deterministic given the score.

    Eligibility is re-checked at acceptance. A previous revision trusted the
    score's -inf sentinels to keep the scan inside the eligible set; once
    suppression covers the grid the scan falls through into the -inf tail and
    would accept corridor pixels -- silently letting the emitter see through a
    hide-and-recover blind. The check is explicit so that failure cannot recur.
    """
    eligible = np.asarray(eligible, bool)
    score = np.where(eligible & np.isfinite(score), score, -np.inf)
    order = np.argsort(score, axis=None)[::-1]
    chosen = np.zeros(score.shape, bool)
    occupied = np.zeros(score.shape, bool)
    elig_flat = eligible.ravel()
    n = 0
    idx = 0
    total = order.size
    while n < budget and idx < total:
        p = order[idx]
        idx += 1
        if occupied.ravel()[p] or not elig_flat[p]:
            continue
        r, c = np.unravel_index(p, score.shape)
        chosen[r, c] = True
        r0, r1 = max(0, r - min_dist_px), min(score.shape[0], r + min_dist_px + 1)
        c0, c1 = max(0, c - min_dist_px), min(score.shape[1], c + min_dist_px + 1)
        occupied[r0:r1, c0:c1] = True
        n += 1
    assert not (chosen & ~eligible).any(), "spread_select emitted an ineligible pixel"
    return chosen


def lattice_select(shape, budget: int, eligible: np.ndarray,
                   rng: np.random.Generator, min_dist_px: int = MIN_DIST_PX) -> np.ndarray:
    """A jittered lattice of the eligible area with min-distance separation.

    Achieves near-uniform coverage for a given count, which is the arrangement
    the metric's max-over-kernel credit rewards.
    """
    H, W = shape
    area = int(eligible.sum())
    spacing = max(min_dist_px, (area / max(budget, 1)) ** 0.5)
    chosen = np.zeros(shape, bool)
    occupied = np.zeros(shape, bool)
    n = 0
    r = rng.uniform(0, spacing)
    while r < H and n < budget:
        c = rng.uniform(0, spacing)
        while c < W and n < budget:
            ri, ci = int(r), int(c)
            if eligible[ri, ci] and not occupied[ri, ci]:
                chosen[ri, ci] = True
                r0, r1 = max(0, ri - min_dist_px), min(H, ri + min_dist_px + 1)
                c0, c1 = max(0, ci - min_dist_px), min(W, ci + min_dist_px + 1)
                occupied[r0:r1, c0:c1] = True
                n += 1
            c += spacing
        r += spacing
    if n < budget:   # sparse eligible geometry: top up with greedy spread
        extra = spread_select(eligible.astype(np.float32), budget - n,
                              eligible & ~chosen)
        chosen |= extra
    return chosen


def score_field(pred, truth, mask):
    r = metric.dti(pred.astype(np.float32), truth, mask=mask)
    return {"dti": r.dti, "tp_w": r.tp_w, "fp_w": r.fp_w, "fn_w": r.fn_w,
            "emitted_px": int((pred > 0).sum()),
            "tp_per_emitted_px": r.tp_per_emitted_px, "break_even": r.break_even}


def coverage_of_truth(pred_bool, truth):
    from scipy import ndimage
    if not truth.any():
        return None
    near = ndimage.maximum_filter(pred_bool.astype(np.uint8), size=2 * metric.R_PX + 1) > 0
    return float(near[truth].mean())


def run_arm(name, truth, mask, eligible, fields, dense, sparse, out):
    print(f"\n=== arm: {name} ===")
    rng = np.random.default_rng(SEED)
    rows = {}
    for cname, score in fields.items():
        # The emitter's support is wherever the field is defined and eligible:
        # arrangement is the free variable, not the support.
        support = eligible & np.isfinite(score)
        for variant, builder in (
                ("clustered_topk", lambda s, b: metric.top_k_mask(s, b, eligible=support)),
                ("spread_nms", lambda s, b: spread_select(s, b, support)),
        ):
            sel = builder(score, dense)
            pred = sel.astype(np.float32)
            res = score_field(pred, truth, mask)
            res["truth_coverage_within_R"] = coverage_of_truth(sel, truth)
            res["support_px"] = int(support.sum())
            key = f"{cname}__{variant}"
            rows[key] = res
            print(f"  {key:44s} DTI={res['dti']:.4f}  cov={res['truth_coverage_within_R']:.3f}  "
                  f"emitted={res['emitted_px']:,}")
            del sel, pred
        gc.collect()

    # lattice-NULL: arrangement-only control over the valid footprint
    lat_support = eligible & np.isfinite(fields.get("NULL_random", np.ones(1)))
    sel = lattice_select(truth.shape, dense, lat_support, rng)
    res = score_field(sel.astype(np.float32), truth, mask)
    res["truth_coverage_within_R"] = coverage_of_truth(sel, truth)
    rows["NULL_random__lattice"] = res
    print(f"  {'NULL_random__lattice':44s} DTI={res['dti']:.4f}  "
          f"cov={res['truth_coverage_within_R']:.3f}  emitted={res['emitted_px']:,}")

    # sparse budget: the same comparison at a perfect detector's pixel count
    sparse_rows = {}
    for cname, score in fields.items():
        support = eligible & np.isfinite(score)
        sel_c = metric.top_k_mask(score, sparse, eligible=support)
        sel_s = spread_select(score, sparse, support)
        sparse_rows[f"{cname}__clustered_topk"] = score_field(sel_c, truth, mask)
        sparse_rows[f"{cname}__spread_nms"] = score_field(sel_s, truth, mask)

    out[name] = {
        "budgets": {"dense_px": dense, "sparse_px": sparse},
        "dense": rows,
        "sparse": sparse_rows,
    }
    return out


def build_fields(labels, valid, feats, blank_zone=None):
    """Score fields, with the blind corridor blanked from the INPUT bands
    before any filtering -- smoothing an unblanked band lets a detector see
    through the hide-and-recover corridor, which would measure interpolation,
    not recovery.

    Matches the frozen protocol in scripts/run_hypotheses.py: the field itself
    is left defined wherever the detector defines it (edge_lineaments fills the
    blank internally); a band-level blank that leaks into the OUTPUT would
    silently forbid emission in the corridor and degenerate the comparison.
    NULL_random is never blanked -- uniform noise has no input to blind.
    """
    rng = np.random.default_rng(SEED)

    fields: dict[str, np.ndarray] = {}
    fields["NULL_random"] = rng.random(labels.shape).astype(np.float32)

    tc = data.load_band(feats, "tc")
    if blank_zone is not None:
        tc = tc.copy()
        tc[blank_zone] = np.nan
    line = detect.edge_lineaments(tc, sigma=2.0)
    line = np.maximum(line, detect.edge_lineaments(tc, sigma=4.0))
    fields["H2_tilt_lineament"] = np.where(np.isfinite(line), line, -np.inf).astype(np.float32)
    del tc, line
    gc.collect()

    sl = data.load_band(feats, "det_elev_slope")
    if blank_zone is not None:
        sl = sl.copy()
        sl[blank_zone] = np.nan
    fields["CTRL_det_elev_slope"] = np.where(np.isfinite(sl), np.abs(sl), -np.inf).astype(np.float32)
    del sl
    gc.collect()
    return fields


def run(data_dir: Path, out_path: Path) -> dict:
    t0 = time.time()
    feats = data_dir / "training_features.tif"
    labels = data.load_labels(data_dir / "labels.tif")
    valid = data.valid_mask(feats)
    if not data.verify_band_layout(feats):
        raise SystemExit("band layout does not match the verified layout")

    report: dict = {
        "hypothesis": "H-COV: coverage-optimal spreading raises DTI at fixed budget",
        "preregistered_predictions": {
            "p1": "spread/lattice NULL >= 1.5x clustered NULL (catalogue holdout, dense); >= 1.2x on surrogate",
            "p2": "spread raises physical fields too",
            "p3": "if physical spread still loses to spread-NULL, rankings carry no signal",
            "direction": "up",
            "rough_size": "catalogue holdout 0.12 -> 0.18-0.25 dense",
        },
        "min_dist_px": MIN_DIST_PX,
        "arms": {},
    }

    # ---- arm 1: catalogue hide-and-recover holdout ------------------------
    # Emission is allowed anywhere on the valid footprint: the blind corridor is
    # removed from the DETECTOR'S INPUTS (build_fields blank_zone), not from the
    # emission map -- the emitter does not know where the withheld truth is, and
    # the organizers' scoring never restricts where predictions may land.
    # Restricting emission to ~corridor is degenerate: the 3-px buffer equals the
    # credit radius, so every eligible pixel would have k(d) = 0 and every field
    # would score exactly zero.
    ho = Holdout(labels)
    fold_rows = []
    for fi, fold in enumerate(ho.folds):
        corr = fold.blinded
        elig = valid
        dense = int(round(DENSE_FRACTION * int(elig.sum())))
        sparse = int(fold.truth_px)
        print(f"\nfold {fi}: building fields with corridor blanked ...")
        fields = build_fields(labels, valid, feats, blank_zone=corr)
        arm = run_arm(f"catalogue_holdout_fold{fi}", fold.truth, fold.masked,
                      elig, fields, dense, sparse, {})
        arm_row = arm[f"catalogue_holdout_fold{fi}"]
        fold_rows.append(arm_row)
        del fields
        gc.collect()

    # aggregate folds
    agg = {}
    for key in fold_rows[0]["dense"]:
        dts = [fr["dense"][key]["dti"] for fr in fold_rows]
        covs = [fr["dense"][key]["truth_coverage_within_R"] for fr in fold_rows
                if fr["dense"][key]["truth_coverage_within_R"] is not None]
        agg[key] = {"mean_DTI": float(np.mean(dts)), "std_DTI": float(np.std(dts)),
                    "fold_DTI": [round(d, 6) for d in dts],
                    "mean_truth_coverage_within_R": float(np.mean(covs)) if covs else None}
    sparse_agg = {}
    for key in fold_rows[0]["sparse"]:
        dts = [fr["sparse"][key]["dti"] for fr in fold_rows]
        sparse_agg[key] = {"mean_DTI": float(np.mean(dts)), "std_DTI": float(np.std(dts)),
                           "fold_DTI": [round(d, 6) for d in dts]}
    report["arms"]["catalogue_hide_and_recover"] = {
        "folds": len(fold_rows),
        "dense_mean": agg,
        "sparse_mean": sparse_agg,
        "per_fold": fold_rows,
    }
    print("\ncatalogue holdout, dense mean DTI:")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["mean_DTI"]):
        print(f"  {k:44s} {v['mean_DTI']:.4f} +-{v['std_DTI']:.4f}  cov={v['mean_truth_coverage_within_R']}")

    # ---- arm 2: off-catalogue surrogate ---------------------------------
    import rasterio
    proxy_path = REPO / "data" / "external" / "proxy_catalogue.tif"
    if proxy_path.exists():
        from scipy import ndimage
        with rasterio.open(proxy_path) as ds:
            proxy = ds.read(1)
        halo = ndimage.maximum_filter(labels, footprint=np.ones((7, 7), bool))
        truth = (proxy == 2) & ~halo & valid
        mask = halo
        elig = valid & ~mask
        dense = int(round(DENSE_FRACTION * int(elig.sum())))
        sparse = int(truth.sum())
        print("\nsurrogate: building fields (no corridor) ...")
        fields = build_fields(labels, valid, feats, blank_zone=None)
        run_arm("surrogate", truth, mask, elig, fields, dense, sparse,
                report["arms"])
        del fields
        gc.collect()
    else:
        print("no surrogate raster; skipping arm 2")

    # ---- predictions check ------------------------------------------------
    def g(arm, key):
        return arm["dense_mean"][key]["mean_DTI"] if "dense_mean" in arm \
            else arm["dense"][key]["dti"]

    cat = report["arms"]["catalogue_hide_and_recover"]
    sur = report["arms"].get("surrogate")
    checks = {}
    base_c = g(cat, "NULL_random__clustered_topk")
    lat_c = g(cat, "NULL_random__lattice")
    checks["p1_catalogue_1.5x"] = bool(lat_c >= 1.5 * base_c)
    if sur:
        base_s = sur["dense"]["NULL_random__clustered_topk"]["dti"]
        lat_s = sur["dense"]["NULL_random__lattice"]["dti"]
        checks["p1_surrogate_1.2x"] = bool(lat_s >= 1.2 * base_s)
        checks["p2_surrogate_H2_spread_gain"] = bool(
            sur["dense"]["H2_tilt_lineament__spread_nms"]["dti"]
            > sur["dense"]["H2_tilt_lineament__clustered_topk"]["dti"])
    checks["p2_catalogue_H2_spread_gain"] = bool(
        g(cat, "H2_tilt_lineament__spread_nms") > g(cat, "H2_tilt_lineament__clustered_topk"))
    report["predictions_held"] = checks

    report["decision"] = {
        "holdout_best_reference_dense": 0.1253,
        "holdout_null_dense": base_c,
        "spread_null_dense": lat_c,
        "beats_current_holdout_best": bool(lat_c > 0.1253),
        "verdict": (
            "Spreading raises DTI at fixed budget (coverage mechanism confirmed). "
            "A spread field that clears the current holdout best (0.1253 dense) is "
            "eligible for a submission slot ONLY together with the discovery screen; "
            "uniform random has no geological grounding."
            if lat_c > 0.1253 else
            "Spreading did not clear the current holdout best. Emission shaping "
            "alone cannot move the score; detection remains the bottleneck."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}  ({report['decision']['runtime_seconds']}s)")
    print("predictions held:", checks)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "evidence" / "coverage_emission.json"))
    args = ap.parse_args()
    run(Path(args.data_dir), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
