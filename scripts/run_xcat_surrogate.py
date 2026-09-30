#!/usr/bin/env python3
"""Off-catalogue surrogate arm (H-XSUR): can a detector find REAL structure that
the competition catalogue lacks?

Why this arm exists
-------------------
The hide-and-recover holdout (scripts/run_hypotheses.py) cannot rank discovery
hypotheses: its truth *is* the supplied catalogue, so a detector that predicts in
the empty space between catalogue strands scores zero by construction
(evidence/h1_gap_diagnostic.json). The natural second arm -- an independent
official fault compilation -- was attempted with the USGS Quaternary Fault and
Fold Database (Qfaults, DOI 10.5066/P9BCVRCK) and quantitatively REFUSED: the
sibling project's cross-catalogue measurement (GEMSDOE/data/evidence/xcat/
transfer_report.json) shows 60,938 of its 60,939 in-footprint pixels are already
within R = 3 px of a label -- the two catalogues are the same lines here, so
there is no 'new fault' population to score against.

The SGMC structure raster is the one remaining surrogate
(GEMSDOE/data/evidence/proxy/proxy_catalogue.tif, USGS State Geologic Map
Compilation, Horton, San Juan and Stoeser, 2017, DOI 10.3133/ds1052). Byte-level census run for
this script (reproduce with the --census flag):

    code 1 (already near a label)     20,491 px   100.0 % within R of labels
    code 2 (new-fault-like)           61,664 px     4.3 % within R of labels
    code 2 beyond R of labels         59,035 px    95.7 %  <-- the surrogate truth

What the arm answers
--------------------
Truth = SGMC code-2 pixels beyond R = 3 px of every competition label. The
catalogue's credit halo (its own R-dilation) is removed from prediction AND
truth before scoring, so every credit pixel must sit in genuinely off-catalogue
terrain. A detector that cannot beat uniform random here has no discovery skill
of any kind; one that can beat it demonstrably finds real mapped structure the
catalogue misses. That is a necessary-not-sufficient condition for finding a
young hidden fault -- SGMC includes bedrock contacts of any age (the sibling's
own caveat, adopted here) -- so this arm is a SCREEN, not the prize metric.

Pre-registered prediction (written before the run)
------------------------------------------------
* H2_tilt_lineament >= 1.5x NULL_random  (geophysical edges express structure
  of any age; SGMC structure was mapped from that expression)
* H3_relay_bridge   >  NULL_random       (Faulds: connecting structures are
  where missing faults live; SGMC should map some of them)
* REFERENCE support <= 1.2x NULL_random after halo-masking (it is built FROM
  the labels; it has no information about structure away from them)

Direction and rough size: NULL dense DTI 0.05-0.10; H2/H3 dense DTI 0.10-0.25.
A result in the opposite direction is a documented negative result and redirects
the next test (it would mean SGMC code-2 is unrelated to our detectors' signals,
pushing toward the radiometric/lidar external-data route).

Output: evidence/xcat_surrogate.json
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

SEED = 17
DENSE_FRACTION = 0.033
HALO_PX = 3          # the credit radius: catalogue + halo is removed from scoring
N_BOOT = 200         # block-bootstrap resamples for error bars


def census(proxy: np.ndarray, labels: np.ndarray) -> dict:
    """Byte-level population census of the SGMC surrogate (see module docstring)."""
    from scipy import ndimage
    k = np.ones((2 * HALO_PX + 1,) * 2, bool)
    lab_halo = ndimage.maximum_filter(labels, footprint=k)
    code2 = proxy == 2
    out = {
        "code1_px": int((proxy == 1).sum()),
        "code2_px": int(code2.sum()),
        "code2_within_halo_px": int((code2 & lab_halo).sum()),
        "code2_beyond_halo_px": int((code2 & ~lab_halo).sum()),
        "label_px": int(labels.sum()),
    }
    return out


def block_bootstrap_dti(pred, truth, mask, n_blocks=4, n_boot=N_BOOT, seed=SEED):
    """Mean and 95 % interval of DTI under spatial-block bootstrap.

    Each block crop is scored once (TP_w/FP_w/FN_w are the sufficient
    statistics of the metric), then resamples of blocks are combined. A
    previous revision resampled the full grid per draw -- 200 draws x 8
    candidates x full-grid distance transforms -- and timed out; block-level
    sufficient statistics give the same inference at a fraction of the cost.
    """
    rng = np.random.default_rng(seed)
    H, W = truth.shape
    bh, bw = H // n_blocks, W // n_blocks
    triples = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            sl = (slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw))
            if truth[sl].sum() == 0:
                continue
            r = metric.dti(pred[sl], truth[sl], mask=mask[sl])
            triples.append((r.tp_w, r.fp_w, r.fn_w))
    if not triples:
        return {"mean": None, "lo": None, "hi": None}
    T = np.array([t for t, _, _ in triples])
    F = np.array([f for _, f, _ in triples])
    N = np.array([n for _, _, n in triples])
    n = len(triples)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tp, fp, fn = T[idx].sum(), F[idx].sum(), N[idx].sum()
        denom = tp + metric.ALPHA * fp + metric.BETA * fn + metric.EPS
        scores.append(tp / denom if denom > 0 else 0.0)
    s = np.asarray(scores, float)
    return {"mean": float(s.mean()), "lo": float(np.percentile(s, 2.5)),
            "hi": float(np.percentile(s, 97.5)),
            "note": "block-bootstrap over 4x4 crop sufficient statistics"}


def run(data_dir: Path, proxy_path: Path, anchor_path: Path | None,
        out_path: Path) -> dict:
    t0 = time.time()
    feats = data_dir / "training_features.tif"
    if not data.verify_band_layout(feats):
        raise SystemExit("band layout does not match the verified layout")

    labels = data.load_labels(data_dir / "labels.tif")
    valid = data.valid_mask(feats)
    import rasterio
    with rasterio.open(proxy_path) as ds:
        proxy = ds.read(1)
    if proxy.shape != labels.shape:
        raise SystemExit(f"surrogate grid {proxy.shape} != label grid {labels.shape}")

    cen = census(proxy, labels)
    print("SGMC surrogate census:", json.dumps(cen))

    from scipy import ndimage
    k = np.ones((2 * HALO_PX + 1,) * 2, bool)
    lab_halo = ndimage.maximum_filter(labels, footprint=k)

    truth = (proxy == 2) & ~lab_halo & valid
    # Discovery-strict masking: the catalogue's entire credit halo is removed
    # from both sides, so no credit can be earned by hugging a known fault.
    mask = lab_halo
    print(f"surrogate truth: {int(truth.sum()):,} px; halo removed: {int(mask.sum()):,} px")

    valid_scorable = valid & ~mask
    dense = int(round(DENSE_FRACTION * int(valid_scorable.sum())))
    sparse = int(truth.sum())          # a perfect detector's pixel count

    rng = np.random.default_rng(SEED)

    # ---- candidate score fields ------------------------------------------
    # Random fields need no feature bands; physical detectors load one band at
    # a time because nine float32 bands on this grid exceed the machine's RAM.
    cands: dict[str, np.ndarray] = {}
    cands["NULL_random"] = rng.random(labels.shape).astype(np.float32)

    cands["BASE_distance_to_catalogue"] = -ndimage.distance_transform_edt(
        ~labels).astype(np.float32)

    print("H2: tilt lineaments (tc) ...")
    tc = data.load_band(feats, "tc")
    line = detect.edge_lineaments(tc, sigma=2.0)
    line = np.maximum(line, detect.edge_lineaments(tc, sigma=4.0))
    cands["H2_tilt_lineament"] = line
    del tc, line
    gc.collect()

    print("H2b: supplied tmi_hg ...")
    hg = data.load_band(feats, "tmi_hg")
    cands["H2b_supplied_tmi_hg"] = np.abs(hg).astype(np.float32)
    del hg
    gc.collect()

    print("H3: relay bridges ...")
    bridges, bm = detect.relay_bridges(labels, min_gap_px=6, max_gap_px=45,
                                       strike_window_deg=(0.0, 45.0))
    b = np.zeros(labels.shape, np.float32)
    b[bridges] = 1.0
    cands["H3_relay_bridge"] = b
    relay_meta = {"n_bridges": len(bm), "bridge_px": int(bridges.sum())}
    del bridges, b
    gc.collect()

    print("H4: joint strain localisation ...")
    sh = data.load_band(feats, "geod_shearrate")
    dl = data.load_band(feats, "geod_dilaterate")
    iv = data.load_band(feats, "geod_2ndinv")
    cands["H4_strain_jog"] = detect.joint_strain_localisation(sh, dl, iv)
    del sh, dl, iv
    gc.collect()

    print("CTRL: det_elev_slope ...")
    sl = data.load_band(feats, "det_elev_slope")
    cands["CTRL_det_elev_slope"] = np.abs(sl).astype(np.float32)
    del sl
    gc.collect()

    if anchor_path and anchor_path.exists():
        with rasterio.open(anchor_path) as ds:
            a = ds.read(1)
        support = np.isfinite(a) & (a > 0)
        cands["REFERENCE_support"] = support.astype(np.float32)
        del a, support
        gc.collect()

    # ---- scoring ----------------------------------------------------------
    results = {}
    for name, field in cands.items():
        f = np.where(np.isfinite(field), field, -np.inf).astype(np.float32)
        row = {}
        for regime, bud in (("sparse_matched_truth_px", sparse),
                            ("dense_competition_like", dense)):
            sel = metric.top_k_mask(f, bud, eligible=valid_scorable)
            pred = sel.astype(np.float32)
            res = metric.dti(pred, truth, mask=mask)
            boot = block_bootstrap_dti(pred, truth, mask)
            row[regime] = {
                "dti": res.dti, "bootstrap": boot,
                "emitted_px": int(sel.sum()),
                "tp_w": res.tp_w, "fp_w": res.fp_w, "fn_w": res.fn_w,
                "tp_per_emitted_px": res.tp_per_emitted_px,
                "break_even": res.break_even,
            }
            del sel, pred
        row["field_support_px"] = int(np.isfinite(f).sum())
        results[name] = row
        print(f"  {name:28s} dense DTI={row['dense_competition_like']['dti']:.4f} "
              f"sparse DTI={row['sparse_matched_truth_px']['dti']:.4f}")

    null_dense = results["NULL_random"]["dense_competition_like"]["dti"]
    null_sparse = results["NULL_random"]["sparse_matched_truth_px"]["dti"]

    ranked = sorted((k for k in results if k != "NULL_random"),
                    key=lambda k: -results[k]["dense_competition_like"]["dti"])
    lifts = {k: {
        "dense_lift_vs_null": (results[k]["dense_competition_like"]["dti"] / null_dense
                               if null_dense > 0 else None),
        "sparse_lift_vs_null": (results[k]["sparse_matched_truth_px"]["dti"] / null_sparse
                                if null_sparse > 0 else None),
    } for k in ranked}

    pred_holds = {
        "H2_at_least_1.5x_null": bool(lifts.get("H2_tilt_lineament", {})
                                      .get("dense_lift_vs_null", 0) >= 1.5),
        "H3_above_null": bool(lifts.get("H3_relay_bridge", {})
                              .get("dense_lift_vs_null", 0) > 1.0),
        "reference_within_1.2x_null": bool(lifts.get("REFERENCE_support", {})
                                           .get("dense_lift_vs_null", 99) <= 1.2),
    }

    report = {
        "arm": "off-catalogue surrogate (SGMC code-2 beyond the catalogue's credit halo)",
        "why": ("Qfaults cross-catalogue arm REFUSED (same lines as labels, "
                "1 independent px); see GEMSDOE/data/evidence/xcat/transfer_report.json"),
        "surrogate_truth": "SGMC code-2 pixels beyond R=3 px of every competition label",
        "surrogate_census": cen,
        "surrogate_caveat": ("SGMC is pre-Quaternary bedrock structure and "
                             "contacts (DOI 10.3133/ds1052): beating NULL here is "
                             "necessary but not sufficient for finding young "
                             "hidden faults. This arm is a screen."),
        "masking": f"catalogue + {HALO_PX}-px credit halo removed from prediction and truth",
        "budgets": {"sparse_matched_truth_px": sparse,
                    "dense_competition_like": dense,
                    "dense_fraction_of_scorable": DENSE_FRACTION},
        "relay_bridge_geometry": relay_meta,
        "hypotheses": results,
        "lifts_vs_null": lifts,
        "ranked_by_dense_DTI": ranked,
        "preregistered_predictions": {
            "text": "H2 >= 1.5x NULL; H3 > NULL; reference <= 1.2x NULL (dense)",
            "direction": "up for H2/H3; flat for reference",
            "rough_size": "NULL 0.05-0.10; H2/H3 0.10-0.25 dense DTI",
        },
        "predictions_held": pred_holds,
        "decision": {
            "best_candidate": ranked[0] if ranked else None,
            "best_dense_DTI": (results[ranked[0]]["dense_competition_like"]["dti"]
                               if ranked else None),
            "null_dense_DTI": null_dense,
            "verdict": (
                "Discovery-capable detectors confirmed: "
                + ", ".join(k for k in ranked
                            if lifts[k]["dense_lift_vs_null"] and lifts[k]["dense_lift_vs_null"] > 1.0)
                + " beat uniform random on off-catalogue structure. These earn "
                  "union-candidate status; the catalogue-holdout cannot rank them, "
                  "this arm can."
                if any(lifts[k]["dense_lift_vs_null"] and lifts[k]["dense_lift_vs_null"] > 1.0
                       for k in ranked) else
                "No detector beat uniform random on off-catalogue structure. "
                "Documented negative result: local signals carry no discovery "
                "skill visible to this screen; external data (radiometrics, 1 m "
                "lidar) is the redirected path."),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out_path}  ({report['runtime_seconds']}s)")
    print("predictions held:", pred_holds)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(REPO / "data"))
    ap.add_argument("--proxy", default=str(REPO / "data" / "external" / "proxy_catalogue.tif"))
    ap.add_argument("--anchor", default="/tmp/gems_build/anchor_submission.tif",
                    help="the reference 0.1563 field (fetched by build_submission.py)")
    ap.add_argument("--out", default=str(REPO / "evidence" / "xcat_surrogate.json"))
    ap.add_argument("--census", action="store_true", help="print the census and exit")
    args = ap.parse_args()

    if args.census:
        import rasterio
        labels = data.load_labels(Path(args.data_dir) / "labels.tif")
        with rasterio.open(args.proxy) as ds:
            proxy = ds.read(1)
        print(json.dumps(census(proxy, labels), indent=1))
        return 0

    run(Path(args.data_dir), Path(args.proxy),
        Path(args.anchor) if Path(args.anchor).exists() else None,
        Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
