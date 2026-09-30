"""The official Driving Distance-Weighted Tversky Index (DTI).

Verified against the competition's own worked example on the problem-description
page: with TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 the published answer is 0.60,
and this module returns 3.00 / (3.00 + 0.2*1.89 + 0.8*2.00) = 0.6027.

Definitions, transcribed from the problem description:

    k(d) = max(1 - d/R, 0)                      R = 300 m = 3 pixels at 100 m
    sigma(x) = max over ground-truth g of k(d(x, g))
    TP_w = sum over g of max over x: p(x) * k(d(x, g))
    FP_w = sum over x with p(x) > 0 of p(x) * (1 - sigma(x))
    FN_w = sum over g of (1 - max over x: p(x) * k(d(x, g)))
    DTI  = TP_w / (TP_w + alpha*FP_w + beta*FN_w + eps),  alpha=0.2, beta=0.8

Two consequences that drive every design decision in this repository, and that
are cheap to verify but easy to miss:

1.  TP_w + FN_w = |G| exactly, because the FN term is one minus the same
    maximum that defines the TP term. The false-negative side is therefore never
    an independent quantity, and a candidate's score is fully determined by
    (weighted true positives) and (false-positive mass).

2.  For a fixed set of pixel *locations*, the score is strictly increasing in a
    uniform scale factor applied to the emitted probabilities: scaling p by c
    replaces TP_w by c*TP_w and FP_w by c*FP_w, leaving FN_w = |G| - c*TP_w,
    and the resulting DTI is larger for larger c up to c = 1. The optimum is
    therefore to emit exactly 1.0 on every chosen pixel, which turns the problem
    into a pure selection problem: *which* pixels, not *what value*.

The only exception to (2) is a pixel that is certainly not near any ground
truth, where emitting 0 is better than emitting anything positive. The expected
margin rule in :func:`expected_margin_rule` handles that case explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

ALPHA = 0.2          # weight on false positives
BETA = 0.8           # weight on false negatives
RADIUS_M = 300.0     # metres, from the problem description
PIXEL_M = 100.0      # metres, the competition raster's cell size
R_PX = 3             # 300 m / 100 m
EPS = 1e-9


# ---------------------------------------------------------------------------
# distance-weighted kernel
# ---------------------------------------------------------------------------
def distance_to_nearest_truth(truth: np.ndarray) -> np.ndarray:
    """Euclidean distance in pixels to the nearest True cell. 0 where truth is True.

    Returned dtype is float32; inf is replaced by a large finite value so that
    downstream arithmetic cannot produce a NaN.
    """
    if not truth.any():
        return np.full(truth.shape, np.float32(1e9), dtype=np.float32)
    d = ndimage.distance_transform_edt(~truth).astype(np.float32)
    return d


def kernel(d: np.ndarray | float) -> np.ndarray | float:
    """Triangular kernel k(d) = max(1 - d/R, 0)."""
    return np.maximum(1.0 - np.asarray(d, dtype=np.float32) / R_PX, 0.0)


# ---------------------------------------------------------------------------
# the score
# ---------------------------------------------------------------------------
@dataclass
class DTIScore:
    dti: float
    tp_w: float
    fp_w: float
    fn_w: float
    truth_px: int
    emitted_px: int
    # diagnostics that make a score interpretable rather than just comparable
    tp_per_emitted_px: float     # "marginal precision" -- compare to break_even
    break_even: float            # needed to keep this score and stay level

    def as_dict(self) -> dict:
        return asdict(self)


def dti(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None,
        alpha: float = ALPHA, beta: float = BETA, r_px: int = R_PX) -> DTIScore:
    """Score a prediction against a truth raster.

    Parameters
    ----------
    pred
        Predicted values in [0, 1]. Values outside that range are a format error
        and are clipped here only so that a scoring call never crashes; the
        submission gate in :mod:`gems.io` is what enforces the range.
    truth
        Boolean ground truth.
    mask
        Optional boolean mask of pixels to *exclude* -- this is how the
        organizers' pixel-exact masking of known faults is reproduced. Both the
        prediction and the truth are zeroed under the mask, so a prediction that
        tries to score by re-drawing known faults gains nothing.

    Returns
    -------
    DTIScore
    """
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=bool)
    if mask is not None:
        keep = ~np.asarray(mask, dtype=bool)
        pred = np.where(keep, pred, 0.0)
        truth = truth & keep

    pred = np.clip(np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    n_truth = int(truth.sum())
    if n_truth == 0:
        return DTIScore(float("nan"), 0.0, 0.0, 0.0, 0, int((pred > 0).sum()),
                        0.0, 0.0)

    d = distance_to_nearest_truth(truth)
    k = np.maximum(1.0 - d / float(r_px), 0.0).astype(np.float32)

    # TP_w: for each truth pixel, the best weighted prediction within R.
    # Computed with a grey dilation (maximum filter) of `pred * k` over a
    # (2R+1)^2 footprint, which is exactly max_{x in R(g)} pred(x) * k(d(x,g)).
    weighted = (pred * k).astype(np.float32)
    size = 2 * r_px + 1
    best_near_g = ndimage.maximum_filter(weighted, size=size, mode="constant",
                                         cval=0.0)
    tp_w = float(best_near_g[truth].sum())

    # FP_w: every positive pixel pays for how far it is from any truth.
    fp_w = float((pred * (1.0 - k)).sum())
    fn_w = float(n_truth) - tp_w

    denom = tp_w + alpha * fp_w + beta * fn_w + EPS
    score = tp_w / denom

    emitted = int((pred > 0).sum())
    be = break_even_marginal_tp(score, alpha) if score > 0 else float("inf")
    return DTIScore(
        dti=float(score), tp_w=tp_w, fp_w=fp_w, fn_w=fn_w,
        truth_px=n_truth, emitted_px=emitted,
        tp_per_emitted_px=float(tp_w / emitted) if emitted else 0.0,
        break_even=float(be),
    )


def break_even_marginal_tp(dti_value: float, alpha: float = ALPHA) -> float:
    """Marginal weighted-TP per fully-wrong emitted pixel needed to stay level.

    Adding one pixel that is at least R away from every truth pixel costs
    ``alpha`` and gains nothing, so the score falls unless the same addition
    also captures ``dT`` of true-positive weight. Differentiating
    ``T / (T + alpha*F + beta*(G - T))`` gives the condition

        dT * (1 - alpha*DTI) > alpha * DTI * dF

    and a pure false positive has dF = 1, hence ``dT > alpha*DTI/(1 - alpha*DTI)``.
    This is the number to quote next to any candidate: a detector whose marginal
    precision per emitted pixel is below it loses score no matter how good its
    map looks, and at DTI = 0.1563 the threshold is 0.0323.
    """
    return float(alpha * dti_value / max(1.0 - alpha * dti_value, 1e-12))


def marginal_decision(d_tp: float, d_fp: float, dti_value: float,
                      alpha: float = ALPHA) -> bool:
    """True if adding weight ``d_tp`` while paying ``d_fp`` raises the score."""
    if d_fp <= 0:
        return d_tp > 0
    return d_tp * (1.0 - alpha * dti_value) > alpha * dti_value * d_fp


# ---------------------------------------------------------------------------
# emission rules
# ---------------------------------------------------------------------------
def expected_margin_rule(pred: np.ndarray, p_near: np.ndarray,
                         dti_value: float, alpha: float = ALPHA) -> np.ndarray:
    """Select pixels by expected marginal score change instead of a quantile.

    ``p_near[x]`` is the modelled probability that at least one ground-truth
    pixel lies within R pixels of ``x``, and ``pred[x]`` is a *shape* score that
    ranks the candidate locations. A fixed quantile rule ignores the metric
    entirely; this rule emits a pixel when

        p_near * (1 - alpha*DTI)  >  alpha*DTI * (1 - p_near)

    i.e. when the expected true-positive weight exceeds the expected
    false-positive cost. Solving for the probability gives the closed form

        p_near > alpha*DTI / (1 + alpha*DTI*(1 - ... ))   ->   p* = alpha*DTI

    because the two sides differ only in the coefficient on p_near. The rule is
    therefore: emit everything above a probability floor of ``alpha*DTI``, which
    at DTI = 0.1563 is a floor of 3.1 %. Note this is a statement about
    *probability*, not about rank -- a detector with 500,000 candidates at 4 %
    probability each is better served than one with 5,000 candidates at 2 %.
    """
    p_near = np.asarray(p_near, dtype=np.float32)
    floor = alpha * dti_value
    return p_near > floor


def quantile_threshold(score: np.ndarray, emit_px: int,
                       eligible: np.ndarray | None = None) -> float:
    """The threshold that emits exactly ``emit_px`` pixels (the naive baseline)."""
    s = np.asarray(score, dtype=np.float32)
    if eligible is not None:
        s = np.where(eligible, s, -np.inf)
    flat = s.ravel()
    n = int(np.isfinite(flat).sum())
    if emit_px >= n:
        return float(np.nanmin(flat[np.isfinite(flat)])) if n else 0.0
    kth = n - int(emit_px)
    return float(np.partition(flat[np.isfinite(flat)], kth)[kth])


def top_k_mask(score: np.ndarray, emit_px: int,
               eligible: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of the ``emit_px`` highest-scoring eligible pixels."""
    s = np.asarray(score, dtype=np.float32)
    elig = np.isfinite(s) if eligible is None else (np.asarray(eligible) & np.isfinite(s))
    n_elig = int(elig.sum())
    if n_elig == 0 or emit_px <= 0:
        return np.zeros(s.shape, dtype=bool)
    if emit_px >= n_elig:
        return elig.copy()
    thr = quantile_threshold(s, emit_px, elig)
    m = elig & (s > thr)
    deficit = emit_px - int(m.sum())
    if deficit > 0:                       # ties at the threshold: take any of them
        tied = elig & (s == thr) & ~m
        idx = np.flatnonzero(tied.ravel())[:deficit]
        m.ravel()[idx] = True
    return m
