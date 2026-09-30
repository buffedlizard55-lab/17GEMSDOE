"""Turning a continuous score field into a submittable binary field.

Three facts drive everything here:

1. For a fixed support, scaling all values by c > 0 increases the DTI
   (derivation (2) in ``metric.py``).  The optimum for a chosen pixel set is
   therefore to emit every chosen pixel at exactly 1.0.  The submission is
   binary.

2. Adding a pixel that is farther than R = 3 px from any truth pixel costs
   ``alpha * DTI / (1 - alpha * DTI)`` of weighted true-positive mass.  At the
   best score the group has measured (0.1563) that break-even is 0.0323; to
   reach 0.3049 a field must average 0.0649.  Budget is not the lever -
   precision at the margin is.

3. Because of (2) the correct way to compare two detectors is at an *identical
   emitted-pixel budget*, reporting the marginal hit rate of the pixels each
   one actually spends.  ``equal_budget_mask`` exists to enforce that.
"""

from __future__ import annotations

import numpy as np

from .metric import break_even_marginal_tp

__all__ = ["top_k_mask", "equal_budget_mask", "thin_field", "BudgetReport"]


def top_k_mask(score: np.ndarray, k: int, eligible: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of the ``k`` highest-scoring eligible pixels.

    Uses ``argpartition`` (O(n)) rather than a full sort.  NaN scores rank last.
    """
    s = np.asarray(score, dtype=np.float64)
    if eligible is None:
        eligible = np.isfinite(s)
    el = np.asarray(eligible, dtype=bool) & np.isfinite(s)
    out = np.zeros(s.shape, dtype=bool)
    n_el = int(el.sum())
    if k <= 0 or n_el == 0:
        return out
    k = min(k, n_el)
    flat_idx = np.flatnonzero(el.ravel())
    vals = s.ravel()[flat_idx]
    if k == n_el:
        out.ravel()[flat_idx] = True
        return out
    part = np.argpartition(vals, -k)[-k:]
    out.ravel()[flat_idx[part]] = True
    return out


def equal_budget_mask(reference_positive: np.ndarray, score: np.ndarray,
                      eligible: np.ndarray) -> np.ndarray:
    """Emit exactly as many pixels as the reference field does *within* ``eligible``.

    The reference is masked the same way the candidate is, so the comparison is
    budget-matched under the metric's own masking rule.
    """
    ref = np.asarray(reference_positive, dtype=bool) & np.asarray(eligible, dtype=bool)
    return top_k_mask(score, int(ref.sum()), eligible=eligible)


def thin_field(score: np.ndarray, mask: np.ndarray, radius_px: int = 1) -> np.ndarray:
    """Greedy non-maximum suppression: keep a pixel only if no kept neighbour within radius.

    Produces spaced nodes rather than a solid band.  The group's own measurement
    (``5GEMSDOE`` leaderboard anchor, deduction D3) found this to be a
    second-order effect on the real board (+0.0041 between spaced nodes and a
    dense ridge at equal budget), so it is offered but is not the lever.
    """
    m = np.asarray(mask, dtype=bool).copy()
    if radius_px <= 0 or not m.any():
        return m
    order = np.argsort(-np.asarray(score, dtype=np.float64).ravel())
    flat = m.ravel()
    keep = np.zeros_like(flat)
    d = np.arange(-radius_px, radius_px + 1)
    yy, xx = np.meshgrid(d, d, indexing="ij")
    struct = (yy * yy + xx * xx) <= radius_px * radius_px
    taken = np.zeros_like(m)
    h, w = m.shape
    for idx in order:
        if not flat[idx]:
            continue
        r, c = divmod(int(idx), w)
        if taken[r, c]:
            continue
        keep[idx] = True
        r0, r1 = max(0, r - radius_px), min(h, r + radius_px + 1)
        c0, c1 = max(0, c - radius_px), min(w, c + radius_px + 1)
        sub = taken[r0:r1, c0:c1]
        sy0 = r0 - (r - radius_px)
        sx0 = c0 - (c - radius_px)
        st = struct[sy0:sy0 + sub.shape[0], sx0:sx0 + sub.shape[1]]
        sub |= st
    return keep.reshape(m.shape)


class BudgetReport:
    """Small helper that formats the marginal-precision decision for the log."""

    def __init__(self, score_reference: float):
        self.tau = break_even_marginal_tp(score_reference)

    def line(self, name: str, n_px: int, tp_w: float) -> str:
        rate = tp_w / n_px if n_px else 0.0
        verdict = "BEATS" if rate > self.tau else "below"
        return (f"{name:28s} px={n_px:>8d} TP_w={tp_w:>10.1f} "
                f"marginal={rate:.4f} vs tau={self.tau:.4f} -> {verdict}")
