"""Hide-and-recover evaluation: withhold whole fault segments plus a buffer.

The failure mode this module is built to avoid is the one that makes a local
score meaningless: scoring a prediction against the same catalogue the detector
was allowed to read. A detector given the known-fault raster will score well by
re-drawing it and will tell you nothing about finding faults that are not in it.

The protocol here mirrors the organizers' confirmed behaviour:

* **Whole segments are withheld**, not individual pixels. Isolated pixels would
  be recoverable by any smoothing operator, so a pixel-level holdout measures
  interpolation, not discovery.
* **A buffer around every withheld segment is removed from the detector's
  inputs.** Without it, a detector sees the trace stop at the buffer edge and
  simply extends the line -- again interpolation, not discovery.
* **The remaining catalogue is masked out pixel-exactly at scoring time.** A
  prediction that lands on a known fault gains nothing, exactly as in the
  competition, where known-fault pixels are removed from both prediction and
  truth before the score is computed.
* **Folds are spatial blocks.** Withholding scattered segments would let a
  detector infer a hidden segment from its immediate neighbours, which is
  geography generalisation rather than fault-finding. Entire 4x4 blocks are
  assigned to folds so a withheld segment's neighbourhood is withheld too.

Known limitation, recorded rather than papered over: **the holdout's truth is
the supplied catalogue**, so a detector that predicts in the empty space between
catalogue strands scores near zero by construction, whatever its geological
merit. See ``evidence/h1_gap_diagnostic.json``. This instrument answers "can the
model generalise over geography?"; it cannot answer "can the model find
unmapped faults?". A second arm built on an independent official compilation is
required for that.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage

MIN_COMPONENT_PX = 20
N_BLOCKS = 4
BUFFER_PX = 3


# ---------------------------------------------------------------------------
def components(labels: np.ndarray, min_px: int = MIN_COMPONENT_PX,
               connectivity: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """8-connected fault components at or above ``min_px``.

    Returns ``(component_id_map, sizes)`` where component ids are 1..n and the
    map is 0 everywhere that is not part of a retained component.
    """
    structure = np.ones((3, 3), bool) if connectivity == 8 else None
    raw, n = ndimage.label(labels, structure=structure)
    if n == 0:
        return np.zeros(labels.shape, np.int32), np.zeros(0, np.int64)
    sizes = np.bincount(raw.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes[1:] >= min_px
    out = np.where(keep[raw], raw, 0).astype(np.int32)
    # relabel so ids stay contiguous after dropping small components
    ids = np.unique(out)
    ids = ids[ids > 0]
    remap = np.zeros(int(out.max()) + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    out = remap[out]
    new_sizes = np.bincount(out.ravel(), minlength=len(ids) + 1)[1:]
    return out, new_sizes


def dilate(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Square (Chebyshev) dilation -- a conservative superset of a disc of the same r."""
    if radius_px <= 0:
        return mask.copy()
    size = 2 * int(radius_px) + 1
    return ndimage.maximum_filter(np.asarray(mask, bool), size=size, mode="constant",
                                  cval=False)


def remove_with_buffer(faults: np.ndarray, radius_px: int = BUFFER_PX):
    """Return ``(visible, hidden_zone)``: the catalogue minus a corridor around it.

    ``hidden_zone`` is the corridor itself, returned so a caller can report how
    much of the map the detector was blinded to.
    """
    corridor = dilate(np.asarray(faults, bool), radius_px)
    return faults & ~corridor, corridor


# ---------------------------------------------------------------------------
@dataclass
class Fold:
    index: int
    truth: np.ndarray            # withheld segments: the thing to be recovered
    visible: np.ndarray          # catalogue the detector may read
    blinded: np.ndarray          # corridor removed from detector inputs
    masked: np.ndarray           # visible catalogue: masked out at scoring time
    n_components: int
    truth_px: int
    visible_px: int

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("truth", "visible", "blinded", "masked"):
            d[k] = f"<{d[k].shape} bool, {int(d[k].sum()):,} True>"
        return d


class Holdout:
    """Build and evaluate a hide-and-recover split."""

    def __init__(self, labels: np.ndarray, features_shape: tuple[int, int] | None = None,
                 min_component_px: int = MIN_COMPONENT_PX, n_blocks: int = N_BLOCKS,
                 buffer_px: int = BUFFER_PX):
        self.labels = np.asarray(labels, bool)
        self.shape = self.labels.shape if features_shape is None else features_shape
        if self.shape != self.labels.shape:
            raise ValueError("labels and feature grid must share a shape")
        self.min_component_px = min_component_px
        self.n_blocks = n_blocks
        self.buffer_px = buffer_px
        self.comp, self.sizes = components(self.labels, min_component_px)
        self.n_components = len(self.sizes)
        self._assign_folds()
        self._build_folds()

    # ------------------------------------------------------------------
    def _assign_folds(self) -> None:
        """Greedy balanced assignment of spatial blocks to folds.

        Blocks are ranked by fault-pixel mass and handed to the currently
        lightest fold, which keeps the folds comparable in size while keeping
        them spatially disjoint.
        """
        if self.n_components == 0:
            self.block_fold = np.zeros((self.n_blocks, self.n_blocks), np.int32)
            return
        # centroid of every retained component
        cents = ndimage.center_of_mass(np.ones_like(self.comp, bool), self.comp,
                                       np.arange(1, self.n_components + 1))
        mass = np.zeros(self.n_blocks * self.n_blocks)
        block_of_comp = np.zeros(self.n_components, np.int32)
        for i, (r, c) in enumerate(cents):
            bi = min(int(r / self.shape[0] * self.n_blocks), self.n_blocks - 1)
            bj = min(int(c / self.shape[1] * self.n_blocks), self.n_blocks - 1)
            b = bi * self.n_blocks + bj
            block_of_comp[i] = b
            mass[b] += self.sizes[i] if i < len(self.sizes) else 0
        order = np.argsort(-mass)
        fold_load = np.zeros(4)
        self.block_fold = np.zeros(self.n_blocks * self.n_blocks, np.int32)
        for b in order:
            f = int(np.argmin(fold_load))
            self.block_fold[b] = f
            fold_load[f] += mass[b]
        self.comp_fold = self.block_fold[block_of_comp]

    # ------------------------------------------------------------------
    def _build_folds(self) -> None:
        self.folds: list[Fold] = []
        for f in range(4):
            ids = np.flatnonzero(self.comp_fold == f) + 1      # component ids are 1-based
            if len(ids) == 0:
                continue
            truth = np.isin(self.comp, ids)
            # The blind corridor surrounds the WITHHELD segments, not the
            # remaining catalogue. The detector may read the rest of the
            # catalogue in full -- that is what the organizers' input raster
            # gives it -- but must not see the hidden fault or its immediate
            # surroundings, because seeing the trace stop at a boundary is
            # enough to extrapolate it.
            blinded = dilate(truth, self.buffer_px)
            visible = self.labels & ~blinded
            self.folds.append(Fold(
                index=f, truth=truth, visible=visible, blinded=blinded,
                masked=self.labels & ~truth,      # remaining catalogue: masked at scoring
                n_components=int(len(ids)),
                truth_px=int(truth.sum()), visible_px=int(visible.sum()),
            ))

    # ------------------------------------------------------------------
    def detector_input(self, fold: int, features: np.ndarray) -> np.ndarray:
        """Features with the withheld corridor blanked out of every channel.

        Blanking sets the cells to NaN rather than zero so a detector that
        filters the array must decide explicitly what to do about missing data;
        silently zeroing would create an artificial edge at the fold boundary
        that a curvature detector would happily report as a fault.
        """
        f = self.folds[fold]
        out = np.array(features, dtype=np.float32, copy=True)
        out[..., f.blinded] = np.nan
        return out

    def evaluate(self, pred: np.ndarray, fold: int) -> dict:
        """Score one fold under the organizers' pixel-exact masking rule."""
        from .metric import dti
        f = self.folds[fold]
        return dti(pred, f.truth, mask=f.masked).as_dict()

    def summary(self) -> dict:
        return {
            "min_component_px": self.min_component_px,
            "n_blocks": self.n_blocks,
            "buffer_px": self.buffer_px,
            "n_components": int(self.n_components),
            "total_fault_px": int(self.labels.sum()),
            "folds": [f.as_dict() for f in self.folds],
        }
