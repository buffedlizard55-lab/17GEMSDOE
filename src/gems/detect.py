"""Deterministic detectors for the pre-registered hypotheses.

Every detector here is a fixed mathematical operator with no fitted parameters,
which is a deliberate constraint rather than a limitation. The reported history
of this project is that local learned proxies and the public leaderboard
disagreed (leave-one-system-out Pearson correlation -0.20 against the public
score), so a detector whose only free choice is a published physical constant is
far more useful for discriminating between ideas than another tuned model.

Shared conventions:

* Fields are float32 on the 3730 x 3292 grid; NaN marks cells the detector must
  not read (the withheld corridor during evaluation, and the invalid margin of
  the competition raster).
* Every operator is *normalised convolution* aware: NaN cells contribute
  nothing to a filter response, so blanking a corridor does not fabricate an
  edge at the corridor boundary -- which is exactly the artefact that would let
  a curvature detector "recover" a withheld fault that is not there.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Great Basin fault strike. Normal faults in the region trend N to NNE; the
# published step-over catalogue puts ~73 % of right-stepping relay ramps on
# N-to-NNE strands (Giddens & Faulds, 2025). Azimuth here is degrees clockwise
# from north of the fault TRACE.
GB_STRIKE_AZIMUTH_DEG = 10.0
GB_STRIKE_SPREAD_DEG = 35.0

# Published relay-ramp widths, in metres (Giddens & Faulds, 2025: 1.6-3.2 km).
RELAY_RAMP_MIN_M = 1600.0
RELAY_RAMP_MAX_M = 3200.0
PIXEL_M = 100.0


# ---------------------------------------------------------------------------
# normalised convolution helpers
# ---------------------------------------------------------------------------
def _fill(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(field, dtype=np.float32)
    valid = np.isfinite(a)
    return np.where(valid, a, 0.0).astype(np.float32), valid


def smooth(field: np.ndarray, sigma: float, order: int = 0,
           axis: int | None = None) -> np.ndarray:
    """Gaussian (derivative) smoothing that ignores NaN cells.

    Normalised convolution: filtering a zero-filled array and then dividing by
    the filtered validity mask reproduces the smoothing that would have been
    obtained from the valid cells alone.
    """
    a, valid = _fill(field)
    m = valid.astype(np.float32)
    kw = dict(sigma=sigma, mode="nearest")
    if order == 0:
        num = ndimage.gaussian_filter(a, **kw)
        den = ndimage.gaussian_filter(m, **kw)
    else:
        if axis is None:
            num = ndimage.gaussian_filter(a, order=order, **kw)
            den = ndimage.gaussian_filter(m, order=order, **kw)
        else:
            num = ndimage.gaussian_filter1d(a, sigma=sigma, order=order, axis=axis,
                                            mode="nearest")
            den = ndimage.gaussian_filter1d(m, sigma=sigma, order=order, axis=axis,
                                            mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def _derivatives(field: np.ndarray, sigma: float):
    gx = smooth(field, sigma, order=1, axis=1)
    gy = smooth(field, sigma, order=1, axis=0)
    return gx, gy


def _second_derivatives(field: np.ndarray, sigma: float):
    gxx = smooth(field, sigma, order=2, axis=1)
    gyy = smooth(field, sigma, order=2, axis=0)
    # d2/dxdy via two first-order passes, which keeps the NaN handling consistent
    gxy = smooth(smooth(field, sigma, order=1, axis=0), sigma, order=1, axis=1)
    return gxx, gxy, gyy


# ---------------------------------------------------------------------------
# hypothesis detector 1: amplitude-invariant magnetic edge lineaments
# ---------------------------------------------------------------------------
def tilt_from_potential(field: np.ndarray, sigma: float) -> np.ndarray:
    """Tilt angle  theta = atan2(dV/dz, sqrt((dV/dx)^2 + (dV/dy)^2)).

    Here ``dV/dz`` is the field itself, the usual convention for a potential
    field measured on a plane, so ``theta = atan2(V, |grad V|)``. The tilt angle
    is dimensionless and amplitude-independent: a weakly magnetised basement
    block and a strongly magnetised one produce the same tilt signature, which
    is why it is more robust than a raw anomaly amplitude for mapping
    magnetisation boundaries across a survey with variable cover.
    """
    gx, gy = _derivatives(field, sigma)
    hg = np.sqrt(gx ** 2 + gy ** 2)
    return np.arctan2(np.asarray(field, dtype=np.float32), hg).astype(np.float32)


def edge_lineaments(field: np.ndarray, sigma: float) -> np.ndarray:
    """Local-energy map of tilt-angle discontinuities.

    A fault that offsets a magnetic unit puts a *step* in the tilt angle; a
    fault that merely changes its gradient puts a *ridge*. Both are captured by
    the tilt-angle total horizontal derivative, blended with a Hessian ridge
    term so that a coherent lineament outranks an isolated step. The result is
    non-negative and unbounded above; callers rank or threshold it.
    """
    tc = tilt_from_potential(field, sigma)
    gx, gy = _derivatives(tc, sigma)
    thdr = np.sqrt(gx ** 2 + gy ** 2)

    gxx, gxy, gyy = _second_derivatives(tc, sigma)

    # Flip the sign of the Hessian so that "bright lineament" is positive in
    # both polarities of the tilt step, then take the ridge strength properly.
    flip = np.where(gxx + gyy < 0, -1.0, 1.0)
    l1 = np.full(tc.shape, np.nan, np.float32)
    l2 = np.full(tc.shape, np.nan, np.float32)
    tr = (gxx + gyy) * flip
    det = (gxx * gyy - gxy ** 2) * flip
    disc = np.sqrt(np.maximum(tr ** 2 / 4.0 - det, 0.0))
    a = tr / 2.0 + disc
    b = tr / 2.0 - disc
    l1, l2 = np.minimum(a, b), np.maximum(a, b)      # |l1| <= |l2|
    l1 = np.where(np.abs(l1) > np.abs(l2), l2, l1)

    eps = 1e-12
    Rb = np.abs(l1) / (np.abs(l2) + eps)
    S = np.sqrt(l1 ** 2 + l2 ** 2)
    vessel = np.exp(-(Rb ** 2) / (2 * 0.5 ** 2)) * (1.0 - np.exp(-(S ** 2) / (2 * 1.0 ** 2)))
    vessel = np.where(np.isfinite(vessel), vessel, 0.0)

    out = np.sqrt(thdr) * vessel.astype(np.float32)
    out = np.where(np.isfinite(out), out, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def strike_alignment(field_orientation_deg: np.ndarray, target_deg: float,
                     spread_deg: float) -> np.ndarray:
    """Cosine-weighting of a lineament's orientation to the regional strike.

    Faults that carry the regional N-to-NNE Basin-and-Range fabric are the ones
    most likely to be normal faults with geothermal significance; a lineament at
    90 degrees to that fabric is more often a cultural or drainage artefact.
    Being an *orientation*, this is a pi-periodic function, so the angular
    difference is folded into [-90, 90] before weighting.
    """
    d = np.abs(((field_orientation_deg - target_deg) + 90.0) % 180.0 - 90.0)
    return np.cos(np.deg2rad(np.minimum(d / spread_deg, 90.0) * 90.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# hypothesis detector 2: relay-ramp and step-over bridging
# ---------------------------------------------------------------------------
def strand_endpoints(catalogue: np.ndarray, min_strand_px: int = 30,
                     max_turn_deg: float = 60.0):
    """Endpoints of catalogue strands, with the local trace direction at each.

    A skeleton endpoint is a fault pixel with exactly one skeleton neighbour.
    Endpoints whose local direction changes by more than ``max_turn_deg`` within
    a few pixels are dropped: those are skeleton spurs at irregular trace ends,
    and their "direction" is noise.
    """
    from skimage.morphology import skeletonize

    cat = np.asarray(catalogue, bool)
    if not cat.any():
        return []
    skel = skeletonize(cat)
    # neighbour count on the skeleton
    nb = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                          mode="constant") - skel.astype(np.uint8)
    ends = skel & (nb == 1)
    labs, n = ndimage.label(skel, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labs.ravel(), minlength=n + 1)

    ys, xs = np.nonzero(ends)
    out = []
    for y, x in zip(ys, xs):
        lab = labs[y, x]
        if sizes[lab] < min_strand_px:
            continue
        # walk inward along the skeleton to estimate the trace direction
        path = [(y, x)]
        cy, cx = y, x
        seen = {(y, x)}
        for _ in range(12):
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] \
                            and skel[ny, nx] and (ny, nx) not in seen:
                        nxt = (ny, nx)
                        break
                if nxt:
                    break
            if nxt is None:
                break
            seen.add(nxt)
            path.append(nxt)
            cy, cx = nxt
        if len(path) < 5:
            continue
        dy = path[0][0] - path[-1][0]
        dx = path[0][1] - path[-1][1]
        ang = np.rad2deg(np.arctan2(dx, -dy)) % 180.0      # azimuth, 0 = north
        out.append({"y": int(y), "x": int(x), "component": int(lab),
                    "azimuth_deg": float(ang), "strand_px": int(sizes[lab])})
    return out


def relay_bridges(catalogue: np.ndarray, min_gap_px: int = 4, max_gap_px: int = 40,
                  max_strike_mismatch_deg: float = 30.0,
                  strike_window_deg: tuple[float, float] = (0.0, 40.0),
                  min_strand_px: int = 30):
    """Connectors between endpoints of *different* strands across a step-over.

    Gates, each tied to a published observation:

    * the two endpoints must belong to different strands -- otherwise the
      operator is drawing a line down the middle of one fault;
    * their traces must be near-parallel (``max_strike_mismatch_deg``), because a
      relay ramp links two strands of the same fault system, not two unrelated
      faults;
    * the connector direction must be oblique to the strands, so that the pair is
      a step-over rather than a collinear gap;
    * both strands keep the regional N-to-NNE strike.

    The returned field is the set of connector pixels; ``meta`` records the
    measured gap of every accepted pair, which is what
    ``scripts/diagnose_h1_gaps.py`` compares against the published relay-ramp
    width range.
    """
    from skimage.draw import line as draw_line

    ends = strand_endpoints(catalogue, min_strand_px=min_strand_px)
    field = np.zeros(catalogue.shape, bool)
    meta: list[dict] = []
    if len(ends) < 2:
        return field, meta

    pts = np.array([[e["y"], e["x"]] for e in ends], float)
    comp = np.array([e["component"] for e in ends])
    az = np.array([e["azimuth_deg"] for e in ends])

    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for i, j in sorted(tree.query_pairs(max_gap_px)):
        if comp[i] == comp[j]:
            continue
        dy, dx = pts[j] - pts[i]
        gap = float(np.hypot(dy, dx))
        if gap < min_gap_px:
            continue
        mism = abs(((az[i] - az[j]) + 90.0) % 180.0 - 90.0)
        if mism > max_strike_mismatch_deg:
            continue
        if not (strike_window_deg[0] <= az[i] <= strike_window_deg[1]
                and strike_window_deg[0] <= az[j] <= strike_window_deg[1]):
            continue
        connector_az = np.rad2deg(np.arctan2(dx, -dy)) % 180.0
        # a step-over connector runs oblique to the strands: between 25 and 90
        # degrees away from them
        obl = abs(((connector_az - az[i]) + 90.0) % 180.0 - 90.0)
        if obl < 25.0:
            continue
        rr, cc = draw_line(int(pts[i][0]), int(pts[i][1]), int(pts[j][0]), int(pts[j][1]))
        ok = ((rr >= 0) & (rr < field.shape[0]) & (cc >= 0) & (cc < field.shape[1]))
        field[rr[ok], cc[ok]] = True
        meta.append({"i": int(i), "j": int(j), "gap_px": gap,
                     "gap_m": gap * PIXEL_M, "strike_mismatch_deg": float(mism),
                     "connector_azimuth_deg": float(connector_az),
                     "obliquity_deg": float(obl)})
    return field, meta


# ---------------------------------------------------------------------------
# hypothesis detector 3: strain localisation
# ---------------------------------------------------------------------------
def localisation(field: np.ndarray, sigma: float = 2.0,
                 clip: float = 3.0) -> np.ndarray:
    """Robust localisation of a smooth regional field.

    The supplied geodetic strain channels vary over hundreds of kilometres; a
    fault is a *localisation* within them. Subtracting the regional trend and
    dividing by the local spread turns the field into a dimensionless anomaly,
    after which only the upper tail carries information. The clip bounds the
    influence of a single outlier cell, which matters because the geodetic
    channels contain sharp edges at model-domain boundaries.
    """
    a = np.asarray(field, dtype=np.float32)
    background = smooth(a, sigma * 8.0)
    resid = a - background
    scale = np.sqrt(np.maximum(smooth(resid ** 2, sigma * 4.0), 1e-12))
    z = resid / scale
    z = np.clip(z, -clip, clip)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def joint_strain_localisation(shear: np.ndarray, dilatation: np.ndarray,
                              second_invariant: np.ndarray,
                              sigma: float = 2.0) -> np.ndarray:
    """Detector for a dilational jog: shear AND volumetric strain AND a second invariant.

    The published association between Great Basin geothermal systems and
    step-overs and terminations is a statement about *dilational* jogs: a
    transfer zone opens while it shears. Requiring all three strain measures to be
    anomalous at the same place is a much stricter condition than any one of
    them, and it is the conjunction, not the individual channels, that carries
    the hypothesis. The geometric mean is used so that a single large channel
    cannot carry the product alone.
    """
    sh = np.clip(localisation(shear, sigma), 0.0, None)
    dl = np.clip(localisation(dilatation, sigma), 0.0, None)
    iv = np.clip(localisation(second_invariant, sigma), 0.0, None)
    out = np.cbrt(sh * dl * iv)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
def zscore(field: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Standardise a field over its valid cells, for combining detectors."""
    a = np.asarray(field, dtype=np.float32)
    m = np.isfinite(a) if valid is None else (np.isfinite(a) & valid)
    if not m.any():
        return np.zeros_like(a)
    mu = float(a[m].mean())
    sd = float(a[m].std()) or 1.0
    out = (a - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
