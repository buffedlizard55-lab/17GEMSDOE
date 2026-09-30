"""Catalogue-topology hypotheses H1, H2 and H4.

Scientific premise
------------------
The scored truth is the set of faults that are **absent** from the supplied
USGS/INGENIOUS catalogue.  A structural compilation is length- and
expression-thresholded: it stores long, clearly-scarped traces and drops short
structures to which surface expression is weak.  The structures that are
dropped are not random.  In the Great Basin they are concentrated at
interaction zones, because that is where short faults are generated:

  * step-overs / relay ramps   ~32 % of catalogued Great Basin geothermal
    systems (Faulds et al. 2011, 2012; Faulds & Hinz 2015).  Giddens & Faulds
    (2025) measured average relay-ramp widths of 2.1-3.2 km in the Great
    Basin, with 73-78 % of step-overs occurring along faults that strike
    north to north-northeast.
  * normal-fault terminations / tip-lines  ~22-25 %, where "horse-tailing
    generates a myriad of closely-spaced faults".
  * fault intersections (normal x strike-slip/oblique-slip)  ~22 %.
  * major range-front faults (near displacement maxima)  only ~1-6 %.

Source verbatim text and links are recorded in ``docs/sources.html`` and
``evidence/sources.json``; the 32/22/22 split is quoted from
https://gdr.openei.org/files/383/Faulds%20et%20al%202012%20GeoNZ%20Paper.pdf
(verified 2026-09-30) and the relay-ramp width from
https://pangea.stanford.edu/ERE/pdf/IGAstandard/SGW/2025/Giddens.pdf
(verified 2026-09-30).

Important scope limit, stated so it is not over-claimed: the Faulds inventory
describes where *geothermal systems* sit.  The competition labels are
*expert-mapped faults missing from the public catalogue*.  The two are not the
same population.  The link used here is mechanical rather than statistical --
short faults are preferentially created and preferentially dropped at
interaction zones -- and each hypothesis is tested on a hide-and-recover
holdout rather than assumed.

How this differs from what the group had already implemented
------------------------------------------------------------
The sibling "extension arm" (GEMSDOE2) and the 5GEMSDOE "GDR corridor"
candidate placed a **uniform offset band along the entire catalogue**.  That is
dilation, not topology: it does not distinguish a mid-segment from a tip, a
single fault from a step-over, or a gap from a continuous trace.  The
detectors here build an explicit skeleton **graph** and act only on its
degrees of freedom -- endpoints, tip pairs and junctions -- and they emit
pixels in the *empty space between* structures, which is where a dropped
connecting fault lives and where a dilation can never place a pixel.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["skeletonise", "skeleton_graph", "trace_tips", "trace_junctions",
           "detect_bridges", "detect_splays", "detect_junctions",
           "rasterise_segments", "line_pixels"]

# 8-neighbour offsets
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeletonise(binary: np.ndarray) -> np.ndarray:
    """1-pixel-wide medial skeleton (Zhang-Suen thinning)."""
    from skimage.morphology import skeletonize
    return skeletonize(np.asarray(binary, dtype=bool))


def _neighbour_count(sk: np.ndarray) -> np.ndarray:
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return ndimage.convolve(sk.astype(np.uint8), k, mode="constant", cval=0)


def skeleton_graph(sk: np.ndarray):
    """Degrees of the skeleton graph and the tip / junction pixel sets."""
    deg = _neighbour_count(sk)
    deg[~sk] = 0
    tips = np.argwhere(sk & (deg == 1))
    junc = np.argwhere(sk & (deg >= 3))
    return deg, tips, junc


def trace_tips(sk: np.ndarray, tips: np.ndarray, back_px: int = 12):
    """For each tip, walk ``back_px`` steps along the skeleton to get a tangent.

    Returns a list of dicts with ``tip`` (r, c) and ``dir`` = unit vector
    pointing *outward* beyond the tip (away from the fault body), plus
    ``strike_deg`` in [0, 180).
    """
    sk = np.asarray(sk, dtype=bool)
    h, w = sk.shape
    out = []
    for (r0, c0) in tips:
        prev = None
        cur = (int(r0), int(c0))
        path = [cur]
        for _ in range(back_px):
            nxt = None
            for dr, dc in _NB:
                rr, cc = cur[0] + dr, cur[1] + dc
                if 0 <= rr < h and 0 <= cc < w and sk[rr, cc] and (rr, cc) != prev \
                        and (rr, cc) not in path:
                    nxt = (rr, cc)
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            path.append(cur)
        if len(path) < 3:
            continue
        # direction from the body toward the tip
        d = np.array([path[0][0] - path[-1][0], path[0][1] - path[-1][1]], float)
        n = np.hypot(*d)
        if n < 1e-9:
            continue
        d /= n
        # skeleton tangent (the fault's local strike) = direction along the path
        t = np.array([path[0][0] - path[min(len(path) - 1, back_px)][0],
                      path[0][1] - path[min(len(path) - 1, back_px)][1]], float)
        tn = np.hypot(*t)
        t = t / tn if tn > 1e-9 else d
        strike = np.degrees(np.arctan2(t[0], t[1])) % 180.0
        out.append({"tip": (int(r0), int(c0)), "dir_rc": d, "strike_deg": float(strike),
                    "path": path})
    return out


def trace_junctions(sk: np.ndarray, junc: np.ndarray, arm_px: int = 15):
    """For each junction pixel, measure the outward directions of its arms."""
    sk = np.asarray(sk, dtype=bool)
    h, w = sk.shape
    out = []
    for (r0, c0) in junc:
        arms = []
        for dr, dc in _NB:
            rr, cc = int(r0) + dr, int(c0) + dc
            if not (0 <= rr < h and 0 <= cc < w) or not sk[rr, cc]:
                continue
            prev, cur = (int(r0), int(c0)), (rr, cc)
            path = [cur]
            for _ in range(arm_px):
                nxt = None
                for dr2, dc2 in _NB:
                    r2, c2 = cur[0] + dr2, cur[1] + dc2
                    if 0 <= r2 < h and 0 <= c2 < w and sk[r2, c2] \
                            and (r2, c2) != prev and (r2, c2) not in path:
                        nxt = (r2, c2)
                        break
                if nxt is None:
                    break
                prev, cur = cur, nxt
                path.append(cur)
            if len(path) >= 3:
                v = np.array([path[-1][0] - r0, path[-1][1] - c0], float)
                n = np.hypot(*v)
                if n > 1e-9:
                    arms.append(v / n)
        if len(arms) >= 2:
            out.append({"junction": (int(r0), int(c0)), "arms": np.array(arms)})
    return out


def line_pixels(r0, c0, r1, c1, max_px: int = 4000):
    """Integer pixel coordinates on the straight segment between two points."""
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    if n > max_px:
        n = max_px
    rr = np.rint(np.linspace(r0, r1, n)).astype(np.int64)
    cc = np.rint(np.linspace(c0, c1, n)).astype(np.int64)
    return rr, cc


def rasterise_segments(shape, segments, thickness: int = 0):
    """Paint a list of ((r0,c0),(r1,c1)) segments into a boolean raster."""
    out = np.zeros(shape, dtype=bool)
    h, w = shape
    for (p0, p1) in segments:
        rr, cc = line_pixels(p0[0], p0[1], p1[0], p1[1])
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        out[rr[ok], cc[ok]] = True
    if thickness > 0:
        d = np.arange(-thickness, thickness + 1)
        yy, xx = np.meshgrid(d, d, indexing="ij")
        out = ndimage.binary_dilation(out, structure=((yy * yy + xx * xx) <= thickness ** 2))
    return out


def _angdiff_180(a, b):
    """Absolute difference between two axial (mod 180) angles, in degrees."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


# ---------------------------------------------------------------------------
# H1: step-over / relay-ramp bridges
# ---------------------------------------------------------------------------
def detect_bridges(catalogue: np.ndarray, max_gap_px: int = 50,
                   min_gap_px: int = 4, max_strike_mismatch_deg: float = 35.0,
                   max_connection_turn_deg: float = 45.0,
                   min_strand_px: int = 30):
    """Predict connecting structures between en-echelon fault tips.

    Parameters are anchored to published Great Basin measurements:
    Giddens & Faulds (2025) report average relay-ramp widths of 2.1-3.2 km
    (21-32 px at 100 m) and 73-78 % of step-overs on N-to-NNE-striking faults.
    ``max_gap_px=50`` therefore covers the measured ramp-width population with
    headroom, and the strike-mismatch gate encodes the near-parallel main
    strands of a relay ramp.

    Two gates are essential and were both found by unit-testing rather than by
    inspection:

    ``different strands``
        The two tips must belong to **different** connected components of the
        catalogue.  Without this gate the detector "bridges" the two ends of a
        single fault -- drawing a straight line down the middle of a fault that
        is already mapped, which is both useless (it lands on masked pixels)
        and a sign the detector is not modelling step-overs at all.

    ``connection turn``
        The tip-to-tip direction must lie within ``max_connection_turn_deg`` of
        the local strike.  A relay ramp links two strands that are roughly
        collinear, so the connecting structure runs *along* the strike axis; a
        pair of tips that are side by side across strike is a different feature
        (handled by the wide sensitivity variant) and in the narrow form is
        rejected.
    """
    sk = skeletonise(catalogue)
    _, tips, _ = skeleton_graph(sk)
    tips, strand_px = _filter_tips(sk, tips, min_strand_px)
    info = trace_tips(sk, tips)
    comp_of = _component_ids_at(sk, [a["tip"] for a in info])

    # Spatial pre-filter: only tip pairs within max_gap can possibly qualify.
    # The full catalogue yields ~6900 tips; the naive double loop is 24 million
    # comparisons and dominated runtime, while only ~0.85 % of pairs are close
    # enough to matter.
    from scipy.spatial import cKDTree
    pts = np.array([a["tip"] for a in info], dtype=float)
    pairs = cKDTree(pts).query_pairs(max_gap_px, output_type="ndarray") if len(pts) else np.zeros((0, 2), int)

    segs, meta = [], []
    for i, j in pairs:
        a, b = info[i], info[j]
        if comp_of[i] == comp_of[j]:
            continue                      # same strand: nothing is missing
        dr = b["tip"][0] - a["tip"][0]
        dc = b["tip"][1] - a["tip"][1]
        gap = float(np.hypot(dr, dc))
        if not (min_gap_px <= gap <= max_gap_px):
            continue
        mismatch = _angdiff_180(a["strike_deg"], b["strike_deg"])
        if mismatch > max_strike_mismatch_deg:
            continue
        conn = np.degrees(np.arctan2(dr, dc)) % 180.0
        align = _angdiff_180(conn, a["strike_deg"])
        if align > max_connection_turn_deg:
            continue
        score = (1.0 - mismatch / 180.0) * (1.0 - align / 90.0)
        segs.append((a["tip"], b["tip"]))
        meta.append({"a": a["tip"], "b": b["tip"], "gap_px": gap,
                     "gap_km": gap * 0.1, "strike_mismatch_deg": float(mismatch),
                     "connection_align_deg": float(align), "score": float(score)})
    field = rasterise_segments(catalogue.shape, segs, thickness=0)
    field = _thin(field)
    return field.astype(np.float32), meta


def _filter_tips(sk: np.ndarray, tips: np.ndarray, min_strand_px: int):
    """Drop tips that belong to strands shorter than ``min_strand_px``.

    Justification is geological, not statistical: a step-over or a horse-tail
    is an interaction between two *faults*.  A two-pixel speck in a rasterised
    compilation is a digitising artefact or a very short mapped break, and
    bridging it produces a structure with no mechanical meaning.  The gate also
    has a large practical effect: the supplied catalogue yields 6938 tips for
    3199 components, i.e. a long tail of fragments, and without the gate the
    detector is dominated by them.
    """
    if min_strand_px <= 0 or len(tips) == 0:
        return tips, None
    lab, n = ndimage.label(np.asarray(sk, dtype=bool),
                           structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(lab.ravel())
    keep = np.array([sizes[lab[r, c]] >= min_strand_px for (r, c) in tips], dtype=bool)
    return tips[keep], sizes


def _component_ids_at(sk: np.ndarray, points):
    """Connected-component id (8-connectivity) of each skeleton point."""
    lab, _ = ndimage.label(np.asarray(sk, dtype=bool),
                           structure=np.ones((3, 3), dtype=bool))
    return [int(lab[r, c]) for (r, c) in points]


# ---------------------------------------------------------------------------
# H2: tip-line horse-tail splays
# ---------------------------------------------------------------------------
def detect_splays(catalogue: np.ndarray, splay_len_px: int = 25,
                  angles_deg=(20.0, 30.0, 40.0), sides=(-1.0, 1.0),
                  min_strand_px: int = 30):
    """Predict horse-tail splays fanning beyond each catalogue tip.

    Faulds et al. (2011, 2012) describe terminations as hosting "a myriad of
    closely-spaced faults" generated by horse-tailing; those are exactly the
    short structures a compilation drops.  Splays are drawn from each tip at
    +/- the given angles to the local strike, out to ``splay_len_px``.
    """
    sk = skeletonise(catalogue)
    _, tips, _ = skeleton_graph(sk)
    tips, _ = _filter_tips(sk, tips, min_strand_px)
    info = trace_tips(sk, tips)
    segs, meta = [], []
    for a in info:
        (r0, c0) = a["tip"]
        th = np.radians(a["strike_deg"])
        # unit vector along strike, in (row, col)
        u = np.array([np.sin(th), np.cos(th)])
        n = np.array([-u[1], u[0]])  # normal
        for side in sides:
            for ang in angles_deg:
                ar = np.radians(ang)
                v = np.cos(ar) * u * side + np.sin(ar) * n
                p1 = (int(round(r0 + v[0] * splay_len_px)),
                      int(round(c0 + v[1] * splay_len_px)))
                segs.append(((int(r0), int(c0)), p1))
                meta.append({"tip": (int(r0), int(c0)), "side": float(side),
                             "angle_deg": float(ang), "strike_deg": a["strike_deg"]})
    field = rasterise_segments(catalogue.shape, segs, thickness=0)
    field = _thin(field)
    return field.astype(np.float32), meta


# ---------------------------------------------------------------------------
# H4: fault-intersection / dilational quadrant connections
# ---------------------------------------------------------------------------
def detect_junctions(catalogue: np.ndarray, arm_len_px: int = 25,
                     min_arm_angle_deg: float = 30.0,
                     min_strand_px: int = 30):
    """Predict short connecting faults across catalogue junctions.

    Faulds et al. (2011, 2012): intersections between normal faults and
    transversely oriented strike-slip or oblique-slip faults are a favourable
    setting for ~22 % of Great Basin geothermal systems, and "multiple minor
    faults typically connect major structures".  Where two catalogue strands
    meet at a high angle, this detector emits a short connector bisecting the
    wider of the two opening quadrants -- the dilational quadrant in which
    fluids concentrate.
    """
    sk = skeletonise(catalogue)
    _, _, junc = skeleton_graph(sk)
    if min_strand_px > 0 and len(junc):
        # keep only junctions whose surrounding strand is long enough to be a
        # real mapped fault rather than a digitising speck
        lab, _ = ndimage.label(np.asarray(sk, dtype=bool),
                               structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(lab.ravel())
        junc = np.array([p for p in junc if sizes[lab[p[0], p[1]]] >= min_strand_px],
                        dtype=int).reshape(-1, 2)
    info = trace_junctions(sk, junc, arm_px=max(8, arm_len_px // 2))
    segs, meta = [], []
    for j in info:
        arms = j["arms"]
        # find the widest-angle pair of arms
        best = None
        for i in range(len(arms)):
            for k in range(i + 1, len(arms)):
                ang = np.degrees(np.arccos(np.clip(np.dot(arms[i], arms[k]), -1, 1)))
                if best is None or ang > best[0]:
                    best = (ang, i, k)
        if best is None or best[0] < min_arm_angle_deg:
            continue
        ang, i, k = best
        bis = arms[i] + arms[k]
        nb = np.hypot(*bis)
        if nb < 1e-9:
            continue
        bis = bis / nb
        r0, c0 = j["junction"]
        p1 = (int(round(r0 + bis[0] * arm_len_px)), int(round(c0 + bis[1] * arm_len_px)))
        segs.append(((int(r0), int(c0)), p1))
        meta.append({"junction": (int(r0), int(c0)), "arm_angle_deg": float(ang)})
    field = rasterise_segments(catalogue.shape, segs, thickness=0)
    field = _thin(field)
    return field.astype(np.float32), meta


def _thin(field: np.ndarray) -> np.ndarray:
    """Collapse a boolean field to a 1-pixel centreline.

    Skeletonisation is the correct reduction here: several bridges or splays can
    overlap near a tip, and the metric charges for every emitted pixel, so the
    redundant interior must not survive.

    (An earlier version of this helper accepted a ``radius_px`` argument that it
    then ignored -- a latent bug, since a caller could reasonably expect a
    non-maximum-suppression radius to change the result. The parameter is gone.)
    """
    if not field.any():
        return field
    try:
        return skeletonise(field)
    except Exception:
        return field
