"""Hypothesis H3: amplitude-invariant magnetic edge network from the tilt angle.

Physical argument
-----------------
Band 6 of the supplied stack is ``tc`` -- documented in the raster's own
per-band tags as "Tilt angle or total curvature - magnetic field derivative for
edge detection".  The tilt angle is

    theta = atan2(dT/dz, |dT/dx, dy|)

and its defining property is that it is *amplitude independent*: a weakly
magnetised unit and a strongly magnetised one produce the same tilt response
over their edges.  The horizontal derivative of the tilt angle peaks directly
over the edge of the magnetised body.  Because a fault trace in this region
commonly juxtaposes two different magnetic basement/lithologic units, that edge
is present even where the fault has **no topographic scarp** -- which is the
population the labelled catalogue is known to miss, and the population that
scarp/ridge detectors (the group's many "h*scarp*" and "h*ridge*" candidates)
cannot reach by construction.

Difference from the existing work
---------------------------------
The magnetic channels have been fed to the CNN ensemble as raw amplitude
inputs (bands 3, 9, 14 and 6 were all present in the 19-band stack the 0.1563
run consumed).  Feeding a band to a CNN is not the same as applying the
operator the band was built for.  Here the *explicit* amplitude-normalised edge
operator is computed and used directly, which is a different inductive bias: it
cannot be dominated by the high-amplitude volcanic centres that dominate the
raw inputs.

Honest limitation: this is the weakest novelty claim of the five hypotheses,
because the source band was already present in the input stack.  It is
registered and tested anyway so that the negative or positive result is on the
record rather than assumed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["gradient_magnitude", "tilt_horizontal_derivative",
           "upward_continuation", "analytic_signal", "edge_score_field"]


def gradient_magnitude(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """|grad a| with optional Gaussian pre-smoothing; NaNs ignored."""
    x = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    if sigma > 0:
        x = ndimage.gaussian_filter(x, sigma=sigma, mode="nearest")
    gy, gx = np.gradient(x)
    return np.hypot(gx, gy)


def tilt_horizontal_derivative(tilt_deg_or_rad: np.ndarray, sigma: float = 1.5,
                               unwrap: bool = True) -> np.ndarray:
    """Total horizontal derivative of the tilt angle (an edge detector).

    The tilt angle is an angle, so a naive gradient is discontinuous where it
    wraps.  We therefore smooth first (which also suppresses flight-line
    corrugation in the GeoDAWN products) and then take the gradient; the
    resulting maxima sit on the edges of magnetic sources.
    """
    x = np.nan_to_num(np.asarray(tilt_deg_or_rad, dtype=np.float64), nan=0.0)
    x = ndimage.gaussian_filter(x, sigma=sigma, mode="nearest")
    if unwrap:
        # Remove 180-degree jumps in a robust way for degree-valued tilt.
        m = np.abs(x) > 90.0
        if m.any():
            x = np.where(m, x - np.sign(x) * 180.0, x)
    return gradient_magnitude(x, sigma=0.0)


def analytic_signal(a: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """2-D analytic signal amplitude sqrt((dA/dx)^2 + (dA/dy)^2).

    With no vertical derivative available on a 2-D grid this is the standard
    practical substitute; it also normalises amplitude.
    """
    return gradient_magnitude(a, sigma=sigma)


def upward_continuation(a: np.ndarray, dz_px: float) -> np.ndarray:
    """Exponential (Wiener) upward continuation by ``dz_px`` pixels.

    Used only inside H5 to build the multi-level source-depth estimate.
    """
    x = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)
    F = np.fft.fft2(x)
    ky = np.fft.fftfreq(x.shape[0])[:, None]
    kx = np.fft.fftfreq(x.shape[1])[None, :]
    k = np.hypot(kx, ky)
    return np.real(np.fft.ifft2(F * np.exp(-2.0 * np.pi * k * dz_px)))


def edge_score_field(tc: np.ndarray, tmi: np.ndarray | None = None,
                     rtp: np.ndarray | None = None, sigma: float = 1.5) -> np.ndarray:
    """Combined amplitude-invariant edge score from the tilt angle (+ magnets).

    Returns a non-negative field; larger = more likely a magnetic source edge.
    The score is the product of the tilt-angle horizontal derivative and a
    light smoothing, optionally multiplied by the analytic-signal amplitude of
    ``tmi`` so that edges of *strong* sources rank above numerical noise in
    quiet areas.  The product rather than the sum is deliberate: an edge must
    be supported by both an amplitude-invariant edge operator and a real
    gradient in the data to survive.
    """
    thdr = tilt_horizontal_derivative(tc, sigma=sigma)
    score = thdr.copy()
    if tmi is not None:
        asig = analytic_signal(tmi, sigma=sigma)
        # Robust scale: compare each field to its own 95th percentile so that
        # neither term simply dominates by having larger raw units.
        t = np.percentile(thdr, 95) or 1.0
        a95 = np.percentile(asig, 95) or 1.0
        score = (thdr / t) * np.sqrt(np.clip(asig / a95, 0.0, None))
    if rtp is not None:
        sig = analytic_signal(rtp, sigma=sigma)
        s95 = np.percentile(sig, 95) or 1.0
        score = score * np.sqrt(np.clip(sig / s95, 0.0, None))
    # Guard the cast: after the ratio the values are O(1), but a percentile of
    # zero (a saturated band) would otherwise blow them past float32 range and
    # silently emit inf, which is not representable in the submission format.
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score.astype(np.float32)
