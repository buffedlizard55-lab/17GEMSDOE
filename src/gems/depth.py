"""Hypothesis H5: reconstruct the magnetic source-depth band the description promises.

Verified discrepancy this hypothesis responds to
-----------------------------------------------
The GEMS problem description (page 967, section "Provided features", verified
2026-09-30) lists the contents of ``training_features.tif`` and includes:

    "Magnetics including reduced-to-pole magnetic anomaly, total magnetic
     intensity, the vertical and horizontal slope of total magnetic intensity,
     and the top-of-crustal magnetic source depth estimate"

The delivered 19-band raster contains **no source-depth band**.  Its bands are
(read from the file's own per-band tags, not from documentation):

    1 mag_anom            8  geod_dilaterate      15 depth_to_base_surf
    2 rtp                 9  tmi_vg               16 ieq_n100a15
    3 tmi_hg             10  deq_n100a15          17 cond_surf
    4 geod_2ndinv        11  iso_grav_anom_vg     18 iso_grav_anom_hg
    5 iso_grav_anom_slope 12 det_elev             19 det_elev_slope
    6 tc                 13  iso_grav_anom
    7 geod_shearrate     14  tmi

The same absence was independently noted by a sibling repository
(``8GEMSDOE/reports/holdout_real.json``: ``"no magnetic-source-depth band
exists"``).  This is flagged as a data irregularity in ``docs/findings.html``.

What H5 does
------------
Rather than treating the missing band as an obstacle, H5 reconstructs it from
the supplied magnetics using the standard spectral depth method (Tanaka et al.,
1999; Okubo et al., 1985): radially averaged log-power spectra of overlapping
windows give the centroid depth from the low-wavenumber slope and the top depth
from the high-wavenumber slope.  The resulting depth map is then used as a
*selector*: an amplitude-normalised magnetic edge is only promoted to a fault
prediction where the estimated source depth is deep enough to be a basement
structure, rather than a surficial volcanic or playa edge.

Why that should help rather than hurt
-------------------------------------
The evidence in the group's own audit is that generic structural/geophysical
detectors had essentially no out-of-sample skill (``8GEMSDOE/reports/
holdout_struct.json``: structural 0.0017 vs random 0.1704).  A plausible reason
is that the Great Basin magnetic field is dominated by shallow volcanic and
surficial sources whose edges have nothing to do with the fault population.  A
depth filter is a physically motivated way to remove exactly that population.

Status note: this module is implemented and unit-tested on synthetic data.  It
is *not* run on the full competition grid in this session -- the sandbox has 2
CPU cores and 3 GB of RAM and the full-surface spectral sweep is the most
expensive of the five candidates.  It is therefore registered as
"pre-registered, not yet executed" in the experiment ledger, which is the
honest status, rather than reported as a result.
"""

from __future__ import annotations

import numpy as np

__all__ = ["radial_power_spectrum", "windowed_source_depth",
           "depth_to_top_from_spectrum"]


def radial_power_spectrum(window: np.ndarray, dx: float = 1.0,
                          detrend: bool = True):
    """Radially averaged log power spectrum of a 2-D window.

    Returns ``(k, log_power)`` where ``k`` is the radial wavenumber in cycles
    per pixel.
    """
    x = np.asarray(window, dtype=np.float64)
    x = np.nan_to_num(x, nan=float(np.nanmean(x)) if np.isfinite(x).any() else 0.0)
    if detrend:
        x = x - x.mean()
    # 2-D Hann taper to limit spectral leakage
    wy = np.hanning(x.shape[0])[:, None]
    wx = np.hanning(x.shape[1])[None, :]
    F = np.fft.fftshift(np.fft.fft2(x * wy * wx))
    P = (F.real ** 2 + F.imag ** 2)

    fy = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=dx))
    fx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=dx))
    KY, KX = np.meshgrid(fy, fx, indexing="ij")
    K = np.hypot(KX, KY)

    nbins = max(x.shape) // 2
    edges = np.linspace(0.0, K.max(), nbins + 1)
    idx = np.digitize(K.ravel(), edges) - 1
    k_out, p_out = [], []
    Pr = P.ravel()
    for b in range(nbins):
        m = idx == b
        if m.sum() < 4:
            continue
        k_out.append(0.5 * (edges[b] + edges[b + 1]))
        p_out.append(float(Pr[m].mean()))
    k_out = np.asarray(k_out)
    p_out = np.asarray(p_out)
    good = (k_out > 0) & (p_out > 0) & np.isfinite(p_out)
    return k_out[good], np.log(p_out[good])


def depth_to_top_from_spectrum(k: np.ndarray, logp: np.ndarray,
                               n_low: int = 12, n_high: int = 12):
    """Two-slope spectral depth estimate from one radial spectrum.

    Returns ``(depth_top, depth_centroid, depth_bottom)`` in pixels.

    Convention, stated explicitly because the literature uses more than one and
    an earlier revision of this module was internally inconsistent about it:

        ln P(k) = const - 4*pi*z_centroid*k     (low-wavenumber slope)
        ln P(k) = const - 4*pi*z_top*k          (high-wavenumber slope)
        z_bottom = 2*z_centroid - z_top         (Okubo et al., 1985)

    i.e. the *power* spectrum (not its square root) is fitted and both depths are
    ``-slope / (4*pi)``.  This is the form most commonly implemented for the
    magnetic-spectral-depth method; some papers fit ``ln sqrt(P)`` and quote
    ``-slope / (2*pi)``, which is the same model.  Both conventions are recorded
    here so a reader can check the arithmetic against whichever source they
    hold.  Note this is a *unit-tested estimator on synthetic spectra*, not a
    calibrated depth inversion -- H5 was pre-registered and never executed, so no
    result anywhere in this repository depends on the absolute scale.
    """
    if len(k) < (n_low + n_high + 4):
        return None
    # low-wavenumber (deep) part -> centroid depth z0
    kl, pl = k[:n_low], logp[:n_low]
    ml = np.polyfit(kl, pl, 1)[0]
    z0 = -ml / (4.0 * np.pi)
    # high-wavenumber (shallow) part -> top depth zt
    kh, ph = k[-n_high:], logp[-n_high:]
    mh = np.polyfit(kh, ph, 1)[0]
    zt = -mh / (4.0 * np.pi)
    if not np.isfinite(z0) or not np.isfinite(zt):
        return None
    return float(max(zt, 0.0)), float(max(z0, 0.0)), float(max(2.0 * z0 - zt, 0.0))


def windowed_source_depth(field: np.ndarray, win: int = 256, stride: int = 128,
                          valid: np.ndarray | None = None, dx: float = 1.0):
    """Gridwise top-depth map.  Returns (depth_map, count_map).

    ``depth_map`` is in pixels (100 m each at the competition resolution).
    Windows that are mostly outside ``valid`` are skipped.
    """
    h, w = field.shape
    dsum = np.zeros((h, w), dtype=np.float64)
    dcnt = np.zeros((h, w), dtype=np.int32)
    ys = list(range(0, max(1, h - win + 1), stride))
    xs = list(range(0, max(1, w - win + 1), stride))
    for y0 in ys:
        for x0 in xs:
            sl = (slice(y0, min(y0 + win, h)), slice(x0, min(x0 + win, w)))
            if valid is not None:
                frac = float(valid[sl].mean())
                if frac < 0.6:
                    continue
            sub = field[sl]
            if sub.shape[0] < 32 or sub.shape[1] < 32:
                continue
            k, lp = radial_power_spectrum(sub, dx=dx)
            est = depth_to_top_from_spectrum(k, lp)
            if est is None:
                continue
            zt = est[0]
            dsum[sl] += zt
            dcnt[sl] += 1
    out = np.zeros((h, w), dtype=np.float32)
    nz = dcnt > 0
    out[nz] = (dsum[nz] / dcnt[nz]).astype(np.float32)
    return out, dcnt
