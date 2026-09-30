"""Tests for the format gate, the emission helpers and the depth/edge transforms.

The format-gate tests are the important ones: the operator's submission was rejected
with "Predicted values must be in range [0, 1]", and `validate_submission` is the
control that prevents a recurrence. Each failure mode that could produce that message
gets an explicit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems import depth, edges, emit, io  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: a miniature but structurally valid competition grid
# ---------------------------------------------------------------------------
SHAPE = (80, 100)
TRANSFORM = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0)


def _write(path, array, dtype="float32", nodata=float("nan"), crs="EPSG:32611",
           transform=TRANSFORM, count=1):
    profile = {"driver": "GTiff", "width": SHAPE[1], "height": SHAPE[0],
               "count": count, "dtype": dtype, "crs": crs, "transform": transform}
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        if count == 1:
            ds.write(array, 1)
        else:
            for i in range(1, count + 1):
                ds.write(array, i)
    return path


@pytest.fixture()
def sample(tmp_path):
    """A sample submission: valid probabilities in the middle band of rows, NaN elsewhere."""
    a = np.full(SHAPE, np.nan, dtype=np.float32)
    a[20:60, :] = 0.0
    return _write(tmp_path / "sample_submission.tif", a)


# ---------------------------------------------------------------------------
# the [0,1] gate
# ---------------------------------------------------------------------------
def test_valid_file_passes_every_check(sample, tmp_path):
    # The whole footprint must carry a finite value: NaN *inside* the footprint
    # is a rejection, which is exactly what the sample's finite rows define.
    a = np.full(SHAPE, np.nan, dtype=np.float32)
    a[20:60, :] = 0.0
    a[30:40, 30:40] = 1.0
    p = _write(tmp_path / "ok.tif", a)
    r = io.validate_submission(p, sample)
    assert r["ok"], r["errors"]
    assert all(c["ok"] for c in r["checks"].values())


def test_value_above_one_is_rejected(sample, tmp_path):
    a = np.full(SHAPE, np.nan, dtype=np.float32)
    a[20:60, :] = 0.0
    a[30, 30] = 1.5
    p = _write(tmp_path / "high.tif", a)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("values_in_0_1" in e for e in r["errors"])


def test_negative_value_is_rejected(sample, tmp_path):
    a = np.full(SHAPE, np.nan, dtype=np.float32)
    a[20:60, :] = 0.0
    a[30, 30] = -9999.0                      # the classic nodata sentinel misuse
    p = _write(tmp_path / "neg.tif", a)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("values_in_0_1" in e or "no_negative" in e for e in r["errors"])


def test_nan_inside_the_footprint_is_rejected(sample, tmp_path):
    a = np.full(SHAPE, np.nan, dtype=np.float32)
    a[20:60, :] = 0.0
    a[40, 40] = np.nan
    p = _write(tmp_path / "hole.tif", a)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("nan_only_outside_footprint" in e for e in r["errors"])


def test_wrong_dtype_is_rejected(sample, tmp_path):
    a = np.zeros(SHAPE, dtype=np.uint8)
    p = _write(tmp_path / "u8.tif", a, dtype="uint8", nodata=None)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("dtype_float32" in e for e in r["errors"])


def test_two_bands_are_rejected(sample, tmp_path):
    a = np.zeros(SHAPE, dtype=np.float32)
    p = _write(tmp_path / "two.tif", a, count=2, nodata=None)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("single_band" in e for e in r["errors"])


def test_wrong_crs_is_rejected(sample, tmp_path):
    a = np.zeros(SHAPE, dtype=np.float32)
    p = _write(tmp_path / "crs.tif", a, crs="EPSG:4326", nodata=None)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("crs" in e for e in r["errors"])


def test_wrong_transform_is_rejected(sample, tmp_path):
    a = np.zeros(SHAPE, dtype=np.float32)
    bad = Affine(100.0, 0.0, 243350.0, 0.0, -100.0, 4508550.0 - 100)
    p = _write(tmp_path / "tf.tif", a, transform=bad, nodata=None)
    r = io.validate_submission(p, sample)
    assert not r["ok"]
    assert any("bounds" in e or "transform" in e for e in r["errors"])


def test_write_submission_clips_out_of_range_values(sample, tmp_path):
    """The writer must make a bad field safe rather than pass it through."""
    field = np.full(SHAPE, np.nan)
    field[20:60, :] = 0.0
    field[30, 30] = 5.0
    field[31, 30] = -3.0
    out = tmp_path / "clipped.tif"
    r = io.write_submission(out, field, sample)
    assert r["ok"], r["errors"]
    assert r["min"] >= 0.0 and r["max"] <= 1.0
    assert r["repairs"]["clipped_into_range"] == 2
    with rasterio.open(out) as ds:
        a = ds.read(1)
    assert a[30, 30] == pytest.approx(1.0)      # clipped up
    assert a[31, 30] == pytest.approx(0.0)      # clipped down


# ---------------------------------------------------------------------------
# emission helpers
# ---------------------------------------------------------------------------
def test_thin_field_produces_spaced_nodes_not_a_solid_band():
    score = np.zeros((40, 40))
    mask = np.zeros((40, 40), bool)
    mask[:, :] = False
    mask[5:35, 20] = True                     # a 30-pixel vertical band
    score[mask] = 1.0
    thin = emit.thin_field(score, mask, radius_px=3)
    assert 0 < thin.sum() < 30
    assert (thin & ~mask).sum() == 0
    # kept pixels must be at least 2 px apart along the band
    idx = np.flatnonzero(thin[:, 20])
    assert np.all(np.diff(idx) >= 2)


def test_top_k_returns_requested_count_when_support_allows():
    score = np.random.default_rng(0).random((30, 30))
    m = emit.top_k_mask(score, 50)
    assert m.sum() == 50


def test_top_k_default_eligibility_is_finite_not_positive():
    """The default treats every finite pixel as available, including zeros.

    That is deliberate: callers that need "only pixels the detector supports" pass
    an explicit eligibility mask, as run_hypotheses.py does with ``field > 0``.
    """
    score = np.zeros((30, 30))
    score[0, :5] = 1.0
    assert emit.top_k_mask(score, 100).sum() == 100          # finite everywhere
    eligible = score > 0
    assert emit.top_k_mask(score, 100, eligible=eligible).sum() == 5


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------
def test_tilt_horizontal_derivative_peaks_on_a_step_edge():
    x = np.zeros((64, 64))
    x[:, 32:] = 1.0                           # a step (a magnetic contact)
    thdr = edges.tilt_horizontal_derivative(x, sigma=1.0)
    assert thdr[:, 30:36].max() > 5.0 * thdr[:, :20].max() + 1e-9


def test_edge_score_field_is_finite_and_non_negative():
    rng = np.random.default_rng(2)
    tc = rng.normal(size=(48, 48))
    tmi = rng.normal(size=(48, 48))
    s = edges.edge_score_field(tc, tmi=tmi)
    assert np.isfinite(s).all()
    assert (s >= 0).all()


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------
def test_radial_power_spectrum_shapes_and_positivity():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(64, 64))
    k, lp = depth.radial_power_spectrum(w)
    assert len(k) == len(lp) > 4
    assert np.all(k > 0)


def test_depth_estimator_recovers_a_known_synthetic_spectrum():
    """Round-trip: recover z_t and z_0 from a spectrum built to the module's own model.

    The module documents the convention ln P(k) = const - 4*pi*z*k for BOTH the
    low- and high-wavenumber slopes (Okubo et al., 1985 form, fitting the power
    spectrum).  This test builds a spectrum that obeys exactly that model and
    checks the estimator inverts it -- which is what a unit test can legitimately
    assert.  It does NOT assert that the convention is the right one physically;
    that is a literature question, and H5 was never executed, so nothing
    downstream depends on it.

    History: an earlier revision of this test built the high-wavenumber branch
    with a 2*pi slope while the code used 4*pi.  The test caught a genuine
    internal inconsistency between depth.py's docstring and its arithmetic.
    """
    z_t, z_0 = 3.0, 9.0                      # pixels
    k = np.linspace(0.004, 0.25, 60)
    logp = np.empty_like(k)
    low = k < 0.06
    logp[low] = 5.0 - 4.0 * np.pi * z_0 * k[low]
    logp[~low] = (5.0 - 4.0 * np.pi * z_0 * 0.06) - 4.0 * np.pi * z_t * (k[~low] - 0.06)
    est = depth.depth_to_top_from_spectrum(k, logp, n_low=12, n_high=12)
    assert est is not None
    zt_hat, z0_hat, zb_hat = est
    assert z0_hat == pytest.approx(z_0, abs=0.35)
    assert zt_hat == pytest.approx(z_t, abs=0.35)
    assert zb_hat == pytest.approx(2 * z_0 - z_t, abs=0.8)


def test_depth_estimator_is_a_no_op_on_a_too_short_spectrum():
    k = np.linspace(0.01, 0.1, 5)
    assert depth.depth_to_top_from_spectrum(k, np.zeros_like(k)) is None


def test_documented_slope_convention_matches_the_arithmetic():
    """The docstring claims -slope/(4*pi) recovers the depth; verify it directly."""
    z = 6.0
    k = np.linspace(0.005, 0.2, 40)
    logp = 2.0 - 4.0 * np.pi * z * k          # a single-slope spectrum
    slope = np.polyfit(k, logp, 1)[0]
    assert -slope / (4.0 * np.pi) == pytest.approx(z, abs=1e-6)


def test_windowed_source_depth_returns_finite_non_negative_depths():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(200, 200))
    d, c = depth.windowed_source_depth(field, win=64, stride=32)
    assert c.max() > 0
    est = d[c > 0]
    assert np.isfinite(est).all() and (est >= 0).all()


def test_windowed_source_depth_runs_and_respects_the_valid_mask():
    rng = np.random.default_rng(5)
    field = rng.normal(size=(200, 200))
    valid = np.zeros((200, 200), bool)
    valid[40:160, 40:160] = True
    d, c = depth.windowed_source_depth(field, win=64, stride=32, valid=valid)
    assert d.shape == field.shape
    assert c.max() > 0
    assert np.all(d[c == 0] == 0.0)           # never estimated => left at zero
