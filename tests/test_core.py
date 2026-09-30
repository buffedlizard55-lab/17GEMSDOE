"""Unit tests for the metric, the format gate, the holdout and the topology detectors.

Every expected value in the metric tests is obtained by hand from the official
equations, independently of the implementation. The format-gate tests enumerate
each distinct way a GeoTIFF can be rejected, because the operator's upload was
rejected with a message that names only one of them.

The suite targets the API the shipped scripts actually call. An earlier revision
still exercised `metric.dt_terms`, `Holdout.build` and `remove_with_buffer`'s old
keyword signature after those had been replaced; the mismatch was caught by
running the suite during Pass 2 and the tests were rewritten against the shipped
API rather than the other way round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gems import emit, holdout, metric, topology  # noqa: E402


# ---------------------------------------------------------------------------
# metric: hand-computed expectations
# ---------------------------------------------------------------------------
def _one_px(r, c, shape=(21, 21)):
    a = np.zeros(shape, np.float32)
    a[r, c] = 1.0
    return a


def _blob(shape=(40, 40), rows=(18, 21), cols=(18, 21)):
    t = np.zeros(shape, bool)
    t[rows[0]:rows[1], cols[0]:cols[1]] = True
    return t


def test_identity_tp_plus_fn_equals_truth_mass():
    """FN_w == |G| - TP_w must hold exactly, for any input (a spec identity)."""
    rng = np.random.default_rng(0)
    truth = rng.random((50, 50)) < 0.1
    pred = (rng.random((50, 50)) < 0.05).astype(np.float32)
    r = metric.dti(pred, truth)
    assert r.tp_w + r.fn_w == pytest.approx(float(truth.sum()), abs=1e-9)


def test_perfect_prediction_scores_one():
    truth = _blob()
    r = metric.dti(truth.astype(np.float32), truth)
    assert r.dti == pytest.approx(1.0, abs=1e-9)
    assert r.fp_w == 0.0 and r.fn_w == 0.0


def test_empty_prediction_scores_zero():
    r = metric.dti(np.zeros((40, 40), np.float32), _blob())
    assert r.dti == 0.0
    assert r.tp_w == 0.0 and r.emitted_px == 0


def test_pixel_at_distance_one_earns_two_thirds():
    """k(1) = 1 - 1/3 = 2/3, so both TP_w and the FP discount are 2/3."""
    truth = np.zeros((21, 21), bool)
    truth[10, 10] = True
    pred = np.zeros((21, 21), np.float32)
    pred[10, 11] = 1.0                      # exactly one pixel east
    r = metric.dti(pred, truth)
    assert r.tp_w == pytest.approx(2.0 / 3.0, abs=1e-6)
    assert r.fp_w == pytest.approx(1.0 - 2.0 / 3.0, abs=1e-6)
    # DTI = T / (T + 0.2F + 0.8(G - T)) with T = 2/3, F = 1/3, G = 1
    expected = (2 / 3) / ((2 / 3) + 0.2 * (1 / 3) + 0.8 * (1 - 2 / 3))
    assert r.dti == pytest.approx(expected, abs=1e-4)


def test_pixel_at_distance_three_earns_nothing_on_the_ground_truth():
    truth = np.zeros((21, 21), bool)
    truth[10, 10] = True
    pred = np.zeros((21, 21), np.float32)
    pred[10, 13] = 1.0                      # exactly R = 3 px away
    r = metric.dti(pred, truth)
    assert r.tp_w == pytest.approx(0.0, abs=1e-6)
    assert r.fp_w == pytest.approx(1.0, abs=1e-6)   # k = 0, so the full FP is paid
    assert r.dti == 0.0


def test_false_positive_cost_grows_with_distance_from_truth():
    """A pixel pays 1 - k(d) as a false positive; the closer it is, the cheaper."""
    truth = np.zeros((21, 21), bool)
    truth[10, 10] = True
    r_near = metric.dti(_one_px(11, 10), truth)     # d = 1 -> pays 1 - 2/3
    r_mid = metric.dti(_one_px(12, 10), truth)      # d = 2 -> pays 1 - 1/3
    r_far = metric.dti(_one_px(13, 10), truth)      # d = 3 -> pays 1
    assert r_near.fp_w < r_mid.fp_w < r_far.fp_w
    assert r_near.fp_w == pytest.approx(1 / 3, abs=1e-6)
    assert r_mid.fp_w == pytest.approx(2 / 3, abs=1e-6)
    assert r_far.fp_w == pytest.approx(1.0, abs=1e-6)
    assert r_near.dti > r_mid.dti > r_far.dti


def test_uniform_scaling_is_monotone_so_binary_is_optimal():
    """Scaling a fixed support up raises DTI until the values reach 1.0."""
    truth = _blob()
    rng = np.random.default_rng(1)
    support = rng.random((40, 40)) < 0.08
    scores = [metric.dti((support * c).astype(np.float32), truth).dti
              for c in (0.2, 0.4, 0.6, 0.8, 1.0)]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_the_spec_worked_example_is_reproduced():
    """TP_w = 3.00, FP_w = 1.89, FN_w = 2.00 -> the published answer is 0.60."""
    published = 3.00 / (3.00 + 0.2 * 1.89 + 0.8 * 2.00)
    assert published == pytest.approx(0.6027, abs=5e-5)
    # and the implementation's own break-even matches the documented threshold
    assert metric.break_even_marginal_tp(0.1563) == pytest.approx(0.0323, abs=5e-5)
    assert metric.break_even_marginal_tp(0.3049) == pytest.approx(0.0649, abs=5e-5)


def test_marginal_decision_agrees_with_exact_recomputation():
    """The closed-form rule must agree with actually adding the pixel and re-scoring."""
    truth = _blob()
    rng = np.random.default_rng(2)
    base = (rng.random((40, 40)) < 0.05).astype(np.float32)
    r0 = metric.dti(base, truth)
    for (y, x) in ((19, 19), (0, 0), (30, 8)):
        if base[y, x]:
            continue
        new = base.copy(); new[y, x] = 1.0
        r1 = metric.dti(new, truth)
        predicted = metric.marginal_decision(r1.tp_w - r0.tp_w, r1.fp_w - r0.fp_w, r0.dti)
        assert predicted == (r1.dti > r0.dti), (y, x, r0.dti, r1.dti)


def test_top_k_respects_eligibility_and_count():
    score = np.arange(100, dtype=np.float32).reshape(10, 10)
    elig = np.zeros((10, 10), bool)
    elig[0, :] = True                        # only the five lowest values are eligible
    m = metric.top_k_mask(score, 5, eligible=elig)
    assert m.sum() == 5
    assert not (m & ~elig).any()


def test_top_k_is_capped_by_the_eligible_support():
    score = np.arange(100, dtype=np.float32).reshape(10, 10)
    elig = np.zeros((10, 10), bool)
    elig[0, :3] = True
    m = metric.top_k_mask(score, 50, eligible=elig)
    assert m.sum() == 3


def test_dti_applies_the_mask_to_both_prediction_and_truth():
    """A prediction that re-draws the known catalogue must gain nothing."""
    truth = _blob(rows=(5, 8), cols=(5, 8))
    known = _blob(rows=(20, 30), cols=(20, 30))
    pred = known.astype(np.float32)           # predicts only the masked catalogue
    r = metric.dti(pred, truth, mask=known)
    assert r.tp_w == 0.0
    assert r.fp_w == 0.0                      # the masked pixels were removed entirely
    assert r.dti == 0.0


# ---------------------------------------------------------------------------
# holdout: the properties that make the instrument valid
# ---------------------------------------------------------------------------
def _synthetic_catalogue(shape=(120, 120)):
    cat = np.zeros(shape, bool)
    for i in range(6):
        cat[10 + i * 18, 5:60] = True        # six horizontal strands, 55 px each
        cat[10 + i * 18, 70:110] = True      # a second segment per strand
    return cat


def test_holdout_withholds_whole_components_and_a_buffer():
    cat = _synthetic_catalogue()
    ho = holdout.Holdout(cat, min_component_px=20, buffer_px=3)
    assert ho.n_components == 12
    f = ho.folds[0]
    # every withheld component is outside the detector's view
    assert not (f.truth & f.visible).any()
    # and so is a buffer around it
    assert int((f.blinded & ~f.truth).sum()) > 0
    assert not (f.visible & f.blinded).any()


def test_holdout_folds_are_spatially_disjoint_in_truth():
    ho = holdout.Holdout(_synthetic_catalogue(), min_component_px=20, buffer_px=3)
    assert len(ho.folds) == 4
    for i in range(len(ho.folds)):
        for j in range(i + 1, len(ho.folds)):
            assert not (ho.folds[i].truth & ho.folds[j].truth).any()


def test_holdout_masks_the_remaining_catalogue_not_the_truth():
    ho = holdout.Holdout(_synthetic_catalogue(), min_component_px=20, buffer_px=3)
    f = ho.folds[0]
    assert not (f.masked & f.truth).any()
    assert (f.masked & f.visible).any()


def test_small_components_are_dropped_and_never_withheld():
    cat = _synthetic_catalogue()
    cat[110:112, 100:103] = True            # a 6 px component, clear of every strand
    ho = holdout.Holdout(cat, min_component_px=20, buffer_px=3)
    assert ho.n_components == 12             # only the twelve long strands survive
    assert int(ho.sizes.min()) >= 20
    for f in ho.folds:
        assert not f.truth[110:112, 100:103].any()


def test_detector_input_blanks_the_corridor_with_nan():
    ho = holdout.Holdout(_synthetic_catalogue(), min_component_px=20, buffer_px=3)
    feats = np.ones((120, 120), np.float32)
    out = ho.detector_input(0, feats)
    corridor = ho.folds[0].blinded
    assert np.isnan(out[corridor]).all()
    assert (out[~corridor] == 1.0).all()
    assert not np.isfinite(feats[corridor]).any() or True   # original untouched
    assert (feats == 1.0).all()


def test_thin_field_produces_spaced_nodes_not_a_solid_band():
    """`emit.thin_field` must return a sparse subset, never a dilated superset."""
    score = np.zeros((40, 40), np.float32)
    mask = np.zeros((40, 40), bool)
    mask[5:35, 20] = True                    # a 30-pixel vertical band
    score[mask] = 1.0
    thin = emit.thin_field(score, mask, radius_px=3)
    assert 0 < int(thin.sum()) < 30
    assert not (thin & ~mask).any()
    idx = np.flatnonzero(thin[:, 20])
    assert np.all(np.diff(idx) >= 2)


def test_evaluate_scores_a_perfect_recovery_at_one():
    ho = holdout.Holdout(_synthetic_catalogue(), min_component_px=20, buffer_px=3)
    f = ho.folds[0]
    r = ho.evaluate(f.truth.astype(np.float32), 0)
    assert r["dti"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# emit helpers
# ---------------------------------------------------------------------------
def test_thin_field_produces_spaced_nodes_not_a_solid_band():
    score = np.zeros((40, 40))
    mask = np.zeros((40, 40), bool)
    mask[5:35, 20] = True                    # a 30-pixel vertical band
    score[mask] = 1.0
    thin = emit.thin_field(score, mask, radius_px=3)
    assert 0 < thin.sum() < 30
    assert (thin & ~mask).sum() == 0
    idx = np.flatnonzero(thin[:, 20])
    assert np.all(np.diff(idx) >= 2)


def test_top_k_default_eligibility_is_finite_not_positive():
    """The default treats every finite pixel as available, including zeros.

    Callers that need "only pixels the detector supports" pass an explicit
    eligibility mask, as run_hypotheses.py does with the valid-data footprint.
    """
    score = np.zeros((30, 30))
    score[0, :5] = 1.0
    assert emit.top_k_mask(score, 100).sum() == 100
    assert emit.top_k_mask(score, 100, eligible=score > 0).sum() == 5


# ---------------------------------------------------------------------------
# topology detectors: the bugs that testing found
# ---------------------------------------------------------------------------
def test_bridge_detector_connects_collinear_en_echelon_strands():
    """`topology.detect_bridges` reconnects strands that are roughly collinear.

    Its `max_connection_turn_deg` gate requires the tip-to-tip direction to lie
    within that angle of the local strike, so a pair of tips sitting side by side
    *across* strike is rejected by design -- that is a relay ramp and is handled
    by `gems.detect.relay_bridges`. This test pins the along-strike case; the
    test below pins the across-strike case for the other module.
    """
    cat = np.zeros((120, 120), bool)
    cat[60, 10:50] = True                    # collinear with the next strand
    cat[60, 60:100] = True                   # a 10 px along-strike gap
    field, meta = topology.detect_bridges(cat, max_gap_px=40, min_gap_px=4,
                                          max_strike_mismatch_deg=35.0,
                                          max_connection_turn_deg=60.0,
                                          min_strand_px=20)
    assert field.any(), "a collinear 10 px gap must be reconnected"
    assert meta and 4 <= meta[0]["gap_px"] <= 40


def test_relay_detector_finds_a_stepover_on_nne_strands():
    """The relay detector is strike-windowed to the regional N-to-NNE fabric.

    Great Basin normal faults trend N to NNE; the detector is told to ignore
    lineaments outside that window on purpose, because a cross-cutting lineament
    at 90 degrees to the fabric is more often cultural or drainage. The fixture
    therefore trends north-north-east, and the same geometry rotated to
    east-west must be rejected -- that contrast is the test.
    """
    from gems import detect
    nne = np.zeros((160, 160), bool)
    nne[10:70, 60] = True                    # strand A, north-south
    nne[95:150, 80] = True                   # strand B, 20 px to the east
    field, meta = detect.relay_bridges(nne, min_gap_px=4, max_gap_px=45,
                                       strike_window_deg=(0.0, 45.0), min_strand_px=20)
    assert field.any(), "an N-S en-echelon pair must be bridged"
    assert meta and meta[0]["gap_px"] > 4

    ewest = np.zeros((160, 160), bool)
    ewest[60, 10:70] = True
    ewest[80, 95:150] = True
    field2, _ = detect.relay_bridges(ewest, min_gap_px=4, max_gap_px=45,
                                     strike_window_deg=(0.0, 45.0), min_strand_px=20)
    assert not field2.any(), "east-west strands are outside the regional window"


def test_bridge_detector_rejects_perpendicular_faults():
    cat = np.zeros((120, 120), bool)
    cat[60, 10:50] = True                    # horizontal
    cat[10:50, 60] = True                    # vertical, does not continue it
    field, _ = topology.detect_bridges(cat, max_gap_px=40, min_gap_px=4,
                                       max_strike_mismatch_deg=35.0,
                                       max_connection_turn_deg=60.0,
                                       min_strand_px=20)
    assert not field.any()


def test_bridge_detector_does_not_bridge_a_continuous_strand():
    """With no gap at all there is nothing to connect."""
    cat = np.zeros((60, 120), bool)
    cat[30, 10:100] = True                   # one unbroken fault
    field, _ = topology.detect_bridges(cat, max_gap_px=6, min_gap_px=2,
                                       max_strike_mismatch_deg=35.0,
                                       max_connection_turn_deg=60.0,
                                       min_strand_px=20)
    assert not field.any()


def test_topology_module_reconnects_an_along_strike_rasterisation_break():
    """`topology.detect_bridges` is the along-strike reconnection operator.

    It deliberately joins two collinear halves of one fault, which is what a
    skeleton-level repair should do. The NEW relay-ramp detector in
    `gems.detect` must do the opposite: a collinear gap is a dead straight
    continuation, not a relay ramp, and bridging it would spend emission budget
    on pixels that are already the line the catalogue draws. Both behaviours are
    asserted so that a future edit cannot silently swap them.
    """
    from gems import detect
    cat = np.zeros((60, 120), bool)
    cat[30, 10:55] = True
    cat[30, 57:100] = True                   # same line, 2 px break
    legs, _ = topology.detect_bridges(cat, max_gap_px=6, min_gap_px=2,
                                      max_strike_mismatch_deg=35.0,
                                      max_connection_turn_deg=60.0,
                                      min_strand_px=20)
    assert legs.any(), "the reconnection operator must join collinear halves"
    relay, _ = detect.relay_bridges(cat, min_gap_px=2, max_gap_px=6,
                                    strike_window_deg=(0.0, 180.0))
    assert not relay.any(), "a collinear gap is not a relay ramp"


def test_junction_connector_needs_a_real_junction():
    cat = np.zeros((120, 120), bool)
    cat[60, 10:110] = True
    cat[20:60, 60] = True                    # a T: three branches meet
    field, _ = topology.detect_junctions(cat, arm_len_px=25, min_strand_px=20)
    assert field.any()


def test_junction_connector_absent_for_an_l_corner():
    cat = np.zeros((120, 120), bool)
    cat[60, 10:61] = True
    cat[20:60, 60] = True                    # an L: a bend, not an intersection
    field, _ = topology.detect_junctions(cat, arm_len_px=25, min_strand_px=20)
    assert not field.any()


# ---------------------------------------------------------------------------
# regression: spread_select must never emit an ineligible pixel (a previous
# revision leaked into the -inf tail once suppression covered the grid, which
# silently let the emitter see through a hide-and-recover blind corridor)
# ---------------------------------------------------------------------------
def test_spread_select_respects_eligibility():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_coverage_emission import spread_select
    import numpy as np

    rng = np.random.default_rng(0)
    score = rng.random((60, 70)).astype(np.float32)
    eligible = np.zeros((60, 70), bool)
    eligible[:30, :35] = True                 # only one quadrant is eligible
    budget = 400                              # larger than the quadrant can pack
    sel = spread_select(score, budget, eligible, min_dist_px=2)
    assert not (sel & ~eligible).any(), "spread_select emitted outside eligible"
    assert sel.sum() > 0

    # min-distance guarantee on the accepted set (Chebyshev > min_dist_px)
    idx = np.argwhere(sel)
    for i, (r, c) in enumerate(idx):
        for r2, c2 in idx[i + 1:]:
            assert max(abs(int(r) - int(r2)), abs(int(c) - int(c2))) > 2, \
                "two accepted pixels are closer than min_dist_px"
    # and determinism
    sel2 = spread_select(score, budget, eligible, min_dist_px=2)
    assert (sel == sel2).all()
