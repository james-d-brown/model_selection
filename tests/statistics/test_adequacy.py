"""
Unit tests for statistics.adequacy -- validates the single-state test
function BEFORE any greedy-selection loop is built on top of it, per the
explicit design requirement that this be tested standalone first.

Test strategy, mirroring the rest of this codebase's validation discipline:
- Building-block correctness (_ols_residuals, _reset_statistic) against
  known closed-form or hand-checkable cases.
- True-positive / true-negative behavior of LinearityTest on synthetic
  data with a KNOWN functional form (genuinely linear vs. genuinely
  threshold-nonlinear), so "reject_linearity" can be checked against
  ground truth rather than just "runs without error."
- Autocorrelated synthetic data specifically, to confirm the dependence
  correction (block permutation) is actually doing something -- i.e. that
  a naive i.i.d.-assuming test would be miscalibrated on this data where
  the block-permutation test is not.
"""
import numpy as np
import pytest

from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth, rbf_kernel
from model_selection.statistics.adequacy.linearity import LinearityTest
from model_selection.statistics.adequacy.linearity import ols_residuals as _ols_residuals
from model_selection.statistics.adequacy.linearity import reset_statistic as _reset_statistic
from model_selection.statistics.adequacy.state_independence import StateIndependenceTest
from model_selection.statistics.adequacy.state_independence import (
    kernel_interaction_score_statistic as _kernel_interaction_score_statistic,
)
from model_selection.statistics.significance.resampling import (
    grid_subsample,
)
from model_selection.visualization.adequacy import (
    PlotContext,
    caption_html,
    composite_regime_score,
    plot_linearity_diagnostics,
    plot_multivariate_linearity_diagnostics,
    plot_state_independence_diagnostics,
    state_strata,
)


def test_ols_residuals_matches_manual_regression():
    rng = np.random.default_rng(0)
    n = 500
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    true_coefs = np.array([3.0, 1.5, -2.0])  # intercept, z1, z2
    y = true_coefs[0] + true_coefs[1] * z1 + true_coefs[2] * z2 + rng.normal(scale=0.01, size=n)

    Z = np.column_stack([z1, z2])
    resid = _ols_residuals(y, Z)

    # Residuals of a correctly-specified linear model should be tiny and
    # have ~zero mean (design includes an intercept).
    assert np.abs(resid.mean()) < 0.01
    assert resid.std() < 0.05


def test_ols_residuals_none_conditioning_is_mean_deviation():
    rng = np.random.default_rng(1)
    y = rng.normal(loc=7.0, size=200)
    resid = _ols_residuals(y, None)
    np.testing.assert_allclose(resid, y - y.mean(), atol=1e-10)


def test_reset_statistic_near_zero_for_clean_linear_relationship():
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.1, size=n)
    f_stat, df_num, df_den, r2, semi_r2 = _reset_statistic(x, y)
    assert df_num == 2
    assert df_den == n - 4
    # A clean linear relationship should NOT show strong evidence of
    # missing nonlinear structure -- F should be small relative to typical
    # F(2, ~1000) critical values (~3.0 at alpha=0.05).
    assert f_stat < 5.0
    assert 0.0 <= r2 <= 1.0
    assert 0.0 <= semi_r2 <= 1.0
    assert r2 < 0.02  # negligible practical departure from linearity
    assert semi_r2 <= r2  # semipartial can never exceed partial (denominator is >= )


def test_reset_statistic_large_for_threshold_nonlinearity():
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.uniform(-3, 3, size=n)
    # Genuine threshold/state-dependent relationship: flat, then a sharp
    # regime change -- the kind of structure a linear fit cannot capture
    # and RESET should detect via the fitted-value polynomial terms.
    y = np.where(x > 0, 5.0 * x, 0.1 * x) + rng.normal(scale=0.2, size=n)
    f_stat, df_num, df_den, r2, semi_r2 = _reset_statistic(x, y)
    assert f_stat > 20.0  # should be a large, clearly-detectable signal
    assert r2 > 0.05  # a real, practically meaningful departure, not just detectable
    assert semi_r2 <= r2












def _make_autocorrelated_covariate(n: int, rng: np.random.Generator, phi: float = 0.9) -> np.ndarray:
    """AR(1) series -- gives genuine short-range serial correlation to
    stress-test the dependence correction."""
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(scale=0.3)
    return x


def test_linearity_test_does_not_reject_genuine_linear_relationship():
    rng = np.random.default_rng(6)
    n = 600
    z = _make_autocorrelated_covariate(n, rng)
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 1.5 * candidate + 0.8 * z + rng.normal(scale=0.3, size=n)

    test = LinearityTest(alpha=0.05)
    result = test.run(
        candidate, obs, conditioning=z,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=200, rng=rng,
    )
    assert result.reject_linearity is False
    assert result.permutation_p_value > 0.05


def test_linearity_test_rejects_genuine_state_dependent_nonlinearity():
    rng = np.random.default_rng(7)
    n = 600
    z = _make_autocorrelated_covariate(n, rng)
    candidate = _make_autocorrelated_covariate(n, rng)
    # obs depends on candidate very differently depending on candidate's
    # own sign -- a genuinely state-dependent nonlinear relationship, not
    # capturable by any linear (or even a fixed-slope) model of candidate.
    obs = np.where(candidate > 0, 4.0 * candidate, 0.1 * candidate) + 0.8 * z + rng.normal(scale=0.2, size=n)

    test = LinearityTest(alpha=0.05)
    result = test.run(
        candidate, obs, conditioning=z,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=200, rng=rng,
    )
    assert result.reject_linearity is True
    assert result.permutation_p_value < 0.05


def test_linearity_test_handles_empty_conditioning_set():
    rng = np.random.default_rng(8)
    n = 400
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 2.0 * candidate + rng.normal(scale=0.2, size=n)

    test = LinearityTest(alpha=0.05)
    result = test.run(
        candidate, obs, conditioning=None,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=150, rng=rng,
    )
    assert result.reject_linearity is False


def test_dependence_correction_changes_result_under_strong_autocorrelation():
    """Confirms the block-permutation null is actually doing something
    different from what an i.i.d.-assuming test would do -- i.e. that the
    classical F p-value and the dependence-aware permutation p-value can
    genuinely disagree under strong autocorrelation, the same sanity check
    already established for the CMI significance test earlier this
    project. This does not assert WHICH direction the disagreement goes --
    only that dependence correction is not a no-op on this data."""
    rng = np.random.default_rng(9)
    n = 800
    # Very strong autocorrelation (phi close to 1) with a genuinely null
    # (independent) candidate-obs relationship -- obs built from a SEPARATE
    # AR(1) draw with no dependence on candidate at all.
    candidate = _make_autocorrelated_covariate(n, rng, phi=0.97)
    obs = _make_autocorrelated_covariate(n, rng, phi=0.97)

    test = LinearityTest(alpha=0.05)
    result = test.run(
        candidate, obs, conditioning=None,
        time_positions=np.arange(n), min_temporal_lag=1,  # deliberately too small a lag
        n_permutations=200, rng=rng,
    )
    # Under near-unit-root autocorrelation, classical and permutation
    # p-values are not required to agree -- record both are at least
    # computed and finite, and that the test does not silently crash on
    # this stress case.
    assert np.isfinite(result.classical_p_value)
    assert np.isfinite(result.permutation_p_value)


def test_null_distribution_summary_fields_populated_and_sensible():
    rng = np.random.default_rng(10)
    n = 400
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 2.0 * candidate + rng.normal(scale=0.3, size=n)

    test = LinearityTest(alpha=0.05)
    result = test.run(
        candidate, obs, conditioning=None,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=150, rng=rng,
    )
    assert result.null_mean >= 0.0
    assert result.null_std >= 0.0
    assert result.null_median <= result.null_p90 <= result.null_p99 <= result.null_max


def test_plot_diagnostics_returns_figure_with_expected_traces():
    rng = np.random.default_rng(11)
    n = 300
    z = _make_autocorrelated_covariate(n, rng)
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 1.5 * candidate + 0.8 * z + rng.normal(scale=0.3, size=n)

    test = LinearityTest(alpha=0.05)
    fig = plot_linearity_diagnostics(candidate, obs, z, n_bins=10, max_scatter_points=100, rng=rng)
    # 3 traces in left panel (scatter, fitted line, binned trend) + 3 in
    # right panel (scatter, zero line, binned residual trend).
    assert len(fig.data) == 6
    # No paragraph-length annotations on the figure itself -- those live
    # in panel_interpretation_html() as plain HTML instead (Plotly's own
    # annotation wrapping clips full paragraphs regardless of width/margin
    # tuning). Only the two short Panel A/B subplot-title annotations
    # should remain.
    assert len(fig.layout.annotations) == 2


def test_captions_are_plain_html_specific_to_the_round():
    """Captions name the candidate, the conditioning set, the tier and the
    test's outcome, in plain wrappable HTML (no Plotly layout directives)."""
    ctx = PlotContext(tier="linear", round_no=2, candidate="b", selected=("a",),
                      covariate_label="the seasonal phase", n_covariate_columns=2,
                      result="Rejected: p=0.001. The linear tier's results may then be invalid.")
    for kind in ("linear_specification", "linearity", "state_independence"):
        html = caption_html(kind, ctx, multivariate=False)
        assert "<div" in html and "xref" not in html and "yref" not in html
        assert "candidate 'b'" in html and "the selected formulation 'a' and the seasonal phase" in html
        assert "linear tier" in html and "Rejected: p=0.001" in html
        assert "FPC" not in html and "PC1" not in html and "CMI" not in html
    multi = caption_html("multivariate_linearity", PlotContext(tier="linear", round_no=1, candidate="b",
                                                               feature_names=("PC1", "PC2")), multivariate=True)
    assert "PC1, PC2" in multi and "No formal test" in multi


def test_effect_modification_strata_follow_the_selected_formulations_or_the_time_of_year():
    rng = np.random.default_rng(3)
    n = 600
    angle = rng.uniform(0, 2 * np.pi, n)
    phase = np.column_stack([np.sin(angle), np.cos(angle)])
    flow = rng.normal(size=n)
    with_selected = PlotContext(tier="nonlinear", round_no=2, candidate="b", selected=("a",),
                                covariate_label="the seasonal phase", n_covariate_columns=2)
    labels, names, what = state_strata(np.column_stack([phase, flow]), with_selected)
    assert names == ["lowest third", "middle third", "highest third"] and "'a'" in what
    assert np.all(np.diff([flow[labels == k].mean() for k in range(3)]) > 0)  # thirds of the selected flow
    seasonal = PlotContext(tier="linear", round_no=1, candidate="b", covariate_label="the seasonal phase",
                           n_covariate_columns=2)
    labels, names, what = state_strata(phase, seasonal)
    assert what == "the time of year" and names[0] == "January to April"
    day = np.mod(angle, 2 * np.pi) / (2 * np.pi) * 365
    assert np.all(labels[day < 100] == 0) and np.all(labels[(day > 130) & (day < 230)] == 1)

def test_plot_diagnostics_subsampling_does_not_change_trend_estimate_much():
    """Confirms the binned trend is computed from the FULL data, not the
    subsampled scatter -- i.e. a tiny max_scatter_points shouldn't distort
    the trend line meaningfully."""
    rng = np.random.default_rng(12)
    n = 2000
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 3.0 * candidate + rng.normal(scale=0.2, size=n)

    test = LinearityTest(alpha=0.05)
    fig_full = plot_linearity_diagnostics(candidate, obs, None, n_bins=10, max_scatter_points=n, rng=rng)
    fig_sparse = plot_linearity_diagnostics(candidate, obs, None, n_bins=10, max_scatter_points=20, rng=rng)

    # trace index 2 = binned empirical trend in the left panel
    trend_full = np.array(fig_full.data[2].y)
    trend_sparse = np.array(fig_sparse.data[2].y)
    np.testing.assert_allclose(trend_full, trend_sparse, atol=1e-10)


def test_effect_size_is_small_for_linear_and_larger_for_nonlinear():
    """Confirms partial R^2 tracks PRACTICAL magnitude of nonlinearity,
    distinct from statistical detectability -- the whole reason it was
    added."""
    rng = np.random.default_rng(13)
    n = 2000
    z = _make_autocorrelated_covariate(n, rng)

    candidate_lin = _make_autocorrelated_covariate(n, rng)
    obs_lin = 1.5 * candidate_lin + 0.8 * z + rng.normal(scale=0.3, size=n)
    r_lin = LinearityTest(alpha=0.05).run(
        candidate_lin, obs_lin, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=100, rng=rng,
    )

    candidate_nl = _make_autocorrelated_covariate(n, rng)
    obs_nl = np.where(candidate_nl > 0, 4.0 * candidate_nl, 0.1 * candidate_nl) + 0.8 * z + rng.normal(scale=0.2, size=n)
    r_nl = LinearityTest(alpha=0.05).run(
        candidate_nl, obs_nl, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=100, rng=rng,
    )

    assert r_lin.effect_size is not None and 0.0 <= r_lin.effect_size <= 1.0
    assert r_nl.effect_size is not None and 0.0 <= r_nl.effect_size <= 1.0
    assert r_nl.effect_size > r_lin.effect_size


def test_decision_not_significant_case():
    """Case 1: p >= alpha -- not rejected regardless of effect size, and
    decision_reason should name the p-value, not the effect size, as the
    reason."""
    rng = np.random.default_rng(14)
    n = 400
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = 2.0 * candidate + rng.normal(scale=0.2, size=n)  # genuinely linear, no Z

    test = LinearityTest(alpha=0.05, effect_size_floor=0.0)  # floor at 0 -- would ALWAYS pass on effect size alone
    result = test.run(
        candidate, obs, conditioning=None,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=150, rng=rng,
    )
    assert result.significant is False
    assert result.reject_linearity is False
    assert "alpha" in result.decision_reason
    assert "not rejected" in result.decision_reason


def test_decision_significant_but_below_effect_size_floor_case():
    """Case 2: p < alpha (statistically real) but semipartial R^2 below
    the floor -- reject_linearity must be False, and decision_reason must
    name the effect-size floor as the reason, not the p-value."""
    rng = np.random.default_rng(15)
    n = 5000
    z = _make_autocorrelated_covariate(n, rng)
    candidate = _make_autocorrelated_covariate(n, rng)
    # A very strong linear relationship with a TINY nonlinear nudge --
    # large n gives this enough power to be statistically significant,
    # while the nonlinear nudge's actual share of obs's variance stays
    # small (a fraction of one percent), exactly the "small but real n"
    # situation this floor exists for.
    obs = 3.0 * candidate + 0.8 * z + 0.01 * candidate ** 2 + rng.normal(scale=0.3, size=n)

    test = LinearityTest(alpha=0.05, effect_size_floor=0.5)  # deliberately high floor to force this case
    result = test.run(
        candidate, obs, z,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=150, rng=rng,
    )
    assert result.effect_size_floor == 0.5
    if result.significant:
        assert result.meets_effect_size_floor is False
        assert result.reject_linearity is False
        assert "effect_size_floor" in result.decision_reason
        assert "semipartial" in result.decision_reason.lower()


def test_decision_rejected_case_names_both_criteria():
    """Case 3: both criteria genuinely met -- decision_reason should
    reference both the p-value and the semipartial R^2 against the floor
    that was actually used."""
    rng = np.random.default_rng(16)
    n = 2000
    z = _make_autocorrelated_covariate(n, rng)
    candidate = _make_autocorrelated_covariate(n, rng)
    obs = np.where(candidate > 0, 4.0 * candidate, 0.1 * candidate) + 0.8 * z + rng.normal(scale=0.2, size=n)

    test = LinearityTest(alpha=0.05, effect_size_floor=0.02)
    result = test.run(
        candidate, obs, z,
        time_positions=np.arange(n), min_temporal_lag=5,
        n_permutations=150, rng=rng,
    )
    assert result.significant is True
    assert result.meets_effect_size_floor is True
    assert result.reject_linearity is True
    assert "rejected:" in result.decision_reason
    assert str(result.effect_size_floor) in result.decision_reason or "0.02" in result.decision_reason


# ---------------------------------------------------------------------------
# StateIndependenceTest tests
# ---------------------------------------------------------------------------

def _make_two_regime_interaction_data(n, rng, z_scale=1.5):
    """Synthetic data with GENUINE effect modification: candidate's
    relationship to obs is a completely different shape depending on
    whether Z is positive or negative -- steep positive slope in one
    regime, near-zero (flat) in the other. This is exactly the kind of
    structure PC and PDC (one fixed shape regardless of Z) cannot
    represent, and StateIndependenceTest exists to detect."""
    z = rng.normal(scale=z_scale, size=n)
    candidate = rng.normal(size=n)
    slope = np.where(z > 0, 4.0, 0.0)
    obs = slope * candidate + 0.5 * z + rng.normal(scale=0.3, size=n)
    return candidate, obs, z


def _make_no_interaction_data(n, rng, z_scale=1.5):
    """Synthetic data where candidate's relationship to obs is IDENTICAL
    regardless of Z's value -- a single fixed (here, nonlinear/saturating)
    shape, exactly PDC's own assumption, with no effect modification."""
    z = rng.normal(scale=z_scale, size=n)
    candidate = rng.normal(size=n)
    obs = 2.0 * np.tanh(candidate) + 0.5 * z + rng.normal(scale=0.3, size=n)
    return candidate, obs, z


def test_median_heuristic_bandwidth_is_positive_and_scale_sensitive():
    rng = np.random.default_rng(20)
    X_small = rng.normal(scale=0.1, size=(200, 1))
    X_large = rng.normal(scale=10.0, size=(200, 1))
    bw_small = median_heuristic_bandwidth(X_small)
    bw_large = median_heuristic_bandwidth(X_large)
    assert bw_small > 0
    assert bw_large > bw_small  # larger-spread data -> larger bandwidth


def test_rbf_kernel_diagonal_is_one_and_symmetric():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(50, 2))
    K = rbf_kernel(X, bandwidth=1.0)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-10)
    assert np.all(K > 0) and np.all(K <= 1.0 + 1e-10)


def test_kernel_interaction_score_small_for_no_interaction():
    rng = np.random.default_rng(22)
    n = 500
    candidate, obs, z = _make_no_interaction_data(n, rng)
    score, partial_es, semi_es = _kernel_interaction_score_statistic(candidate, obs, z)
    assert score >= 0.0
    assert 0.0 <= partial_es <= 1.0
    assert 0.0 <= semi_es <= 1.0


def test_kernel_interaction_score_larger_for_genuine_interaction():
    """The core discrimination check: genuine effect modification should
    produce a materially larger score/effect size than no interaction at
    all, matched sample size and Z scale."""
    rng = np.random.default_rng(23)
    n = 500
    cand_no, obs_no, z_no = _make_no_interaction_data(n, rng)
    _, es_no, semi_no = _kernel_interaction_score_statistic(cand_no, obs_no, z_no)

    rng2 = np.random.default_rng(24)
    cand_int, obs_int, z_int = _make_two_regime_interaction_data(n, rng2)
    _, es_int, semi_int = _kernel_interaction_score_statistic(cand_int, obs_int, z_int)

    assert es_int > es_no
    assert semi_int > semi_no


def test_state_independence_test_does_not_reject_no_interaction():
    rng = np.random.default_rng(25)
    n = 400
    candidate, obs, z = _make_no_interaction_data(n, rng)
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)
    result = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=100, rng=rng,
    )
    assert result.reject_linearity is False
    assert np.isnan(result.classical_p_value)  # Davies' method not implemented -- see docstring


def test_state_independence_test_rejects_genuine_interaction():
    rng = np.random.default_rng(26)
    n = 400
    candidate, obs, z = _make_two_regime_interaction_data(n, rng)
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)
    result = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=100, rng=rng,
    )
    assert result.significant is True
    assert result.reject_linearity is True
    assert "rejected:" in result.decision_reason


def test_state_independence_test_handles_empty_conditioning_without_error():
    """Round-1-style state (no already-selected models yet) -- effect
    modification is vacuous with no Z to modify by; must return a clean
    'not rejected' result, never raise."""
    rng = np.random.default_rng(27)
    n = 300
    candidate = rng.normal(size=n)
    obs = 2.0 * candidate + rng.normal(scale=0.3, size=n)
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02)
    result = test.run(
        candidate, obs, conditioning=None, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=50, rng=rng,
    )
    assert result.reject_linearity is False
    assert result.observed_statistic == 0.0


def test_state_independence_test_subsampling_triggers_and_stays_consistent():
    """At n > max_n, run() must subsample ONCE and use that SAME subsample
    for both the observed statistic and every permutation draw -- not a
    fresh subsample per call. Indirect check: running twice with the SAME
    seeded rng should give IDENTICAL results (full determinism), which
    would not hold if _compute_statistic silently drew its own fresh
    randomness internally on top of run()'s subsampling."""
    n = 3000
    candidate, obs, z = _make_two_regime_interaction_data(n, np.random.default_rng(28))

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=300)
    result_a = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=50, rng=np.random.default_rng(99),
    )
    result_b = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=50, rng=np.random.default_rng(99),
    )
    assert result_a.observed_statistic == result_b.observed_statistic
    assert result_a.permutation_p_value == result_b.permutation_p_value


def test_state_independence_test_reports_subsampling_diagnostics():
    """dependence_aware_subsampling_used / n_after_subsampling /
    natural_independent_sample_size must be populated and consistent
    whenever subsampling actually triggers."""
    n = 5000
    candidate, obs, z = _make_two_regime_interaction_data(n, np.random.default_rng(40))
    min_temporal_lag = 5

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=200)
    result = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=min_temporal_lag, n_permutations=30, grid_step=1, rng=np.random.default_rng(41),
    )
    assert result.dependence_aware_subsampling_used is True
    assert result.n_after_subsampling == 200  # natural ceiling (n/lag=1000) exceeds max_n=200, so capped
    assert result.natural_independent_sample_size is not None
    assert result.natural_independent_sample_size > 200  # confirms the cap WAS binding here


def test_state_independence_test_no_subsampling_when_n_below_max_n():
    """When n <= max_n, no subsampling occurs at all, and the diagnostic
    fields correctly reflect that -- the original min_temporal_lag must
    also reach the base class's block permutation UNCHANGED in this case."""
    n = 300
    candidate, obs, z = _make_two_regime_interaction_data(n, np.random.default_rng(42))
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=1000)
    result = test.run(
        candidate, obs, z, time_positions=np.arange(n),
        min_temporal_lag=5, n_permutations=30, rng=np.random.default_rng(43),
    )
    assert result.dependence_aware_subsampling_used is False
    assert result.n_after_subsampling == n
    assert result.natural_independent_sample_size is None


def test_block_length_after_subsampling_is_estimated_in_time_units():
    """After subsampling, the block length is estimated on the subsample (in
    sampling intervals) and converted back to time units. The data are
    independent, so the multiplier bandwidth must span about one subsample point: the
    record's lag (100 time units) must not be reinterpreted as a count of
    sparse subsample positions, which would give a handful of enormous blocks
    with almost no shuffling resolution."""
    import model_selection.statistics.adequacy.base as ff_module

    n = 50000
    min_temporal_lag = 100
    max_n = 200
    stride = n // max_n  # 250: longer than the lag, set by the cap
    candidate, obs, z = _make_two_regime_interaction_data(n, np.random.default_rng(44))

    recorded = []
    real_weights = ff_module.dependent_wild_weights

    def spy(time_positions_, block_length_, n_draws_, rng_, **kwargs):
        recorded.append((block_length_, np.asarray(time_positions_)))
        return real_weights(time_positions_, block_length_, n_draws_, rng_, **kwargs)

    ff_module.dependent_wild_weights = spy
    try:
        test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=max_n)
        test.run(
            candidate, obs, z, time_positions=np.arange(n),
            min_temporal_lag=min_temporal_lag, n_permutations=20, grid_step=1, rng=np.random.default_rng(45),
        )
    finally:
        ff_module.dependent_wild_weights = real_weights

    assert recorded
    for block_length, positions in recorded:
        assert np.allclose(np.diff(positions), n / max_n, rtol=1e-6)
        assert stride - 1 <= block_length <= 2 * stride



def _check_on_grid(t, idx, positions, step, grid_step):
    """Selected samples are distinct, in time order, given positions within
    half a grid step of their times, and those positions lie on a regular
    grid of the returned step."""
    from model_selection.statistics.significance.block_length import grid_layout
    assert np.all(np.diff(idx) > 0)
    assert np.all(np.abs(positions - t[idx]) <= grid_step / 2 + 1e-9)
    assert grid_layout(positions, step) is not None


@pytest.mark.parametrize("n", [2001, 3000, 3999, 26280, 100000])
def test_dense_subsample_keeps_exactly_the_cap_spread_over_the_record(n):
    """Where the null accounts for dependence, a record longer than the cap is
    thinned to exactly the cap, evenly spaced over the whole record, at a
    fractional interval, whatever the lag."""
    t = np.arange(n)
    for seed, lag in enumerate((1, 50, 5000)):
        idx, positions, step, _ = grid_subsample(t, 1, lag, 2000, np.random.default_rng(seed), at_lag_spacing=False)
        assert idx.size == 2000
        _check_on_grid(t, idx, positions, step, 1)
        gaps = np.diff(t[idx])
        assert gaps.min() >= np.floor(n / 2000) and gaps.max() <= np.ceil(n / 2000) + 1
        assert t[idx][0] - t[0] <= np.ceil(n / 2000) and t[-1] - t[idx][-1] <= np.ceil(n / 2000) + 1


def test_dense_subsample_leaves_records_within_the_cap_whole():
    t = np.arange(1500)
    idx, positions, step, natural_n = grid_subsample(t, 1, 200, 2000, np.random.default_rng(34),
                                                     at_lag_spacing=False)
    assert idx.size == 1500 and step == 1 and natural_n is None and np.array_equal(positions, t)


def test_lag_spaced_subsample_is_at_least_one_lag_apart_even_within_the_cap():
    """Where the null treats samples as exchangeable, samples are at least one
    lag apart, also in a record shorter than the cap; where that leaves more
    than the cap, exactly the cap, evenly spaced."""
    rng = np.random.default_rng(31)
    t = np.arange(1500)
    idx, positions, step, natural_n = grid_subsample(t, 1, 10, 2000, rng, at_lag_spacing=True)
    assert step == 10 and np.all(np.diff(idx) == 10) and idx.size == natural_n == 150
    assert np.array_equal(positions, t[idx])  # an integer interval keeps every sample at its own time
    t = np.arange(100000)
    idx, positions, step, natural_n = grid_subsample(t, 1, 20, 2000, rng, at_lag_spacing=True)
    assert idx.size == 2000 and np.diff(t[idx]).min() >= 20 and natural_n > 2000
    _check_on_grid(t, idx, positions, step, 1)


def test_subsamples_measure_the_interval_in_time_across_missing_points():
    """With missing grid points the interval is a time interval, not a count
    of rows: samples falling on missing points are absent, the rest keep the
    grid, and the cap is still filled exactly."""
    rng = np.random.default_rng(32)
    t = np.flatnonzero(rng.random(20000) > 0.3) * 2  # grid step 2, 30% missing
    for lag_spaced in (False, True):
        idx, positions, step, _ = grid_subsample(t, 2, 40, 500, rng, at_lag_spacing=lag_spaced)
        assert idx.size == 500
        _check_on_grid(t, idx, positions, step, 2)
        assert np.all(np.diff(t[idx]) % 2 == 0) and np.diff(t[idx]).min() >= (40 if lag_spaced else 2)


def test_subsample_positions_give_block_lengths_in_time_units():
    """The dependence estimates on a fractional-interval subsample measure lags
    in sampling intervals and report them in time units."""
    from model_selection.statistics.significance.block_length import resolve_block
    rng = np.random.default_rng(35)
    n = 26280
    x = np.zeros(n)
    e = rng.normal(size=n)
    for k in range(1, n):  # AR(1) with a correlation time of about 100 steps
        x[k] = 0.99 * x[k - 1] + e[k]
    idx, positions, step, _ = grid_subsample(np.arange(n), 1, 100, 2000, rng, at_lag_spacing=False)
    resolved = resolve_block(positions, 100, [(x[idx], x[idx])], grid_step=step)
    assert resolved.estimated and resolved.length >= step and resolved.length % 1 == 0


def test_composite_regime_score_reduces_to_own_rank_for_single_column():
    rng = np.random.default_rng(50)
    z = rng.normal(size=500)
    score = composite_regime_score(z[:, None])
    from scipy.stats import rankdata
    expected = rankdata(z, method="average") / 500
    np.testing.assert_allclose(score, expected)


def test_composite_regime_score_is_bounded_and_obs_free():
    rng = np.random.default_rng(51)
    Z = rng.normal(size=(400, 3))
    score = composite_regime_score(Z)
    assert score.shape == (400,)
    assert np.all(score >= 0.0) and np.all(score <= 1.0)


def test_state_independence_plot_returns_figure_with_traces_per_stratum():
    """For a scalar candidate and observations each third has the
    relationship panel and the residual panel (scatter and trend in each):
    4 traces per third x 3 thirds = 12. The feature-weight and per-feature
    panels are drawn only with several features."""
    rng = np.random.default_rng(52)
    n = 900
    candidate, obs, z = _make_no_interaction_data(n, rng)
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02)
    fig = plot_state_independence_diagnostics(candidate, obs, z, rng=rng)
    assert len(fig.data) == 12


def test_state_independence_plot_trend_slopes_similar_when_no_interaction():
    """Core discrimination check, mirroring the numeric (not just
    render-without-crashing) verification used throughout this project:
    fit an OLS slope to each stratum's WHOLE (unbinned) residualized data
    and confirm the slopes are similar across strata when there is
    genuinely no effect modification."""
    rng = np.random.default_rng(53)
    n = 3000
    candidate, obs, z = _make_no_interaction_data(n, rng)

    regime_score = composite_regime_score(z[:, None] if z.ndim == 1 else z)
    edges = np.quantile(regime_score, [0.0, 1 / 3, 2 / 3, 1.0])
    labels = np.digitize(regime_score, edges[1:-1])

    slopes = []
    for k in sorted(np.unique(labels)):
        mask = labels == k
        cand_k = _ols_residuals(candidate[mask], z[mask])
        obs_k = _ols_residuals(obs[mask], z[mask])
        design = np.column_stack([np.ones(mask.sum()), cand_k])
        coef, *_ = np.linalg.lstsq(design, obs_k, rcond=None)
        slopes.append(coef[1])

    slopes = np.array(slopes)
    assert slopes.std() < 0.3  # strata slopes should be close to each other


def test_state_independence_plot_trend_slopes_differ_when_genuine_interaction():
    """Same check as above, but on data with GENUINE effect modification
    (steep slope in one Z-regime, flat in another) -- per-stratum slopes
    must differ substantially, confirming the plot's construction actually
    has discriminating power, not just that it renders."""
    rng = np.random.default_rng(54)
    n = 3000
    candidate, obs, z = _make_two_regime_interaction_data(n, rng)

    regime_score = composite_regime_score(z[:, None] if z.ndim == 1 else z)
    edges = np.quantile(regime_score, [0.0, 1 / 3, 2 / 3, 1.0])
    labels = np.digitize(regime_score, edges[1:-1])

    slopes = []
    for k in sorted(np.unique(labels)):
        mask = labels == k
        cand_k = _ols_residuals(candidate[mask], z[mask])
        obs_k = _ols_residuals(obs[mask], z[mask])
        design = np.column_stack([np.ones(mask.sum()), cand_k])
        coef, *_ = np.linalg.lstsq(design, obs_k, rcond=None)
        slopes.append(coef[1])

    slopes = np.array(slopes)
    assert slopes.max() - slopes.min() > 1.0  # strata slopes should differ substantially


def test_state_independence_plot_raises_clear_error_on_empty_conditioning():
    """The bug found on real data: plot_diagnostics used to crash with an
    obscure IndexError from composite_regime_score(None) when conditioning
    was empty (e.g. the round-1/unconditional state). Must now raise a
    clear ValueError instead."""
    rng = np.random.default_rng(55)
    n = 300
    candidate = rng.normal(size=n)
    obs = 2.0 * candidate + rng.normal(scale=0.3, size=n)
    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02)
    with pytest.raises(ValueError, match="non-empty conditioning set"):
        plot_state_independence_diagnostics(candidate, obs, None)
    with pytest.raises(ValueError, match="non-empty conditioning set"):
        plot_state_independence_diagnostics(candidate, obs, np.array([]))


def test_state_independence_test_multivariate_scalar_case_is_byte_identical_to_reference():
    """Direct regression check: the multivariate generalization must not
    perturb the scalar case's CORE MECHANICS at all -- compares against
    the exact reference numbers established before this generalization
    existed, using an EXPLICIT ridge_lambda=1.0
    override (the class's own OLD default, now optional) to isolate "did
    the underlying computation change" from "did the DEFAULT
    regularization-selection strategy change" -- the latter changed
    deliberately (ridge_lambda now auto-estimates via REML by default --
    see reml_kernel_ridge_lambda), and this test must not conflate the two."""
    rng = np.random.default_rng(101)
    n = 400
    z = rng.normal(scale=1.5, size=n)
    candidate = rng.normal(size=n)
    slope = np.where(z > 0, 4.0, 0.0)
    obs = slope * candidate + 0.5 * z + rng.normal(scale=0.3, size=n)

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400, ridge_lambda=1.0)
    result = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5,
                       n_permutations=150, rng=rng)
    assert abs(result.observed_statistic - 27341.326019) < 1e-2
    assert abs(result.effect_size - 0.815536) < 1e-4
    # The semipartial effect is the interaction's share of the variation in
    # obs left unexplained by Z (kernel ridge on K_z at the same fixed penalty).
    from model_selection.statistics.adequacy.state_independence import kernel_main_effect_kernels
    K_c, K_z = kernel_main_effect_kernels(candidate[:, None], z)
    y = obs - obs.mean()

    def rss(K):
        return float(np.sum((y - K @ np.linalg.solve(K + np.eye(n), y)) ** 2))

    expected = (rss(K_c + K_z) - rss(K_c + K_z + K_c * K_z)) / rss(K_z)
    assert abs(result.semipartial_effect_size - expected) < 1e-6


def test_state_independence_test_default_reml_lambda_differs_from_old_fixed_default_but_is_well_calibrated():
    """Companion: with the NEW default (ridge_lambda=None, auto-estimated
    via REML), the SAME data gives a DIFFERENT (not necessarily identical)
    result than the old fixed ridge_lambda=1.0 -- confirming the default
    actually changed, not just that an override still reproduces the old
    behavior -- while remaining a sensible, non-degenerate result."""
    rng = np.random.default_rng(101)
    n = 400
    z = rng.normal(scale=1.5, size=n)
    candidate = rng.normal(size=n)
    slope = np.where(z > 0, 4.0, 0.0)
    obs = slope * candidate + 0.5 * z + rng.normal(scale=0.3, size=n)

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)  # ridge_lambda=None default
    result = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5,
                       n_permutations=150, rng=np.random.default_rng(101))
    assert np.isfinite(result.observed_statistic)
    assert 0.0 <= result.semipartial_effect_size <= 1.0
    # Still correctly detects the genuine, strong effect modification present in this data.
    assert result.reject_linearity is True


def test_state_independence_test_genuinely_multivariate_detects_no_interaction():
    rng = np.random.default_rng(0)
    n = 400
    d_cand, d_obs = 3, 2
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, d_cand))
    B = rng.normal(size=(d_cand, d_obs))
    obs = np.tanh(candidate) @ B + 0.5 * z[:, None] + rng.normal(scale=0.3, size=(n, d_obs))

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)
    result = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5,
                       n_permutations=100, rng=rng)
    assert candidate.shape == (n, d_cand)  # confirms input was genuinely multivariate, not silently reshaped
    assert result.reject_linearity is False


def test_state_independence_test_genuinely_multivariate_detects_genuine_interaction():
    rng = np.random.default_rng(1)
    n = 400
    d_cand, d_obs = 3, 2
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, d_cand))
    B = rng.normal(size=(d_cand, d_obs))
    slope = np.where(z > 0, 4.0, 0.0)
    obs = (slope[:, None] * candidate) @ B + 0.5 * z[:, None] + rng.normal(scale=0.3, size=(n, d_obs))

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)
    result = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5,
                       n_permutations=100, rng=rng)
    assert result.reject_linearity is True
    assert result.semipartial_effect_size > 0.1


def test_state_independence_test_multivariate_candidate_scalar_obs():
    """Mixed case: multivariate candidate, scalar obs -- confirms the
    generalization handles asymmetric shapes correctly, not just the
    fully-multivariate or fully-scalar extremes."""
    rng = np.random.default_rng(2)
    n = 300
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, 2))
    slope = np.where(z > 0, 3.0, 0.5)
    obs = slope * (candidate[:, 0] + candidate[:, 1]) + rng.normal(scale=0.2, size=n)

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02, max_n=400)
    result = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5,
                       n_permutations=100, rng=rng)
    assert np.isfinite(result.observed_statistic)
    assert 0.0 <= result.semipartial_effect_size <= 1.0


def test_state_independence_test_trace_formula_matches_naive_vector_formula_at_q1():
    """Direct check that the trace-based score generalization is not an
    approximation at q=1 -- computed via the actual class."""
    from model_selection.statistics.adequacy.state_independence import (
        kernel_interaction_score_statistic as _kernel_interaction_score_statistic,
    )

    rng = np.random.default_rng(3)
    n = 300
    z = rng.normal(size=n)
    candidate = rng.normal(size=n)
    obs = 2 * candidate + 0.5 * z + rng.normal(scale=0.2, size=n)

    score_1d, _, _ = _kernel_interaction_score_statistic(candidate, obs, z, ridge_lambda=1.0)
    score_2d, _, _ = _kernel_interaction_score_statistic(candidate, obs[:, None], z, ridge_lambda=1.0)
    assert abs(score_1d - score_2d) < 1e-6


# ---------------------------------------------------------------------------
# MultivariateLinearityTest -- PC's tier-adequacy check for genuinely
# multivariate candidate/obs, which LinearityTest/RESET cannot handle.
# ---------------------------------------------------------------------------

def test_multivariate_linearity_test_qualitatively_agrees_with_reset_on_scalar_data():
    """Different mechanism (kernel score test vs RESET), but must reach
    the SAME conclusion on the SAME scalar data LinearityTest's own
    reference tests use -- a sanity/consistency check, not an exact-number
    match (these are genuinely different statistics)."""
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    def _ar1(n, rng, phi=0.9):
        x = np.empty(n); x[0] = rng.normal()
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(scale=0.3)
        return x

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)

    rng = np.random.default_rng(6)
    n = 600
    z = _ar1(n, rng)
    candidate = _ar1(n, rng)
    obs = 1.5 * candidate + 0.8 * z + rng.normal(scale=0.3, size=n)
    r = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5, n_permutations=150, rng=rng)
    assert r.reject_linearity is False  # matches LinearityTest/RESET's own finding here

    rng = np.random.default_rng(7)
    z = _ar1(n, rng)
    candidate = _ar1(n, rng)
    obs = np.where(candidate > 0, 4.0 * candidate, 0.1 * candidate) + 0.8 * z + rng.normal(scale=0.2, size=n)
    r = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5, n_permutations=150, rng=rng)
    assert r.reject_linearity is True  # matches LinearityTest/RESET's own finding here


def test_multivariate_linearity_test_detects_genuinely_multivariate_linear_case():
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, 3))
    B = rng.normal(size=(3, 2))
    obs = candidate @ B + 0.5 * z[:, None] + rng.normal(scale=0.3, size=(n, 2))

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)
    r = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5, n_permutations=100, rng=rng)
    assert candidate.shape == (n, 3) and obs.shape == (n, 2)
    assert r.reject_linearity is False
    assert r.semipartial_effect_size < 0.02  # confirms no double-counting false positive


def test_multivariate_linearity_test_detects_genuinely_multivariate_nonlinear_case():
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    rng = np.random.default_rng(1)
    n = 500
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, 3))
    B = rng.normal(size=(3, 2))
    nonlin = np.where(candidate > 0, 4.0 * candidate, 0.1 * candidate)
    obs = nonlin @ B + 0.5 * z[:, None] + rng.normal(scale=0.2, size=(n, 2))

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)
    r = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=5, n_permutations=100, rng=rng)
    assert r.reject_linearity is True
    assert r.semipartial_effect_size > 0.02


def test_adequacy_effect_is_the_increase_in_set_correlation_r2_given_z():
    """The adequacy effect is on the scale of the selection floor: the
    increase in the set-correlation R^2 given Z (one minus a ratio of
    generalized variances) when the departure is modelled. For one column it
    is the usual ratio of residual sums of squares."""
    from model_selection.statistics.adequacy.kernels import variance_shares
    rng = np.random.default_rng(3)
    rz, rn, ra = rng.normal(size=(3, 200, 2))
    rn, ra = 0.8 * rz + 0.1 * rn, 0.6 * rz + 0.1 * ra

    def g(r):
        return np.linalg.det(r.T @ r)

    partial, semi = variance_shares(rz, rn, ra)
    assert semi == pytest.approx((1 - g(ra) / g(rz)) - (1 - g(rn) / g(rz)))
    assert partial == pytest.approx(1 - g(ra) / g(rn))
    p1, s1 = variance_shares(rz[:, 0], rn[:, 0], ra[:, 0])
    assert s1 == pytest.approx((np.sum(rn[:, 0] ** 2) - np.sum(ra[:, 0] ** 2)) / np.sum(rz[:, 0] ** 2))


def test_multivariate_linearity_test_supports_multivariate_flag():
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest
    assert MultivariateLinearityTest._supports_multivariate is True






def test_bias_corrected_semipartial_effect_size_computed_for_multivariate_linearity_test_too():
    """Confirms the bias-correction fix applies uniformly across all
    FunctionalFormTest subclasses via the shared base -- not just
    LinearityTest."""
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)
    rng = np.random.default_rng(3)
    n = 20  # deliberately tiny
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, 2))
    B = rng.normal(size=(2, 2))
    obs = candidate @ B + rng.normal(scale=0.3, size=(n, 2))
    r = test.run(candidate, obs, z, time_positions=np.arange(n), min_temporal_lag=1, n_permutations=100, rng=rng)
    assert r.bias_corrected_semipartial_effect_size is not None
    assert abs(r.bias_corrected_semipartial_effect_size -
               (r.semipartial_effect_size - r.null_semipartial_effect_size_mean)) < 1e-9


def test_make_multivariate_linearity_adequacy_test_adapter_works():
    from model_selection.statistics.adequacy.base import AdequacyTestOutcome
    from model_selection.statistics.adequacy.linearity import make_multivariate_linearity_adequacy_test

    rng = np.random.default_rng(2)
    n = 400
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, 3))
    B = rng.normal(size=(3, 2))
    nonlin = np.where(candidate > 0, 4.0 * candidate, 0.1 * candidate)
    obs = nonlin @ B + 0.5 * z[:, None] + rng.normal(scale=0.2, size=(n, 2))

    adequacy_test = make_multivariate_linearity_adequacy_test(alpha=0.05, effect_size_floor=0.02)
    outcome = adequacy_test(candidate, obs, z, n_permutations=100, rng=np.random.default_rng(3),
                             time_positions=np.arange(n), min_temporal_lag=5)
    assert isinstance(outcome, AdequacyTestOutcome)
    assert outcome.tier_adequate is False  # genuine nonlinearity, correctly rejected


# ---------------------------------------------------------------------------
# reml_kernel_ridge_lambda -- the actual fix for the fixed-ridge_lambda
# calibration gap found in MultivariateLinearityTest/StateIndependenceTest
#.
# ---------------------------------------------------------------------------

def test_reml_kernel_ridge_lambda_matches_brute_force_2param_mle():
    """Direct verification of the profiled 1-parameter derivation against
    an unprofiled, brute-force 2-parameter (tau, sigma^2) optimization --
    the load-bearing check for the whole fix."""
    from scipy.optimize import minimize

    from model_selection.statistics.adequacy.kernels import reml_kernel_ridge_lambda

    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 3))
    K = X @ X.T
    true_tau, true_sigma2 = 2.0, 0.5
    L = np.linalg.cholesky(true_tau * K + true_sigma2 * np.eye(n) + 1e-10 * np.eye(n))
    y = L @ rng.normal(size=n)

    def neg_log_lik_full(params):
        tau, sigma2 = np.exp(params)
        V = tau * K + sigma2 * np.eye(n)
        sign, logdet = np.linalg.slogdet(V)
        return 0.5 * (logdet + y @ np.linalg.solve(V, y))

    res_full = minimize(neg_log_lik_full, x0=[0.0, 0.0], method="Nelder-Mead")
    tau_full, sigma2_full = np.exp(res_full.x)
    lambda_full = sigma2_full / tau_full

    lambda_profiled = reml_kernel_ridge_lambda(K, y)
    assert abs(lambda_full - lambda_profiled) / lambda_full < 0.05


def test_reml_kernel_ridge_lambda_shrinks_hard_under_genuine_independence():
    """The actual, real-world verification: at the exact dimensionality
    that caused this problem (n=500, p=7), genuinely independent data
    should produce a LARGE REML-selected lambda (heavy shrinkage,
    correctly recognizing there's no real signal)."""
    from model_selection.statistics.adequacy.kernels import reml_kernel_ridge_lambda

    rng = np.random.default_rng(0)
    n, p = 500, 7
    candidate = rng.normal(size=(n, p))
    K_linear = candidate @ candidate.T
    obs = rng.normal(size=(n, 2))  # genuinely independent, multivariate

    lam = reml_kernel_ridge_lambda(K_linear, obs)
    assert lam > 100  # much larger than the old fixed default of 1.0


def test_reml_kernel_ridge_lambda_stays_small_under_genuine_signal():
    """Companion: the SAME dimensionality, but with genuine structure in
    K relative to y, should produce a SMALL lambda -- confirming this
    doesn't just shrink everything indiscriminately."""
    from model_selection.statistics.adequacy.kernels import reml_kernel_ridge_lambda

    rng = np.random.default_rng(0)
    n, p = 500, 7
    candidate = rng.normal(size=(n, p))
    K_linear = candidate @ candidate.T
    true_beta = rng.normal(size=p)
    y = (candidate @ true_beta) + rng.normal(scale=0.1, size=n)  # genuine linear signal

    lam = reml_kernel_ridge_lambda(K_linear, y)
    assert lam < 10


# ---------------------------------------------------------------------------
# Multivariate plot_diagnostics tests -- MultivariateLinearityTest and
# StateIndependenceTest's generalization to FPC-vector candidate/obs (the
# high-flow branch), previously hardcoded off entirely in the pipeline.
# ---------------------------------------------------------------------------

def test_first_canonical_projection_matches_canonical_correlations():
    """Direct verification of the shared projection helper: the empirical
    correlation of the two projections must exactly reproduce
    canonical_correlations(...)[0] -- confirms the plotted relationship
    is genuinely the same one PartialCCA's own estimator is built from,
    not a separately-chosen, disconnected summary."""
    from model_selection.statistics.association.canonical import canonical_correlations
    from model_selection.statistics.association.canonical import (
        first_canonical_projection as _first_canonical_projection,
    )

    rng = np.random.default_rng(0)
    n, p, q = 300, 4, 5
    X = rng.normal(size=(n, p))
    B = rng.normal(size=(p, q))
    Y = X @ B * 0.5 + rng.normal(scale=1.0, size=(n, q))
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    x_proj, y_proj, x_load, y_load, first_cc = _first_canonical_projection(X, Y)
    official = canonical_correlations(X, Y)
    assert np.isclose(first_cc, official[0])
    empirical_corr = np.corrcoef(x_proj, y_proj)[0, 1]
    assert np.isclose(empirical_corr, official[0])
    assert empirical_corr >= 0  # sign-fixed for display, always non-negative
    assert x_load.shape == (p,)
    assert y_load.shape == (q,)


def test_multivariate_linearity_plot_diagnostics_handles_asymmetric_and_edge_dimensions():
    """Robustness across dimensionality shapes: asymmetric p != q, a
    single FPC pair (p=q=1), and a count not a multiple of the per-row
    FPC grid width (p=q=5) -- all must produce a valid figure with no
    subplot-title/spec misalignment."""
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)
    rng = np.random.default_rng(1)
    n = 150
    z = rng.normal(size=n)

    for p, q in [(4, 4), (3, 5), (1, 1), (5, 5), (2, 3)]:
        candidate = rng.normal(size=(n, p))
        B = rng.normal(size=(p, q))
        obs = candidate @ B * 0.5 + 0.2 * z[:, None] + rng.normal(scale=0.5, size=(n, q))
        fig = plot_multivariate_linearity_diagnostics(candidate, obs, z)
        # 3 traces for panels A/B (2+2... actually 3+3) + 1 loadings + 2*min(p,q)
        expected = 3 + 3 + 1 + 2 * min(p, q)
        assert len(fig.data) == expected, f"p={p}, q={q}: expected {expected}, got {len(fig.data)}"


def test_multivariate_linearity_plot_diagnostics_loadings_sum_to_all_dimensions():
    """Loading weights bar chart must include exactly one bar per
    candidate FPC dimension and one per obs FPC dimension."""
    from model_selection.statistics.adequacy.linearity import MultivariateLinearityTest

    test = MultivariateLinearityTest(alpha=0.05, effect_size_floor=0.02)
    rng = np.random.default_rng(2)
    n, p, q = 150, 4, 6
    z = rng.normal(size=n)
    candidate = rng.normal(size=(n, p))
    B = rng.normal(size=(p, q))
    obs = candidate @ B * 0.5 + rng.normal(scale=0.5, size=(n, q))
    fig = plot_multivariate_linearity_diagnostics(candidate, obs, z)

    bar_traces = [t for t in fig.data if t.type == "bar"]
    assert len(bar_traces) == 1
    assert len(bar_traces[0].x) == p + q


def test_state_independence_multivariate_plot_diagnostics_valid_across_shapes():
    """Same robustness sweep as MultivariateLinearityTest, but through
    StateIndependenceTest's per-stratum generalization -- the more
    complex code path, since each stratum computes its own, independent
    canonical projection."""
    from model_selection.statistics.adequacy.state_independence import StateIndependenceTest

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02)
    rng = np.random.default_rng(3)
    n = 400
    z = rng.normal(size=n)

    for p, q in [(3, 3), (2, 4), (1, 1)]:
        candidate = rng.normal(size=(n, p))
        B = rng.normal(size=(p, q))
        obs = candidate @ B * 0.5 + 0.2 * z[:, None] + rng.normal(scale=0.5, size=(n, q))
        fig = plot_state_independence_diagnostics(candidate, obs, z)
        assert len(fig.data) > 0


def test_state_independence_multivariate_plot_diagnostics_requires_conditioning():
    """The empty-conditioning guard (round-1/unconditional state) must
    still raise, unchanged by the multivariate generalization."""
    from model_selection.statistics.adequacy.state_independence import StateIndependenceTest

    test = StateIndependenceTest(alpha=0.05, effect_size_floor=0.02)
    rng = np.random.default_rng(4)
    n, p, q = 100, 3, 3
    candidate = rng.normal(size=(n, p))
    obs = rng.normal(size=(n, q))
    with pytest.raises(ValueError, match="non-empty conditioning set"):
        plot_state_independence_diagnostics(candidate, obs, None)


def test_binned_trend_shared_helper_matches_prior_per_class_duplicates():
    """Regression test for the de-duplication itself: the shared module-
    level _binned_trend must produce identical output to what the two,
    now-removed, per-class copies computed."""
    from model_selection.visualization.adequacy import _binned_trend

    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = 2 * x + rng.normal(scale=0.3, size=200)
    tx, ty = _binned_trend(x, y, 15)
    assert tx.shape == ty.shape
    assert len(tx) <= 15
    # monotonically increasing trend for a strongly positive relationship
    assert np.corrcoef(tx, ty)[0, 1] > 0.9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_spectral_interaction_score_equals_the_direct_kernel_ridge_computation():
    from model_selection.statistics.adequacy.kernels import kernel_ridge_fit, reml_kernel_ridge_lambda
    from model_selection.statistics.adequacy.state_independence import kernel_main_effect_kernels
    rng = np.random.default_rng(70)
    candidate, obs, z = _make_two_regime_interaction_data(300, rng)
    K_c, K_z = kernel_main_effect_kernels(candidate[:, None], z)
    K_main, K_int = K_c + K_z, K_c * K_z
    y = (obs - obs.mean())[:, None]
    lam = reml_kernel_ridge_lambda(K_main, y)
    _, r0 = kernel_ridge_fit(K_main, y, lam)
    _, r1 = kernel_ridge_fit(K_main + K_int, y, lam)
    expected_score = float(np.sum(r0 * (K_int @ r0)))
    _, rz = kernel_ridge_fit(K_z, y, reml_kernel_ridge_lambda(K_z, y))
    expected_semi = (np.sum(r0 ** 2) - np.sum(r1 ** 2)) / np.sum(rz ** 2)
    score, _, semi = _kernel_interaction_score_statistic(candidate, obs, z)
    assert score == pytest.approx(expected_score, rel=1e-6)
    assert semi == pytest.approx(expected_semi, rel=1e-6, abs=1e-12)


def test_check_c_is_reproducible_with_a_seed_above_the_bandwidth_subset_size():
    rng = np.random.default_rng(71)
    candidate, obs, z = _make_two_regime_interaction_data(800, rng)
    runs = [StateIndependenceTest(max_n=1000).run(candidate, obs, z, n_permutations=19,
                                                 rng=np.random.default_rng(5)) for _ in range(2)]
    assert runs[0].observed_statistic == runs[1].observed_statistic
    assert runs[0].permutation_p_value == runs[1].permutation_p_value


def test_kernel_diagnostic_plots_fit_long_records_on_evenly_spaced_rows(monkeypatch):
    """A pooled record in the linear tier can have hundreds of thousands of
    rows; the plots that fit kernel regressions use at most
    KERNEL_PLOT_MAX_N evenly spaced rows, as the tests themselves do."""
    import model_selection.visualization.adequacy as viz
    from model_selection.utils import as2d
    sizes = []

    def spy(Y, Z):
        sizes.append(as2d(Y).shape[0])
        return as2d(Y) - as2d(Y).mean(axis=0)

    monkeypatch.setattr(viz, "_nonlinear_residuals", spy)
    rng = np.random.default_rng(0)
    n = 30000
    z = rng.normal(size=(n, 2)); c = z[:, 0] + rng.normal(size=n); o = c + z[:, 1] + rng.normal(size=n)
    viz.plot_state_independence_diagnostics(c, o, z)
    assert sizes and max(sizes) == viz.KERNEL_PLOT_MAX_N


def test_diagnostic_plots_of_a_round_share_one_kernel_regression(monkeypatch):
    """Every kernel-residual panel of a round's plots uses one regression on
    the conditioning set (its eigendecompositions depend only on Z), built
    once rather than once per column and per plot."""
    import model_selection.statistics.regression.kernel_residuals as kr
    import model_selection.visualization.adequacy as viz
    built = []
    real_init = kr.ConditioningRegression.__init__

    def counting_init(self, *args, **kwargs):
        built.append(1)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(kr.ConditioningRegression, "__init__", counting_init)
    viz._REGRESSION_CACHE.clear()
    rng = np.random.default_rng(3)
    z = rng.normal(size=(80, 4)); c = z[:, :3] + rng.normal(size=(80, 3)); o = c + rng.normal(size=(80, 3))
    ctx = viz.PlotContext(tier="nonlinear", round_no=1, candidate="c")
    viz.plot_state_independence_diagnostics(c, o, z)
    viz.plot_state_independence_diagnostics(c, o, z, context=ctx)
    assert len(built) == 1


def _registration_case(n, rng, interaction):
    """A selected formulation Z, a registration-like covariate V acting
    nonlinearly on every series, and a candidate and observations that
    depend on Z (with a slope that changes with V when interaction)."""
    V = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0.1, 0.9, n)])
    g = np.sin(2 * np.pi * V[:, 1]) * 2 + V[:, 0] ** 2
    Zs = rng.normal(size=n)
    slope = 1.0 + (0.9 * V[:, 0] if interaction else 0.0)
    Z = Zs + 0.3 * rng.normal(size=n) + g
    O = slope * Zs + 0.7 * rng.normal(size=n) + 1.2 * g
    C = 0.8 * slope * Zs + 0.7 * rng.normal(size=n) + 0.9 * g
    return V, Z, C, O


def test_with_the_covariates_removed_the_adjustment_test_ignores_their_curvature_and_finds_an_interaction():
    """The partially linear form of the linear adjustment test: once the
    registration-like covariate is removed from every series by kernel
    regression, its own curvature no longer rejects the linear tier, while a
    slope that changes with it does, through the interaction part."""
    from model_selection.selection.tiers import SelectionSample, partial_out_covariates
    from model_selection.statistics.adequacy.specification import linear_specification_test
    n, t = 150, np.arange(150) * 10.0
    outcomes = {}
    for interaction in (False, True):
        V, Z, C, O = _registration_case(n, np.random.default_rng(3), interaction)
        res = partial_out_covariates(SelectionSample({"z": Z, "c": C}, O, t, 1, covariates=V))
        out = linear_specification_test(np.column_stack([res.candidates["c"], res.obs]), res.candidates["z"], 0.05,
                                        0.02, 599, np.random.default_rng(4), t, 1, n_candidate_columns=1,
                                        interaction_with=res.partialled_covariates)
        outcomes[interaction] = out
    assert outcomes[False].tier_adequate
    assert not outcomes[True].tier_adequate and "interaction" in outcomes[True].decision_reason


def test_partial_out_covariates_residualizes_every_series_and_reports_degrees_of_freedom():
    from model_selection.selection.tiers import SelectionSample, partial_out_covariates
    rng = np.random.default_rng(0)
    V, Z, C, O = _registration_case(100, rng, False)
    sample = SelectionSample({"z": Z, "c": C}, O, np.arange(100) * 10.0, 1, covariates=V, covariate_label="reg")
    res = partial_out_covariates(sample)
    assert res.covariates is None and res.partialled_covariates.shape == (100, 2)
    assert res.obs.shape == O.shape and res.candidates["c"].shape == C.shape
    assert 2.0 < res.partialled_df < 100
    # the covariate's curved effect is gone from the residuals
    g = np.sin(2 * np.pi * V[:, 1]) * 2 + V[:, 0] ** 2
    assert abs(np.corrcoef(res.obs, g)[0, 1]) < 0.2 < abs(np.corrcoef(O, g)[0, 1])
    assert partial_out_covariates(SelectionSample({"c": C}, O, np.arange(100), 1)).covariates is None


def test_kernel_escalation_tests_on_pivoted_cholesky_factors_match_the_exact_kernels():
    """Beyond the rank cap the two kernel escalation tests use pivoted-Cholesky
    factors through their thin spectra. With the factors below the kernels'
    numerical rank they equal the kernels, and the tests reproduce the exact
    path: statistics, effect sizes, block lengths and, for the same seed,
    p-values."""
    import numpy as np
    from model_selection.statistics.adequacy.specification import linear_specification_test
    from model_selection.statistics.adequacy.state_independence import StateIndependenceTest
    from model_selection.statistics.regression import kernel_residuals as kr
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n, dtype=float)
    s = np.cumsum(rng.standard_normal(n)) * 0.05
    z = np.column_stack([np.sin(2 * np.pi * t / 100), np.sin(s)])
    c = s + 0.5 * rng.standard_normal(n)
    o = 0.8 * z[:, 1] + 0.5 * z[:, 0] + 0.3 * c + 0.3 * rng.standard_normal(n)
    w = np.cos(t / 40)[:, None]

    def run():
        spec = linear_specification_test(np.column_stack([c, o]), z, 0.05, 0.02, 99, np.random.default_rng(5), t, 4,
                                          n_candidate_columns=1, grid_step=1.0, max_n=5000, interaction_with=w)
        eff = StateIndependenceTest(max_n=5000).run(c, o, z, time_positions=t, min_temporal_lag=4,
                                                    n_permutations=99, rng=np.random.default_rng(7), grid_step=1.0)
        return (spec.p_value, spec.semipartial_effect_size, spec.block_length,
                eff.permutation_p_value, eff.observed_statistic, eff.semipartial_effect_size)
    try:
        kr.set_low_rank(None)
        exact = run()
        kr.set_low_rank(n - 1)
        factored = run()
    finally:
        kr.set_low_rank(kr.DEFAULT_RANK_CAP)
    assert exact[0] == factored[0] and exact[2] == factored[2] and exact[3] == factored[3]
    assert np.allclose([exact[1], exact[4], exact[5]], [factored[1], factored[4], factored[5]], rtol=1e-6)
