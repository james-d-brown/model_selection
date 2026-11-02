"""
Tests of the adequacy tests' residual-resampling nulls and of the linear
specification test: size when the simpler tier is adequate (including a
candidate that closely tracks the conditioning set), and detection of the
departure each test targets.
"""
import numpy as np
import pytest

from model_selection.statistics.adequacy.linearity import LinearityTest, make_linearity_adequacy_test
from model_selection.statistics.adequacy.specification import linear_specification_test
from model_selection.statistics.adequacy.state_independence import (
    StateIndependenceTest, make_state_independence_adequacy_test,
)

N = 120


def _rate(fn, reps):
    return float(np.mean([fn() for _ in range(reps)]))


def test_null_fit_shapes_match_obs():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(N, 1)); c = rng.normal(size=N); o = rng.normal(size=N)
    assert LinearityTest()._null_fit(c, o, z).shape == o.shape
    assert StateIndependenceTest()._null_fit(c, o, z).shape == o.shape
    assert StateIndependenceTest()._null_fit(c, o, None).shape == o.shape


def test_residual_linearity_size_when_candidate_tracks_conditioning():
    rng = np.random.default_rng(1)
    test = make_linearity_adequacy_test(alpha=0.05, effect_size_floor=0.0)

    def once():
        z = rng.normal(size=N); c = z + 0.3 * rng.normal(size=N)
        o = z + 0.8 * c + 0.5 * rng.normal(size=N)
        return not test(c, o, z[:, None], n_permutations=79, rng=rng, time_positions=np.arange(N)).tier_adequate
    assert _rate(once, 40) <= 0.15


def test_state_independence_size_and_detection():
    rng = np.random.default_rng(2)
    test = make_state_independence_adequacy_test(alpha=0.05, effect_size_floor=0.0)

    def once(modified):
        z = rng.normal(size=N); c = np.tanh(z) + 0.5 * rng.normal(size=N)
        effect = c * (z > 0) * 2 if modified else 0.8 * np.tanh(c)
        o = np.sin(z) + effect + 0.3 * rng.normal(size=N)
        return not test(c, o, z[:, None], n_permutations=199, rng=rng, time_positions=np.arange(N)).tier_adequate
    assert _rate(lambda: once(False), 25) <= 0.2
    assert _rate(lambda: once(True), 20) >= 0.7


def test_linear_specification_size_and_detection():
    rng = np.random.default_rng(4)

    def once(nonlinear):
        z = rng.normal(size=(N, 2))
        y = (np.tanh(1.5 * z[:, 0]) + 0.3 * z[:, 1] ** 2) if nonlinear else z @ np.array([1.0, 0.5])
        y = y + 0.3 * rng.normal(size=N)
        return not linear_specification_test(y, z, 0.05, 0.0, 79, rng, np.arange(N), 0).tier_adequate
    assert _rate(lambda: once(False), 40) <= 0.15
    assert _rate(lambda: once(True), 10) >= 0.9


def test_linear_specification_effect_size_floor_blocks_trivial_departures():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(2000, 1))
    y = z[:, 0] + 0.02 * z[:, 0] ** 2 + rng.normal(size=2000)
    out = linear_specification_test(y, z, 0.05, 0.02, 49, rng, np.arange(2000), 0)
    assert out.tier_adequate
    assert "not rejected" in out.decision_reason


def test_linear_specification_reason_names_the_rejecting_column():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(400, 1))
    linear = z[:, 0] + 0.3 * rng.normal(size=400)
    curved = np.tanh(2.0 * z[:, 0]) + z[:, 0] ** 2 + 0.3 * rng.normal(size=400)
    labels = ["candidate 'a', feature 1 (PC1)", "candidate 'a', feature 2 (PC2)"]
    out = linear_specification_test(np.column_stack([linear, curved]), z, 0.05, 0.02, 49, rng,
                                    np.arange(400), 0, column_labels=labels)
    assert not out.tier_adequate
    assert "candidate 'a', feature 2 (PC2)" in out.decision_reason


def test_nonlinear_dependence_check_effect_is_a_share_of_the_unexplained_variation():
    """The effect size is the departure's share of the variation the linear
    fit on the selected formulations leaves unexplained (docs/methods.md, The
    cascade), not of the column's total variation: a strong linear part does
    not shrink it."""
    rng = np.random.default_rng(3)
    z = rng.normal(size=(400, 1))
    noise = 0.5 * rng.normal(size=400)
    weak = linear_specification_test(0.3 * z[:, 0] ** 2 + noise, z, 0.05, 0.0, 99, rng, np.arange(400), 0)
    strong = linear_specification_test(30 * z[:, 0] + 0.3 * z[:, 0] ** 2 + noise, z, 0.05, 0.0, 99, rng,
                                       np.arange(400), 0)
    assert strong.semipartial_effect_size == pytest.approx(weak.semipartial_effect_size, abs=0.02)
    assert 0.1 < weak.semipartial_effect_size < 0.5


def test_nonlinear_dependence_check_sees_directions_without_a_linear_slope():
    """Nonlinearity in a direction of Z in which the linear fit has no slope
    (here a symmetric effect of the second column) is detected; tests built on
    powers of the fitted values cannot see it."""
    rng = np.random.default_rng(8)
    n = 300
    z = rng.normal(size=(n, 2))
    y = 0.6 * z[:, 0] + 0.4 * (z[:, 1] ** 2 - 1) + rng.normal(size=n)
    assert not linear_specification_test(y, z, 0.05, 0.02, 199, rng, np.arange(n), 0).tier_adequate


def test_nonlinear_dependence_check_size_when_residual_spread_depends_on_z():
    rng = np.random.default_rng(9)
    n, reps = 200, 60
    rejected = 0
    for _ in range(reps):
        z = rng.normal(size=(n, 2))
        y = z @ np.array([0.6, 0.3]) + np.exp(0.7 * z[:, 0]) * rng.normal(size=n)
        rejected += not linear_specification_test(y, z, 0.05, 0.0, 199, rng, np.arange(n), 0).tier_adequate
    assert rejected / reps <= 0.12


def test_state_independence_residual_scaling_is_the_exact_residual_variance_of_the_fit():
    """The bootstrap divides each residual by its standard deviation under the
    null fit: 1 - leverage equals the diagonal of (I - H)(I - H)', with H the
    smoother matrix of centring followed by kernel ridge regression on
    K_c + K_z at the fixed penalty, built here explicitly."""
    from model_selection.statistics.adequacy.state_independence import kernel_main_effect_kernels
    rng = np.random.default_rng(11)
    n = 50
    z = rng.normal(size=(n, 2)); c = rng.normal(size=(n, 2)); o = rng.normal(size=(n, 2))
    test = StateIndependenceTest(ridge_lambda=0.7)
    K_c, K_z = kernel_main_effect_kernels(c, z)
    K = K_c + K_z
    centre = np.eye(n) - np.ones((n, n)) / n
    residual_operator = (np.eye(n) - K @ np.linalg.inv(K + 0.7 * np.eye(n))) @ centre
    expected = 1.0 - np.diag(residual_operator @ residual_operator.T)
    assert np.allclose(test._null_leverage(c, o, z), expected, atol=1e-8)
    assert np.allclose(o - test._null_fit(c, o, z), residual_operator @ o, atol=1e-8)


def test_state_independence_penalties_are_estimated_once_per_test(monkeypatch):
    """The penalties are estimated by REML on the observed data only, and held
    fixed in every bootstrap draw."""
    import model_selection.statistics.adequacy.state_independence as module
    calls = []
    real = module.reml_lambda_from_spectrum
    monkeypatch.setattr(module, "reml_lambda_from_spectrum", lambda *a, **k: calls.append(1) or real(*a, **k))
    rng = np.random.default_rng(12)
    z = rng.normal(size=(60, 1)); c = rng.normal(size=60); o = z[:, 0] + c + rng.normal(size=60)
    StateIndependenceTest().run(c, o, z, time_positions=np.arange(60) * 50.0, min_temporal_lag=10,
                                n_permutations=49, rng=rng)
    assert len(calls) == 2  # the additive null model and the regression on Z alone


def test_state_independence_size_on_multivariate_event_samples():
    """Multivariate candidate and observations, five conditioning variables,
    60 independent events, everything linear: the kernel null fit uses many
    degrees of freedom, so unscaled residuals would make the null too narrow
    (about a third of such tests rejected before the residual scaling)."""
    rng = np.random.default_rng(13)
    test = make_state_independence_adequacy_test(alpha=0.05, effect_size_floor=0.0)

    def once():
        n = 60
        z = rng.normal(size=(n, 5)); latent = z[:, :3] + rng.normal(size=(n, 3))
        c = 0.7 * latent + 0.7 * rng.normal(size=(n, 3))
        o = latent @ rng.normal(size=(3, 3)) * 0.7 + 0.6 * rng.normal(size=(n, 3))
        t = np.cumsum(rng.uniform(50, 150, size=n))
        return not test(c, o, z, n_permutations=99, rng=rng, time_positions=t, min_temporal_lag=10).tier_adequate
    assert _rate(once, 25) <= 0.12
