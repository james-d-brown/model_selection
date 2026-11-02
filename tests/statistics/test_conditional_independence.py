"""
Tests of the three conditional independence tiers: the shared-resampling
contract, null calibration under H0 (including candidates nearly identical
to the selected set) and power under H1.
Calibration checks use small replicate counts and loose bounds so they stay
fast; they guard against gross miscalibration.
"""
import numpy as np
import pytest

from model_selection.statistics.conditional_independence import CMITest, LinearResidualTest, NonlinearResidualTest
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights


def _max_p(res):
    w = int(np.argmax(res.observed))
    m = res.null.max(axis=1)
    return (np.sum(m >= res.observed[w]) + 1) / (len(m) + 1)


def _data(rng, n=60, d=2, h1=False, nonlinear=False):
    z = rng.normal(size=(n, d))
    f = (lambda x: np.tanh(1.5 * x) + 0.3 * x ** 2) if nonlinear else (lambda x: x)
    cands = {"dup": f(z) + 0.1 * rng.normal(size=(n, d)), "sim": f(z) + rng.normal(size=(n, d)),
             "ind": rng.normal(size=(n, d))}
    obs = f(z) + rng.normal(size=(n, d))
    if h1:
        u = rng.normal(size=(n, d))
        cands["info"] = z + u
        obs = obs + 0.8 * u
    return cands, obs, z


def _reject_rate(test, reps, seed, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        c, o, z = _data(rng, **kw)
        out.append(_max_p(test.run(c, o, z, 99, rng, np.arange(len(o)), 0)) < 0.05)
    return float(np.mean(out))


ALL = [LinearResidualTest(), NonlinearResidualTest(), CMITest(workers=1)]


@pytest.mark.parametrize("test", ALL, ids=lambda t: t.name)
def test_result_contract(test):
    rng = np.random.default_rng(0)
    c, o, z = _data(rng)
    res = test.run(c, o, z, 20, rng, np.arange(60), 0)
    assert res.names == list(c)
    assert res.observed.shape == (3,) and res.null.shape == (20, 3)
    assert set(res.diagnostics) == set(c)
    assert res.winner in c


@pytest.mark.parametrize("test", ALL, ids=lambda t: t.name)
def test_empty_conditioning_runs(test):
    rng = np.random.default_rng(1)
    c, o, _ = _data(rng)
    res = test.run(c, o, None, 20, rng, np.arange(60), 3)
    assert np.all(np.isfinite(res.observed)) and np.all(np.isfinite(res.null))


@pytest.mark.parametrize("test", [LinearResidualTest(), NonlinearResidualTest()], ids=lambda t: t.name)
def test_size_with_linear_main_effects(test):
    assert _reject_rate(test, 60, 11) <= 0.12


def test_nonlinear_tier_size_with_nonlinear_main_effects():
    assert _reject_rate(NonlinearResidualTest(), 60, 12, nonlinear=True) <= 0.12


def test_linear_tier_is_invalid_with_nonlinear_main_effects():
    """The linear tier's assumption fails here, which its adequacy tests must
    catch; this documents why escalation is required."""
    assert _reject_rate(LinearResidualTest(), 30, 13, nonlinear=True) > 0.3


@pytest.mark.parametrize("test", ALL, ids=lambda t: t.name)
def test_power_against_an_informative_candidate(test):
    assert _reject_rate(test, 10, 14, h1=True) >= 0.8


def test_linear_tier_null_is_valid_when_residual_spread_depends_on_z():
    """The multiplier null reproduces the variance of the residual products
    when their spread varies with Z, where a residual permutation is too
    narrow (Winkler et al., 2014)."""
    rng = np.random.default_rng(5)
    n, rejections, reps = 300, 0, 150
    for _ in range(reps):
        z = rng.normal(size=n)
        s = np.exp(0.7 * z)
        c = 0.6 * z + s * rng.normal(size=n)
        o = 0.6 * z + s * rng.normal(size=n)
        res = LinearResidualTest().run({"c": c}, o, z[:, None], 199, rng, np.arange(n), 1)
        rejections += (np.sum(res.null[:, 0] >= res.observed[0]) + 1) / 200 < 0.05
    assert rejections / reps <= 0.10


def test_linear_tier_null_matches_the_degrees_of_freedom_at_small_n():
    """With few samples and several conditioning columns the null mean of the
    statistic matches its expectation under independence, u / (n - k - 1)."""
    rng = np.random.default_rng(6)
    n, k = 60, 3
    z = rng.normal(size=(n, k))
    res = LinearResidualTest().run({"c": rng.normal(size=(n, 2))}, rng.normal(size=(n, 2)), z, 4000, rng,
                                   np.arange(n), 1)
    assert res.null[:, 0].mean() == pytest.approx(4 / (n - k - 1), rel=0.15)


def test_dependent_wild_weights_have_bartlett_correlation():
    rng = np.random.default_rng(7)
    w = dependent_wild_weights(np.arange(200), 10, 4000, rng)
    assert abs(w.var() - 1.0) < 0.05
    assert abs(np.corrcoef(w[:, 100], w[:, 105])[0, 1] - 0.5) < 0.06
    assert abs(np.corrcoef(w[:, 100], w[:, 115])[0, 1]) < 0.06


def test_dependent_wild_weights_independent_at_unit_lag():
    w = dependent_wild_weights(np.arange(50), 1, 3000, np.random.default_rng(8))
    assert abs(np.corrcoef(w[:, 10], w[:, 11])[0, 1]) < 0.06


def test_neighbor_index_excludes_temporal_neighbours():
    from model_selection.statistics.significance.neighborhood import compute_neighbor_index
    z = np.linspace(0, 1, 200)[:, None]  # neighbours in Z are also neighbours in time
    nn, pool = compute_neighbor_index(z, 5, time_positions=np.arange(200), min_temporal_lag=10, workers=1)
    valid = nn >= 0
    rows = np.repeat(np.arange(200)[:, None], nn.shape[1], axis=1)
    assert np.all(np.abs(nn[valid] - rows[valid]) >= 10)
    assert pool.min() >= 1


def test_local_permutation_draws_every_value_from_the_samples_own_neighbourhood():
    """Runge (2018, Algorithm 1): each sample receives a neighbour's value,
    never a value from outside its neighbourhood, and values repeat only
    when a neighbourhood is exhausted."""
    from model_selection.statistics.significance.neighborhood import compute_neighbor_index, local_permutation
    rng = np.random.default_rng(10)
    nn, _ = compute_neighbor_index(rng.normal(size=(300, 3)), 5, workers=1)
    index, n_repeats = local_permutation(nn, rng)
    assert all(index[i] in nn[i] for i in range(300))
    assert len(set(index.tolist())) == 300 - n_repeats


def test_local_permutation_keeps_own_value_without_neighbours():
    from model_selection.statistics.significance.neighborhood import local_permutation
    nn = np.full((4, 3), -1)
    index, _ = local_permutation(nn, np.random.default_rng(0))
    assert index.tolist() == [0, 1, 2, 3]


def test_cmi_neighbour_count_scales_with_sample_size():
    t = CMITest(workers=1)
    assert [t.estimator_k(n) for n in (20, 60, 200, 2000)] == [2, 6, 20, 200]
    assert CMITest(k=5).estimator_k(2000) == 5
    rng = np.random.default_rng(12)
    c, o, z = _data(rng, n=80)
    res = t.run(c, o, z, 10, rng, np.arange(80), 0)
    assert res.diagnostics["dup"]["estimator_k"] == 8


def test_permutation_p_value_is_exact_and_never_zero():
    from model_selection.statistics.significance.base import minimum_resamples, permutation_p_value
    null = np.arange(99, dtype=float)
    assert permutation_p_value(1000.0, null) == 1 / 100
    assert permutation_p_value(89.5, null) == 10 / 100  # 9 draws (90..98) plus one
    assert minimum_resamples(0.05) == 200 and minimum_resamples(0.01) == 1000


def test_coarse_resampling_warns():
    import warnings
    from model_selection.selection.config import AdequacyConfig, SelectionConfig
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SelectionConfig(n_permutations=100)
        AdequacyConfig()
        SelectionConfig()
    assert len(caught) == 1 and "10 / alpha" in str(caught[0].message)


def test_adequacy_resamples_derived_from_alpha():
    from model_selection.selection.config import AdequacyConfig
    from model_selection.statistics.significance.base import resamples_for_relative_error
    assert AdequacyConfig().resolved_n_permutations == 1900
    assert AdequacyConfig(alpha=0.01).resolved_n_permutations == resamples_for_relative_error(0.01, 0.1) == 9900
    assert AdequacyConfig(n_permutations=500).resolved_n_permutations == 500


def test_nonlinear_tier_magnitude_is_on_the_set_correlation_scale():
    """The effect compared with the floor is the same quantity in tiers 1 and
    2 when the conditional means are linear: the share of the observations'
    remaining generalized variance the candidate explains. The RV coefficient
    is not (it is about a third of it here, with three features)."""
    from model_selection.statistics.conditional_independence import NonlinearResidualTest
    rng = np.random.default_rng(40)
    n = 600
    z = rng.normal(size=(n, 1))
    c = rng.normal(size=(n, 3)) + z
    o = np.column_stack([0.5 * c[:, 0] + z[:, 0] + rng.normal(size=n)]
                        + [z[:, 0] + rng.normal(size=n) for _ in range(2)])
    set_r2 = LinearResidualTest().run({"c": c}, o, z, 10, rng, np.arange(n), 0).observed[0]
    test = NonlinearResidualTest()
    res = test.run({"c": c}, o, z, 10, rng, np.arange(n), 0)
    effect, null_mean = test.magnitude(c, o, z, np.arange(n), 0, res.observed[0], res.null[:, 0])
    assert null_mean is None
    assert abs(effect - set_r2) < 0.05  # cross-fitted, so slightly below the in-sample value
    assert res.observed[0] < 0.5 * set_r2


def test_multiplier_weights_have_the_bartlett_correlation_at_sparse_positions():
    """Weights are generated only at the samples' positions, with unit
    variance and correlation max(0, 1 - |t - t'| / lag)."""
    rng = np.random.default_rng(41)
    t = np.array([0, 10, 25, 1000, 175000])
    w = dependent_wild_weights(t, 40, 40000, rng)
    c = np.corrcoef(w.T)
    assert np.allclose(w.var(axis=0), 1.0, atol=0.03)
    assert c[0, 1] == pytest.approx(0.75, abs=0.02) and c[0, 2] == pytest.approx(0.375, abs=0.02)
    assert abs(c[2, 3]) < 0.02 and abs(c[3, 4]) < 0.02


def test_ties_are_separated_for_the_nearest_neighbour_estimator():
    from model_selection.statistics.conditional_independence.cmi import break_ties
    rng = np.random.default_rng(51)
    x = np.column_stack([np.r_[np.zeros(50), rng.normal(size=50)], rng.normal(size=100)])
    out = break_ties(x, rng)
    assert np.unique(out[:, 0]).size == 100 and np.max(np.abs(out[:, 0] - x[:, 0])) < 1e-8
    assert np.array_equal(out[:, 1], x[:, 1])


def test_calibration_surrogate_count_detects_three_times_alpha():
    from scipy.stats import binom
    from model_selection.statistics.conditional_independence.calibration import (
        binomial_rejection_count, surrogates_needed)
    n = surrogates_needed(0.05, 3.0)
    k = binomial_rejection_count(n, 0.05)
    assert n == 52 and binom.sf(k - 1, n, 0.05) <= 0.05 and binom.sf(k - 1, n, 0.15) >= 0.8
    assert binom.sf(k - 1, n - 1, 0.15) < 0.8 or binom.sf(k - 2, n - 1, 0.05) > 0.05


def test_sequential_p_value_stops_early_only_when_clearly_not_significant():
    from model_selection.statistics.conditional_independence import CMITest
    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=(n, 1))
    o = z + 0.5 * rng.normal(size=(n, 1))
    null_c = z + 0.5 * rng.normal(size=(n, 1))
    test = CMITest(n_neighbors=10)
    pos = np.arange(n)
    _, draws, p = test.sequential_p_value(null_c, o, z, 200, np.random.default_rng(1), pos, 1, 10)
    assert len(draws) < 200 and p == pytest.approx(10 / len(draws))
    informative = o + 0.1 * rng.normal(size=(n, 1))
    _, draws, p = test.sequential_p_value(informative, o, z, 200, np.random.default_rng(2), pos, 1, 10)
    assert len(draws) == 200 and p < 0.05



def test_sequential_stopping_gives_the_full_tests_decision():
    from model_selection.statistics.significance.base import stopping_exceedances
    for alpha, n in [(0.05, 199), (0.05, 200), (0.05, 500), (0.05, 1900), (0.01, 999)]:
        h = stopping_exceedances(alpha, n)
        assert (h + 1) / (n + 1) >= alpha          # h exceedances: not significant with all draws
        assert h / (n + 1) < alpha or h == 1        # h - 1 exceedances: significant


def test_cmi_theiler_window_applies_only_to_the_local_permutation(monkeypatch):
    """A Theiler window shorter than the dependence lag is used for the
    local-permutation neighbour search; with nothing to condition on, the
    time-block permutation keeps the dependence lag."""
    import model_selection.statistics.conditional_independence.cmi as cmi_module
    from model_selection.statistics.conditional_independence import CMITest
    seen = {}
    real_neighbours, real_blocks = cmi_module.compute_neighbor_index, cmi_module.time_block_permutation

    def neighbours(*args, **kwargs):
        seen["window"] = kwargs["min_temporal_lag"]
        return real_neighbours(*args, **kwargs)

    def blocks(n, time_positions, lag, rng):
        seen["block"] = lag
        return real_blocks(n, time_positions, lag, rng)

    monkeypatch.setattr(cmi_module, "compute_neighbor_index", neighbours)
    monkeypatch.setattr(cmi_module, "time_block_permutation", blocks)
    n = 120
    t = np.arange(n, dtype=float) * 10
    z = np.random.default_rng(0).normal(size=(n, 1))
    test = CMITest(theiler_window=15)
    test._permuter(z, n, t, 50)
    assert seen["window"] == 15
    test._permuter(None, n, t, 50)(np.random.default_rng(1))
    assert seen["block"] == 50
    CMITest()._permuter(z, n, t, 50)
    assert seen["window"] == 50


def test_rademacher_multipliers_only_where_every_pair_of_samples_is_independent():
    rng = np.random.default_rng(0)
    apart = np.arange(0, 500, 10)  # every pair at least the lag (10) apart
    w = dependent_wild_weights(apart, 10, 200, rng, independent="rademacher")
    assert set(np.unique(w)) == {-1.0, 1.0}
    w = dependent_wild_weights(np.arange(50), 1, 200, rng, independent="rademacher")
    assert set(np.unique(w)) == {-1.0, 1.0}
    close = np.arange(0, 100, 2)  # neighbours closer than the lag: correlated, standard normal increments
    w = dependent_wild_weights(close, 10, 200, rng, independent="rademacher")
    assert np.unique(w).size > 2
    assert np.unique(dependent_wild_weights(apart, 10, 200, rng)).size > 2  # default stays standard normal
    with pytest.raises(ValueError):
        dependent_wild_weights(apart, 10, 5, rng, independent="uniform")


def test_independent_multiplier_distributions_of_the_escalation_tests():
    """Rademacher for the regression statistic and for the effect-modification
    score (each residual keeps its own square); Gaussian for the other
    kernel statistic."""
    from model_selection.statistics.adequacy.linearity import LinearityTest, MultivariateLinearityTest
    from model_selection.statistics.adequacy.state_independence import StateIndependenceTest
    assert LinearityTest.independent_multipliers == "rademacher"
    assert MultivariateLinearityTest.independent_multipliers == "gaussian"
    assert StateIndependenceTest.independent_multipliers == "rademacher"


def test_the_linear_tier_requests_rademacher_multipliers(monkeypatch):
    import model_selection.statistics.conditional_independence.linear as linear_module
    seen = []
    real = linear_module.dependent_wild_weights

    def spy(*args, **kwargs):
        seen.append(kwargs.get("independent", "gaussian"))
        return real(*args, **kwargs)

    monkeypatch.setattr(linear_module, "dependent_wild_weights", spy)
    rng = np.random.default_rng(1)
    z = rng.normal(size=(60, 1))
    LinearResidualTest().run({"c": z[:, 0] + rng.normal(size=60)}, z[:, 0] + rng.normal(size=60), z, 49, rng,
                             np.arange(60), 1)
    assert seen and set(seen) == {"rademacher"}


def test_cmi_estimator_excludes_temporal_neighbours_exactly():
    """With a Theiler window, samples closer in time than the window are not
    neighbours in the estimate (brute force); without one, nothing changes."""
    from scipy.special import digamma
    from model_selection.statistics.association.information import CachedCMI
    rng = np.random.default_rng(0)
    n, k = 200, 10
    x, y, z = (np.cumsum(rng.normal(size=(n, 1)), axis=0) for _ in range(3))
    t = np.arange(n, dtype=float)
    assert CachedCMI(y, z, k=k)(x) == CachedCMI(y, z, k=k, time_positions=t, theiler_window=0)(x)
    for w in (3, 10):
        J, XZ, YZ = np.hstack([x, y, z]), np.hstack([x, z]), np.hstack([y, z])
        total = 0.0
        for i in range(n):
            ok = np.abs(t - t[i]) >= w
            eps = np.sort(np.abs(J[ok] - J[i]).max(axis=1))[k - 1]
            count = lambda S: int(np.sum(np.abs(S[ok] - S[i]).max(axis=1) < eps))
            total += digamma(count(z) + 1) - digamma(count(XZ) + 1) - digamma(count(YZ) + 1)
        expected = digamma(k) + total / n
        assert CachedCMI(y, z, k=k, time_positions=t, theiler_window=w)(x) == pytest.approx(expected, abs=1e-10)
