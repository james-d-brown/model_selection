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
    """Runge (2018, Algorithm 1): each sample receives the value of a member
    of its neighbourhood (itself or a neighbour), never a value from outside
    it, and values repeat only when a neighbourhood is exhausted. As in
    Runge's implementation, a sample may keep its own value, also when the
    neighbour index excludes temporal neighbours."""
    from model_selection.statistics.significance.neighborhood import compute_neighbor_index, local_permutation
    rng = np.random.default_rng(10)
    for lag in (0, 5):
        nn, _ = compute_neighbor_index(rng.normal(size=(300, 3)), 4, np.arange(300), lag, workers=1)
        assert not np.any(nn == np.arange(300)[:, None])
        kept = 0
        for _ in range(20):
            index, n_repeats = local_permutation(nn, rng)
            assert all(index[i] == i or index[i] in nn[i] for i in range(300))
            assert len(set(index.tolist())) == 300 - n_repeats
            kept += np.sum(index == np.arange(300))
        # Each of a sample's five members is about equally likely; far from never keeping its own value.
        assert 0.1 < kept / (20 * 300) < 0.4


def test_local_permutation_test_holds_its_level_with_two_conditioning_variables():
    """Candidate and observations both depend strongly on two conditioning
    variables and are conditionally independent. With every sample forced
    to take another sample's value (the former defect) the rate at the 5%
    level was 0.13 to 0.15 at 150 to 600 samples; with the sample itself in
    its neighbourhood, as in Runge's implementation, it was 0.04 to 0.05,
    and Runge's implementation gave 0.05 to 0.07 (paired runs on identical
    data, 400 to 1000 replicates). 99 null draws resolve the 5% level
    exactly; the bound 0.10 separates the two rates (Monte Carlo SE 0.018
    and 0.029 at 150 replicates)."""
    from model_selection.statistics.conditional_independence import CMITest
    rng = np.random.default_rng(11)
    reps, n, B = 150, 150, 99
    rejected = 0
    for _ in range(reps):
        z = rng.normal(size=(n, 2))
        s = z.sum(axis=1, keepdims=True)
        c, o = s + rng.normal(size=(n, 1)), s + rng.normal(size=(n, 1))
        r = CMITest(k=30, n_neighbors=5, workers=1).run({"c": c}, o, z, B, rng, np.arange(n), 0)
        rejected += (1 + np.sum(r.null[:, 0] >= r.observed[0])) / (B + 1) < 0.05
    assert rejected / reps <= 0.10


def test_local_permutation_keeps_own_value_without_neighbours():
    from model_selection.statistics.significance.neighborhood import local_permutation
    nn = np.full((4, 3), -1)
    index, _ = local_permutation(nn, np.random.default_rng(0))
    assert index.tolist() == [0, 1, 2, 3]


def test_cmi_neighbour_count_is_fixed():
    """The estimator's neighbour count is fixed (k = 5), so neighbourhoods
    shrink as the sample grows; it is capped at n - 1 on tiny samples."""
    t = CMITest(workers=1)
    assert [t.estimator_k(n) for n in (4, 20, 200, 2000)] == [3, 5, 5, 5]
    assert CMITest(k=8).estimator_k(2000) == 8
    with pytest.raises(ValueError):
        CMITest(k=0)
    rng = np.random.default_rng(12)
    c, o, z = _data(rng, n=80)
    res = t.run(c, o, z, 10, rng, np.arange(80), 0)
    assert res.diagnostics["dup"]["estimator_k"] == 5


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
    remaining generalized variance the candidate explains. The nonlinear
    effect is a held-out lower bound whose tuning never sees the block it
    predicts, so it reads below the in-sample value at this sample size (600
    samples, three columns), but on its scale; the RV coefficient is not on
    that scale (about a third of it here, with three features)."""
    from model_selection.statistics.conditional_independence import NonlinearResidualTest
    rng = np.random.default_rng(40)
    n = 600
    z = rng.normal(size=(n, 1))
    c = rng.normal(size=(n, 3)) + z
    o = np.column_stack([0.5 * c[:, 0] + z[:, 0] + rng.normal(size=n)]
                        + [z[:, 0] + rng.normal(size=n) for _ in range(2)])
    set_r2 = LinearResidualTest().run({"c": c}, o, z, 10, rng, np.arange(n), 0).diagnostics["c"][
        "partial_canonical_correlation"]
    test = NonlinearResidualTest()
    res = test.run({"c": c}, o, z, 10, rng, np.arange(n), 0)
    effect, null_mean = test.magnitude(c, o, z, np.arange(n), 0, res.observed[0], res.null[:, 0])
    assert null_mean is None
    assert 0.6 * set_r2 < effect < set_r2 + 0.01
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


def test_standardize_keeps_tied_values_together_and_equalizes_scale():
    """A point mass (e.g. zero flows or a censoring threshold) stays one value
    after standardization, and no new ties appear: the rescaling is linear."""
    from model_selection.statistics.conditional_independence.cmi import standardize
    rng = np.random.default_rng(51)
    x = np.column_stack([np.r_[np.zeros(50), rng.normal(size=50) + 5], 100 * rng.normal(size=100), np.ones(100)])
    s = standardize(x)
    assert np.unique(s[:50, 0]).size == 1 and np.unique(s[:, 0]).size == 51
    assert np.allclose(s[:, :2].mean(axis=0), 0) and np.allclose(s[:, :2].std(axis=0), 1)
    assert np.all(s[:, 2] == 0)

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


def test_the_linear_tier_uses_gaussian_multipliers(monkeypatch):
    """Pillai's trace is a quadratic form in the multipliers, so they are
    standard normal even where samples are independent, as in the nonlinear
    tier (docs/methods.md, Linear tier, Null distribution)."""
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
    assert seen and set(seen) == {"gaussian"}


def test_cmi_estimator_excludes_temporal_neighbours_exactly():
    """With a Theiler window, samples closer in time than the window are not
    neighbours in the estimate (brute force, Mesner and Shalizi, 2021, eqs.
    14, 18, 19); without one, nothing changes. Half
    the samples take one of two values in every variable, so some samples'
    k-th neighbours coincide with them and both branches of eq. 19 are
    exercised."""
    from scipy.special import digamma
    from model_selection.statistics.association.information import CachedCMI
    rng = np.random.default_rng(0)
    n, k = 200, 10
    x, y, z = (np.cumsum(rng.normal(size=(n, 1)), axis=0) for _ in range(3))
    for v in (x, y, z):
        v[::2] = rng.integers(0, 2, size=(n // 2, 1))
    t = np.arange(n, dtype=float)
    assert CachedCMI(y, z, k=k)(x) == CachedCMI(y, z, k=k, time_positions=t, theiler_window=0)(x)
    branches = set()
    for w in (3, 10):
        J, XZ, YZ = np.hstack([x, y, z]), np.hstack([x, z]), np.hstack([y, z])
        total = 0.0
        for i in range(n):
            ok = np.abs(t - t[i]) >= w
            eps = np.sort(np.abs(J[ok] - J[i]).max(axis=1))[k - 1]
            count = lambda S: int(np.sum(np.abs(S[ok] - S[i]).max(axis=1) <= eps))
            k_tilde = count(J)
            branches.add(k_tilde == k)
            if k_tilde == k:
                total += digamma(k) + digamma(count(z)) - digamma(count(XZ)) - digamma(count(YZ))
            else:
                total += np.log(k_tilde) + np.log(count(z)) - np.log(count(XZ)) - np.log(count(YZ))
        expected = total / n
        assert CachedCMI(y, z, k=k, time_positions=t, theiler_window=w)(x) == pytest.approx(expected, abs=1e-10)
    assert branches == {True, False}


def _dominant_events(n, d, beta, rng, sigma=1.0):
    """Independent events whose magnitude, shared by every series, is
    lognormal: a few large events dominate the residual products."""
    scale = np.exp(sigma * rng.normal(size=(n, 1)))
    z = rng.normal(size=(n, 2)) * scale
    c = 0.5 * z @ rng.normal(size=(2, d)) + rng.normal(size=(n, d)) * scale
    o = 0.5 * z @ rng.normal(size=(2, d)) + beta * c + rng.normal(size=(n, d)) * scale
    return c, o, z


def test_the_linear_statistic_is_pillais_trace_and_the_effect_is_pc_less_its_chance_level():
    from model_selection.statistics.association.canonical import canonical_correlations
    from model_selection.statistics.conditional_independence.linear import chance_level
    from model_selection.statistics.regression.linear import ols_fit
    rng = np.random.default_rng(21)
    n = 120
    z = rng.normal(size=(n, 2))
    c = rng.normal(size=(n, 3)) + z[:, :1]
    o = 0.4 * c + rng.normal(size=(n, 3))
    test = LinearResidualTest()
    res = test.run({"c": c}, o, z, 199, rng, np.arange(n), 0)
    rho = canonical_correlations(ols_fit(c, z)[1], ols_fit(o, z)[1])
    assert res.observed[0] == pytest.approx(np.sum(rho ** 2))
    pc = 1 - np.prod(1 - rho ** 2)
    assert res.diagnostics["c"]["partial_canonical_correlation"] == pytest.approx(pc)
    effect, chance = test.magnitude(c, o, z, np.arange(n), 0, res.observed[0], res.null[:, 0])
    assert chance == pytest.approx(chance_level(res.null[:, 0], 3, 3)) and 0 < chance < 0.5
    assert effect == pytest.approx(pc - chance)
    # For a single feature the statistic is the squared partial correlation.
    r1 = test.run({"c": c[:, 0]}, o[:, 0], z, 9, rng, np.arange(n), 0)
    rc, ro = ols_fit(c[:, 0], z)[1].ravel(), ols_fit(o[:, 0], z)[1].ravel()
    assert r1.observed[0] == pytest.approx(np.corrcoef(rc, ro)[0, 1] ** 2)


def test_the_chance_level_matches_the_null_mean_of_one_minus_wilks_lambda():
    """For independent normal samples the chance level from the null mean of
    Pillai's trace is the mean of 1 - Wilks' lambda (Bartlett's
    approximation, within a few per cent)."""
    from model_selection.statistics.association.canonical import canonical_correlations
    from model_selection.statistics.conditional_independence.linear import chance_level
    rng = np.random.default_rng(22)
    n, p, k = 80, 3, 2
    pt, wilks = [], []
    for _ in range(3000):
        z = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
        resid = lambda y: y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        rho = canonical_correlations(resid(rng.normal(size=(n, p))), resid(rng.normal(size=(n, p))))
        pt.append(np.sum(rho ** 2))
        wilks.append(1 - np.prod(1 - rho ** 2))
    assert np.mean(pt) == pytest.approx(p * p / (n - 1 - k), rel=0.03)
    assert chance_level(np.array(pt), p, p) == pytest.approx(np.mean(wilks), rel=0.05)
    assert chance_level(np.zeros(5), p, p) == 0.0 and chance_level(np.full(5, 50.0), p, p) == 1.0


def test_the_effective_sample_size_recovers_n_for_independent_multivariate_nulls():
    from model_selection.statistics.association.canonical import canonical_correlations
    from model_selection.statistics.power import effective_sample_size
    rng = np.random.default_rng(23)
    n, p, k = 100, 3, 2
    draws = []
    for _ in range(3000):
        z = np.column_stack([np.ones(n), rng.normal(size=(n, k))])
        resid = lambda y: y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        draws.append(np.sum(canonical_correlations(resid(rng.normal(size=(n, p))), resid(rng.normal(size=(n, p)))) ** 2))
    assert effective_sample_size(10 * n, np.array(draws), p, p, k) == pytest.approx(n, rel=0.03)


def test_a_strong_contribution_carried_by_a_few_dominant_events_is_detected():
    """When a few large events carry most of the residual products, many
    reweighted null draws have singular values above one. Pillai's trace
    stays defined for them, so a strong contribution is not held at a floor
    set by the share of such draws (the earlier Wilks' statistic, capped at
    one, could not fall below it)."""
    rng = np.random.default_rng(24)
    detected, above_one = 0, 0
    reps = 20
    for _ in range(reps):
        c, o, z = _dominant_events(140, 4, 2.0, rng)
        res = LinearResidualTest().run({"c": c}, o, z, 199, rng, np.arange(140), 0)
        detected += float(np.mean(res.null[:, 0] >= res.observed[0])) < 0.05
        above_one += np.any(res.null[:, 0] > 1.0)
    assert above_one > 0          # draws past a perfect fit occur, and are kept
    assert detected >= 0.8 * reps


def test_the_linear_tier_holds_its_level_with_dominant_events():
    rng = np.random.default_rng(25)
    rejections, reps = 0, 300
    for _ in range(reps):
        c, o, z = _dominant_events(100, 3, 0.0, rng)
        res = LinearResidualTest().run({"c": c}, o, z, 99, rng, np.arange(100), 0)
        rejections += (1 + np.sum(res.null[:, 0] >= res.observed[0])) / 100 < 0.05
    assert rejections / reps <= 0.08


def test_the_nonlinear_effect_size_sees_contributions_that_are_not_monotone_in_the_candidate():
    """The effect's direction is built from the candidate's residual kernel
    features with coefficients that may vary with the state, so it recovers
    a contribution that depends on the square of the candidate's own error,
    which a direction linear in the candidate's residual cannot see; with no
    contribution it is near zero. True shares: 0.453 and 0."""
    from model_selection.statistics.conditional_independence import NonlinearResidualTest
    rng = np.random.default_rng(3)
    n = 1500
    z = rng.uniform(-2, 2, size=n)
    e1, e2 = rng.standard_normal(n), rng.standard_normal(n)
    c = np.sin(2 * z) + e1
    base = np.cos(1.5 * z) + 0.5 * z + e2
    t = np.arange(n, dtype=float)
    quadratic, _ = NonlinearResidualTest().magnitude(c, base + 0.7 * (e1 ** 2 - 1), z[:, None], t, 0, 0.0,
                                                     np.zeros(1))
    null, _ = NonlinearResidualTest().magnitude(c, base, z[:, None], t, 0, 0.0, np.zeros(1))
    assert abs(quadratic - 0.453) < 0.06
    assert 0.0 <= null < 0.01


def test_the_effect_sizes_tuning_never_sees_the_block_it_predicts():
    """With enough blocks, each group of water years has its bandwidth and
    penalty chosen by cross-validation over the other groups, outside its
    buffer, so changing the observations' residuals in one block cannot change
    which fit predicts that block; with fewer blocks the tuning falls back to
    all blocks and says so."""
    import model_selection.statistics.conditional_independence.nonlinear as nl
    from model_selection.statistics.regression.kernel_residuals import crossfit_blocks, _folds
    rng = np.random.default_rng(5)
    n = 400
    t = np.arange(n, dtype=float)
    X1, X2 = rng.normal(size=(n, 3)), rng.normal(size=(n, 6))
    e = X2[:, :2] @ np.array([0.5, -0.5]) + rng.normal(size=n)
    block, k, _ = crossfit_blocks(t, 10.0)
    folds, _ = _folds(block, k, 1)
    assert len(folds) >= nl.MIN_NESTED_BLOCKS
    h, nested = nl._ridge_direction(e, [X1, X2], folds, t, 10.0)
    assert nested
    # Perturb the first block's residuals: its group is tuned on the other
    # groups, by fits without it, and it is predicted by a fit without it, so
    # its predictions do not change.
    first = folds[0][0]
    e2 = e.copy()
    e2[first] += 5.0 * rng.normal(size=first.size)
    h2, _ = nl._ridge_direction(e2, [X1, X2], folds, t, 10.0)
    assert np.allclose(h2[first], h[first])
    few, _ = _folds(*crossfit_blocks(t, 80.0)[:2], 1)
    assert len(few) < nl.MIN_NESTED_BLOCKS
    assert nl._ridge_direction(e, [X1, X2], few, t, 80.0)[1] is False


def test_the_refit_keeps_each_blocks_tuning():
    """_ridge_refit predicts each block with the bandwidth (design) and penalty
    _ridge_direction chose for it: on the same designs it reproduces the
    tuned predictions, so a refit on half-rank factors differs only through
    the rank (docs/methods.md, Nonlinear tier, Low-rank kernels)."""
    import model_selection.statistics.conditional_independence.nonlinear as nl
    from model_selection.statistics.regression.kernel_residuals import crossfit_blocks, _folds
    rng = np.random.default_rng(6)
    n = 400
    t = np.arange(n, dtype=float)
    X1, X2 = rng.normal(size=(n, 4)), rng.normal(size=(n, 8))
    e = X2[:, :2] @ np.array([0.5, -0.5]) + rng.normal(size=n)
    for lag in (10.0, 80.0):                                        # nested and not nested
        folds, _ = _folds(*crossfit_blocks(t, lag)[:2], 1)
        chosen = []
        h, _ = nl._ridge_direction(e, [X1, X2], folds, t, lag, chosen)
        assert len(chosen) == len(folds)
        assert np.allclose(nl._ridge_refit(e, [X1, X2], folds, chosen), h)
        half = nl._ridge_refit(e, [X1[:, :2], X2[:, :4]], folds, chosen)
        assert half.shape == h.shape and not np.allclose(half, h)


def _tuning_reference(e, designs, folds, t, lag):
    """A direct, slow implementation of the effect size's nested tuning, from
    raw rows: groups of TUNING_WATER_YEARS blocks, each tuned on the other
    groups outside its buffer by fits without both groups and both buffers;
    each block predicted by its usual cross-fitted fit."""
    import model_selection.statistics.conditional_independence.nonlinear as nl
    from model_selection.statistics.regression.kernel_residuals import RIDGE_PENALTIES
    e2 = e if e.ndim == 2 else e[:, None]
    n, rows, size = e2.shape[0], [p for p, _ in folds], nl.TUNING_WATER_YEARS
    K = len(rows)
    starts = list(range(0, K, size))
    if len(starts) > 1 and K - starts[-1] < size:
        starts.pop()
    groups = [list(range(a, starts[j + 1] if j + 1 < len(starts) else K)) for j, a in enumerate(starts)]
    R, B = [], []
    for blocks in groups:
        r = np.zeros(n, bool)
        r[np.concatenate([rows[b] for b in blocks])] = True
        lo, hi = t[r].min(), t[r].max()
        R.append(r)
        B.append((((t >= lo - lag) & (t < lo)) | ((t > hi) & (t <= hi + lag))) & ~r)

    def ridge(X, keep, lam):
        G = X[keep].T @ X[keep]
        return np.linalg.solve(G + lam * np.trace(G) / X.shape[1] * np.eye(X.shape[1]), X[keep].T @ e2[keep])

    out = np.zeros_like(e2)
    for g, blocks in enumerate(groups):
        best = (np.inf, None)
        for X in designs:
            for lam in RIDGE_PENALTIES:
                err = 0.0
                for h in range(len(groups)):
                    evaluate = R[h] & ~B[g]
                    if h != g and evaluate.any():
                        keep = ~(R[g] | B[g] | R[h] | B[h])
                        err += float(((e2[evaluate] - X[evaluate] @ ridge(X, keep, lam)) ** 2).sum())
                if err < best[0]:
                    best = (err, (X, lam))
        X, lam = best[1]
        for i in blocks:
            keep = np.ones(n, bool)
            keep[folds[i][0]] = False
            keep[np.setdiff1d(np.arange(n), folds[i][1])] = False
            out[rows[i]] = X[rows[i]] @ ridge(X, keep, lam)
    return out


def test_the_effect_sizes_tuning_matches_a_direct_implementation():
    """The fast tuning (block and segment Gram matrices, shared
    eigendecompositions) equals the direct one, with one and two observation
    columns, a remainder block, and a buffer longer than a block."""
    import model_selection.statistics.conditional_independence.nonlinear as nl
    from model_selection.statistics.regression.kernel_residuals import _folds
    rng = np.random.default_rng(4)
    for n, k, c, lag in ((400, 10, 1, 7.0), (400, 10, 2, 15.0), (440, 11, 1, 3.0), (480, 12, 1, 60.0)):
        t = np.arange(n, dtype=float)
        folds, _ = _folds(np.repeat(np.arange(k), n // k), k, 1)
        designs = [rng.normal(size=(n, 6)), 0.5 * rng.normal(size=(n, 9))]
        e = designs[0][:, :2] @ rng.normal(size=(2, c)) + rng.normal(size=(n, c))
        e = e[:, 0] if c == 1 else e
        h, nested = nl._ridge_direction(e, designs, folds, t, lag)
        assert nested
        assert np.max(np.abs(np.atleast_2d(h.T).T - _tuning_reference(e, designs, folds, t, lag))) < 1e-8


def test_fixed_b_draws_match_their_mean_and_kiefer_vogelsangs_critical_values():
    """The fixed-b variance functional for the Bartlett kernel has mean
    1 - b + b^2 / 3, and |Z| / sqrt(Q) has the two-sided 5% critical values
    Kiefer and Vogelsang (2005) tabulate as 1.96 + 2.9694 b + 0.4160 b^2
    - 0.5324 b^3. A 95% point from 20,000 draws varies by about 1%; the
    tolerance allows for that."""
    from model_selection.statistics.conditional_independence.multipliers import fixed_b_draws
    rng = np.random.default_rng(13)
    for b in (0.025, 0.1):
        Q = fixed_b_draws(b, 20000, rng)
        assert abs(Q.mean() - (1 - b + b * b / 3)) < 0.004
        cv = np.quantile(np.abs(rng.standard_normal(Q.size)) / np.sqrt(Q), 0.95)
        assert abs(cv / (1.96 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3) - 1) < 0.025
    assert np.all(fixed_b_draws(None, 5, rng) == 1.0) and np.all(fixed_b_draws(0.0, 5, rng) == 1.0)


def test_fixed_b_draws_at_one_step_on_a_long_record_are_cheap_and_near_one():
    """A bandwidth of one step on a 20-year hourly record (b = 1/N) is
    simulated on the record's own N points, not on 400 / b of them, with
    bounded memory, and Q is then close to one."""
    import time
    from model_selection.statistics.conditional_independence.multipliers import fixed_b_draws
    rng = np.random.default_rng(1)
    t0 = time.time()
    Q = fixed_b_draws(1 / 175_320, 200, rng, n_steps=175_320)
    assert time.time() - t0 < 30
    assert abs(Q.mean() - 1) < 0.001 and Q.std() < 0.01


def test_width_draws_and_degrees_of_freedom_follow_the_fixed_b_draws():
    from model_selection.statistics.conditional_independence.base import JointTestResult
    from model_selection.statistics.conditional_independence.multipliers import fixed_b_draws
    rng = np.random.default_rng(4)
    b = 0.05
    q = fixed_b_draws(b, 20000, rng)
    undivided = rng.chisquare(1, size=q.size)
    r = JointTestResult(["c"], np.array([1.0]), (undivided / q)[:, None], fixed_b=q)
    assert np.isclose(r.width_draws(0).mean(), undivided.mean() / q.mean())
    assert abs(r.null_degrees_of_freedom() * b / 1.5 - 1) < 0.1     # var(Q) ~ 4b/3, so nu ~ 1.5 / b
    # Clusters: the number of clusters less one, whatever the restrictions (the tiers' statistics invert no
    # variance matrix); the effective number of clusters is a diagnostic only.
    units = JointTestResult(["c"], np.array([1.0]), undivided[:, None],
                            diagnostics={"c": {"effective_clusters": 7.5}},
                            dependence={"independent_units": True, "n_blocks": 12})
    assert units.null_degrees_of_freedom(0, 1) == 11 and units.null_degrees_of_freedom(0, 9) == 11
    assert np.all(units.width_draws(0) == undivided)


def test_unrestricted_width_on_a_grid_is_not_widened_by_a_strong_effect():
    """On a regular grid the width of the null from the unrestricted residuals
    matches the test's own without an effect, and is not widened by a strong
    one (LinearResidualTest.unrestricted_width_draws)."""
    from scipy.signal import lfilter
    from model_selection.statistics.power import effective_sample_size
    n = 3000
    test = LinearResidualTest()
    pos = np.arange(n)

    def ar(rng):
        return lfilter([1], [1, -0.98], rng.normal(size=n)) * np.sqrt(1 - 0.98 ** 2)

    out = {}
    for label, rho2 in (("none", 0.0), ("strong", 0.8)):
        rs, us = [], []
        for i in range(6):
            rng = np.random.default_rng(i)
            z, c, e = ar(rng)[:, None], ar(rng), ar(rng)
            o = np.sqrt(rho2) * c + np.sqrt(1 - rho2) * e
            r = test.run({"c": c}, o, z, 199, rng, pos, 0, grid_step=1)
            rs.append(effective_sample_size(n, r.width_draws(0), 1, 1, 1))
            us.append(effective_sample_size(n, test.unrestricted_width_draws(c, o, z, 199, rng, pos, 0, grid_step=1),
                                            1, 1, 1))
        out[label] = (np.median(rs), np.median(us))
    assert abs(out["none"][1] / out["none"][0] - 1) < 0.2
    assert out["strong"][0] < 0.8 * out["strong"][1]
    assert abs(out["strong"][1] / out["none"][1] - 1) < 0.2
