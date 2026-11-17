"""
Conditional means of censored series by fitting the uncensored series and
then applying the censoring (statistics.regression.censored), the records
that tie a censored array to its uncensored values
(preprocessing.censoring), and their use by the nonlinear tier
(docs/methods.md, Censoring at high flow).
"""
import numpy as np
import pytest
from scipy.stats import norm

from model_selection.preprocessing.censoring import (
    CensoredSeries, Censoring, censor_and_record, censoring_record,
)
from model_selection.selection import SelectionSample
from model_selection.statistics.conditional_independence.nonlinear import NonlinearResidualTest
from model_selection.statistics.regression.censored import (
    SCALE_BOUNDS, _feature_table, censored_feature_means, censored_mean, location_scale, reduced_pool,
)
from model_selection.statistics.regression.kernel_residuals import ConditioningRegression, feature_factor_with_cap

N = 600


def _data(seed=0, n=N):
    """U = sin(2 z) + noise of sd 0.3 + 0.2 |z|, z uniform on (-2, 2)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-2, 2, size=n)
    mu, sd = np.sin(2 * z), 0.3 + 0.2 * np.abs(z)
    return z, mu, sd, mu + sd * rng.standard_normal(n)


def _regression(z):
    return ConditioningRegression(z[:, None], np.arange(z.size), buffer=0)


def test_the_pool_is_centred_and_scaled_and_the_scale_positive():
    z, _, _, U = _data()
    mu, scale, pool, info = location_scale(U, _regression(z))
    assert abs(pool.mean()) < 1e-12 and np.sqrt(np.mean(pool ** 2)) == pytest.approx(1.0)
    assert np.all(scale > 0)
    assert info["pool_max_abs"] == pytest.approx(np.abs(pool).max())
    lo, hi = np.min(scale) / np.median(scale), np.max(scale) / np.median(scale)
    assert SCALE_BOUNDS[0] - 1e-12 <= lo and hi <= SCALE_BOUNDS[1] + 1e-12


def test_the_reduced_pool_keeps_the_mean_and_barely_moves_the_censored_mean():
    rng = np.random.default_rng(9)
    pool = rng.standard_t(4, size=16000)
    pool = (pool - pool.mean()) / np.sqrt(np.mean((pool - pool.mean()) ** 2))
    values, weights = reduced_pool(pool)
    assert values.size == 400 and weights.sum() == pytest.approx(1.0)
    assert values @ weights == pytest.approx(0.0, abs=1e-12)
    for t in (-1.0, 0.0, 1.0, 2.0, 3.0):
        assert abs(np.maximum(values, t) @ weights - np.maximum(pool, t).mean()) < 2e-3


def test_a_near_zero_scale_does_not_put_an_extreme_value_in_the_pool():
    class Planted:
        """Regression whose fit of the log squared residual is planted: one
        time step's scale is a millionth of the others'."""
        def residuals(self, Y, *extra):
            y = Y[:, 0]
            if self.calls == 0:
                self.calls += 1
                return (y[:, None] - 0.0,)
            fit = np.full(y.size, np.log(np.mean(np.exp(y))))
            fit[0] = fit[0] - 2 * np.log(1e6)
            return ((y - fit)[:, None],)
    reg = Planted()
    reg.calls = 0
    U = np.random.default_rng(10).standard_normal(500)
    U[0] = 3.0
    _, scale, pool, info = location_scale(U, reg)
    assert info["scale_share_at_bound"] > 0 and info["pool_max_abs"] < 20


def test_without_censoring_the_mean_is_the_regression_fit():
    z, _, _, U = _data()
    reg = _regression(z)
    fit = U - reg.residuals(U[:, None])[0][:, 0]
    assert np.allclose(censored_mean(U, U.min() - 100.0, reg), fit)


def test_the_censored_mean_is_close_to_the_exact_mean():
    z, mu, sd, U = _data(seed=1, n=1500)
    t = float(np.quantile(U, 0.8))
    d = (mu - t) / sd
    exact = mu * norm.cdf(d) + sd * norm.pdf(d) + t * norm.cdf(-d)
    est = censored_mean(U, t, _regression(z))
    assert np.all(est >= t - 1e-12)
    assert np.sqrt(np.mean((est - exact) ** 2)) < 0.05


def test_the_feature_table_reproduces_the_factor_at_the_record_values():
    _, _, _, U = _data(seed=2)
    t = float(np.quantile(U, 0.9))
    X = Censoring.apply(U, t)
    L, _ = feature_factor_with_cap(X)
    grid, table = _feature_table(X, L, t, float(X.max()) + 1.0)
    at_x = np.stack([np.interp(X, grid, table[:, k]) for k in range(L.shape[1])], axis=1)
    assert np.abs(at_x - L).max() < 1e-4 * np.abs(L).max()


def test_feature_means_agree_with_the_mean_for_the_linear_feature():
    """The kernel contains the standardized variable linearly, so the mean of
    the features determines the mean of the variable; both come from the
    same draws."""
    z, _, _, U = _data(seed=3)
    t = float(np.quantile(U, 0.9))
    X = censor_and_record(U, t)
    reg = _regression(z)
    L, _ = feature_factor_with_cap(X)
    means, mean, _ = censored_feature_means(X, L, U, t, reg)
    assert means.shape == L.shape
    assert np.allclose(mean, censored_mean(U, t, reg))
    # Project the features on the standardized variable: its mean is recovered,
    # up to the extension's approximation for draws beyond the record's values.
    xs = (X - X.mean()) / X.std()
    w = np.linalg.lstsq(L, xs, rcond=None)[0]
    d = means @ w - (mean - X.mean()) / X.std()
    assert np.sqrt(np.mean(d ** 2)) < 1e-3 and np.abs(d).max() < 5e-3


def test_a_censored_array_carries_its_uncensored_values_and_threshold():
    U = np.array([0.1, 2.0, -1.0, 3.0])
    X = censor_and_record(U, 1.0)
    assert np.allclose(X, [1.0, 2.0, 1.0, 3.0])
    unc, t = censoring_record(X)
    assert np.allclose(unc, U) and t == 1.0
    assert censoring_record(X.copy()) is None and censoring_record(U) is None


def test_subsamples_and_surrogates_are_censored_with_records():
    rng = np.random.default_rng(4)
    unc = {"a": rng.normal(size=40)}
    obs_unc = rng.normal(size=40)
    c = Censoring(0.8)
    cs = CensoredSeries(unc, {"a": c.threshold(unc["a"])}, c.threshold(obs_unc), obs_unc)
    s = SelectionSample(cs.censored_candidates(), cs.censored_obs(), np.arange(40), 0, censored=cs)
    idx = np.arange(0, 40, 2)
    sub = s.subset(idx, idx.astype(float), 2.0)
    assert np.allclose(sub.candidates["a"], s.candidates["a"][idx]) and np.allclose(sub.obs, s.obs[idx])
    assert np.allclose(censoring_record(sub.obs)[0], obs_unc[idx])
    assert np.allclose(censoring_record(sub.candidates["a"])[0], unc["a"][idx])
    surrogate = cs.censor("a", rng.normal(size=40))
    assert censoring_record(surrogate)[1] == cs.thresholds["a"]


def test_the_nonlinear_tier_fits_then_censors_a_recorded_series():
    z, _, _, U = _data(seed=5)
    t = float(np.quantile(U, 0.9))
    X = censor_and_record(U, t)
    reg = _regression(z)
    test = NonlinearResidualTest()
    E, E_own, _ = test._residuals(X, reg)
    L = test._factor(X)
    means, mean, _ = censored_feature_means(X, L, U, t, reg)
    expected = L - means
    assert np.allclose(E, expected - expected.mean(axis=0))
    own = (X - mean) / X.std()
    assert np.allclose(E_own[:, 0], own - own.mean())
    assert test.censored_path(X, reg)["censored_means"] == "fit_then_censor"
    # An unrecorded array with the same values is treated as uncensored.
    Y = X.copy()
    E_direct, _, _ = test._residuals(Y, reg)
    assert not np.allclose(E_direct, E) and test.censored_path(Y, reg) == {}


def test_uncensored_feature_means_come_from_the_series_own_fit_by_default():
    """Without censoring, the features' means are those of the series' own
    smeared distribution given Z (no threshold), not separate regressions;
    the own column's residual is the regression residual."""
    z, _, _, U = _data(seed=7)
    reg = _regression(z)
    test = NonlinearResidualTest()
    assert test.uncensored_feature_means == "smear"
    E, E_own, _ = test._residuals(U, reg)
    L = test._factor(U)
    means, mean, _ = censored_feature_means(U, L, U, None, reg)
    expected = L - means
    assert np.allclose(E, expected - expected.mean(axis=0))
    fit = U - reg.residuals(U[:, None])[0][:, 0]
    assert np.allclose(mean, fit)
    own = (U - fit) / U.std()
    assert np.allclose(E_own[:, 0], own - own.mean())
    # The earlier method, kept for diagnostics, regresses each feature on Z.
    test.uncensored_feature_means = "regression"
    E_reg, _, _ = NonlinearResidualTest._residuals(test, U, reg)
    assert not np.allclose(E_reg, E)


def test_a_censored_sample_refuses_series_without_records():
    rng = np.random.default_rng(11)
    unc = {"a": rng.normal(size=30)}
    obs_unc = rng.normal(size=30)
    cs = CensoredSeries(unc, {"a": 0.5}, 0.5, obs_unc)
    SelectionSample(cs.censored_candidates(), cs.censored_obs(), np.arange(30), 0, censored=cs)
    with pytest.raises(ValueError, match="uncensored values"):
        SelectionSample({"a": Censoring.apply(unc["a"], 0.5)}, cs.censored_obs(), np.arange(30), 0, censored=cs)


def test_censored_effect_of_an_uninformative_candidate_is_small():
    """C and O both depend on z only, through the same bend; the effect of C
    on O is near zero (or negative) when O's means are fitted then censored."""
    rng = np.random.default_rng(6)
    z, mu, sd, U_o = _data(seed=6, n=800)
    U_c = mu + sd * rng.standard_normal(z.size)
    t_o, t_c = float(np.quantile(U_o, 0.9)), float(np.quantile(U_c, 0.9))
    O, C = censor_and_record(U_o, t_o), censor_and_record(U_c, t_c)
    effect, _ = NonlinearResidualTest().magnitude(C, O, z[:, None], np.arange(z.size), 0, 0.0, np.zeros(1))
    assert effect < 0.02
