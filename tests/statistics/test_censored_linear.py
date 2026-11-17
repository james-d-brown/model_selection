"""The linear tier on censored series: the fit-then-censor means of the
censored candidate and observations, by least squares, added to the
conditioning set (conditional_independence.linear)."""
import numpy as np
import pytest
from scipy.stats import norm

from model_selection.preprocessing.censoring import censor_and_record
from model_selection.statistics.conditional_independence.linear import (
    LinearResidualTest, augmented_conditioning, censored_mean_columns, partial_canonical_correlation,
)
from model_selection.statistics.regression.linear import ols_fit


def _pair(seed, n=20000):
    """Uncensored C and O, both linear in z with normal errors and
    independent given z; their thresholds at the 0.9 quantile."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    U_c = 1.0 + 0.8 * z + 0.6 * rng.normal(size=n)
    U_o = -0.5 + 1.2 * z + 0.5 * rng.normal(size=n)
    return z, U_c, U_o, float(np.quantile(U_c, 0.9)), float(np.quantile(U_o, 0.9))


def test_without_censoring_the_test_equals_the_uncensored_linear_tier():
    z, U_c, U_o, _, _ = _pair(0, n=3000)
    t = np.arange(z.size, dtype=float)
    low_c, low_o = -1e6, -1e6   # below every value and every draw of the smearing
    C, O = censor_and_record(U_c, low_c), censor_and_record(U_o, low_o)
    a = LinearResidualTest().run({"c": C}, O, z[:, None], 49, np.random.default_rng(1), t, 1.0)
    b = LinearResidualTest().run({"c": U_c.copy()}, U_o.copy(), z[:, None], 49, np.random.default_rng(1), t, 1.0)
    assert np.allclose(a.observed, b.observed, rtol=1e-8)
    assert np.allclose(a.null, b.null, rtol=1e-6)


def test_the_censored_mean_column_matches_the_tobit_mean_under_normal_errors():
    z, U_c, _, t_c, _ = _pair(1)
    C = censor_and_record(U_c, t_c)
    col = censored_mean_columns(C, U_c.copy(), z[:, None])[:, 0]
    mu, s = 1.0 + 0.8 * z, 0.6
    d = (mu - t_c) / s
    tobit = mu * norm.cdf(d) + s * norm.pdf(d) + t_c * norm.cdf(-d)
    assert np.sqrt(np.mean((col - tobit) ** 2)) < 0.01


def test_each_column_depends_only_on_its_own_series_and_the_conditioning_set():
    z, U_c, U_o, t_c, t_o = _pair(2, n=4000)
    C, O = censor_and_record(U_c, t_c), censor_and_record(U_o, t_o)
    U_o2 = U_o + np.random.default_rng(9).normal(size=U_o.size)
    O2 = censor_and_record(U_o2, float(np.quantile(U_o2, 0.9)))
    a, b = censored_mean_columns(C, O, z[:, None]), censored_mean_columns(C, O2, z[:, None])
    assert np.allclose(a[:, 0], b[:, 0]) and not np.allclose(a[:, 1], b[:, 1])
    assert censored_mean_columns(U_c.copy(), U_o.copy(), z[:, None]) is None


@pytest.mark.parametrize("seed", [3, 4])
def test_the_added_columns_remove_the_bias_of_regressing_censored_series_directly(seed):
    """Censored C and O, independent given z: regressed on z alone, both keep
    the bend at their thresholds and the squared partial correlation is far
    above the acceptance floor; with the censored means added it is near
    zero."""
    z, U_c, U_o, t_c, t_o = _pair(seed)
    C, O = censor_and_record(U_c, t_c), censor_and_record(U_o, t_o)
    direct = partial_canonical_correlation(ols_fit(C, z[:, None])[1], ols_fit(O, z[:, None])[1])
    zz = augmented_conditioning(C, O, z[:, None])
    assert zz.shape[1] == 3
    augmented = partial_canonical_correlation(ols_fit(C, zz)[1], ols_fit(O, zz)[1])
    assert direct > 0.05
    assert augmented < 0.002
