"""Tests of the regression utilities used by the regression-based tiers."""
import numpy as np

from model_selection.statistics.regression.kernel_residuals import (
    ConditioningRegression, feature_kernel, kernel_factor,
)
from model_selection.statistics.regression.linear import ols_fit


def test_ols_residuals_are_orthogonal_to_design():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(100, 3))
    y = z @ rng.normal(size=(3, 2)) + rng.normal(size=(100, 2))
    fitted, resid = ols_fit(y, z)
    assert np.allclose(np.column_stack([np.ones(100), z]).T @ resid, 0.0, atol=1e-9)
    assert np.allclose(fitted + resid, y)


def test_ols_without_conditioning_centres():
    y = np.arange(10.0)
    _, resid = ols_fit(y, None)
    assert np.isclose(resid.mean(), 0.0) and np.allclose(resid.ravel(), y - y.mean())


def test_kernel_ridge_recovers_a_smooth_nonlinear_function():
    rng = np.random.default_rng(1)
    z = rng.uniform(-2, 2, size=(300, 1))
    y = np.sin(2 * z) + 0.05 * rng.normal(size=(300, 1))
    resid, _ = ConditioningRegression(z, np.arange(300)).residuals(y)
    assert np.sqrt(np.mean(resid ** 2)) < 0.1


def _refit_residuals(Z, y, t, buffer, reg):
    """Residuals by refitting kernel ridge with intercept, at the bandwidth and
    penalty the regression chose, without each sample's buffer set."""
    from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth
    from model_selection.statistics.regression.kernel_residuals import (
        BANDWIDTH_MULTIPLIERS, RIDGE_PENALTIES, _sq_dists, standardize)
    Zs = standardize(Z)
    yc = y - y.mean(axis=0)
    best = None
    for (s, U), mult in zip(reg.spectra, BANDWIDTH_MULTIPLIERS):
        for lam in RIDGE_PENALTIES:
            shrink = s / (s + lam)
            E = yc - U @ (shrink[:, None] * (U.T @ yc))
            h = 1.0 / len(y) + (U ** 2) @ shrink
            err = float(np.sum((E / (1 - h)[:, None]) ** 2))
            if best is None or err < best[0]:
                best = (err, mult, lam)
    _, mult, lam = best
    K = np.exp(-_sq_dists(Zs, Zs) / (2.0 * (median_heuristic_bandwidth(Zs) * mult) ** 2))
    out = np.empty(len(y))
    for i in range(len(y)):
        T = np.flatnonzero(np.abs(t - t[i]) > buffer)
        KT = K[np.ix_(T, T)]
        A = np.zeros((T.size + 1, T.size + 1))
        A[0, 0], A[0, 1:], A[1:, 0], A[1:, 1:] = T.size, KT.sum(0), KT.sum(1), KT @ KT + lam * KT
        sol = np.linalg.lstsq(A, np.concatenate([[y[T, 0].sum()], KT @ y[T, 0]]), rcond=None)[0]
        out[i] = y[i, 0] - (sol[0] + K[i, T] @ sol[1:])
    return out


def test_cross_fitted_residuals_equal_refits_without_each_samples_buffer():
    """The leave-set-out identity gives exactly the residuals of refitting
    without the samples within the buffer of each sample (h-block)."""
    rng = np.random.default_rng(5)
    t = np.arange(80, dtype=float)
    Z = np.cumsum(rng.normal(size=(80, 2)), axis=0) * 0.2
    y = (np.sin(Z[:, 0]) + 0.3 * rng.normal(size=80))[:, None]
    for buffer in (0.0, 3.0):
        reg = ConditioningRegression(Z, t, buffer=buffer)
        fast = reg.residuals(y)[0][:, 0]
        assert np.max(np.abs(fast - _refit_residuals(Z, y, t, buffer, reg))) < 1e-4 * np.max(np.abs(fast))


def test_out_of_fold_residuals_reflect_fit():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(150, 1))
    reg = ConditioningRegression(z, np.arange(150))
    y = z ** 2 + 0.1 * rng.normal(size=(150, 1))
    noise = rng.normal(size=(150, 1))
    r2 = 1 - np.sum(reg.residuals(y)[0] ** 2) / np.sum((y - y.mean()) ** 2)
    r2_noise = 1 - np.sum(reg.residuals(noise)[0] ** 2) / np.sum((noise - noise.mean()) ** 2)
    assert r2 > 0.9 and r2_noise < 0.1


def test_kernel_factor_reproduces_the_kernel():
    """The factor's inner products are the kernel to working precision, so
    quantities computed from it are the exact-kernel quantities."""
    rng = np.random.default_rng(3)
    for x in (rng.normal(size=(200, 1)), rng.normal(size=(80, 4))):
        K = feature_kernel(x)
        L = kernel_factor(K)
        assert np.max(np.abs(L @ L.T - K)) < 1e-8 * np.max(np.abs(K))


def test_regression_is_linear_in_its_targets():
    """Residuals of a linear combination of targets are that combination of
    the residuals when the smoothers are fixed by the same selection target,
    which is what makes residual feature maps exact."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=(120, 2))
    T = rng.normal(size=(120, 3)) + z[:, :1]
    reg = ConditioningRegression(z, np.arange(120))
    E, extra = reg.residuals(T, T @ np.array([[1.0], [2.0], [-1.0]]))
    assert np.allclose(extra, E @ np.array([[1.0], [2.0], [-1.0]]))


def test_buffer_is_dropped_only_where_too_few_samples_lie_outside_it():
    z = np.random.default_rng(0).standard_normal((60, 1))
    assert ConditioningRegression(z, np.arange(60), buffer=3).buffer_dropped_samples == 0
    # A buffer longer than the record leaves no other sample to fit from.
    assert ConditioningRegression(z, np.arange(60), buffer=100).buffer_dropped_samples == 60
