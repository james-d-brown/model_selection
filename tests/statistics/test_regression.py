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


def _refit_residuals(Z, y, t, lag):
    """Blocked cross-fitted residuals by brute force: kernel ridge with an
    intercept refitted on each block's fitting samples (outside the block and
    its two neighbours), with the bandwidth and penalty that minimize the
    blocked error, all from the kernel matrix itself."""
    from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth
    from model_selection.statistics.regression.kernel_residuals import (
        BANDWIDTH_MULTIPLIERS, RIDGE_PENALTIES, _sq_dists, crossfit_blocks, standardize)
    Zs = standardize(Z)
    block, k, _ = crossfit_blocks(t, lag)
    best = None
    for mult in BANDWIDTH_MULTIPLIERS:
        K = np.exp(-_sq_dists(Zs, Zs) / (2.0 * (median_heuristic_bandwidth(Zs) * mult) ** 2))
        for lam in RIDGE_PENALTIES:
            out = np.empty(len(y))
            for b in range(k):
                P, T = np.flatnonzero(block == b), np.flatnonzero(np.abs(block - b) > 1)
                if P.size == 0:
                    continue
                KT = K[np.ix_(T, T)]
                A = np.zeros((T.size + 1, T.size + 1))
                A[0, 0], A[0, 1:], A[1:, 0], A[1:, 1:] = T.size, KT.sum(0), KT.sum(1), KT @ KT + lam * KT
                sol = np.linalg.lstsq(A, np.concatenate([[y[T, 0].sum()], KT @ y[T, 0]]), rcond=None)[0]
                out[P] = y[P, 0] - (sol[0] + K[np.ix_(P, T)] @ sol[1:])
            err = float(np.sum(out ** 2))
            if best is None or err < best[0]:
                best = (err, out)
    return best[1]


def test_cross_fitted_residuals_equal_refits_without_each_block_and_its_neighbours():
    """The residuals equal those of refitting kernel ridge on each block's
    fitting samples, with the bandwidth and penalty chosen by the same
    blocked cross-validation."""
    rng = np.random.default_rng(5)
    t = np.arange(120, dtype=float)
    Z = np.cumsum(rng.normal(size=(120, 2)), axis=0) * 0.2
    y = (np.sin(Z[:, 0]) + 0.3 * rng.normal(size=120))[:, None]
    for lag in (0.0, 9.0):
        fast = ConditioningRegression(Z, t, buffer=lag).residuals(y)[0][:, 0]
        assert np.max(np.abs(fast - _refit_residuals(Z, y, t, lag))) < 1e-4 * np.max(np.abs(fast))


def test_crossfit_blocks_are_at_least_one_dependence_lag_long():
    from model_selection.statistics.regression.kernel_residuals import crossfit_blocks
    t = np.arange(1000, dtype=float)
    assert crossfit_blocks(t, 10)[1:] == (20, False)
    assert crossfit_blocks(t, 99)[1:] == (10, False)      # 10 blocks of 99.9 steps
    assert crossfit_blocks(t, 300)[1:] == (4, True)       # fewer than 4 lags: still 4 blocks, flagged
    block, k, _ = crossfit_blocks(t, 99)
    assert np.all(np.diff(block) >= 0) and np.bincount(block).min() >= 99


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


def test_short_records_are_recorded():
    z = np.random.default_rng(0).standard_normal((60, 1))
    assert not ConditioningRegression(z, np.arange(60), buffer=3).blocks_short
    assert ConditioningRegression(z, np.arange(60), buffer=100).blocks_short


def test_shared_regression_reuses_a_conditioning_set_whatever_its_column_order():
    """The shared regression returns the same fit for the same conditioning
    set, in any column order, and its residuals are a fresh fit's."""
    import numpy as np
    from model_selection.statistics.regression.kernel_residuals import ConditioningRegression, shared_regression
    rng = np.random.default_rng(0)
    n = 120
    t = np.arange(n) * 5.0
    Z = rng.normal(size=(n, 3))
    y = np.sin(Z[:, 0]) + Z[:, 1] * Z[:, 2] + 0.3 * rng.normal(size=n)
    a = shared_regression(Z, t, buffer=10.0)
    b = shared_regression(Z[:, [2, 0, 1]], t, buffer=10.0)
    assert a is b
    fresh = ConditioningRegression(Z, t, buffer=10.0)
    assert np.allclose(a.residuals(y)[0], fresh.residuals(y)[0])
    assert shared_regression(Z, t, buffer=5.0) is not a


def test_parallel_and_sequential_runs_of_the_nonlinear_tier_agree():
    """The nonlinear tier's statistic, null draws and effect sizes do not
    depend on the number of workers, apart from rounding: the linear-algebra
    library sums in a different order with a different number of threads."""
    import numpy as np
    from model_selection.parallel import set_workers
    from model_selection.statistics.conditional_independence.nonlinear import NonlinearResidualTest
    rng = np.random.default_rng(0)
    n = 150
    t = np.arange(n) * 5.0
    z = rng.normal(size=(n, 2))
    obs = np.sin(z[:, 0]) + rng.normal(size=n)
    cands = {f"c{i}": obs + rng.normal(size=n) for i in range(3)}
    out = []
    for workers in (1, 3):
        set_workers(workers)
        try:
            test = NonlinearResidualTest()
            res = test.run(cands, obs, z, 99, np.random.default_rng(1), t, 2.0)
            effects = [test.magnitude(cands[nm], obs, z, t, 2.0, 0.0, res.null[:, 0])[0] for nm in res.names]
            out.append((res.observed, res.null, effects))
        finally:
            set_workers(None)
    assert np.allclose(out[0][0], out[1][0], rtol=1e-10, atol=1e-12)
    assert np.allclose(out[0][1], out[1][1], rtol=1e-10, atol=1e-12)
    assert np.allclose(out[0][2], out[1][2], rtol=1e-10, atol=1e-12)


def test_pooled_effect_sizes_equal_one_at_a_time_effect_sizes():
    import numpy as np
    from model_selection.parallel import set_workers
    from model_selection.statistics.conditional_independence.nonlinear import NonlinearResidualTest
    import model_selection.statistics.regression.kernel_residuals as kr
    rng = np.random.default_rng(2)
    n = 120
    t = np.arange(n) * 5.0
    z = rng.normal(size=(n, 2))
    obs = np.cos(z[:, 1]) + rng.normal(size=n)
    cands = {f"c{i}": obs + rng.normal(size=n) for i in range(4)}
    test = NonlinearResidualTest()
    one = {nm: test.magnitude(cands[nm], obs, z, t, 2.0, 0.0, np.zeros(1))[0] for nm in cands}
    kr._SHARED.clear()
    set_workers(3)
    try:
        pooled = {nm: v[0] for nm, v in NonlinearResidualTest().magnitudes(cands, obs, z, t, 2.0).items()}
    finally:
        set_workers(None)
    assert set(pooled) == set(one) and all(np.isclose(pooled[nm], one[nm]) for nm in one)


def test_low_rank_path_matches_exact_kernels():
    """Under set_low_rank, pivoted-Cholesky factors below the cap are the
    kernels' to working precision, so the regression and the nonlinear test
    (computed without n x n matrices) agree with the exact path."""
    import numpy as np
    from model_selection.statistics.regression import kernel_residuals as kr
    from model_selection.statistics.conditional_independence.nonlinear import NonlinearResidualTest
    rng = np.random.default_rng(0)
    n = 400
    t = np.arange(n, dtype=float)
    s = np.cumsum(rng.standard_normal(n)) * 0.1
    z = np.column_stack([np.sin(2 * np.pi * t / 100), np.sin(s)])
    cands = {"a": s + 0.5 * rng.standard_normal(n), "b": rng.standard_normal(n)}
    o = np.sin(s) + 0.2 * cands["a"] ** 2 + 0.3 * rng.standard_normal(n)
    results = []
    try:
        for rank in (None, n - 1):
            kr.set_low_rank(rank)
            kr._SHARED.clear()
            test = NonlinearResidualTest()
            r = test.run(cands, o, z, 99, np.random.default_rng(1), t, 3, grid_step=1.0)
            results.append((r.observed, r.null, r.block_length,
                            kr.ConditioningRegression(z, t, buffer=3).residuals(o)[0]))
    finally:
        kr.set_low_rank(None)
        kr._SHARED.clear()
    (obs0, null0, b0, e0), (obs1, null1, b1, e1) = results
    assert b0 == b1
    assert np.allclose(obs0, obs1, atol=1e-8) and np.allclose(null0, null1, atol=1e-8)
    assert np.allclose(e0, e1, atol=1e-8)


def test_matrix_free_trace_autocovariance():
    import numpy as np
    from model_selection.statistics.significance.block_length import (
        trace_autocovariance, trace_autocovariance_features)
    rng = np.random.default_rng(2)
    F = rng.standard_normal((200, 7))
    gi = np.sort(rng.choice(300, 200, replace=False))
    assert np.allclose(trace_autocovariance(F @ F.T, gi, 300), trace_autocovariance_features(F, gi, 300))


def test_a_capped_factor_records_its_trace_norm_error():
    """At the rank cap, the share recorded is the trace-norm error of the
    approximation relative to the kernel's trace (the residual is positive
    semi-definite, so its trace is its trace norm); below the cap it is 0."""
    from model_selection.statistics.regression.kernel_residuals import _rbf_column, pivoted_cholesky
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((300, 3))
    sq = (Z ** 2).sum(1)
    K = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0) / 2.0)
    G, share = pivoted_cholesky(np.ones(300), _rbf_column(Z, sq, 1.0), 40)
    R = K - G @ G.T
    assert G.shape[1] == 40 and share > 0
    assert np.linalg.eigvalsh(R).min() > -1e-8
    assert np.isclose(np.trace(R) / np.trace(K), share, rtol=1e-6)
    G, share = pivoted_cholesky(np.ones(300), _rbf_column(Z, sq, 1.0), 300)
    assert share == 0.0 and np.abs(K - G @ G.T).max() < 1e-8


def test_rank_change_is_measured_only_when_the_chosen_factor_is_capped():
    import numpy as np
    from model_selection.statistics.regression import kernel_residuals as kr
    rng = np.random.default_rng(6)
    n = 500
    t = np.arange(n, dtype=float)
    z = np.column_stack([np.sin(t / 30), rng.standard_normal(n)])
    y = np.sin(3 * z[:, 1]) + z[:, 0] + 0.2 * rng.standard_normal(n)
    try:
        kr.set_low_rank(None)
        exact = kr.ConditioningRegression(z, t, buffer=5)
        exact.residuals(y)
        assert exact.rank_change == 0.0 and exact.cap_share == 0.0
        kr.set_low_rank(8)                          # far below the kernels' rank: capped
        capped = kr.ConditioningRegression(z, t, buffer=5)
        capped.residuals(y)
        assert capped.cap_share > 0 and capped.rank_change > 0
    finally:
        kr.set_low_rank(kr.DEFAULT_RANK_CAP)
