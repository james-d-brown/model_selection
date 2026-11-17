"""
Conditional means of a censored series, by fitting the uncensored series and
then applying the censoring (docs/methods.md, Censoring at high flow).

A series censored at a threshold t is max(U, t), U its uncensored values.
Its conditional mean given Z is not a smooth function that a regression of
the censored values can follow: it is flat at t where U rarely exceeds t and
bends where exceedances become likely. It is instead estimated from U, whose
conditional mean is smooth, by location-scale smearing (Duan, 1983):

  1. U's conditional mean, mu(Z), by the tier's cross-fitted regression;
  2. U's conditional scale, s(Z), from the same regression of the log of the
     squared residual (Harvey, 1976), which is positive by construction,
     bounded to SCALE_BOUNDS times its median so that a near-zero estimate
     cannot place an extreme value in the pool;
  3. the pool of standardized residuals e_j = (U_j - mu_j) / s_j, centred and
     scaled to unit root mean square, reduced to POOL_POINTS equal-count
     bins of its sorted values, each represented by its mean and weighted by
     its count;
  4. at each time step i, the weighted mean of g(max(mu_i + s_i e_k, t)) over
     the reduced pool, for g the identity (the series itself) or each of the
     tier's kernel features.

Without censoring (t below every draw) the mean of the series is mu, the
regression's fit, because the reduced pool keeps the pool's mean of zero.
Reducing the pool changes the censored means by a small fraction of s (bins
of a few dozen sorted residuals at the operating sample sizes) and makes the
cost proportional to n rather than n squared. Time steps at which no draw
exceeds t have the censored mean t exactly and are not computed.

A kernel feature at a value that is not in the record is the Nystrom
extension of the tier's factor L of the series' feature kernel,
phi(v) = k(v, X) L (L'L)^-1, which reproduces L at the record's values
(regression.kernel_residuals.feature_kernel_at). It is tabulated on a fine
grid of values and interpolated linearly.
"""
from __future__ import annotations

import numpy as np

from model_selection.utils import as2d

GRID_POINTS = 4000
"""Values at which the kernel features are tabulated, from the threshold to
the largest draw."""
POOL_POINTS = 400
"""Equal-count bins to which the pool of standardized residuals is reduced."""
SCALE_BOUNDS = (0.2, 5.0)
"""Bounds on the conditional scale, as multiples of its median."""
_CHUNK_ENTRIES = 4_000_000
"""Draws times features held at once, to bound memory."""


def location_scale(U: np.ndarray, regression) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """(mu, s, pool, diagnostics): U's cross-fitted conditional mean and
    scale given the regression's conditioning set, and the full pool of
    standardized residuals (mean zero, unit root mean square). Diagnostics:
    the share of scales at a bound, the pool's largest absolute value and
    its kurtosis."""
    U = np.asarray(U, dtype=float).ravel()
    eps = regression.residuals(U[:, None])[0][:, 0]
    mu = U - eps
    log_sq = np.log(eps ** 2 + 1e-6 * np.mean(eps ** 2))
    scale = np.exp(0.5 * (log_sq - regression.residuals(log_sq[:, None])[0][:, 0]))
    med = float(np.median(scale))
    lo, hi = SCALE_BOUNDS[0] * med, SCALE_BOUNDS[1] * med
    bounded = float(np.mean((scale < lo) | (scale > hi)))
    scale = np.clip(scale, lo, hi)
    pool = eps / scale
    pool = pool - pool.mean()
    r = float(np.sqrt(np.mean(pool ** 2)))
    if r <= 0:
        return mu, np.zeros_like(scale), np.zeros_like(pool), {
            "scale_share_at_bound": bounded, "pool_max_abs": 0.0, "pool_kurtosis": float("nan")}
    pool = pool / r
    info = {"scale_share_at_bound": bounded, "pool_max_abs": float(np.abs(pool).max()),
            "pool_kurtosis": float(np.mean(pool ** 4))}
    return mu, scale * r, pool, info


def reduced_pool(pool: np.ndarray, points: int = POOL_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """(values, weights): the sorted pool in `points` equal-count bins (all
    of it when smaller), each bin's mean with weight its share of the pool.
    The weighted mean equals the pool's mean."""
    groups = np.array_split(np.sort(pool), min(points, pool.size))
    return np.array([g.mean() for g in groups]), np.array([g.size for g in groups], dtype=float) / pool.size


def _row_chunks(n: int, per_row: int):
    step = max(1, _CHUNK_ENTRIES // max(per_row, 1))
    for i0 in range(0, n, step):
        yield i0, min(n, i0 + step)


def censored_mean(U: np.ndarray, threshold: float, regression) -> np.ndarray:
    """E[max(U, t) | Z] at each sample, by location-scale smearing."""
    mu, scale, pool, _ = location_scale(U, regression)
    values, weights = reduced_pool(pool)
    out = np.full(mu.size, float(threshold))
    live = np.flatnonzero(mu + scale * values.max() > threshold)
    for i0, i1 in _row_chunks(live.size, values.size):
        rows = live[i0:i1]
        out[rows] = np.maximum(mu[rows, None] + scale[rows, None] * values[None, :], threshold) @ weights
    return out


def _feature_table(X: np.ndarray, L: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(grid, features on it): the Nystrom extension of the factor L of X's
    feature kernel at GRID_POINTS values from lo to hi."""
    from model_selection.statistics.regression.kernel_residuals import feature_kernel_at
    grid = np.linspace(lo, hi, GRID_POINTS)
    # L (L'L)^-1 is the transposed pseudo-inverse of L; computed from L's own
    # singular values, not by inverting L'L, whose condition number is squared
    # and would amplify rounding (Golub and Van Loan, 2013, section 5.3).
    project = np.linalg.pinv(L).T
    table = np.empty((grid.size, L.shape[1]))
    for i0, i1 in _row_chunks(grid.size, L.shape[0]):
        table[i0:i1] = feature_kernel_at(X, grid[i0:i1]) @ project
    return grid, table


def censored_feature_means(X: np.ndarray, L: np.ndarray, U: np.ndarray, threshold: float | None,
                           regression) -> tuple[np.ndarray, np.ndarray, dict]:
    """(E[phi(max(U, t)) | Z], E[max(U, t) | Z], diagnostics) at each sample:
    the conditional means of the censored series X = max(U, t)'s kernel
    features (the columns of the tier's factor L of X's feature kernel) and
    of X itself, by location-scale smearing; diagnostics of the scale and
    pool (location_scale). threshold None: no censoring (X = U), the
    features' conditional means derived from U's in the same way
    (experimental; review F3)."""
    x = as2d(np.asarray(X, dtype=float))[:, 0]
    mu, scale, pool, info = location_scale(U, regression)
    values, weights = reduced_pool(pool)
    hi = max(float(np.max(mu + scale * values.max())), float(np.max(x)))
    if threshold is None:
        lo, floor = min(float(np.min(mu + scale * values.min())), float(np.min(x))), -np.inf
    else:
        lo = floor = float(threshold)
        hi = max(hi, lo)
    if hi <= lo:
        hi = lo + 1.0
    grid, table = _feature_table(X, L, lo, hi)
    # With censoring the grid starts at t, so a time step with no draw above
    # t has the features at t: the table's first row.
    means, mean = np.tile(table[0], (mu.size, 1)), np.full(mu.size, lo)
    live = (np.arange(mu.size) if threshold is None
            else np.flatnonzero(mu + scale * values.max() > lo))
    for i0, i1 in _row_chunks(live.size, values.size * L.shape[1]):
        rows = live[i0:i1]
        V = np.maximum(mu[rows, None] + scale[rows, None] * values[None, :], floor)
        j = np.clip(np.searchsorted(grid, V) - 1, 0, grid.size - 2)
        f = ((V - grid[j]) / (grid[j + 1] - grid[j]))[..., None]
        means[rows] = np.einsum("ckr,k->cr", table[j] * (1.0 - f) + table[j + 1] * f, weights)
        mean[rows] = V @ weights
    return means, mean, info
