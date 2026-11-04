"""
Data-driven dependence lengths for autocorrelated samples.

Two lengths are estimated from a series' autocorrelation by the procedure of
Politis and White (2004), with the correction of Patton, Politis and White
(2009):

- the decorrelation lag m_hat, the lag beyond which the autocorrelation is
  statistically indistinguishable from zero (Politis, 2003): the first lag
  that starts a run of `run_length` consecutive autocorrelations inside
  +/- c sqrt(log10(n) / n), with c = 2 (Politis and White, 2004, Remark 2.3);
- the block length b, the circular-block-bootstrap block length that
  minimizes the mean squared error of the bootstrap variance of a mean:
  b = (2 G^2 / D_CB)^(1/3) n^(1/3), with the flat-top lag window
  lambda(s) = 1 for |s| <= 1/2, 2(1 - |s|) for 1/2 < |s| <= 1 over
  M = 2 m_hat lags, G = sum lambda(k/M) |k| R(k) and
  D_CB = (4/3) (sum lambda(k/M) R(k))^2.

Departures from the authors' reference code, which caps the lag search at
sqrt(n) + run_length lags and the block length at 3 sqrt(n): neither cap is
part of the method's theory, and both grow only as sqrt(n), while
hydrologic memory is a property of the basin, not of the record length. Here the search
extends to n/4 lags, the longest lag at which autocorrelations are usefully
estimated (Box and Jenkins, 1976), and the block length is not capped. Because
the lag window spans at most n/4 lags, the block length stays below about
0.45 n whenever the long-run variance estimate is positive, so at least two
blocks always fit.

When no run of insignificant autocorrelations occurs within the searched lags
(the dependence does not decay within the record), m_hat is the largest
significant lag, as in the reference code, and the condition is reported.

The series analysed for a test is the one whose sum forms the statistic: the
products of the two residual series being related: for two independent
series, the autocorrelation of their products at each lag is the product of
their autocorrelations, which sets the variance of the sum (Bayley and
Hammersley, 1946; Clifford, Richardson and Hémon, 1989). With
several product series the largest block length is used, because all
candidates share one resampling.

The estimate applies to samples on a regular time grid: evenly spaced
samples, evenly spaced subsamples (resampling.grid_subsample), and either of
these with missing values (samples absent from grid points). Lags are counted in grid
steps and the lengths converted back to time units. With missing values the
autocovariance at each lag is estimated from the pairs of observed values at
that lag (Parzen, 1963), multiplied by 1 - k/N as the standard estimator is,
so that it reduces to the standard estimator when nothing is missing; the
significance band and the block length use the number of observed values. For
samples not on a grid (event samples) the caller's dependence lag is used
unchanged.

See docs/methods.md, Dependence between samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_selection.utils import as2d

C_THRESHOLD = 2.0  # Politis and White (2004), Remark 2.3
DEFAULT_RUN_LENGTH = 5  # Politis and White (2004), Remark 2.3
# Fewest blocks before a test is flagged, a convention. With summable
# autocovariances and fourth cumulants, the Bartlett-weighted variance the
# resampling reproduces has relative bias -G/(b J) and relative variance
# (4/3) b/n (Andrews, 1991, Proposition 1), and at the Politis-White block
# length (b^3 = 2 G^2 n / D, D = (4/3) J^2) these are -sqrt(2 / (3B)) and
# 4 / (3B) with B = n / b blocks: a relative root-mean-square error of
# sqrt(2 / B), 20% at B = 50, of which the null is expected to be about 12%
# too narrow. Holds for estimated block lengths only, not for the caller's
# lag (event samples).
MIN_BLOCKS = 50


def expected_variance_shortfall(n_blocks: float) -> float:
    """Expected relative shortfall of the resampled null's variance at the
    Politis-White block length, sqrt(2 / (3B)) (see MIN_BLOCKS)."""
    return float(np.sqrt(2.0 / (3.0 * n_blocks)))
# Fewest observations from which autocorrelations are estimated: Box and
# Jenkins (1976) advise about 50 for a useful estimate of the autocorrelation
# function, a convention. Below it, a sample's dependence is not estimated and
# the caller's dependence lag (the spacing that made the samples approximately
# independent) is used, as for samples not on a grid.
MIN_ACF_N = 50


@dataclass
class DependenceEstimate:
    """Dependence lengths of one series (lags in samples).

    decorrelation_lag : m_hat.
    block_length : circular-block length.
    decays : a run of insignificant autocorrelations was found.
    max_lag_searched : the longest lag examined (n / 4).
    """
    decorrelation_lag: int
    block_length: int
    decays: bool
    max_lag_searched: int
    assessed: bool = True  # False when the series is too short (MIN_ACF_N) to estimate


def _autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocovariances of x at lags 0..max_lag. Missing values (NaN) are
    allowed: the lag-k value is the mean product over the pairs observed at
    lag k (Parzen, 1963) times (1 - k / N), N = len(x), which is the biased
    (divide-by-N) estimator when nothing is missing. Lags with no observed
    pair are 0."""
    N = x.size
    a = np.isfinite(x).astype(float)
    e = np.where(a > 0, x - np.nanmean(x), 0.0)
    size = 1 << int(np.ceil(np.log2(2 * N - 1)))

    def corr(v):
        f = np.fft.rfft(v, size)
        return np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]

    num = corr(e)
    if a.all():
        return num / N
    pairs = np.rint(corr(a))
    k = np.arange(max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        acov = np.where(pairs > 0, num / pairs, 0.0) * (1.0 - k / N)
    return acov


def searched_lags(N: int, run_length: int = DEFAULT_RUN_LENGTH) -> int:
    """Longest lag examined: N / 4 (Box and Jenkins, 1976), but at least
    run_length (when the series is that long), so that a run of
    insignificant autocorrelations can be found at all on short series."""
    return max(1, N // 4, min(max(1, int(run_length)), N - 1))


def estimate_dependence(x: np.ndarray, run_length: int = DEFAULT_RUN_LENGTH) -> DependenceEstimate:
    """Decorrelation lag and block length of one series on a regular grid
    (NaN marks grid points without a value)."""
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    n = int(np.isfinite(x).sum())
    run_length = max(1, int(run_length))
    max_lag = searched_lags(N, run_length)
    if n < MIN_ACF_N:
        return DependenceEstimate(1, 1, True, max_lag, assessed=False)
    return dependence_from_autocovariance(_autocovariances(x, max_lag), N, n, run_length)


def dependence_from_autocovariance(acov: np.ndarray, N: int, n: int,
                                   run_length: int = DEFAULT_RUN_LENGTH) -> DependenceEstimate:
    """The Politis–White decorrelation lag and block length from the
    autocovariances acov[0..max_lag] of a series spanning N grid steps with n
    observed values (see the module docstring)."""
    run_length = max(1, int(run_length))
    max_lag = acov.size - 1
    if n < MIN_ACF_N or acov[0] <= 0 or max_lag < 1:
        return DependenceEstimate(1, 1, True, max(max_lag, 1), assessed=False)

    rho = np.abs(acov[1:] / acov[0])  # lags 1..max_lag
    crit = C_THRESHOLD * np.sqrt(np.log10(n) / n)
    insignificant = (rho < crit).astype(int)
    decays = False
    if run_length <= insignificant.size:
        runs = np.convolve(insignificant, np.ones(run_length, dtype=int), mode="valid")
        starts = np.flatnonzero(runs == run_length)  # index j: lags j+1 .. j+run_length
        if starts.size:
            m_hat = int(starts[0]) + 1
            decays = True
    if not decays:
        significant = np.flatnonzero(insignificant == 0)
        m_hat = int(significant.max()) + 1 if significant.size else 1

    M = min(2 * m_hat, max_lag)
    k = np.arange(1, M + 1)
    s = k / M
    lam = np.where(s <= 0.5, 1.0, 2.0 * (1.0 - s))
    G = 2.0 * float(np.sum(lam * k * acov[1:M + 1]))
    long_run = acov[0] + 2.0 * float(np.sum(lam * acov[1:M + 1]))
    if long_run <= 0 or G == 0:
        b = 1.0
    else:
        b = (2.0 * G ** 2 / ((4.0 / 3.0) * long_run ** 2)) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b = int(max(1, np.ceil(b)))
    # Guard only: with M <= N/4 and a positive long-run variance b < N/2.
    return DependenceEstimate(m_hat, min(b, max(1, N // 2)), decays, max_lag)


def sampling_stride(time_positions: np.ndarray) -> float | None:
    """The constant time step between consecutive samples, or None when the
    samples are not evenly spaced."""
    t = np.sort(np.asarray(time_positions, dtype=float))
    if t.size < 2:
        return None
    d = np.diff(t)
    return float(d[0]) if d[0] > 0 and np.all(d == d[0]) else None


def grid_layout(time_positions: np.ndarray, grid_step: float | None = None) -> tuple[float, np.ndarray, int] | None:
    """(step, grid index of each sample, grid length) when the samples lie on
    a regular grid: the declared grid_step (samples on a grid with missing
    points), or else a constant spacing. None for samples not on a grid."""
    t = np.asarray(time_positions, dtype=float)
    if t.size < 2:
        return None
    step = grid_step if grid_step is not None else sampling_stride(t)
    if step is None or step <= 0:
        return None
    g = (t - t.min()) / step
    gi = np.rint(g).astype(np.int64)
    if not np.allclose(g, gi):
        raise ValueError(f"time positions do not lie on a grid of step {step}")
    return float(step), gi, int(gi.max()) + 1


def on_grid(values: np.ndarray, grid_index: np.ndarray, length: int) -> np.ndarray:
    """values placed at grid_index on a grid of the given length, NaN elsewhere."""
    out = np.full(length, np.nan)
    out[grid_index] = values
    return out


@dataclass
class ResolvedBlock:
    """Block length used by one resampling, and how it was obtained.

    length, decorrelation_lag : in time units.
    estimated : True when it is a Politis–White estimate (evenly spaced
        samples); False when it is the caller's dependence lag.
    decays : as in DependenceEstimate, over all product series (False if any
        series failed to decay).
    n_blocks : the number of blocks in the sample, its span divided by the
        block length (for samples not on a grid, by the caller's lag); None
        when the samples are resampled independently or are on a grid but
        too few to assess.
    n_samples : the number of samples resampled.
    too_few_to_assess : the samples lie on a grid but are fewer than
        MIN_ACF_N, so their dependence was not estimated.
    """
    length: int
    estimated: bool
    decorrelation_lag: int | None = None
    decays: bool = True
    n_blocks: float | None = None
    n_samples: int | None = None
    too_few_to_assess: bool = False

    @property
    def few_blocks(self) -> bool:
        return self.n_blocks is not None and self.n_blocks < MIN_BLOCKS

    def to_record(self) -> dict:
        return {"block_length": self.length, "estimated": self.estimated,
                "decorrelation_lag": self.decorrelation_lag, "dependence_decays": self.decays,
                "n_blocks": self.n_blocks, "few_blocks": self.few_blocks,
                "n_samples": self.n_samples, "too_few_to_assess": self.too_few_to_assess}


def trace_autocovariance(P: np.ndarray, grid_index: np.ndarray, length: int) -> np.ndarray:
    """Autocovariance function of a vector-valued series from the inner
    products P[i, j] = <u_i, u_j> of its (centred) values: at lag k, the mean
    of P[i, j] over the pairs of samples k grid steps apart, times (1 - k/N)
    as in _autocovariances. This is the trace of the lag-k autocovariance
    matrix, the quantity that sets the variance of the vector's mean, and
    therefore of a statistic built as its squared norm."""
    max_lag = searched_lags(length)
    D = grid_index[None, :] - grid_index[:, None]
    keep = (D >= 0) & (D <= max_lag)
    lags = D[keep]
    total = np.bincount(lags, weights=P[keep], minlength=max_lag + 1)
    count = np.bincount(lags, minlength=max_lag + 1)
    k = np.arange(max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, 0.0) * (1.0 - k / length)


def trace_autocovariance_features(F: np.ndarray, grid_index: np.ndarray, length: int,
                                  chunk: int = 256) -> np.ndarray:
    """trace_autocovariance from the (centred) per-sample vectors themselves,
    rows of F, instead of their inner products: the same function, computed
    by fast Fourier transforms of the vectors placed on their grid, so that
    no n x n matrix is formed."""
    max_lag = searched_lags(length)
    nfft = 1 << int(np.ceil(np.log2(2 * length)))
    power = np.zeros(nfft // 2 + 1)
    for j in range(0, F.shape[1], chunk):
        X = np.zeros((length, min(chunk, F.shape[1] - j)))
        X[grid_index] = F[:, j:j + chunk]
        power += (np.abs(np.fft.rfft(X, n=nfft, axis=0)) ** 2).sum(axis=1)
    total = np.fft.irfft(power, n=nfft)[:max_lag + 1]
    ind = np.zeros(length)
    ind[grid_index] = 1.0
    count = np.rint(np.fft.irfft(np.abs(np.fft.rfft(ind, n=nfft)) ** 2, n=nfft)[:max_lag + 1])
    k = np.arange(max_lag + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), 0.0) * (1.0 - k / length)


def _caller_lag_block(time_positions: np.ndarray, fallback_lag: float) -> "ResolvedBlock":
    """Block of the caller's dependence lag for samples not on a regular
    grid (event samples). n_blocks is the samples' time span divided by the
    lag, as for samples on a grid; None when the samples are resampled
    independently (lag of zero or less) or span no time."""
    t = np.asarray(time_positions, dtype=float)
    length = int(round(max(fallback_lag, 0)))
    span = float(t.max() - t.min()) if t.size else 0.0
    n_blocks = span / length if length > 0 and span > 0 else None
    return ResolvedBlock(length, estimated=False, n_samples=int(t.size), n_blocks=n_blocks)


def resolve_block_from_inner_products(
        time_positions: np.ndarray,
        fallback_lag: float,
        inner_products: list[np.ndarray],
        run_length: int = DEFAULT_RUN_LENGTH,
        grid_step: float | None = None,
) -> "ResolvedBlock":
    """Block length for a statistic that is the squared norm of the mean of
    vector-valued per-sample terms u_i, given each candidate's matrix of
    centred inner products <u_i, u_j>: the largest Politis–White estimate
    over the candidates, from the trace autocovariance (trace_autocovariance),
    when the samples lie on a regular grid; otherwise the caller's
    dependence lag. An item may instead be a callable returning the centred
    vectors u_i as rows, so that no n x n matrix is formed
    (trace_autocovariance_features)."""
    n_samples = int(np.asarray(time_positions).size)
    layout = grid_layout(time_positions, grid_step) if inner_products else None
    if layout is None:
        return _caller_lag_block(time_positions, fallback_lag)
    step, gi, length = layout
    n = gi.size
    estimates = [dependence_from_autocovariance(
        trace_autocovariance_features(P(), gi, length) if callable(P) else trace_autocovariance(P, gi, length),
        length, n, run_length) for P in inner_products]
    if not all(e.assessed for e in estimates):
        return ResolvedBlock(int(round(max(fallback_lag, 0))), estimated=False, n_samples=n_samples,
                             too_few_to_assess=n < MIN_ACF_N)
    steps = max(e.block_length for e in estimates)
    return ResolvedBlock(
        length=int(round(steps * step)), estimated=True,
        decorrelation_lag=int(round(max(e.decorrelation_lag for e in estimates) * step)),
        decays=all(e.decays for e in estimates),
        n_blocks=float(length / steps), n_samples=n_samples,
    )


def product_dependence(left: np.ndarray, right: np.ndarray, run_length: int = DEFAULT_RUN_LENGTH,
                       layout: tuple[np.ndarray, int] | None = None) -> list:
    """Dependence estimates of the products of every column of `left` with
    every column of `right` (each column centred). layout = (grid index,
    grid length) places the products on their grid first."""
    L = as2d(left)
    R = as2d(right)
    L = L - L.mean(axis=0)
    R = R - R.mean(axis=0)
    out = []
    for i in range(L.shape[1]):
        for j in range(R.shape[1]):
            x = L[:, i] * R[:, j]
            if layout is not None:
                x = on_grid(x, *layout)
            out.append(estimate_dependence(x, run_length))
    return out


def resolve_block(
        time_positions: np.ndarray,
        fallback_lag: float,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        run_length: int = DEFAULT_RUN_LENGTH,
        grid_step: float | None = None,
) -> ResolvedBlock:
    """Block length for one resampling: the largest Politis–White estimate
    over the residual products in `pairs` when the samples lie on a regular
    grid (evenly spaced, or on the declared grid_step with missing points),
    otherwise the caller's dependence lag. n_blocks is the grid's span in
    blocks."""
    n_samples = int(np.asarray(time_positions).size)
    layout = grid_layout(time_positions, grid_step) if pairs else None
    if layout is None:
        return _caller_lag_block(time_positions, fallback_lag)
    step, gi, length = layout
    estimates = [e for a, b in pairs for e in product_dependence(a, b, run_length, (gi, length))]
    if not all(e.assessed for e in estimates):
        return ResolvedBlock(int(round(max(fallback_lag, 0))), estimated=False, n_samples=n_samples,
                             too_few_to_assess=gi.size < MIN_ACF_N)
    steps = max(e.block_length for e in estimates)
    return ResolvedBlock(
        length=int(round(steps * step)), estimated=True,
        decorrelation_lag=int(round(max(e.decorrelation_lag for e in estimates) * step)),
        decays=all(e.decays for e in estimates),
        n_blocks=float(length / steps), n_samples=n_samples,
    )


def samples_independent(
        time_positions: np.ndarray, series: list[np.ndarray], run_length: int = DEFAULT_RUN_LENGTH,
        grid_step: float | None = None,
) -> dict | None:
    """For samples on a regular grid, whether each series is
    indistinguishable from independent sampling: its decorrelation lag is one
    grid step. None for samples not on a grid."""
    layout = grid_layout(time_positions, grid_step)
    if layout is None:
        return None
    step, gi, length = layout
    estimates = []
    for x in series:
        X = as2d(x)
        estimates.extend(estimate_dependence(on_grid(X[:, j], gi, length), run_length) for j in range(X.shape[1]))
    if not all(e.assessed for e in estimates):
        return {"estimated": False, "too_few_to_assess": True, "n_samples": int(gi.size)}
    worst = max(e.decorrelation_lag for e in estimates)
    return {"estimated": True, "independent": worst <= 1, "decorrelation_lag": int(round(worst * step))}


