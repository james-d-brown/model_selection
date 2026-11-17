"""
Data-driven dependence lengths for autocorrelated samples.

Two lengths are estimated from a series' autocorrelation by the procedure of
Politis and White (2004), with the correction of Patton, Politis and White
(2009). Only the first, the decorrelation lag, is used by the tests (it sets
the dependence lag and the window of the multiplier bandwidth below); the
second is not used by the tests:

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

The decorrelation lag of each formulation's series sets the dependence lag
(pipeline.shared). The tests' multiplier bandwidth is set differently
(resolve_block): the statistic is a sum of products of two residual series,
and the bandwidth must be chosen for testing, not for estimating a variance.
The products' persistence a (their autocovariance-weighted mean lag) is
estimated from their own autocovariances, fitted by a non-increasing,
non-negative sequence (Geyer, 1992; Berg and Song, 2023), over the lags
before the shorter of the two residual series' decorrelation lags or the
products' own decorrelation lag, whichever is later. Two series that are
independent at every lag have products that cannot stay correlated beyond
either factor's dependence (Bartlett, 1935), and each factor's
autocorrelation is measured far more precisely than the products' weak
tail, which a search on the products themselves would cut short. The null
hypothesis concerns matched time steps and does not make the series
independent at other lags, so the products' own lag is kept as well. The
bandwidth is then c sqrt(a N) on a grid of N steps (Sun,
Phillips and Jin, 2008; Lazarus, Lewis, Stock and Watson, 2018), and the
tests refer their statistics to the fixed-b distribution at b = l / N
(Kiefer and Vogelsang, 2005; conditional_independence.multipliers.
fixed_b_draws). With several candidates the longest bandwidth is used,
because all share one resampling.

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
from model_selection.statistics.significance.clusters import smallest_p_value

C_THRESHOLD = 2.0  # Politis and White (2004), Remark 2.3
DEFAULT_RUN_LENGTH = 5  # Politis and White (2004), Remark 2.3
# Fewest blocks or clusters before a test on samples not on a grid (event
# samples, resampled with the caller's lag or one multiplier per cluster) is
# flagged, a convention: with few clusters the wild cluster bootstrap can
# fail (MacKinnon and Webb, 2017). Tests on evenly spaced samples are flagged
# by their expected variance shortfall instead (SHORTFALL_TOLERANCE).
MIN_BLOCKS = 50


# Fewest observations from which autocorrelations are estimated: Box and
# Jenkins (1976) advise about 50 for a useful estimate of the autocorrelation
# function, a convention. Below it, a sample's dependence is not estimated and
# the caller's dependence lag (the spacing that made the samples approximately
# independent) is used, as for samples not on a grid.
MIN_ACF_N = 50

TESTING_BANDWIDTH_CONSTANT = 0.79
"""c in the multipliers' bandwidth l = c sqrt(a N) (testing_bandwidth): the
testing-optimal rate for the Bartlett kernel (Sun, Phillips and Jin, 2008),
with the constant of Lazarus, Lewis, Stock and Watson (2018), whose rule
1.3 sqrt(N) is for a first-order autoregression of coefficient 0.7
(a = 2.75), here scaled by each series' own persistence a. Their loss
weighs errors in the test's level nine times errors in its power, and the
rule accepts a shortfall of the null's variance of about sqrt(a / N) / c
(ResolvedBlock.expected_shortfall, implied_level). They fix the persistence
at a reference value; estimating it, as here, is an extension."""
SHORTFALL_TOLERANCE = 0.12
"""Expected relative shortfall of the resampled null's variance above which
a test on evenly spaced samples is flagged (few_blocks), a convention: the
truncation bias a / l of the Bartlett-weighted variance (Andrews, 1991)."""


def _non_increasing_fit(y: np.ndarray) -> np.ndarray:
    """The least-squares fit of y by a non-increasing sequence (isotonic
    regression by pooling adjacent violators; Barlow, Bartholomew, Bremner
    and Brunk, 1972)."""
    y = np.asarray(y, dtype=float)
    values, weights, counts = [], [], []
    for v in y:
        values.append(v); weights.append(1.0); counts.append(1)
        while len(values) > 1 and values[-2] < values[-1]:
            w = weights[-2] + weights[-1]
            values[-2] = (weights[-2] * values[-2] + weights[-1] * values[-1]) / w
            weights[-2] = w
            counts[-2] += counts[-1]
            del values[-1], weights[-1], counts[-1]
    return np.repeat(values, counts)


def persistence(acov: np.ndarray, window: int) -> float:
    """The persistence a = G / J of a series whose autocovariances at lags
    0.. are `acov`, from its autocovariances at lags 1..window fitted by the
    closest non-increasing, non-negative sequence g (Geyer, 1992; Berg and
    Song, 2023): G = 2 sum k g(k), J = acov[0] + 2 sum g(k). a is the
    autocovariance-weighted mean lag, in lag units; it is zero only when no
    autocovariance in the window is positive (independent terms). The window
    comes from the factors of a product series (resolve_block)."""
    acov = np.asarray(acov, dtype=float)
    M = int(min(max(window, 0), acov.size - 1))
    if acov[0] <= 0 or M < 1:
        return 0.0
    g = np.clip(_non_increasing_fit(acov[1:M + 1]), 0.0, None)
    k = np.arange(1, M + 1)
    G = 2.0 * float(np.sum(k * g))
    J = float(acov[0]) + 2.0 * float(np.sum(g))
    return G / J if G > 0 and J > 0 else 0.0


def testing_bandwidth(a: float, N: int) -> int:
    """The multipliers' bandwidth, in grid steps, for a statistic whose terms
    have persistence a over a grid of N steps: c sqrt(a N)
    (TESTING_BANDWIDTH_CONSTANT), at least 1 (independent multipliers when
    a = 0) and at most N / 2."""
    if a <= 0 or N < 2:
        return 1
    return int(min(max(1, np.ceil(TESTING_BANDWIDTH_CONSTANT * np.sqrt(a * N))), max(1, N // 2)))


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
                                   run_length: int = DEFAULT_RUN_LENGTH,
                                   bartlett: bool = False) -> DependenceEstimate:
    """The Politis–White decorrelation lag and block length from the
    autocovariances acov[0..max_lag] of a series spanning N grid steps with n
    observed values (see the module docstring).

    bartlett : scale the band at lag k by Bartlett's (1946) standard error of
        an autocorrelation beyond lag k - 1, sqrt(1 + 2 sum_{j<k} rho_j^2),
        rather than that of white noise (Box and Jenkins, 1976): an
        autocorrelation at lag k is then judged against its own sampling
        variability given the autocorrelation at shorter lags."""
    run_length = max(1, int(run_length))
    max_lag = acov.size - 1
    if n < MIN_ACF_N or acov[0] <= 0 or max_lag < 1:
        return DependenceEstimate(1, 1, True, max(max_lag, 1), assessed=False)

    rho = np.abs(acov[1:] / acov[0])  # lags 1..max_lag
    crit = C_THRESHOLD * np.sqrt(np.log10(n) / n)
    if bartlett:
        crit = crit * np.sqrt(1.0 + 2.0 * np.concatenate([[0.0], np.cumsum(rho[:-1] ** 2)]))
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
    estimated : True when it is estimated from the products (evenly spaced
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
    independent_units : samples not on a grid whose distinct positions are
        all at least one lag apart (clusters of events, or events all
        independent); n_blocks is then the number of those units.
    """
    length: int
    estimated: bool
    decorrelation_lag: int | None = None
    decays: bool = True
    n_blocks: float | None = None
    n_samples: int | None = None
    too_few_to_assess: bool = False
    independent_units: bool = False
    persistence: float | None = None
    window: int | None = None
    factor_window: int | None = None
    """The window the factors alone give (estimated bandwidths only); `window`
    is longer where the products stay autocorrelated beyond it."""
    bandwidth_fraction: float | None = None
    grid_length: int | None = None
    """Steps of the regular grid the samples lie on (estimated bandwidths only)."""

    @property
    def expected_shortfall(self) -> float | None:
        """For an estimated bandwidth, the expected relative shortfall of the
        resampled null's variance, a / l: the dependence beyond the
        bandwidth that the Bartlett weights leave out (Andrews, 1991)."""
        if not self.estimated or self.persistence is None or self.length <= 0:
            return None
        return float(self.persistence / self.length)

    def implied_level(self, alpha: float = 0.05) -> float | None:
        """Level of a test of nominal level alpha on one sum whose null variance
        falls short by expected_shortfall: 2 (1 - Phi(z sqrt(1 - shortfall))),
        z the normal quantile at 1 - alpha / 2. First order in the shortfall
        this is the size distortion of Lazarus et al. (2018) for one
        restriction; a statistic built from several sums is distorted more.
        None without an estimated bandwidth."""
        from scipy import stats
        shortfall = self.expected_shortfall
        if shortfall is None:
            return None
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return float(2.0 * stats.norm.sf(z * np.sqrt(max(0.0, 1.0 - shortfall))))

    @property
    def few_blocks(self) -> bool:
        if self.estimated and self.expected_shortfall is not None:
            return self.expected_shortfall >= SHORTFALL_TOLERANCE
        return self.n_blocks is not None and self.n_blocks < MIN_BLOCKS

    def to_record(self) -> dict:
        return {"block_length": self.length, "estimated": self.estimated,
                "decorrelation_lag": self.decorrelation_lag, "dependence_decays": self.decays,
                "n_blocks": self.n_blocks, "few_blocks": self.few_blocks,
                "n_samples": self.n_samples, "too_few_to_assess": self.too_few_to_assess,
                "independent_units": self.independent_units, "persistence": self.persistence,
                "persistence_window": self.window, "factor_window": self.factor_window,
                "window_extended_by_products": (None if self.window is None or self.factor_window is None
                                                else bool(self.window > self.factor_window)),
                "bandwidth_fraction": self.bandwidth_fraction,
                "expected_shortfall": self.expected_shortfall,
                "implied_level_at_5pct": self.implied_level(0.05),
                "smallest_p_value": (smallest_p_value(int(self.n_blocks)) if self.independent_units
                                     and self.n_blocks is not None else None)}


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
    lag, as for samples on a grid, or, when every distinct position is at
    least one lag from the next (clusters of events, or events that are all
    independent), the number of distinct positions, each an independent
    unit; None when the samples are resampled independently (lag of zero or
    less) or span no time."""
    t = np.asarray(time_positions, dtype=float)
    length = int(round(max(fallback_lag, 0)))
    span = float(t.max() - t.min()) if t.size else 0.0
    n_blocks = span / length if length > 0 and span > 0 else None
    units = False
    if n_blocks is not None:
        positions = np.unique(t)
        if positions.size > 1 and float(np.min(np.diff(positions))) >= length:
            n_blocks, units = float(positions.size), True
    return ResolvedBlock(length, estimated=False, n_samples=int(t.size), n_blocks=n_blocks, independent_units=units)


def _factor_window(acovs: list, length: int, n: int, run_length: int) -> tuple[int, bool, bool]:
    """(window, decays, assessed) for a product of two factors: the lags
    before the shorter of the factors' decorrelation lags, each the largest
    over its columns (or its trace autocovariance), so that a factor with no
    significant autocorrelation (decorrelation lag 1) gives an empty window
    and independent terms. When the two factors are independent as series
    (at every lag) given the conditioning set, the products' autocovariance
    at a lag is the product of the factors' autocovariances there
    (Bartlett, 1935; Clifford, Richardson and Hemon, 1989), which vanishes
    where either factor's does. The null hypothesis concerns matched time
    steps only and does not imply this: dependence between the factors at
    other lags adds the product of their cross-covariances at lags +k and
    -k. The window is therefore extended to the products' own decorrelation
    lag where that is later (_product_window)."""
    sides = []
    for side in acovs:
        estimates = [dependence_from_autocovariance(a, length, n, run_length) for a in side]
        if not all(e.assessed for e in estimates):
            return 1, True, False
        sides.append((max(e.decorrelation_lag for e in estimates), all(e.decays for e in estimates)))
    return min(m for m, _ in sides) - 1, all(d for _, d in sides), True


def _product_window(terms: np.ndarray, length: int, n: int, run_length: int) -> tuple[int, bool]:
    """(window, decays) from the products themselves: the lags before their
    own decorrelation lag (Politis, 2003), from their (trace) autocovariances
    `terms`. It needs no assumption about the factors, and covers what the
    factors' window (_factor_window) leaves out: autocovariance of the
    products that comes from dependence between the factors at other lags.
    The factors' window covers what this one leaves out: a weak tail that
    the factors' autocorrelations show and the products' do not. The longer
    of the two is used."""
    e = dependence_from_autocovariance(np.asarray(terms, dtype=float), length, n, run_length, bartlett=True)
    if not e.assessed:
        return 0, True
    return e.decorrelation_lag - 1, e.decays


def _resolved(results: list, step: float, length: int, n_samples: int) -> "ResolvedBlock":
    """The ResolvedBlock shared by all candidates from each candidate's
    (bandwidth in steps, persistence, window, decays): the longest bandwidth,
    since all candidates share one set of multipliers."""
    steps, a, window, _, factor_window = max(results, key=lambda r: r[0])
    return ResolvedBlock(
        length=int(round(steps * step)), estimated=True,
        decorrelation_lag=int(round((max(r[2] for r in results) + 1) * step)),
        decays=all(r[3] for r in results),
        n_blocks=float(length / steps), n_samples=n_samples,
        persistence=float(a * step), window=int(round(window * step)),
        factor_window=int(round(factor_window * step)),
        bandwidth_fraction=float(steps / length), grid_length=int(length),
    )


def resolve_block_from_inner_products(
        time_positions: np.ndarray,
        fallback_lag: float,
        inner_products: list,
        run_length: int = DEFAULT_RUN_LENGTH,
        grid_step: float | None = None,
        factors: list | None = None,
) -> "ResolvedBlock":
    """Multiplier bandwidth for a statistic that is the squared norm of the
    sum of vector-valued per-sample terms u_i = vec(c_i o_i'), products of a
    candidate's and the observations' (centred) per-sample vectors, when the
    samples lie on a regular grid; otherwise the caller's dependence lag.

    inner_products: per candidate, the matrix of centred inner products
    <u_i, u_j>, or a callable returning the centred u_i as rows (so that no
    n x n matrix is formed). factors: per candidate, (candidate's vectors,
    observations' vectors), each as rows or as a matrix of inner products
    <c_i, c_j>. The persistence of the terms is estimated from their own
    trace autocovariance within the window set by the factors (persistence,
    _factor_window), and the bandwidth is testing_bandwidth of it."""
    n_samples = int(np.asarray(time_positions).size)
    layout = grid_layout(time_positions, grid_step) if inner_products else None
    if layout is None:
        return _caller_lag_block(time_positions, fallback_lag)
    step, gi, length = layout
    n = gi.size

    def trace_acov(X):
        X = X() if callable(X) else X
        if X.ndim == 2 and X.shape[0] == X.shape[1] == n and factors_are_gram:
            return trace_autocovariance(X, gi, length)
        return trace_autocovariance_features(X - X.mean(axis=0), gi, length)

    results = []
    for item, pair in zip(inner_products, factors if factors is not None else [None] * len(inner_products)):
        terms = trace_autocovariance_features(item(), gi, length) if callable(item) else \
            trace_autocovariance(item, gi, length)
        if pair is None:
            raise ValueError("resolve_block_from_inner_products needs each candidate's factors")
        factors_are_gram = not callable(item)
        window, decays, assessed = _factor_window([[trace_acov(pair[0])], [trace_acov(pair[1])]], length, n,
                                                  run_length)
        if not assessed or n < MIN_ACF_N:
            return ResolvedBlock(int(round(max(fallback_lag, 0))), estimated=False, n_samples=n_samples,
                                 too_few_to_assess=n < MIN_ACF_N)
        factor_window = window
        own, own_decays = _product_window(terms, length, n, run_length)
        window, decays = max(window, own), decays and own_decays
        a = persistence(terms, window)
        results.append((testing_bandwidth(a, length), a, window, decays, factor_window))
    return _resolved(results, step, length, n_samples)


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
    """Multiplier bandwidth for one resampling of a statistic built from the
    sums of the products of two residual series, `pairs` holding each
    candidate's (left, right) residuals, when the samples lie on a regular
    grid (evenly spaced, or on the declared grid_step with missing points);
    otherwise the caller's dependence lag.

    For each candidate the products of every left column with every right
    column (each centred) form the statistic's terms. Their persistence is
    estimated from their summed autocovariance within the window set by the
    factors (persistence, _factor_window), and the bandwidth is
    testing_bandwidth of it. The longest bandwidth over the candidates is
    used, since they share one set of multipliers. n_blocks is the grid's
    span in bandwidths."""
    n_samples = int(np.asarray(time_positions).size)
    layout = grid_layout(time_positions, grid_step) if pairs else None
    if layout is None:
        return _caller_lag_block(time_positions, fallback_lag)
    step, gi, length = layout
    n = gi.size
    results = []
    for left, right in pairs:
        L, R = as2d(left), as2d(right)
        L, R = L - L.mean(axis=0), R - R.mean(axis=0)
        max_lag = searched_lags(length, run_length)
        if n < MIN_ACF_N:
            return ResolvedBlock(int(round(max(fallback_lag, 0))), estimated=False, n_samples=n_samples,
                                 too_few_to_assess=True)
        acovs = [[_autocovariances(on_grid(X[:, j], gi, length), max_lag) for j in range(X.shape[1])]
                 for X in (L, R)]
        window, decays, assessed = _factor_window(acovs, length, n, run_length)
        if not assessed:
            return ResolvedBlock(int(round(max(fallback_lag, 0))), estimated=False, n_samples=n_samples,
                                 too_few_to_assess=n < MIN_ACF_N)
        terms = sum(_autocovariances(on_grid(L[:, i] * R[:, j], gi, length), max_lag)
                    for i in range(L.shape[1]) for j in range(R.shape[1]))
        factor_window = window
        own, own_decays = _product_window(terms, length, n, run_length)
        window, decays = max(window, own), decays and own_decays
        a = persistence(terms, window)
        results.append((testing_bandwidth(a, length), a, window, decays, factor_window))
    return _resolved(results, step, length, n_samples)


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


