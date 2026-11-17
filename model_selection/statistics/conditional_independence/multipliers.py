"""
Multiplier weights for the bootstrap of residual products.

Weights have mean 0 and variance 1. Samples closer than the dependence lag
in time receive correlated weights with the Bartlett correlation
max(0, 1 - |t - t'| / lag), the dependent wild bootstrap of Shao (2010);
samples further apart, and all samples when the lag is at most one time
unit, receive independent weights. Samples at the same position, the events
of one cluster, share one weight (the wild cluster bootstrap; Cameron,
Gelbach and Miller, 2008; statistics.significance.clusters). Independent weights are standard normal,
or Rademacher (+1 or -1 with equal probability) on request when every pair
of samples is independent: Davidson and Flachaire (2008) recommend
Rademacher weights for the wild bootstrap of regression statistics.

Samples not on a time grid whose distinct positions are all at least one
lag apart (independent units: clusters of events, or events all one lag
apart) receive one Rademacher sign per unit, and every test then uses the
wild cluster restricted bootstrap (Cameron, Gelbach and Miller, 2008;
Djogbenou, MacKinnon and Nielsen, 2019): the observations' data are rebuilt
under the null as the null fit's fitted values plus its residuals with one
sign per unit; the fit is applied again, and the statistic recomputed as the
observed one is. In the linear tier and the adequacy tests the residuals are
first scaled by cluster (CR2; significance.clusters.cluster_scaled_residuals).
The nonlinear tier rebuilds its observations' feature map from its unscaled
cross-fitted residuals and residualizes it with the observed fit's bandwidth
and penalty (nonlinear.NonlinearResidualTest). Recomputing reproduces the coupling
between clusters that estimating the means and regressions creates, which
multipliers applied to the residual products do not: with few clusters and
several features those rejected too often (in simulations, 7-8% at a
nominal 5% in the linear tier and 14-19% in the nonlinear tier with three
features and ten clusters). Gaussian weights per unit on centred products,
the earlier null, gave a normal reference with the units' sample variance,
which also rejects too often with few units (MacKinnon and Webb, 2017).
These nulls are justified as the number of clusters grows; no published
result fixes their level with a few clusters of a few events
(significance.clusters).

On a regular grid the tiers' tests and the adjustment test
keep standard normal weights, correlated or not; with the many samples of a
grid the two give the same null. The candidate linearity test, a regression
F statistic, and the effect-modification test take Rademacher weights
wherever samples are independent: the effect-modification score depends on
each residual's own square, which standard normal weights spread more widely
than the errors do (for normal errors, three times the variance of the
squared errors).

See docs/methods.md, Dependence between samples.
"""
from __future__ import annotations

import numpy as np


def dependent_wild_weights(
        time_positions: np.ndarray, dependence_lag: float, n_draws: int, rng: np.random.Generator,
        independent: str = "gaussian",
) -> np.ndarray:
    """(n_draws, n) weights with the Bartlett correlation over `dependence_lag`.

    independent : distribution of the weights when every pair of samples is
        independent (lag at most one time unit, or all samples at least one
        lag apart): "gaussian" or "rademacher". Correlated weights are always
        built from standard normal increments.
    """
    if independent not in ("gaussian", "rademacher"):
        raise ValueError(f"independent must be 'gaussian' or 'rademacher' -- got {independent!r}")
    t = np.asarray(time_positions, dtype=np.int64)
    n = t.size
    lag = int(round(dependence_lag))
    # Samples at the same position (the events of one cluster; statistics.significance.clusters) share a
    # multiplier; distinct positions at least one lag apart are independent.
    positions, shared = np.unique(t, return_inverse=True)
    all_apart = positions.size < 2 or float(np.min(np.diff(positions))) >= lag
    if independent == "rademacher" and (lag <= 1 or all_apart):
        if lag <= 1:
            return rng.choice(np.array([-1.0, 1.0]), size=(n_draws, n))
        return rng.choice(np.array([-1.0, 1.0]), size=(n_draws, positions.size))[:, shared]
    if lag <= 1:
        return rng.standard_normal((n_draws, n))
    # w_i = (S(t_i + lag) - S(t_i)) / sqrt(lag), with S the running sum of iid
    # standard normals, one per time step. S is needed only at the points
    # t_i and t_i + lag; between consecutive such points its increment is
    # normal with variance equal to their separation, so it is generated
    # there directly (exactly the same distribution, without a value per step).
    points = np.unique(np.concatenate([t, t + lag]))
    steps = np.diff(points).astype(float)
    increments = rng.standard_normal((n_draws, steps.size)) * np.sqrt(steps)
    S = np.concatenate([np.zeros((n_draws, 1)), np.cumsum(increments, axis=1)], axis=1)
    return (S[:, np.searchsorted(points, t + lag)] - S[:, np.searchsorted(points, t)]) / np.sqrt(lag)


FIXED_B_MIN_BANDWIDTH = 400
"""Bandwidth, in steps, at which the fixed-b functional is simulated when the
record has more steps than that needs (fixed_b_draws), so that its
discretization is negligible."""

FIXED_B_MAX_VALUES = 2_000_000
"""Most values held at once by fixed_b_draws (draws times points), to bound
its memory."""


def fixed_b_draws(bandwidth_fraction: float | None, n_draws: int, rng: np.random.Generator,
                  n_steps: int | None = None, chunk: int = 32) -> np.ndarray:
    """Draws of the fixed-b variance functional Q at b = l / N, the
    multipliers' bandwidth as a fraction of the record (Kiefer and
    Vogelsang, 2005): the variance a Bartlett-weighted estimator, of
    bandwidth b m, reports for the centred mean of m independent standard
    normal values, e'Ke / m with K the multipliers' Bartlett correlation
    (dependent_wild_weights). Its mean is 1 - b + b^2 / 3 in the limit.

    m is the record's own number of grid steps, n_steps, so that Q is the
    finite-sample functional at the multipliers' own bandwidth, or fewer
    where the bandwidth exceeds FIXED_B_MIN_BANDWIDTH steps: m =
    ceil(FIXED_B_MIN_BANDWIDTH / b), at which the discretization is
    negligible. Without n_steps, m = ceil(FIXED_B_MIN_BANDWIDTH / b).

    A statistic that is a quadratic form in the sums of per-sample terms is
    asymptotically its numerator's distribution scaled by the long-run
    variance; the multiplier bootstrap reproduces that numerator at the
    Bartlett-weighted variance estimate instead, which is itself random and
    biased by the centring. Dividing each null draw by an independent draw of
    Q gives the statistic's fixed-b reference distribution. Ones when the
    multipliers are independent (no bandwidth) or b is unknown."""
    if bandwidth_fraction is None or bandwidth_fraction <= 0 or bandwidth_fraction >= 1:
        return np.ones(n_draws)
    b = float(bandwidth_fraction)
    m = int(np.ceil(FIXED_B_MIN_BANDWIDTH / b))
    if n_steps is not None:
        m = max(2, min(m, int(n_steps)))
    L = max(1, int(round(b * m)))
    rows = max(1, min(chunk, FIXED_B_MAX_VALUES // (m + 2 * L + 1)))
    out = np.empty(n_draws)
    for start in range(0, n_draws, rows):
        stop = min(start + rows, n_draws)
        e = rng.standard_normal((stop - start, m))
        e -= e.mean(axis=1, keepdims=True)
        # e'Ke = (1/L) sum over every window of L consecutive positions,
        # including those running past either end, of the squared sum of e in it.
        S = np.concatenate([np.zeros((stop - start, 1)), np.cumsum(e, axis=1)], axis=1)
        S = np.concatenate([np.zeros((stop - start, L)), S, np.repeat(S[:, -1:], L, axis=1)], axis=1)
        windows = S[:, L:] - S[:, :-L]
        out[start:stop] = np.sum(windows ** 2, axis=1) / (L * m)
    return out
