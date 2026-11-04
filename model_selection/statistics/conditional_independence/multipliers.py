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
Rademacher weights for the wild bootstrap of regression statistics. The
linear and nonlinear tiers' statistics and the linear adjustment test,
quadratic forms in the weights, keep standard normal weights: with +-1 weights the diagonal
of a quadratic form would not vary. The effect-modification test's score
depends on each residual's own square, which standard normal weights spread
more widely than the errors do (for normal errors, three times the variance
of the squared errors), so it uses Rademacher weights.

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
