"""P-value and bias-correction helpers shared by every permutation and bootstrap test."""
from __future__ import annotations


import numpy as np



def permutation_p_value(observed: float, null_stats: np.ndarray) -> float:
    """Upper-tail permutation p-value (M + 1) / (N + 1), where M of the N null
    draws are at least as large as the observed value. Exactly valid when the
    permutations are drawn at random (Phipson and Smyth, 2010); the smallest
    attainable value is 1 / (N + 1)."""
    null_stats = np.asarray(null_stats)
    return float((np.sum(null_stats >= observed) + 1) / (null_stats.shape[0] + 1))


def resamples_for_relative_error(alpha: float, relative_error: float) -> int:
    """Resamples N for which the Monte Carlo standard error of a permutation
    p-value at the threshold, sqrt(alpha (1 - alpha) / N), is at most
    relative_error times alpha: N = (1 - alpha) / (relative_error^2 alpha)."""
    return int(np.ceil((1.0 - alpha) / (relative_error ** 2 * alpha)))


def stopping_exceedances(alpha: float, n_resamples: int) -> int:
    """Null draws at or above the observed value after which a test with
    n_resamples draws can no longer reject at level alpha: the smallest M
    with (M + 1) / (n_resamples + 1) >= alpha. Stopping the draws there
    (Besag and Clifford, 1991) gives the same decision at alpha, and at any
    smaller level, as drawing all n_resamples."""
    return max(1, int(np.ceil(alpha * (n_resamples + 1) - 1)))


def minimum_resamples(alpha: float) -> int:
    """Resamples needed for at least 10 null draws to fall in a rejection
    region of size alpha, so that decisions near alpha rest on more than a
    handful of draws."""
    return int(np.ceil(10.0 / alpha))


def bias_correct_association(observed: float, null_draws: np.ndarray) -> tuple[float, float]:
    """Remove an association measure's finite-sample bias before comparing it
    with a fixed effect-size floor.

    A permutation p-value is a rank, so a bias shared by the observed value
    and every null draw cannot affect it. A comparison against a fixed floor
    has no such protection. Under exchangeability, E[observed | H0] equals the
    mean of the null draws, so subtracting that mean gives an estimate that is
    zero in expectation under independence, whatever the source of the bias
    (cf. the normalized enrichment score, Subramanian et al. 2005).

    Returns (bias_corrected_observed, null_mean).
    """
    null_mean = float(np.mean(null_draws))
    return float(observed) - null_mean, null_mean
