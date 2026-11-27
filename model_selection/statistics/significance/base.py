"""P-value and bias-correction helpers shared by every permutation and bootstrap test."""
from __future__ import annotations


import numpy as np
from scipy import stats



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


def resamples_at_level(alpha: float, n_at_alpha: int, level: float) -> int:
    """Resamples that give, at a level below alpha, the relative Monte Carlo
    error that n_at_alpha gives at alpha (resamples_for_relative_error), and
    never fewer than n_at_alpha: the most a sequential test at that level
    draws (draw_until_decided)."""
    relative_error_squared = (1.0 - alpha) / (n_at_alpha * alpha)
    return max(int(n_at_alpha), int(np.ceil((1.0 - level) / (relative_error_squared * level))))


def sequential_decision(exceedances: int, n_draws: int, level: float, risk: float) -> bool | None:
    """The confidence-sequence method (Ding, Gandy and Hahn, 2020) after
    n_draws null draws, of which `exceedances` are at least the observed
    value: None while the level lies in the confidence set
    {p : (n + 1) Binom(exceedances; n, p) > risk}, else whether the set lies
    below the level, True meaning significant. These sets cover the exact
    p-value at every n at once with probability at least 1 - risk (Robbins,
    1970; Lai, 1976), so a decision made at any number of draws differs from
    the exact p-value's with probability at most risk. The set contains
    exceedances / n, which therefore gives its side of the level."""
    if (n_draws + 1) * stats.binom.pmf(exceedances, n_draws, level) > risk:
        return None
    return bool(exceedances < level * n_draws)


def draw_until_decided(observed: float, first: tuple, draw_more, level: float, risk: float, n_max: int,
                       chunk: int) -> tuple[tuple, bool]:
    """Extend a test's null draws until its decision at `level` is settled
    (sequential_decision) or n_max draws are reached (docs/methods.md,
    Resamples per test).

    first : arrays of the first draws, the statistic's null draws first and
        any companions (as the semipartial effect's draws) after it.
    draw_more : draw_more(k) returns the same arrays for k further draws.
    Returns the concatenated arrays and whether the decision was settled
    before n_max; the caller decides an unsettled test by its p-value from
    all n_max draws, as a test of that fixed size."""
    parts = [list(first)]
    exceedances = int(np.sum(np.asarray(first[0]) >= observed))
    n = len(first[0])
    while True:
        if sequential_decision(exceedances, n, level, risk) is not None:
            settled = True
            break
        if n >= n_max:
            settled = False
            break
        more = draw_more(min(chunk, n_max - n))
        parts.append(list(more))
        exceedances += int(np.sum(np.asarray(more[0]) >= observed))
        n += len(more[0])
    joined = tuple(np.concatenate([p[i] for p in parts]) for i in range(len(first)))
    return joined, settled


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
