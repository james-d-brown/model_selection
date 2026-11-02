"""
Calibration of the CMI tier on the sample at hand, with surrogate candidates.

The CMI tier's local permutation is valid when the candidate's distribution
changes slowly with the conditioning set on the scale of the neighbourhoods
(Kim et al., 2022), which the data cannot confirm (Shah and Peters, 2020).
Its false-acceptance rate on a given sample can nevertheless be measured
under a stated model of the candidate. The candidate is regressed on the
conditioning set by the nonlinear tier's cross-fitted kernel ridge
regression; each surrogate is the fitted part plus the cross-fitted residuals
permuted in time blocks of the dependence lag. A surrogate keeps the
candidate's dependence on the conditioning set and, by construction, carries
no information about the observations beyond it, so the CMI test should
accept it at most at rate alpha. Each surrogate is tested alone with the
tier's two gates (p < alpha and the effect-size floor). The tier is
calibrated on the sample unless the count of accepted surrogates is
significant at level alpha under a binomial(n, alpha) null.

The number of surrogates is the smallest for which that binomial test
detects a false-acceptance rate of rate_ratio * alpha with 80% power. Each
surrogate's p-value is a sequential Monte Carlo p-value (Besag and
Clifford, 1991): its null draws stop once the surrogate can no longer be
significant at alpha, which gives the same decision as drawing them all,
so surrogates that are clearly not significant, most of them under a
calibrated test, cost a fraction of the full number of draws; the effect
size is computed only for significant surrogates. Using
an estimated conditional distribution of the candidate to examine a
conditional independence test follows the conditional randomization and
permutation tests (Candes et al., 2018; Berrett et al., 2020), whose
validity depends on how well that distribution is estimated; here the
estimate is used only to check calibration.

See docs/methods.md, Free-form tier.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from model_selection.statistics.regression.kernel_residuals import ConditioningRegression
from model_selection.statistics.significance.base import minimum_resamples, stopping_exceedances
from model_selection.statistics.significance.resampling import time_block_permutation
from model_selection.utils import as2d

POWER = 0.80


def binomial_rejection_count(n: int, alpha: float) -> int:
    """Smallest count k with P(X >= k) <= alpha for X ~ binomial(n, alpha)."""
    k = int(binom.isf(alpha, n, alpha)) + 1
    while k > 1 and binom.sf(k - 2, n, alpha) <= alpha:
        k -= 1
    while binom.sf(k - 1, n, alpha) > alpha:
        k += 1
    return k


def surrogates_needed(alpha: float, rate_ratio: float, power: float = POWER) -> int:
    """Smallest number of surrogates for which the binomial test at level alpha
    detects a false-acceptance rate of rate_ratio * alpha with the given power."""
    for n in range(10, 10000):
        k = binomial_rejection_count(n, alpha)
        if binom.sf(k - 1, n, min(rate_ratio * alpha, 1.0)) >= power:
            return n
    raise ValueError("no surrogate count up to 10,000 reaches the requested power")


def cmi_calibration(
        test,
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray,
        time_positions: np.ndarray,
        dependence_lag: float,
        alpha: float,
        effect_size_floor: float,
        rng: np.random.Generator,
        rate_ratio: float = 3.0,
        grid_step: float | None = None,
) -> dict:
    """False-acceptance rate of the CMI test `test` on surrogates of
    `candidate` given `conditioning` (see the module docstring). Each
    surrogate test uses at most minimum_resamples(alpha) null draws."""
    C = as2d(np.asarray(candidate, dtype=float))
    n = C.shape[0]
    regression = ConditioningRegression(conditioning, time_positions, buffer=dependence_lag)
    resid, _ = regression.residuals(C)
    fitted = C - resid
    n_surrogates = surrogates_needed(alpha, rate_ratio)
    n_resamples = minimum_resamples(alpha)
    accepted = 0
    for _ in range(n_surrogates):
        surrogate = fitted + resid[time_block_permutation(n, time_positions, dependence_lag, rng)]
        observed, null, p_value = test.sequential_p_value(
            surrogate, obs, conditioning, n_resamples, rng, time_positions, dependence_lag,
            stopping_exceedances(alpha, n_resamples), grid_step=grid_step)
        if p_value < alpha:
            effect, _ = test.magnitude(surrogate, obs, conditioning, time_positions, dependence_lag, observed, null,
                                       grid_step=grid_step)
            accepted += bool(effect >= effect_size_floor)
    p_value = float(binom.sf(accepted - 1, n_surrogates, alpha)) if accepted > 0 else 1.0
    return {"n_surrogates": n_surrogates, "false_acceptances": accepted,
            "false_acceptance_rate": accepted / n_surrogates, "p_value": p_value,
            "calibrated": bool(p_value > alpha)}
