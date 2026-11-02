"""
Functional-form adequacy testing: does a simpler tier's structural
assumption hold for one state (candidate, obs, conditioning)?

H0 is always "the simpler tier is adequate". Each test fits its null model
(the simpler structure) and computes a statistic for the departure from it.
The null distribution is built by a wild bootstrap of the null model's
residuals (Wu, 1986; Mammen, 1993): each residual (for a least-squares null
model, divided by sqrt(1 - h_t) with h_t its leverage, the HC2 correction;
Davidson and Flachaire, 2008) is multiplied by a multiplier of mean zero and
variance one, correlated over the dependence lag (Shao, 2010), added back
to the fitted values, and the statistic is recomputed. The null model's own
structure, including its dependence on the conditioning set, is therefore
preserved in every draw, and, unlike a permutation of residuals, the draws
reproduce residual spread that varies with the regressors (Winkler et al.,
2014). The p-value is (M + 1) / (N + 1) (Phipson and Smyth, 2010). Rejection also requires a practically meaningful
departure: the bias-corrected semipartial R^2 (the departure's share of all
variance in obs after conditioning) must reach effect_size_floor, so large
samples do not reject a simpler, more powerful tier for trivially small
departures.

Subclasses implement _compute_statistic and _null_fit, and _null_leverage
when the null model is a least-squares fit.

See docs/methods.md, Escalation tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

from model_selection.statistics.significance.base import bias_correct_association, permutation_p_value
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights
from model_selection.utils import as2d

DRAW_CHUNK = 128  # bootstrap draws whose multipliers are generated at a time


@dataclass
class AdequacyTestOutcome:
    """Decision of one adequacy check, as consumed by greedy selection.

    tier_adequate is True when the simpler measure's assumption is NOT
    rejected for this state; False means the whole selection trial under
    that measure must be abandoned.
    """
    tier_adequate: bool
    decision_reason: str
    p_value: float
    effect_size: float | None = None
    semipartial_effect_size: float | None = None
    bias_corrected_semipartial_effect_size: float | None = None
    block_length: int | None = None
    dependence: dict | None = None


AdequacyTestFunc = Callable[
    [np.ndarray, np.ndarray, "np.ndarray | None", int,
     "np.random.Generator | None", "np.ndarray | None", int],
    AdequacyTestOutcome,
]
"""adequacy_test(candidate, obs, conditioning, n_permutations=..., rng=...,
time_positions=..., min_temporal_lag=...) -> AdequacyTestOutcome"""

AdequacyPlotFunc = Callable[..., "str | None"]
"""adequacy_plot(candidate, obs, conditioning, round_label, context=None) -> written path or None.
Called with exactly the state the adequacy decision was made on; context
(visualization.adequacy.PlotContext) describes the round and the outcome."""


@dataclass
class FunctionalFormTestResult:
    """Full result of one FunctionalFormTest.run().

    classical_p_value is the i.i.d.-theory reference (NaN where none is
    implemented) and is reported for comparison only. permutation_p_value is
    the decision p-value, (M + 1) / (N + 1). reject_linearity is the final decision: significant AND
    meets_effect_size_floor (the name is historical; it applies to every
    subclass). effect_size is the partial R^2; semipartial_effect_size is the
    decision-relevant semipartial R^2, compared against effect_size_floor
    after subtracting its null mean. The null_* summaries describe the full
    permutation null of the statistic.
    """
    test_name: str
    observed_statistic: float
    df_numerator: int
    df_denominator: int
    classical_p_value: float
    permutation_p_value: float
    n_permutations: int
    alpha: float
    reject_linearity: bool
    significant: bool
    effect_size: float | None = None
    semipartial_effect_size: float | None = None
    effect_size_floor: float | None = None
    meets_effect_size_floor: bool = True
    null_semipartial_effect_size_mean: float | None = None
    bias_corrected_semipartial_effect_size: float | None = None
    decision_reason: str = ""
    dependence_aware_subsampling_used: bool = False
    n_after_subsampling: int | None = None
    natural_independent_sample_size: int | None = None
    block_length: int | None = None
    dependence: dict | None = None
    null_mean: float = 0.0
    null_std: float = 0.0
    null_median: float = 0.0
    null_p90: float = 0.0
    null_p99: float = 0.0
    null_max: float = 0.0


class FunctionalFormTest(ABC):
    """Tests one state (candidate, obs, conditioning). Subclasses supply the
    statistic and the null model; this class supplies the residual-resampling
    null, p-value correction, effect-size gate and result packaging."""

    # Subclasses that accept multivariate candidate/obs set this True.
    _supports_multivariate: bool = False

    independent_multipliers = "gaussian"
    """Distribution of independent bootstrap weights (multipliers module):
    Rademacher for regression statistics and for the effect-modification
    test (state_independence), standard normal for the other kernel
    statistics, which are quadratic forms in the weights."""

    def __init__(self, alpha: float = 0.05, effect_size_floor: float = 0.02,
                 run_length: int = DEFAULT_RUN_LENGTH):
        self.alpha = alpha
        self.effect_size_floor = effect_size_floor
        self.run_length = run_length

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _compute_statistic(
            self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
    ) -> tuple[float, int, int, float | None, float | None]:
        """Return (statistic, df_numerator, df_denominator, effect_size,
        semipartial_effect_size). The degrees of freedom are used only for
        the classical p-value; return None for an effect size a test has no
        analogue for (the effect-size gate is then skipped)."""
        ...

    @abstractmethod
    def _null_fit(self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None) -> np.ndarray:
        """Fitted values of obs under the null model (same shape as obs)."""
        ...

    def _null_leverage(self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None) -> np.ndarray:
        """1 - Var(r_t) / sigma^2 for the null model's residuals r_t under
        homoscedastic errors, for the scaling of the bootstrapped residuals:
        the leverages for a least-squares null model (the HC2 correction);
        state_independence gives the kernel ridge equivalent. Default: no
        scaling."""
        return np.zeros(as2d(candidate).shape[0])

    def _classical_p_value(self, statistic: float, df_num: int, df_den: int) -> float:
        """F-distribution reference p-value. Subclasses without a closed-form
        reference return NaN."""
        if df_den > 0:
            return float(stats.f.sf(statistic, df_num, df_den))
        return float("nan")

    def run(
            self,
            candidate: np.ndarray,
            obs: np.ndarray,
            conditioning: np.ndarray | None,
            time_positions: np.ndarray | None = None,
            min_temporal_lag: int = 0,
            n_permutations: int = 500,
            rng: np.random.Generator | None = None,
            grid_step: float | None = None,
    ) -> FunctionalFormTestResult:
        """Test one fully specified state. conditioning may be None or (n, k);
        grid_step as in ConditionalIndependenceTest."""
        rng = rng or np.random.default_rng()
        candidate = np.asarray(candidate, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)
        if not self._supports_multivariate:
            for label, arr in (("candidate", candidate), ("obs", obs)):
                if arr.ndim > 1 and arr.shape[1] > 1:
                    raise ValueError(
                        f"{type(self).__name__} supports scalar candidate/obs only; got "
                        f"multivariate {label} with shape {arr.shape}. Use "
                        "MultivariateLinearityTest or StateIndependenceTest for "
                        "multivariate (e.g. FPC score) samples."
                    )
            candidate = candidate.ravel()
            obs = obs.ravel()
        n = candidate.shape[0]

        if time_positions is None:
            time_positions = np.arange(n)
        else:
            time_positions = np.asarray(time_positions)
            if time_positions.shape[0] != n:
                raise ValueError("time_positions must have the same length as candidate")

        observed_stat, df_num, df_den, effect_size, semipartial_effect_size = self._compute_statistic(
            candidate, obs, conditioning,
        )
        classical_p = self._classical_p_value(observed_stat, df_num, df_den)

        # Residual resampling under the fitted null model. The semipartial
        # effect size of every draw is kept so the effect-size gate can be
        # bias-corrected on its own (R^2) scale.
        fitted_null = self._null_fit(candidate, obs, conditioning)
        resid_null = obs - fitted_null
        leverage = np.asarray(self._null_leverage(candidate, obs, conditioning), dtype=float)
        hc2 = 1.0 / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
        scaled = resid_null * (hc2 if resid_null.ndim == 1 else hc2[:, None])
        # Block length from the products of the resampled residuals with the
        # candidate, from which the tested departure terms are built.
        resolved = resolve_block(time_positions, min_temporal_lag, [(resid_null, candidate)], self.run_length,
                                 grid_step)
        block = resolved.length
        null_stats = np.empty(n_permutations)
        null_semipartial_effect_sizes = np.empty(n_permutations)
        for start in range(0, n_permutations, DRAW_CHUNK):
            stop = min(start + DRAW_CHUNK, n_permutations)
            weights = dependent_wild_weights(time_positions, block, stop - start, rng,
                                             independent=self.independent_multipliers)
            for b, w in zip(range(start, stop), weights):
                obs_b = fitted_null + scaled * (w if scaled.ndim == 1 else w[:, None])
                stat_b, _, _, _, semipartial_b = self._compute_statistic(candidate, obs_b, conditioning)
                null_stats[b] = stat_b
                null_semipartial_effect_sizes[b] = semipartial_b if semipartial_b is not None else 0.0

        p_used = permutation_p_value(observed_stat, null_stats)
        significant = bool(p_used < self.alpha)

        if semipartial_effect_size is not None and self.effect_size_floor is not None:
            bias_corrected_semipartial, null_semipartial_effect_size_mean = bias_correct_association(
                semipartial_effect_size, null_semipartial_effect_sizes)
            meets_floor = bool(bias_corrected_semipartial >= self.effect_size_floor)
        else:
            bias_corrected_semipartial = None
            null_semipartial_effect_size_mean = None
            meets_floor = True

        reject = significant and meets_floor

        if not significant:
            decision_reason = f"not rejected: p={p_used:.4g} >= alpha={self.alpha:.4g}"
        elif not meets_floor:
            decision_reason = (
                f"not rejected: statistically significant (p={p_used:.4g}) but "
                f"bias-corrected semipartial R^2={bias_corrected_semipartial:.4g} < "
                f"effect_size_floor={self.effect_size_floor:.4g} "
                f"(raw semipartial R^2={semipartial_effect_size:.4g}, "
                f"null mean={null_semipartial_effect_size_mean:.4g})"
            )
        else:
            floor_note = (
                f"bias-corrected semipartial R^2={bias_corrected_semipartial:.4g} >= "
                f"effect_size_floor={self.effect_size_floor:.4g} "
                f"(raw semipartial R^2={semipartial_effect_size:.4g}, "
                f"null mean={null_semipartial_effect_size_mean:.4g})"
                if semipartial_effect_size is not None else "no effect-size floor applied"
            )
            decision_reason = f"rejected: significant (p={p_used:.4g}) and {floor_note}"

        return FunctionalFormTestResult(
            test_name=self.name,
            observed_statistic=observed_stat,
            df_numerator=df_num,
            df_denominator=df_den,
            classical_p_value=classical_p,
            permutation_p_value=p_used,
            n_permutations=n_permutations,
            alpha=self.alpha,
            reject_linearity=reject,
            significant=significant,
            effect_size=effect_size,
            semipartial_effect_size=semipartial_effect_size,
            effect_size_floor=self.effect_size_floor,
            meets_effect_size_floor=meets_floor,
            null_semipartial_effect_size_mean=null_semipartial_effect_size_mean,
            bias_corrected_semipartial_effect_size=bias_corrected_semipartial,
            decision_reason=decision_reason,
            block_length=block,
            dependence=resolved.to_record(),
            null_mean=float(np.mean(null_stats)),
            null_std=float(np.std(null_stats, ddof=1)),
            null_median=float(np.median(null_stats)),
            null_p90=float(np.percentile(null_stats, 90)),
            null_p99=float(np.percentile(null_stats, 99)),
            null_max=float(np.max(null_stats)),
        )


def as_adequacy_test(test: FunctionalFormTest) -> AdequacyTestFunc:
    """Adapt a FunctionalFormTest instance to the AdequacyTestFunc convention
    used by greedy selection."""

    def _adequacy(
            candidate: np.ndarray,
            obs: np.ndarray,
            conditioning: np.ndarray | None,
            n_permutations: int = 100,
            rng: np.random.Generator | None = None,
            time_positions: np.ndarray | None = None,
            min_temporal_lag: int = 0,
            grid_step: float | None = None,
    ) -> AdequacyTestOutcome:
        result = test.run(
            candidate, obs, conditioning,
            time_positions=time_positions, min_temporal_lag=min_temporal_lag,
            n_permutations=n_permutations, rng=rng, grid_step=grid_step,
        )
        return AdequacyTestOutcome(
            tier_adequate=not result.reject_linearity,
            decision_reason=result.decision_reason,
            p_value=result.permutation_p_value,
            effect_size=result.effect_size,
            semipartial_effect_size=result.semipartial_effect_size,
            bias_corrected_semipartial_effect_size=result.bias_corrected_semipartial_effect_size,
            block_length=result.block_length,
            dependence=result.dependence,
        )

    return _adequacy
