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
2014). On a regular time grid, a statistic that divides a quadratic form in the
residuals by an estimate of their scale (an F statistic) is redrawn as its
numerator over the observed denominator: in the data the denominator is an
average over many samples and barely varies, whereas the draws' sum of
squared multiplied residuals varies with the correlated multipliers
(relative variance about 4b/3 at bandwidth fraction b), so recomputing it
would give the draws a variability the statistic does not have. The draws
then reproduce the numerator, a quadratic form in score sums, as the tiers'
tests do, with the score regressors held fixed (the score bootstrap; Kline
and Santos, 2012), and each is divided by a draw of the fixed-b variance
functional (Kiefer and Vogelsang, 2005; multipliers.fixed_b_draws).

Samples that are independent units (clusters of events) take the wild
cluster restricted bootstrap instead (Cameron, Gelbach and Miller, 2008):
one Rademacher sign per unit, each cluster's residuals scaled by the CR2
correction of the null fit where the test supplies its residual maker
(significance.clusters.cluster_scaled_residuals), and each draw's statistic
recomputed whole, with its own denominator. With one sign per unit the
draws' sum of squares does not vary with the signs, so the reason for the
observed denominator does not arise; and refitting each draw removes the
fit's projection of the multiplied residuals, which is larger than a
per-sample correction restores when the regressors are shared within
clusters, so draws over the observed denominator would be too small.
The p-value is (M + 1) / (N + 1) (Phipson and Smyth, 2010). Rejection also requires a practically meaningful
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

from model_selection.statistics.significance.base import (
    bias_correct_association, draw_until_decided, permutation_p_value, resamples_at_level, sequential_decision)
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights, fixed_b_draws
from model_selection.statistics.significance.clusters import cluster_scaled_residuals, smallest_p_value
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
    n_draws: int | None = None
    level: float | None = None
    settled: bool | None = None
    """n_draws: null draws used. level: the level the test decided at when
    it was below alpha (tier_adequate is then the decision at that level),
    else None. settled: for such a test, whether the sequential test settled
    the decision before the most resamples (False: decided by the p-value
    from all of them); None otherwise."""


AdequacyTestFunc = Callable[
    [np.ndarray, np.ndarray, "np.ndarray | None", int,
     "np.random.Generator | None", "np.ndarray | None", int],
    AdequacyTestOutcome,
]
"""adequacy_test(candidate, obs, conditioning, n_permutations=..., rng=...,
time_positions=..., min_temporal_lag=..., level=None, resampling_risk=None) -> AdequacyTestOutcome

level, resampling_risk : for a test made at a level below alpha, the level
and the sequential test's resampling risk (FunctionalFormTest.run)."""

AdequacyPlotFunc = Callable[..., "str | None"]
"""adequacy_plot(candidate, obs, conditioning, round_label, context=None) -> written path or None.
Called with exactly the state the adequacy decision was made on; context
(visualization.adequacy.PlotContext) describes the round and the outcome."""


@dataclass
class FunctionalFormTestResult:
    """Full result of one FunctionalFormTest.run().

    classical_p_value is the i.i.d.-theory reference (NaN where none is
    implemented) and is reported for comparison only. permutation_p_value is
    the decision p-value, (M + 1) / (N + 1), over the n_permutations draws
    used. level and settled: as in AdequacyTestOutcome. reject_linearity is the final decision: significant AND
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
    level: float | None = None
    settled: bool | None = None


class FunctionalFormTest(ABC):
    """Tests one state (candidate, obs, conditioning). Subclasses supply the
    statistic and the null model; this class supplies the residual-resampling
    null, p-value correction, effect-size gate and result packaging."""

    def _run_state(self):
        """State that holds for one call of run, kept per thread, so that one
        test object can run in several threads at once (the power tests of a
        round's remaining candidates run in parallel)."""
        import threading
        return self.__dict__.setdefault("_thread_state", threading.local())

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

    def _statistic_parts(
            self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
    ) -> tuple[float, float, float, float] | None:
        """For a statistic that divides a quadratic form in the residuals by
        an estimate of their scale: (numerator, denominator, semipartial
        numerator, semipartial denominator), with statistic = numerator /
        denominator and semipartial effect = semipartial numerator /
        semipartial denominator. Each null draw then recomputes the numerators
        only and divides them by the observed denominators (module docstring).
        None (the default) for score statistics, which have no denominator."""
        return None

    def _score_terms(self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
                     resid_null: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """The factors of the statistic's per-sample terms, as (left, right)
        pairs whose column products, summed over samples, are the sums the
        statistic is a quadratic form in: the null model's residuals and the
        tested regressors (a regression test), or the residuals and their
        kernel smooth, column by column (a kernel score r'Kr = sum_t r_t
        (Kr)_t). The multipliers' bandwidth is set from their persistence
        (significance.block_length.resolve_block), as for the tiers' tests
        and the linear adjustment test. Subclasses implement it."""
        raise NotImplementedError

    def _null_residual_maker(self, candidate: np.ndarray, obs: np.ndarray,
                             conditioning: np.ndarray | None) -> np.ndarray | None:
        """The null model's residual maker R (n, n), r = R y, for the
        cluster-level scaling of the bootstrapped residuals when samples are
        independent units (significance.clusters.cluster_scaled_residuals).
        None (the default): the per-sample scaling of _null_leverage is used."""
        return None

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
            level: float | None = None,
            resampling_risk: float | None = None,
    ) -> FunctionalFormTestResult:
        """Test one fully specified state. conditioning may be None or (n, k);
        grid_step as in ConditionalIndependenceTest.

        level, resampling_risk : for a test made at a level below alpha (the
        escalation family's Bonferroni level), the level and the resampling
        risk of the sequential test that decides at it: n_permutations draws
        first, then further draws until the decision is settled, up to the
        draws that give at the level the relative Monte Carlo error that
        n_permutations gives at alpha (significance.base.draw_until_decided;
        docs/methods.md, Resamples per test). Omitted, or at alpha or above,
        the test draws n_permutations and decides at alpha."""
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
        # Bandwidth from the persistence of the statistic's own per-sample terms
        # (_score_terms); the longest over the pairs, which share the multipliers.
        resolved = resolve_block(time_positions, min_temporal_lag,
                                 self._score_terms(candidate, obs, conditioning, resid_null), self.run_length,
                                 grid_step)
        block = resolved.length
        units = resolved.independent_units
        maker = self._null_residual_maker(candidate, obs, conditioning) if units else None
        if maker is not None:
            # Independent units (clusters of events): each cluster's residuals
            # scaled by its block of the fit's residual covariance (CR2;
            # significance.clusters.cluster_scaled_residuals).
            scaled = cluster_scaled_residuals(resid_null, maker, time_positions)
        else:
            leverage = np.asarray(self._null_leverage(candidate, obs, conditioning), dtype=float)
            hc2 = 1.0 / np.sqrt(np.clip(1.0 - leverage, 1e-12, None))
            scaled = resid_null * (hc2 if resid_null.ndim == 1 else hc2[:, None])
        # Independent units: one Rademacher multiplier per unit (multipliers
        # module), and each draw's statistic recomputed whole, with its own
        # denominator. The observed denominator is kept only on a regular grid
        # (module docstring).
        independent = "rademacher" if units else self.independent_multipliers
        parts = None if units else self._statistic_parts(candidate, obs, conditioning)

        def bootstrap_draws(count: int) -> tuple[np.ndarray, np.ndarray]:
            """`count` null draws of the statistic and of the semipartial effect."""
            stats_b = np.empty(count)
            semis_b = np.empty(count)
            for start in range(0, count, DRAW_CHUNK):
                stop = min(start + DRAW_CHUNK, count)
                weights = dependent_wild_weights(time_positions, block, stop - start, rng, independent=independent)
                for b, w in zip(range(start, stop), weights):
                    obs_b = fitted_null + scaled * (w if scaled.ndim == 1 else w[:, None])
                    if parts is None:
                        stat_b, _, _, _, semipartial_b = self._compute_statistic(candidate, obs_b, conditioning)
                    else:
                        # The draw's numerators over the observed denominators.
                        num_b, _, semi_num_b, _ = self._statistic_parts(candidate, obs_b, conditioning)
                        stat_b = num_b / parts[1] if parts[1] > 0 else 0.0
                        semipartial_b = semi_num_b / parts[3] if parts[3] > 0 else 0.0
                    stats_b[b] = stat_b
                    semis_b[b] = semipartial_b if semipartial_b is not None else 0.0
            return stats_b, semis_b

        # The fixed-b reference distribution (multipliers.fixed_b_draws), as in
        # the tiers' tests: each draw divided by a draw of the variance
        # functional; ones off a regular grid. The semipartial draws serve only
        # the chance level of the observed effect, their mean, so they are
        # divided by the functional's mean, which corrects the bandwidth's
        # downward bias in the multipliers' variance (statistics.power).
        def referenced_draws(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            stats_b, semis_b = bootstrap_draws(count)
            q_b = fixed_b_draws(resolved.bandwidth_fraction, count, rng, n_steps=resolved.grid_length)
            return stats_b / q_b, semis_b, q_b

        null_stats, null_semipartial_effect_sizes, q = referenced_draws(n_permutations)
        # A test made at a level below alpha (the escalation family's Bonferroni
        # level) draws on until its decision at that level is settled
        # (significance.base.draw_until_decided; docs/methods.md, Resamples per test).
        sequential = level is not None and resampling_risk is not None and level < self.alpha
        settled = None
        if sequential:
            (null_stats, null_semipartial_effect_sizes, q), settled = draw_until_decided(
                observed_stat, (null_stats, null_semipartial_effect_sizes, q), referenced_draws, level,
                resampling_risk, resamples_at_level(self.alpha, n_permutations, level), DRAW_CHUNK)
        null_semipartial_effect_sizes = null_semipartial_effect_sizes / float(np.mean(q))
        n_drawn = null_stats.shape[0]

        p_used = permutation_p_value(observed_stat, null_stats)
        # Too few independent units for any test of this kind to reject at alpha
        # (significance.clusters): a smaller p-value is an artefact of the
        # residuals' scaling, and the check is untestable, not significant.
        untestable = bool(units and resolved.n_blocks is not None
                          and smallest_p_value(int(resolved.n_blocks)) >= self.alpha)
        threshold = level if sequential else self.alpha
        if sequential and settled:
            below = bool(sequential_decision(int(np.sum(null_stats >= observed_stat)), n_drawn, level,
                                             resampling_risk))
        else:
            below = bool(p_used < threshold)
        significant = below and not untestable

        if semipartial_effect_size is not None and self.effect_size_floor is not None:
            bias_corrected_semipartial, null_semipartial_effect_size_mean = bias_correct_association(
                semipartial_effect_size, null_semipartial_effect_sizes)
            meets_floor = bool(bias_corrected_semipartial >= self.effect_size_floor)
        else:
            bias_corrected_semipartial = None
            null_semipartial_effect_size_mean = None
            meets_floor = True

        reject = significant and meets_floor

        if untestable:
            decision_reason = (f"not rejected: untestable, {resolved.n_blocks:.0f} independent clusters cannot give a "
                               f"p-value below alpha={self.alpha:.4g} (p={p_used:.4g} not used)")
        elif not significant:
            label = "level" if sequential else "alpha"
            decision_reason = f"not rejected: p={p_used:.4g} >= {label}={threshold:.4g}"
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
        if sequential:
            decision_reason += (f" [{n_drawn} resamples, tested at level {level:.4g}; "
                                + ("decision settled by the sequential test]" if settled else
                                   "the most resamples drawn without settling the decision: decided by the p-value "
                                   "from all of them]"))

        return FunctionalFormTestResult(
            test_name=self.name,
            observed_statistic=observed_stat,
            df_numerator=df_num,
            df_denominator=df_den,
            classical_p_value=classical_p,
            permutation_p_value=p_used,
            n_permutations=n_drawn,
            alpha=self.alpha,
            level=level if sequential else None,
            settled=settled,
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
            level: float | None = None,
            resampling_risk: float | None = None,
    ) -> AdequacyTestOutcome:
        result = test.run(
            candidate, obs, conditioning,
            time_positions=time_positions, min_temporal_lag=min_temporal_lag,
            n_permutations=n_permutations, rng=rng, grid_step=grid_step,
            level=level, resampling_risk=resampling_risk,
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
            n_draws=result.n_permutations,
            level=result.level,
            settled=result.settled,
        )

    return _adequacy
