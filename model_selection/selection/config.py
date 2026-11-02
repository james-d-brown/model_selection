"""Configuration of the hierarchical selection framework."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH

TIERS = ("linear", "nonlinear", "cmi")
DEFAULT_ADEQUACY_FLOOR = 0.02
CASCADE = "cascade"
SUPPORTED_MEASURES = TIERS + (CASCADE,)


def _warn_if_coarse(n_permutations: int, alpha: float, owner: str) -> None:
    from model_selection.statistics.significance.base import minimum_resamples

    needed = minimum_resamples(alpha)
    if n_permutations < needed:
        warnings.warn(
            f"{owner}: n_permutations={n_permutations} is below {needed} (10 / alpha at alpha={alpha}); "
            "fewer than 10 null draws fall in the rejection region, so decisions near alpha are coarse.",
            stacklevel=3,
        )


@dataclass(frozen=True)
class SelectionConfig:
    """How models are selected and how significance is tested.

    measure : "linear", "nonlinear" or "cmi" to use one tier throughout, or
        "cascade" to start at "linear" and escalate when a tier's escalation
        tests reject (see docs/methods.md, The cascade).
    n_permutations : resamples per significance test (dependent wild
        bootstrap draws for the linear and nonlinear tiers, local
        permutations for the free-form tier); None
        (default) derives it from alpha so that the Monte Carlo standard
        error of the p-value at the threshold is at most
        monte_carlo_relative_error times alpha (1,900 at alpha = 0.05), as
        for the escalation tests. A warning is issued below 10 / alpha.
    monte_carlo_relative_error : see n_permutations.
    seed : seed of the random generator for resampling and tie-breaking.
        None (default) draws a fresh seed for each run; the seed used is
        logged and recorded in the result, so any run can be repeated by
        passing it here.
    alpha : significance level.
    effect_size_floor : minimum bias-corrected effect size for acceptance on
        the R^2-like scale; converted to nats for CMI.
    min_temporal_lag : dependence lag in timesteps; None estimates it
        obs-free as the largest Politis decorrelation lag of the
        formulations. Resampling of samples on a regular time grid uses a
        Politis–White block length estimated within each test instead
        (docs/methods.md, Dependence between samples).
    dependence_run_length : number of consecutive statistically insignificant
        autocorrelations that ends the search for the decorrelation lag in
        the Politis–White estimate. Default 5, as Politis and White (2004,
        Remark 2.3) suggest; it guards against isolated insignificant values,
        not against cycles, and is counted in samples, so its duration in
        time depends on the sampling interval (docs/methods.md, Dependence
        between samples).
    cmi_k_fraction : the CMI estimator's neighbour count as a fraction of
        the sample size, k = max(1, round(cmi_k_fraction * n)); Runge (2018)
        recommends 0.1 to 0.2 (docs/methods.md, Free-form tier).
    cmi_k : fixed neighbour count overriding cmi_k_fraction; None (default)
        uses the fraction.
    n_neighbors : local-permutation neighbourhood size (CMI tier); Runge
        (2018) recommends 5 to 10.
    theiler_window : the CMI tier's Theiler window: neighbours closer than
        this in time are not local-permutation partners (Theiler, 1986).
        None (default) uses the dependence lag. The high-flow pathway sets a
        shorter window when the dependence lag would leave an event too few
        partners; every other use of the dependence lag keeps the full lag.
    cmi_max_n : largest pooled sample the CMI tier is run on. Its local
        permutation treats samples as exchangeable, so it runs on samples at
        least one dependence lag apart, at most this many. 2000 is the upper
        end of the sample sizes over which Runge (2018) validated the test.
    meaningful_r2 : the smallest association, as a set-correlation R^2, that
        matters for your application: the share of the variation in the
        observations, beyond what the selected formulations explain, that a
        candidate must account for to be worth selecting. With one feature it
        is the fraction by which adding the candidate would reduce the
        unexplained variance (R^2 = 0.2 reduces it by 20%, the RMSE by about
        11%). Every result reports the smallest contribution the data can
        detect with 80% power; when meaningful_r2 is set and that is larger,
        the result is flagged as unable to resolve contributions at the
        meaningful level. Selection is never stopped by it. When set, it is
        also the cascade's escalation floor (AdequacyConfig.effect_size_floor). See
        docs/methods.md, Detectable effect and the meaningful-effect threshold.
    cmi_calibration : check the CMI tier's calibration on surrogate
        candidates before it accepts a candidate or the cascade escalates to
        it. On by default in both pathways; turning it off saves
        computation (52 surrogate tests per use at alpha = 0.05), and a CMI
        selection made without it is flagged for review.
    equivalence_checks : after selection, compare each selected formulation
        with every other candidate in both directions, to report which are
        interchangeable with it (docs/methods.md, Interchangeable
        formulations). Reported only; costs two tests per candidate per
        accepted round.
    cmi_calibration_rate_ratio : the CMI tier's calibration on surrogate
        candidates uses as many surrogates as needed to detect, with 80%
        power, a false-acceptance rate this many times alpha
        (docs/methods.md, Free-form tier).
    min_samples_per_dimension : a selection step is flagged for review when
        its samples per dimension (candidate, observations and selected set
        together) fall below this value. A heuristic, by analogy with
        events-per-variable guidance in regression (Peduzzi et al., 1996),
        not a derived requirement.
    nonlinear_max_n : largest pooled sample the nonlinear tier is run on,
        because kernel regression scales as O(n^3); longer records give it an
        evenly spaced subsample of this many points, whose dependence the
        tier's bootstrap accounts for.
    """
    measure: str = "cascade"
    n_permutations: int | None = None
    monte_carlo_relative_error: float = 0.1
    seed: int | None = None
    alpha: float = 0.05
    effect_size_floor: float = 0.02
    dependence_run_length: int = DEFAULT_RUN_LENGTH
    min_temporal_lag: int | None = None
    cmi_k_fraction: float = 0.1
    cmi_k: int | None = None
    n_neighbors: int = 10
    theiler_window: int | None = None
    cmi_max_n: int = 2000
    nonlinear_max_n: int = 2000
    cmi_calibration: bool = True
    cmi_calibration_rate_ratio: float = 3.0
    equivalence_checks: bool = True
    min_samples_per_dimension: float = 10.0
    meaningful_r2: float | None = None

    def __post_init__(self):
        if self.measure not in SUPPORTED_MEASURES:
            raise ValueError(f"measure must be one of {SUPPORTED_MEASURES} -- got {self.measure!r}")
        if not 0.0 < self.cmi_k_fraction < 1.0:
            raise ValueError(f"cmi_k_fraction must be in (0, 1) -- got {self.cmi_k_fraction}")
        _warn_if_coarse(self.resolved_n_permutations, self.alpha, "SelectionConfig")
        if self.meaningful_r2 is not None and not 0.0 < self.meaningful_r2 < 1.0:
            raise ValueError(f"meaningful_r2 must be in (0, 1) -- got {self.meaningful_r2}")

    @property
    def resolved_n_permutations(self) -> int:
        """Resamples used per significance test."""
        if self.n_permutations is not None:
            return self.n_permutations
        from model_selection.statistics.significance.base import resamples_for_relative_error

        return resamples_for_relative_error(self.alpha, self.monte_carlo_relative_error)


@dataclass(frozen=True)
class AdequacyConfig:
    """Settings of the cascade's adequacy tests (used only when
    SelectionConfig.measure == "cascade").

    alpha : the adequacy tests' significance level.
    effect_size_floor : the smallest departure from a tier's assumption that
        triggers escalation, as the share of the observations' variation left
        unexplained by the selected formulations that the simpler tier would
        miss (0.02 = 2%). None (default) uses SelectionConfig.meaningful_r2
        when that is set, and 0.02 otherwise. See docs/methods.md, Selection.
    n_permutations : resamples per adequacy test; None (default) derives it
        from alpha so that the Monte Carlo standard error of the p-value at
        the threshold is at most monte_carlo_relative_error times alpha
        (1,900 at alpha = 0.05). A warning is issued below 10 / alpha.
    monte_carlo_relative_error : see n_permutations.
    ridge_lambda : kernel ridge penalty of the kernel adequacy tests; None
        estimates it by REML.
    kernel_check_max_n : sample cap of the two kernel-based escalation
        tests, the linear adjustment test (linear tier) and the candidate
        effect-modification test. Longer records are subsampled evenly to this
        many points, because their kernels are n x n; their bootstraps account
        for the dependence that remains. Matched to nonlinear_max_n, so in the
        nonlinear tier the effect-modification test runs on that tier's sample
        without further thinning.
    """
    alpha: float = 0.05
    effect_size_floor: float | None = None
    n_permutations: int | None = None
    monte_carlo_relative_error: float = 0.1
    ridge_lambda: float | None = None
    kernel_check_max_n: int = 2000

    def __post_init__(self):
        _warn_if_coarse(self.resolved_n_permutations, self.alpha, "AdequacyConfig")

    def resolved_effect_size_floor(self, selection: "SelectionConfig") -> float:
        """Escalation floor used: this config's value, else the meaningful-effect
        threshold, else the conventional 0.02."""
        if self.effect_size_floor is not None:
            return self.effect_size_floor
        if selection.meaningful_r2 is not None:
            return selection.meaningful_r2
        return DEFAULT_ADEQUACY_FLOOR

    @property
    def resolved_n_permutations(self) -> int:
        """Resamples used per adequacy test."""
        if self.n_permutations is not None:
            return self.n_permutations
        from model_selection.statistics.significance.base import resamples_for_relative_error

        return resamples_for_relative_error(self.alpha, self.monte_carlo_relative_error)
