"""Configuration of the hierarchical selection framework."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH

TIERS = ("linear", "nonlinear", "cmi")
DEFAULT_ADEQUACY_FLOOR = 0.02
DEFAULT_NONLINEAR_MAX_N = 16000
"""Default sample cap of the nonlinear tier (pooled pathway), read when a
SelectionConfig is created (the test suite lowers it for speed)."""
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
    min_temporal_lag : dependence lag in timesteps; None estimates it as
        the largest Politis decorrelation lag of the formulations and the
        observations. Resampling of samples on a regular time grid uses a
        bandwidth estimated within each test instead (docs/methods.md,
        Dependence between samples).
    dependence_run_length : number of consecutive statistically insignificant
        autocorrelations that ends the search for a decorrelation lag
        (Politis, 2003). Default 5, as Politis and White (2004,
        Remark 2.3) suggest; it guards against isolated insignificant values,
        not against cycles, and is counted in samples, so its duration in
        time depends on the sampling interval (docs/methods.md, Dependence
        between samples).
    cmi_k : the CMI estimator's neighbour count, fixed and small so that
        the neighbourhoods stay local and shrink as the sample grows
        (Kraskov et al., 2004; Gao et al., 2018; Holmes and Nemenman, 2019);
        5, the size of the local permutation's neighbourhood (Runge, 2018).
        At most n - 1 on a sample of n (docs/methods.md, Free-form tier).
    n_neighbors : local-permutation neighbourhood size (CMI tier), counting
        the sample itself, which may keep its own value: 2, the sample and its
        nearest neighbour, so that the permutation needs the least smoothness
        of C's distribution in Z (Kim et al., 2022); small neighbourhoods make
        the null conservative (Runge, 2018).
    theiler_window : the CMI tier's Theiler window: neighbours closer than
        this in time are not local-permutation partners (Theiler, 1986).
        None (default) uses the dependence lag. The high-flow event pathway sets a
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
        11%). Selection is never stopped by it. When set, it is the cascade's
        escalation floor for the candidate linearity and effect-modification
        tests (AdequacyConfig.effect_size_floor). See docs/methods.md, Step
        outcomes and the meaningful-effect threshold.
    cmi_calibration : check the CMI tier's calibration on surrogate
        candidates before it accepts a candidate or the cascade escalates to
        it. On by default in both pathways; turning it off saves
        computation (52 surrogate tests per use at alpha = 0.05), and a CMI
        selection made without it is flagged for review.
    backward_pass : after forward selection, test each selected formulation,
        in selection order, given the other formulations still selected, and
        remove it at once if it adds no information (docs/methods.md,
        Backward pass). At most one test per selected formulation.
    equivalence_checks : after selection, compare each selected formulation
        with every formulation not selected, given the other selected
        formulations, and report the class of every formulation: distinct,
        exchangeable, redundant or not tested (docs/methods.md,
        Interchangeable formulations). Reported only; one test per selected
        formulation per formulation not selected, plus the tests that verify
        equally supported sets.
    cmi_calibration_rate_ratio : the CMI tier's calibration on surrogate
        candidates uses as many surrogates as needed to detect, with 80%
        power, a false-acceptance rate this many times alpha
        (docs/methods.md, Free-form tier).
    min_samples_per_dimension : a selection step is flagged for review when
        its samples per dimension (candidate, observations and selected set
        together) fall below this value. A heuristic, by analogy with
        events-per-variable guidance in regression (Peduzzi et al., 1996),
        not a derived requirement.
    nonlinear_max_n : largest pooled sample the nonlinear tier is run on
        (16,000); longer records give it an evenly spaced subsample of this
        many points, whose dependence the tier's bootstrap accounts for. With
        low-rank kernels its cost grows as n * rank^2 (docs/methods.md,
        Nonlinear tier).
    nonlinear_rank : cap on the rank of the nonlinear tier's kernel factors.
        Samples up to this size get exact factors; longer ones pivoted-Cholesky
        factors, which equal the kernels to working precision below the cap
        (docs/methods.md, Nonlinear tier). Their cost grows
        as n * rank^2. None uses exact factors at every size (O(n^3)).
    n_jobs : threads for the analysis's independent computations
        (model_selection.parallel): the candidates' effect sizes, the
        eigendecompositions and kernel factors a round needs, the
        comparisons and the checks of equally supported sets. None (default)
        uses every core, 1 runs them in sequence. Results do not depend on
        it, apart from rounding.
    """
    measure: str = "cascade"
    n_permutations: int | None = None
    monte_carlo_relative_error: float = 0.1
    seed: int | None = None
    alpha: float = 0.05
    effect_size_floor: float = 0.02
    dependence_run_length: int = DEFAULT_RUN_LENGTH
    min_temporal_lag: int | None = None
    cmi_k: int = 5
    n_neighbors: int = 2
    theiler_window: int | None = None
    cmi_max_n: int = 2000
    nonlinear_max_n: int = field(default_factory=lambda: DEFAULT_NONLINEAR_MAX_N)
    cmi_calibration: bool = True
    cmi_calibration_rate_ratio: float = 3.0
    backward_pass: bool = True
    equivalence_checks: bool = True
    min_samples_per_dimension: float = 10.0
    nonlinear_rank: int | None = 2000
    n_jobs: int | None = None
    meaningful_r2: float | None = None

    def __post_init__(self):
        if self.measure not in SUPPORTED_MEASURES:
            raise ValueError(f"measure must be one of {SUPPORTED_MEASURES} -- got {self.measure!r}")
        if self.cmi_k < 1:
            raise ValueError(f"cmi_k must be at least 1 -- got {self.cmi_k}")
        if self.nonlinear_rank is not None and self.nonlinear_rank < 1:
            raise ValueError(f"nonlinear_rank must be None or at least 1 -- got {self.nonlinear_rank}")
        if self.n_jobs is not None and self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None or at least 1 -- got {self.n_jobs}")
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
        when that is set, and 0.02 otherwise. The linear adjustment test,
        which concerns validity, uses the smaller of this and the acceptance
        floor (resolved_adjustment_floor). See docs/methods.md, Selection.
    n_permutations : resamples per adequacy test; None (default) derives it
        from alpha so that the Monte Carlo standard error of the p-value at
        the threshold is at most monte_carlo_relative_error times alpha
        (1,900 at alpha = 0.05). A warning is issued below 10 / alpha.
    monte_carlo_relative_error : see n_permutations.
    ridge_lambda : kernel ridge penalty of the kernel adequacy tests; None
        estimates it by REML.
    kernel_check_max_n : no longer used: the two kernel-based escalation
        tests share the nonlinear tier's sample cap,
        SelectionConfig.nonlinear_max_n (docs/methods.md, Dependence between
        samples). Accepted, with a warning, so that older scripts still run.
    """
    alpha: float = 0.05
    effect_size_floor: float | None = None
    n_permutations: int | None = None
    monte_carlo_relative_error: float = 0.1
    ridge_lambda: float | None = None
    kernel_check_max_n: int | None = None

    def __post_init__(self):
        _warn_if_coarse(self.resolved_n_permutations, self.alpha, "AdequacyConfig")
        if self.kernel_check_max_n is not None:
            warnings.warn("AdequacyConfig.kernel_check_max_n is no longer used: the kernel-based escalation tests "
                          "share SelectionConfig.nonlinear_max_n.", DeprecationWarning, stacklevel=3)

    def resolved_effect_size_floor(self, selection: "SelectionConfig") -> float:
        """Escalation floor used: this config's value, else the meaningful-effect
        threshold, else the conventional 0.02."""
        if self.effect_size_floor is not None:
            return self.effect_size_floor
        if selection.meaningful_r2 is not None:
            return selection.meaningful_r2
        return DEFAULT_ADEQUACY_FLOOR

    def resolved_adjustment_floor(self, selection: "SelectionConfig") -> float:
        """Floor of the linear adjustment test, the one escalation test that
        concerns validity: the smaller of the escalation floor and the
        acceptance floor. With either regression's nonlinear share below the
        acceptance floor, a spurious contribution cannot reach that floor
        (the Cauchy-Schwarz inequality; docs/methods.md, Escalation tests),
        so this test's floor must not exceed it, whatever the escalation
        floor of the tests that concern power."""
        return min(self.resolved_effect_size_floor(selection), selection.effect_size_floor)

    @property
    def resolved_n_permutations(self) -> int:
        """Resamples used per adequacy test."""
        if self.n_permutations is not None:
            return self.n_permutations
        from model_selection.statistics.significance.base import resamples_for_relative_error

        return resamples_for_relative_error(self.alpha, self.monte_carlo_relative_error)
