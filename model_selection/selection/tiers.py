"""
The three tiers of the selection cascade, each pairing a conditional
independence test with the escalation tests of its assumptions and its
effect-size floor (see docs/methods.md, The three tiers).

  linear    : linear-regression residual test; the linear adjustment test
              (can the selected formulations and covariates be removed
              linearly? a kernel score test of each regression), the
              candidate linearity test (RESET for single series, a kernel
              score test for several features) and the candidate
              effect-modification test (below).
  nonlinear : nonlinear-regression residual test; the candidate
              effect-modification test (does the candidate's relationship
              with the observations vary across the states of the
              conditioning set? a kernel interaction test).
  cmi       : the free-form tier, conditional mutual information; no
              escalation test, and a calibration test before it is used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from model_selection.selection.config import TIERS, AdequacyConfig, SelectionConfig
from model_selection.statistics.adequacy.base import AdequacyPlotFunc, AdequacyTestFunc
from model_selection.statistics.adequacy.linearity import (
    make_linearity_adequacy_test, make_multivariate_linearity_adequacy_test,
)
from model_selection.statistics.adequacy.specification import linear_specification_test
from model_selection.statistics.conditional_independence.calibration import cmi_calibration
from model_selection.statistics.adequacy.state_independence import make_state_independence_adequacy_test
from model_selection.statistics.association.base import StructuralCheckFunc
from model_selection.statistics.association.canonical import partial_cca_structural_adequacy
from model_selection.statistics.conditional_independence import (
    CMITest, ConditionalIndependenceTest, LinearResidualTest, NonlinearResidualTest,
)
from model_selection.utils import as2d

AdequacyPlotFactory = Callable[[str, str], "Callable | None"]
"""(plot_kind, tier_name) -> plot function or None (see
visualization.adequacy.make_adequacy_plot_func for the kinds)."""


@dataclass
class AdequacyCheck:
    """One adequacy test of a tier, with its optional diagnostic plot.

    stage "removal": checks how the selected formulations were removed; it runs
        before the significance test and a rejection escalates the tier.
    stage "relationship": checks the form of the relationship between the
        candidate and the observations; it runs after the significance test,
        and a rejection escalates the tier only if the candidate was not
        accepted (docs/methods.md, Selection).
    """
    name: str
    test: AdequacyTestFunc
    plot: AdequacyPlotFunc | None = None
    needs_conditioning: bool = False
    stage: str = "removal"
    labels_columns: bool = False  # test accepts column_labels for its decision reason


@dataclass
class SelectionSample:
    """The data one selection trial runs on.

    candidates : {formulation: (n,) or (n, d)}; obs : matching array;
    time_positions : original-time index of each row; dependence_lag : in
    the same time units; feature_names : names of the columns of each
    vector-valued series (e.g. PC1, PC2), used in reports;
    grid_step : step of the regular time grid the rows lie on, with missing
    grid points allowed (timestep samples); None for event samples;
    covariates : (n, m) observation-free variables that every test conditions
        on in addition to the selected formulations, from round 1 (the
        parameters of the event samples' shared registration); None for none.
    covariate_label : the covariates in words, for diagnostic pages.
    """
    candidates: dict[str, np.ndarray]
    obs: np.ndarray
    time_positions: np.ndarray
    dependence_lag: int
    feature_names: tuple[str, ...] | None = None
    grid_step: float | None = None
    covariates: np.ndarray | None = None
    covariate_label: str | None = None


@dataclass
class Tier:
    """A tier: its test, adequacy checks, effect-size floor on the test's own
    scale, optional structural check, and optionally its own sample (used to
    subsample long pooled records for the nonlinear tier)."""
    name: str
    test: ConditionalIndependenceTest
    effect_size_floor: float
    adequacy_checks: list[AdequacyCheck] = field(default_factory=list)
    adequacy_n_permutations: int = 200
    adequacy_alpha: float = 0.05
    adequacy_floor: float = 0.02
    calibration: Callable | None = None
    escalation_guard: Callable | None = None
    structural_check: StructuralCheckFunc | None = None
    sample: SelectionSample | None = None
    diagnostic_plots: list[AdequacyPlotFunc] = field(default_factory=list)


def effect_size_floor_for_tier(effect_size_floor: float, tier: str) -> float:
    """Convert the R^2-like floor to a tier's scale. The linear tier's effect
    (one minus Wilks' lambda) and the nonlinear tier's cross-fitted magnitude
    are on the floor's scale; for CMI the floor becomes nats via
    I = -0.5 ln(1 - R^2), exact for Gaussian variables (Linfoot, 1957;
    docs/methods.md, p-values and effect sizes)."""
    if tier in ("linear", "nonlinear"):
        return effect_size_floor
    if tier == "cmi":
        return float(-0.5 * np.log(1.0 - effect_size_floor))
    raise ValueError(f"tier must be one of {TIERS} -- got {tier!r}")


def _effect_modification_check(a: AdequacyConfig, floor: float, run_length: int, plot) -> AdequacyCheck:
    """The candidate effect-modification test (linear and nonlinear tiers):
    does the candidate's relationship with the observations vary with the
    selected formulations' state? Both tiers' statistics average over states,
    so an interaction whose direction changes across states can cancel them."""
    return AdequacyCheck(
        "effect modification by the selected formulations",
        make_state_independence_adequacy_test(alpha=a.alpha, effect_size_floor=floor, ridge_lambda=a.ridge_lambda,
                                              max_n=a.kernel_check_max_n, run_length=run_length),
        plot, needs_conditioning=True, stage="relationship")


def _specification_check(a: AdequacyConfig, floor: float, run_length: int) -> AdequacyTestFunc:
    def _check(candidate, obs, conditioning, n_permutations=100, rng=None, time_positions=None,
               min_temporal_lag=0, column_labels=None, grid_step=None):
        columns = np.column_stack([as2d(candidate), as2d(obs)])
        return linear_specification_test(
            columns, conditioning, a.alpha, floor, n_permutations,
            rng or np.random.default_rng(), time_positions if time_positions is not None else np.arange(len(columns)),
            min_temporal_lag, column_labels=column_labels, run_length=run_length, grid_step=grid_step,
            n_candidate_columns=as2d(candidate).shape[1], max_n=a.kernel_check_max_n,
        )
    return _check


def build_tier(
        name: str,
        selection: SelectionConfig,
        adequacy: AdequacyConfig | None = None,
        multivariate: bool = False,
        plot_factory: AdequacyPlotFactory | None = None,
        workers: int = -1,
) -> Tier:
    """The tier `name`. Adequacy checks are attached only when `adequacy` is
    given (cascade mode); diagnostic plots, when plot_factory is given, are
    attached in every mode (docs/methods.md, Diagnostics)."""
    floor = effect_size_floor_for_tier(selection.effect_size_floor, name)
    a = adequacy
    plot = (lambda kind: plot_factory(kind, name)) if plot_factory else (lambda kind: None)
    adequacy_floor = a.resolved_effect_size_floor(selection) if a is not None else None
    run = selection.dependence_run_length
    if name == "linear":
        tier = Tier("linear", LinearResidualTest(run_length=run), floor,
                    structural_check=partial_cca_structural_adequacy)
        if a is not None:
            if multivariate:
                residual = make_multivariate_linearity_adequacy_test(
                    alpha=a.alpha, effect_size_floor=adequacy_floor, ridge_lambda=a.ridge_lambda, run_length=run)
                kind = "multivariate_linearity"
            else:
                residual = make_linearity_adequacy_test(alpha=a.alpha, effect_size_floor=adequacy_floor,
                                                        run_length=run)
                kind = "linearity"
            tier.adequacy_checks = [
                AdequacyCheck("nonlinear dependence on the selected formulations", _specification_check(a, adequacy_floor, run),
                              plot("linear_specification"), needs_conditioning=True, stage="removal",
                              labels_columns=True),
                AdequacyCheck("nonlinear candidate-observation relation", residual, plot(kind),
                              stage="relationship"),
                _effect_modification_check(a, adequacy_floor, run, plot("state_independence")),
            ]
        else:
            tier.diagnostic_plots = [p for p in (plot("linear_specification"),
                                                 plot("multivariate_linearity" if multivariate else "linearity"),
                                                 plot("state_independence"))
                                     if p is not None]
    elif name == "nonlinear":
        tier = Tier("nonlinear", NonlinearResidualTest(run_length=run), floor)
        if a is not None:
            tier.adequacy_checks = [_effect_modification_check(a, adequacy_floor, run, plot("state_independence"))]
        else:
            tier.diagnostic_plots = [p for p in (plot("state_independence"),) if p is not None]
    elif name == "cmi":
        tier = Tier("cmi", CMITest(k=selection.cmi_k, k_fraction=selection.cmi_k_fraction,
                                   n_neighbors=selection.n_neighbors, workers=workers,
                                   run_length=run, theiler_window=selection.theiler_window), floor)
        if selection.cmi_calibration:
            tier.calibration = _cmi_calibration(tier, selection)
        if a is None:
            # without the cascade there is no rejected nonlinear trial to show
            # whether effect modification is present, so show it here
            tier.diagnostic_plots = [p for p in (plot("state_independence"),) if p is not None]
    else:
        raise ValueError(f"tier must be one of {TIERS} -- got {name!r}")
    if a is not None:
        tier.adequacy_n_permutations = a.resolved_n_permutations
        tier.adequacy_alpha = a.alpha
        tier.adequacy_floor = adequacy_floor
    return tier


def _cmi_calibration(tier: Tier, selection: SelectionConfig) -> Callable:
    """The CMI tier's calibration on surrogate candidates for one state
    (statistics.conditional_independence.calibration)."""
    def _calibrate(candidate, obs, conditioning, time_positions, dependence_lag, rng, grid_step=None):
        return cmi_calibration(tier.test, candidate, obs, conditioning, time_positions, dependence_lag,
                               selection.alpha, tier.effect_size_floor, rng,
                               rate_ratio=selection.cmi_calibration_rate_ratio,
                               grid_step=grid_step)
    return _calibrate


def build_cascade(
        selection: SelectionConfig,
        adequacy: AdequacyConfig,
        multivariate: bool,
        tier_samples: dict[str, SelectionSample] | None = None,
        plot_factory: AdequacyPlotFactory | None = None,
        workers: int = -1,
) -> list[Tier]:
    """The linear, nonlinear and CMI tiers in cascade order. tier_samples maps
    a tier name to a sample replacing the trial sample for that tier (used to
    subsample long pooled records for the costlier tiers)."""
    tiers = [build_tier(name, selection, adequacy, multivariate, plot_factory, workers) for name in TIERS]
    for tier in tiers:
        tier.sample = (tier_samples or {}).get(tier.name)
    tiers[2].adequacy_checks = []
    # Escalation to the CMI tier only where that tier is calibrated on the sample.
    tiers[1].escalation_guard = _guard_on_own_sample(tiers[2])
    return tiers


def _guard_on_own_sample(cmi: Tier) -> Callable | None:
    """The CMI tier's calibration, judged on the sample that tier would use.
    When the CMI tier has its own sample (a long pooled record), the
    candidate and the conditioning set of the round that leads there are
    rebuilt from it by name, so the check is not made on the nonlinear tier's
    sample, which is spaced differently."""
    if cmi.calibration is None:
        return None
    if cmi.sample is None:
        return lambda *args, names=None, **kwargs: cmi.calibration(*args, **kwargs)

    def _guard(candidate, obs, conditioning, time_positions, dependence_lag, rng, grid_step=None, names=None):
        own = cmi.sample
        name, selected = names
        parts = ([] if own.covariates is None else [as2d(own.covariates)]) + [as2d(own.candidates[s]) for s in selected]
        z = np.hstack(parts) if parts else None
        return cmi.calibration(own.candidates[name], own.obs, z, own.time_positions, own.dependence_lag, rng,
                               grid_step=own.grid_step)
    return _guard
