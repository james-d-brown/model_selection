"""The selection stage shared by the pooled and high-flow event pathways: run one
tier or the cascade on a prepared sample."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from model_selection.selection.backward import backward_pass
from model_selection.selection.cascade import run_tiered_selection
from model_selection.selection.config import CASCADE, AdequacyConfig, SelectionConfig
from model_selection.selection.equivalence import (
    equally_supported_sets, formulation_classes, information_equivalence, summarize_sets,
)
from model_selection.selection.greedy import SelectionResult, greedy_forward_selection, step_outcome
from model_selection import timing
from model_selection.selection.tiers import AdequacyPlotFactory, SelectionSample, Tier, build_cascade, build_tier

logger = logging.getLogger(__name__)


@dataclass
class SelectionOutcome:
    """Result of the selection stage. tier is the tier that produced
    `selection` (the committed one in cascade mode), for follow-up tests that
    must use the same test; committed_tier and attempted_tiers are set only in
    cascade mode."""
    selection: SelectionResult
    tier: Tier
    committed_tier: str | None = None
    attempted_tiers: list[dict] = field(default_factory=list)


def run_selection_stage(
        sample: SelectionSample,
        selection: SelectionConfig,
        adequacy: AdequacyConfig,
        multivariate: bool,
        rng: np.random.Generator,
        tier_samples: dict[str, SelectionSample] | None = None,
        plot_factory: AdequacyPlotFactory | None = None,
        label: str = "",
        events: bool = False,
) -> SelectionOutcome:
    """Select formulations from `sample` according to selection.measure.

    multivariate : samples are vectors (functional scores) rather than
        scalars; chooses the linear tier's residual-linearity test.
    tier_samples : per-tier replacement samples (subsampled long records).
    label : included in log messages.
    events : the samples are high-flow events (selection.tiers.build_tier).
    """
    from model_selection.parallel import set_workers
    from model_selection.statistics.regression.kernel_residuals import set_low_rank
    set_workers(selection.n_jobs)
    set_low_rank(selection.nonlinear_rank)
    if selection.measure == CASCADE:
        tiers = build_cascade(selection, adequacy, multivariate, tier_samples, plot_factory, events=events)
        logger.info(f"[*] Tiered (linear -> nonlinear -> CMI) forward selection ({label})")
        tiered = run_tiered_selection(sample, tiers, selection.resolved_n_permutations, selection.alpha, rng=rng,
                                      min_samples_per_dimension=selection.min_samples_per_dimension)
        logger.info(f"[+] Completed tiered selection ({label}): committed tier {tiered.committed_tier}")
        committed = next(t for t in tiers if t.name == tiered.committed_tier)
        _assign_outcomes(tiered.selection, selection)
        contexts = {t.name: (t, t.sample or sample) for t in tiers}
        _shrink(tiered.selection, committed.sample or sample, committed, selection, rng, contexts)
        _compare_alternatives(tiered.selection, committed.sample or sample, committed, selection, rng, contexts)
        _attach_timings(tiered.selection)
        return SelectionOutcome(tiered.selection, committed, tiered.committed_tier,
                                [a.to_record() for a in tiered.attempted_tiers])

    tier = build_tier(selection.measure, selection, plot_factory=plot_factory, events=events)
    tier.sample = (tier_samples or {}).get(tier.name)
    logger.info(f"[*] Greedy forward selection ({label}, tier {tier.name})")
    result = greedy_forward_selection(tier.sample or sample, tier, selection.resolved_n_permutations, selection.alpha,
                                      rng=rng, min_samples_per_dimension=selection.min_samples_per_dimension)
    _assign_outcomes(result, selection)
    _shrink(result, tier.sample or sample, tier, selection, rng)
    _compare_alternatives(result, tier.sample or sample, tier, selection, rng)
    _attach_timings(result)
    return SelectionOutcome(result, tier)


def _assign_outcomes(result: SelectionResult | None, selection: SelectionConfig) -> None:
    """Each step's outcome, from its test's decision alone (greedy.step_outcome)."""
    if result is None:
        return
    for step in result.trajectory:
        step.outcome = step_outcome(step, selection.alpha)


def _attach_timings(result: SelectionResult | None) -> None:
    """The wall-clock time of each stage so far (model_selection.timing)."""
    if result is not None:
        result.timings = timing.records()


def _shrink(result: SelectionResult | None, sample: SelectionSample, tier: Tier, selection: SelectionConfig,
            rng: np.random.Generator, tiers: dict | None = None) -> None:
    """The backward pass (selection.backward), from an independent random
    stream so that later steps' draws are unchanged. Without it, the final
    set is the forward selection."""
    if result is None:
        return
    if selection.backward_pass and not result.tier_rejected and not result.ill_posed:
        with timing.timed("backward pass"):
            backward_pass(result, sample, tier, selection.resolved_n_permutations, selection.alpha,
                          rng.spawn(1)[0], tiers)
    else:
        result.forward_selected = [st.formulation for st in result.trajectory if st.accepted]


def _compare_alternatives(result: SelectionResult, sample: SelectionSample, tier: Tier, selection: SelectionConfig,
                          rng: np.random.Generator, tiers: dict | None = None) -> None:
    """The comparisons of each selected formulation with the other candidates,
    the equally supported sets and the class of every formulation
    (selection.equivalence), from an independent random stream so that later
    steps' draws are unchanged. In cascade mode, tiers holds every tier with
    its sample, and each formulation is compared with the tier that accepted
    it."""
    if selection.equivalence_checks and result.selected:
        stream = rng.spawn(1)[0]
        with timing.timed("comparisons with the formulations not selected"):
            result.equivalence = information_equivalence(result, sample, tier, selection.resolved_n_permutations,
                                                         selection.alpha, stream, tiers)
        with timing.timed("equally supported sets"):
            result.equivalent_sets = equally_supported_sets(result, result.equivalence, sample, tier,
                                                            selection.resolved_n_permutations, selection.alpha,
                                                            stream, tiers)
        result.classes = formulation_classes(result, result.equivalence, result.equivalent_sets,
                                             list(sample.candidates))
        result.equivalent_sets_summary = summarize_sets(result, result.equivalence, result.equivalent_sets,
                                                        result.classes)
        logger.info(f"[i] {result.equivalent_sets_summary}")
