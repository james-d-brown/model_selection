"""The selection stage shared by the pooled and high-flow pathways: run one
tier or the cascade on a prepared sample."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from model_selection.selection.cascade import run_tiered_selection
from model_selection.selection.config import CASCADE, AdequacyConfig, SelectionConfig
from model_selection.selection.equivalence import equally_supported_sets, information_equivalence, summarize_sets
from model_selection.selection.greedy import SelectionResult, greedy_forward_selection
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
) -> SelectionOutcome:
    """Select formulations from `sample` according to selection.measure.

    multivariate : samples are vectors (functional scores) rather than
        scalars; chooses the linear tier's residual-linearity test.
    tier_samples : per-tier replacement samples (subsampled long records).
    label : included in log messages.
    """
    if selection.measure == CASCADE:
        tiers = build_cascade(selection, adequacy, multivariate, tier_samples, plot_factory)
        logger.info(f"[*] Tiered (linear -> nonlinear -> CMI) forward selection ({label})")
        tiered = run_tiered_selection(sample, tiers, selection.resolved_n_permutations, selection.alpha, rng=rng,
                                      min_samples_per_dimension=selection.min_samples_per_dimension)
        logger.info(f"[+] Completed tiered selection ({label}): committed tier {tiered.committed_tier}")
        committed = next(t for t in tiers if t.name == tiered.committed_tier)
        _compare_alternatives(tiered.selection, committed.sample or sample, committed, selection, rng,
                              {t.name: (t, t.sample or sample) for t in tiers})
        return SelectionOutcome(tiered.selection, committed, tiered.committed_tier,
                                [a.to_record() for a in tiered.attempted_tiers])

    tier = build_tier(selection.measure, selection, plot_factory=plot_factory)
    tier.sample = (tier_samples or {}).get(tier.name)
    logger.info(f"[*] Greedy forward selection ({label}, tier {tier.name})")
    result = greedy_forward_selection(tier.sample or sample, tier, selection.resolved_n_permutations, selection.alpha,
                                      rng=rng, min_samples_per_dimension=selection.min_samples_per_dimension)
    _compare_alternatives(result, tier.sample or sample, tier, selection, rng)
    return SelectionOutcome(result, tier)


def equivalence_margin(selection: SelectionConfig) -> float:
    """The contribution an equivalence must be able to detect, on the R^2
    scale of the detectable contributions it is compared with: the
    meaningful-effect threshold when set, otherwise the acceptance floor
    (docs/methods.md, Interchangeable formulations). The R^2-scale floor is
    used in every tier; a tier's own floor can be in other units (nats for
    the free-form tier)."""
    return selection.meaningful_r2 if selection.meaningful_r2 is not None else selection.effect_size_floor


def _compare_alternatives(result: SelectionResult, sample: SelectionSample, tier: Tier, selection: SelectionConfig,
                          rng: np.random.Generator, tiers: dict | None = None) -> None:
    """Information equivalence of each selected formulation with the other
    candidates (selection.equivalence), from an independent random stream so
    that later steps' draws are unchanged. In cascade mode, tiers holds every
    tier with its sample, and each formulation is compared with the tier that
    accepted it."""
    if selection.equivalence_checks and any(st.accepted for st in result.trajectory):
        stream = rng.spawn(1)[0]
        margin = equivalence_margin(selection)
        result.equivalence = information_equivalence(result, sample, tier, selection.resolved_n_permutations,
                                                     selection.alpha, stream, margin, tiers)
        result.equivalent_sets = equally_supported_sets(result, result.equivalence, sample, tier,
                                                        selection.resolved_n_permutations, selection.alpha, stream,
                                                        margin, tiers)
        result.equivalent_sets_summary = summarize_sets(result, result.equivalence, result.equivalent_sets, margin,
                                                        selection.alpha)
        logger.info(f"[i] {result.equivalent_sets_summary}")
