"""
The linear -> nonlinear -> CMI cascade: select under the most restrictive
tier whose assumptions hold (see docs/methods.md, Selection).

Each tier runs a greedy selection trial. If a check calls for escalation at
any round, selection moves to the next tier. When the check concerns the
tier's validity (the nonlinear-dependence check), the trial is discarded and
selection restarts from an empty set under the next tier, because its
acceptances may be spurious. When it concerns power only (the
nonlinear-relation or effect-modification check, for a candidate the tier
did not accept), the trial's acceptances remain valid, and the next tier
continues from them. A trial that stops on the sample-size
requirement ends the cascade, with no selection in round 1 and the selection
so far otherwise, because later tiers need more samples, not fewer.

A tier whose validity assumption was rejected is never used for the
selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model_selection.selection.greedy import SelectionResult, greedy_forward_selection
from model_selection.selection.tiers import SelectionSample, Tier


@dataclass
class TierAttempt:
    """Audit record of one tier tried by the cascade."""
    tier: str
    committed: bool
    reason: str
    diagnostic_plot_paths: list[str] = field(default_factory=list)
    provisional_selection: list[str] = field(default_factory=list)
    """Formulations a rejected trial had accepted before its assumptions were
    rejected; for information only, never the result."""
    escalation_test: dict | None = None
    """The test that led from this tier to the next (SelectionResult.escalation_test)."""

    def to_record(self) -> dict:
        return {"tier": self.tier, "committed": self.committed, "reason": self.reason,
                "diagnostic_plot_paths": self.diagnostic_plot_paths,
                "provisional_selection": self.provisional_selection,
                "escalation_test": self.escalation_test}


@dataclass
class TieredSelectionResult:
    """The committed tier's selection plus the audit of every tier tried."""
    committed_tier: str | None
    selection: SelectionResult | None
    attempted_tiers: list[TierAttempt] = field(default_factory=list)

    @property
    def ill_posed(self) -> bool:
        return bool(self.selection and self.selection.ill_posed)


def run_tiered_selection(
        sample: SelectionSample,
        tiers: list[Tier],
        n_permutations: int = 500,
        alpha: float = 0.05,
        max_selected: int | None = None,
        rng: np.random.Generator | None = None,
        min_samples_per_dimension: float = 10.0,
) -> TieredSelectionResult:
    """Try each tier in order and commit to the first trial that is not
    tier_rejected. Raises RuntimeError if every tier rejects (the last tier
    must have no adequacy checks)."""
    rng = rng or np.random.default_rng()
    attempted: list[TierAttempt] = []
    carried: list = []
    unconditional: dict = {}
    unconditional_effect: dict = {}
    for i, tier in enumerate(tiers):
        result = greedy_forward_selection(tier.sample or sample, tier, n_permutations=n_permutations,
                                          alpha=alpha, max_selected=max_selected, rng=rng,
                                          min_samples_per_dimension=min_samples_per_dimension, carried_steps=carried)
        if carried and not result.unconditional_association:
            result.unconditional_association = unconditional  # round 1 was tested by the tier it was made in
        unconditional = result.unconditional_association
        if carried and not result.unconditional_effect:
            result.unconditional_effect = unconditional_effect
        unconditional_effect = result.unconditional_effect
        restarts = result.tier_rejected and not result.escalation_continues
        attempted.append(TierAttempt(
            tier=tier.name, committed=not result.tier_rejected,
            reason=result.stopped_reason,
            diagnostic_plot_paths=result.diagnostic_plot_paths,
            provisional_selection=[st.formulation for st in result.trajectory if st.accepted] if restarts else [],
            escalation_test=result.escalation_test,
        ))
        if not result.tier_rejected:
            return TieredSelectionResult(tier.name, result, attempted)
        carried = [st for st in result.trajectory if st.accepted] if result.escalation_continues else []
    raise RuntimeError(f"Every tier was rejected: {[a.tier for a in attempted]}. The last tier "
                       "must have no adequacy checks.")
