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
selection. When the tier the escalation leads to is not calibrated on its
own sample (the CMI tier's guard, at the round that leads there), it is not
run: after a power-only escalation the current tier's selection stands,
possibly incomplete; after a validity escalation no selection is reported.
A tier escalated to that runs but could detect no contribution, however
large, in its first round (smallest detectable R^2 of 1) is reported as
such.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from model_selection.selection.greedy import SelectionResult, escalated_from_text, greedy_forward_selection
from model_selection.selection.tiers import SelectionSample, Tier


@dataclass
class TierAttempt:
    """Audit record of one tier tried by the cascade."""
    tier: str
    committed: bool
    reason: str
    diagnostic_plot_paths: list[str] = field(default_factory=list)
    run: bool = True
    """False for a tier committed to without running it, because it cannot be
    used on its sample (module docstring)."""
    provisional_selection: list[str] = field(default_factory=list)
    """Formulations a rejected trial had accepted before its assumptions were
    rejected; for information only, never the result."""
    escalation_test: dict | None = None
    """The test that led from this tier to the next (SelectionResult.escalation_test)."""

    def to_record(self) -> dict:
        return {"tier": self.tier, "committed": self.committed, "reason": self.reason,
                "diagnostic_plot_paths": self.diagnostic_plot_paths, "run": self.run,
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
    escalated: dict | None = None
    for i, tier in enumerate(tiers):
        result = greedy_forward_selection(tier.sample or sample, tier, n_permutations=n_permutations,
                                          alpha=alpha, max_selected=max_selected, rng=rng,
                                          min_samples_per_dimension=min_samples_per_dimension, carried_steps=carried)
        if carried and not result.unconditional_association:
            result.unconditional_association = unconditional  # round 1 was tested by the tier it was made in
        unconditional = result.unconditional_association
        restarts = result.tier_rejected and not result.escalation_continues
        attempted.append(TierAttempt(
            tier=tier.name, committed=not result.tier_rejected,
            reason=result.stopped_reason,
            diagnostic_plot_paths=result.diagnostic_plot_paths,
            provisional_selection=[st.formulation for st in result.trajectory if st.accepted] if restarts else [],
            escalation_test=result.escalation_test,
        ))
        if escalated is not None and len(result.trajectory) == escalated["round"]:
            result.escalated_from = escalated
            extra = escalated_from_text(result)
            if extra:
                result.stopped_reason = f"{result.stopped_reason}. {extra}"
        if not result.tier_rejected:
            return TieredSelectionResult(tier.name, result, attempted)
        if result.next_tier_unsupported is not None and i + 1 < len(tiers):
            nxt = tiers[i + 1].name
            if result.escalation_continues:
                # A power-only departure: this tier's acceptances remain valid, so its selection stands.
                reason = (f"not run: a departure in the {tier.name} tier concerns power only, and the {nxt} tier "
                          f"cannot be used: {result.next_tier_unsupported}; the {tier.name} tier's selection stands, "
                          "possibly incomplete")
                attempted[-1].committed = True
                attempted.append(TierAttempt(tier=nxt, committed=False, reason=reason, run=False))
                return TieredSelectionResult(tier.name, replace(result, tier_rejected=False,
                                                                stopped_reason=f"{result.stopped_reason}; {reason}"),
                                             attempted)
            # The tier's validity was rejected, and the next tier cannot be used: no tier whose assumptions hold
            # can make the selection.
            reason = (f"not run: the {tier.name} tier's assumptions were rejected, and the {nxt} tier cannot be "
                      f"used: {result.next_tier_unsupported}; no selection can be supported")
            attempted.append(TierAttempt(tier=nxt, committed=True, reason=reason, run=False))
            return TieredSelectionResult(nxt, SelectionResult(stopped_reason=reason), attempted)
        carried = [st for st in result.trajectory if st.accepted] if result.escalation_continues else []
        test = attempted[-1].escalation_test
        escalated = ({"tier": tier.name, "round": test["round"], "detectable_r2": test["detectable_r2"]}
                     if result.escalation_continues and test is not None and test["detectable_r2"] is not None
                     else None)
    raise RuntimeError(f"Every tier was rejected: {[a.tier for a in attempted]}. The last tier "
                       "must have no adequacy checks.")
