"""
The backward pass: after forward selection, remove the selected formulations
that no longer add information given the others (docs/methods.md, Backward
pass).

One pass over the selected formulations in the order they were selected, as
the shrinking phase of grow-shrink (Margaritis and Thrun, 2000) and IAMB
(Tsamardinos et al., 2003). Each formulation W is tested given all the other
formulations still selected, with the acceptance test and gates of a tier
(p < alpha and effect size at least the acceptance floor, against its own
null), and is removed at once if it is not accepted; later tests condition
on the smaller set. A selection of k formulations costs at most k tests.
When nothing was removed before it, the last formulation faces exactly its
forward test, whose result is reused.

Which tier. W is tested with the tier that accepted it. When that tier has a
test of its validity for the conditioning set (the linear tier's adjustment
test), it is run first for W's backward conditioning set, and a departure
moves W's test to the next, more general tier, as forward selection would.
The tests that concern power are not repeated: W was already found, and the
question is only whether it is still needed.

A test that cannot be made (too few clusters of events, or a departure with
no more general tier) keeps W, reported as not tested: removal rests on a
test's non-rejection.

Why one pass suffices. Formulations rejected in forward selection are never
re-tested. Forward selection stopped because no remaining formulation added
information given the forward selection; each removed formulation added none
given the formulations left. By the contraction property of conditional
independence, which holds for every distribution (Dawid, 1979; Pearl, 1988),
the rejected formulations then add none given the final set either.
Likewise, every formulation kept still adds information given the final
set's other members. That the final set is the smallest set carrying the
information (the Markov boundary) needs, in addition, faithfulness
(Margaritis and Thrun, 2000).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from model_selection.selection.equivalence import TierContexts, _contexts
from model_selection.selection.greedy import SelectionResult
from model_selection import timing
from model_selection.selection.tiers import SelectionSample, Tier

logger = logging.getLogger(__name__)


def _validity_departure(ctx, name: str, z, rng: np.random.Generator) -> list[dict]:
    """The tier's validity tests (adequacy checks of the "removal" stage, the
    linear adjustment test) for candidate `name` given z, as forward selection
    runs them before its acceptance test. Returns their records."""
    done = []
    if z is None:
        return done
    partialled = ctx.sample.partialled_covariates
    for check in ctx.tier.adequacy_checks:
        if check.stage != "removal":
            continue
        extra = {} if partialled is None else {"interaction_with": partialled}
        outcome = check.test(ctx.candidates[name], ctx.obs, z, n_permutations=ctx.tier.adequacy_n_permutations,
                             rng=rng, time_positions=ctx.time_positions, min_temporal_lag=ctx.lag,
                             grid_step=ctx.grid_step, **extra)
        done.append({"tier": ctx.tier.name, "check": check.name, "adequate": bool(outcome.tier_adequate),
                     "p_value": outcome.p_value, "decision_reason": outcome.decision_reason})
    return done


def backward_pass(
        selection: SelectionResult,
        sample: SelectionSample,
        tier: Tier,
        n_permutations: int,
        alpha: float,
        rng: np.random.Generator,
        tiers: TierContexts | None = None,
) -> None:
    """Run the backward pass on `selection` in place (module docstring):
    selection.forward_selected keeps the forward selection, selection.selected
    becomes the final set, and selection.backward holds one record per
    formulation: {"formulation", "given", "tier", "validity", "p_value",
    "effect", "adds", "untestable", "reused_forward", "removed",
    "not_tested_reason"}. tiers: every tier of a cascade with its sample, in
    order of generality; otherwise every test uses `tier` on `sample`."""
    steps = [st for st in selection.trajectory if st.accepted]
    members = [st.formulation for st in steps]
    selection.forward_selected = list(members)
    selection.backward = []
    if not members:
        return
    context_of, _ = _contexts(sample, tier, tiers)
    order = list(tiers or {})
    current = list(members)
    streams = rng.spawn(2 * len(members))
    for i, step in enumerate(steps):
        w = step.formulation
        others = [m for m in current if m != w]
        record = {"formulation": w, "given": others, "tier": step.tier, "validity": [], "p_value": None,
                  "effect": None, "adds": True, "untestable": False, "reused_forward": False, "removed": False,
                  "not_tested_reason": None}
        if current == members and i == len(members) - 1:
            # Nothing was removed before it: the last formulation faces exactly its forward test.
            record.update(p_value=float(step.p_value), effect=float(step.effect), reused_forward=True)
            selection.backward.append(record)
            continue
        name = step.tier
        ctx = context_of(name)
        z = ctx.z(others)
        t_test = time.perf_counter()
        while True:
            checks = _validity_departure(ctx, w, z, streams[2 * i])
            record["validity"].extend(checks)
            if all(c["adequate"] for c in checks):
                break
            nxt = order.index(ctx.tier.name) + 1 if ctx.tier.name in order else len(order)
            if nxt >= len(order):
                record["not_tested_reason"] = (f"the {ctx.tier.name} tier's validity test departs for this "
                                               "conditioning set, and no more general tier is available")
                ctx = None
                break
            ctx = context_of(order[nxt])
            z = ctx.z(others)
        if ctx is not None:
            record["tier"] = ctx.tier.name
            result = ctx.adds(w, ctx.candidates[w], z, n_permutations, streams[2 * i + 1], alpha)
            record.update(p_value=result["p_value"], effect=result["effect"], adds=result["adds"],
                          untestable=result["untestable"])
            if result["untestable"]:
                record["not_tested_reason"] = "too few clusters of events for a p-value below alpha"
        timing.elapsed_since(f"backward pass: test of '{w}' ({record['tier']} tier)", t_test)
        if record["not_tested_reason"] is None and not record["adds"]:
            record["removed"] = True
            current.remove(w)
            logger.info(f"[i] Backward pass: {w} adds no information given {{{', '.join(others)}}} "
                        f"(p={record['p_value']:.3g}); removed.")
        elif record["not_tested_reason"] is not None:
            record["adds"] = None
            logger.info(f"[i] Backward pass: {w} not tested ({record['not_tested_reason']}); kept.")
        selection.backward.append(record)
    selection.selected = current
    removed = [r["formulation"] for r in selection.backward if r["removed"]]
    logger.info(f"[+] Backward pass: final set {{{', '.join(current)}}}"
                + (f"; removed {', '.join(removed)}." if removed else "; nothing removed."))
