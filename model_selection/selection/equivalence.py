"""
The comparisons of each selected formulation with the other candidates, the
sets of formulations the data support equally, and the class of every
formulation after selection.

Greedy forward selection tests whether each round's winner is informative
about the observations given the formulations already selected, not whether
it is better than the other candidates. When two candidates carry the same
information, the winner can be decided by noise, and the other then adds
nothing in the next round only because the winner already carries what it
has. The selected set is then one of several equally valid sets, as with a
distribution that has several Markov boundaries (Statnikov et al., 2013).

For each accepted round, with Z the conditioning set before the winner W,
every other candidate C not already selected is compared with W in both
directions, with the test and acceptance gates of the tier that accepted W,
on that tier's sample, each at level alpha. After a power-only escalation a
selection combines steps of two tiers, and the tier that accepted W remains
valid for it:

    C adds beyond W:  is C informative about O given Z and W?
    W adds beyond C:  is W informative about O given Z and C?

W adding and C not is encompassing (Mizon and Richard, 1986): W explains
everything C does and more. C adding and W not means C encompasses W. Both
adding means each carries information of its own. Neither adding means the
two are not distinguished: two non-rejections, which are not evidence of no
contribution.

The comparisons are reported and do not change the selection. The
selected_encompassed review flag, the one outcome suggesting the selection
may have picked the wrong formulation, requires its rejection to hold at
alpha divided by the number of tests in the round (Bonferroni).

Equally supported sets. Replacing selected formulations by candidates not
distinguished from them gives alternative sets of the same size. Each is
verified as a whole, as alternative Markov boundaries are (Statnikov et al.,
2013): it is supported if the selected set adds nothing beyond it (the
selected formulations it lacks, jointly, are not informative given it) and
every member adds information given the others. A pairwise relation at one
round is judged given the formulations selected before it, so a combination
of replacements is only a candidate until it is verified.

Classes (formulation_classes). Distinct: a selected formulation that adds
information beyond every candidate compared with it, an intersection-union
finding at level alpha (Berger, 1982). Exchangeable: a selected formulation
with the candidates that replace it in a supported set. Redundant: not
selected and not exchangeable, when the last round tested every formulation
not selected, given the selected set, and accepted none. Not tested: not
selected, when selection ended before such a test was completed. A selected
formulation that is neither distinct nor exchangeable is not shown to be
distinct. Distinct rests on rejections; exchangeable and redundant rest on
non-rejections, whose chance of missing a real difference is not
controlled.

Single-candidate tests use sequential Monte Carlo p-values where the tier
provides them (Besag and Clifford, 1991): draws stop once the candidate can
no longer be significant at alpha, which gives the same decision as drawing
them all.

Limitations: the winner was chosen for its large statistic, which biases
the test of W beyond C towards finding that W adds (post-selection
inference; Berk et al., 2013); and the tier's adequacy checks were made for
the selection's conditioning sets, not for the sets with C in place of W.

See docs/methods.md, Interchangeable formulations.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np

from model_selection.parallel import parallel_map
from model_selection.selection.greedy import SelectionResult, _stack_selected
from model_selection.selection.tiers import SelectionSample, Tier
from model_selection.statistics.significance.base import permutation_p_value, stopping_exceedances
from model_selection.utils import as2d

logger = logging.getLogger(__name__)

def _relation(forward: dict, reverse: dict) -> str:
    if forward["adds"] and reverse["adds"]:
        return "distinct"
    if forward["adds"]:
        return "encompasses_selected"
    if reverse["adds"]:
        return "encompassed_by_selected"
    return "not_distinguished"


def _adds(tier: Tier, name: str, values, obs, z, n_permutations, rng, time_positions, lag, grid_step,
          alpha) -> dict:
    """Whether one candidate passes the tier's gates given z (its own null,
    no adjustment across candidates)."""
    untestable = False
    if hasattr(tier.test, "sequential_p_value"):
        observed, null, p_value = tier.test.sequential_p_value(
            values, obs, z, n_permutations, rng, time_positions, lag,
            stopping_exceedances(alpha, n_permutations), grid_step=grid_step)
    else:
        result = tier.test.run({name: values}, obs, z, n_permutations, rng, time_positions, lag,
                               grid_step=grid_step)
        observed, null = float(result.observed[0]), result.null[:, 0]
        p_value = permutation_p_value(observed, null)
        null = result.width_draws(0)  # for the effect size's chance level
        dep = result.dependence or {}
        if dep.get("independent_units"):
            # Clusters of events: untestable when no p-value below alpha is attainable
            # (significance.clusters); the comparison then never finds that a formulation adds information.
            smallest = dep.get("smallest_p_value")
            untestable = smallest is not None and smallest >= alpha
    effect = float("nan")
    if p_value < alpha:  # the effect matters only for a significant candidate
        effect, _ = tier.test.magnitude(values, obs, z, time_positions, lag, observed, null, grid_step=grid_step)
    return {"p_value": float(p_value), "effect": float(effect),
            "adds": bool(p_value < alpha and effect >= tier.effect_size_floor and not untestable),
            "untestable": bool(untestable)}


TierContexts = dict[str, tuple[Tier, SelectionSample]]
"""Each tier of a cascade, by name, with the sample its tests use."""


class _Context:
    """The test and sample of the tier that made an acceptance. After a
    power-only escalation a selection combines steps of two tiers; each is
    compared with the tier that accepted it, which remains valid for it
    (docs/methods.md, Interchangeable formulations)."""

    def __init__(self, tier: Tier, sample: SelectionSample):
        self.tier, self.sample = tier, sample
        self.candidates, self.obs = sample.candidates, sample.obs
        n = len(sample.obs)
        self.n = n
        self.time_positions = sample.time_positions if sample.time_positions is not None else np.arange(n)
        self.lag, self.grid_step = sample.dependence_lag, sample.grid_step

    def z(self, names: list[str]):
        return _stack_selected(self.sample.conditioning_series, names, self.sample.covariates)

    def adds(self, name, values, z, n_permutations, rng, alpha) -> dict:
        return _adds(self.tier, name, values, self.obs, z, n_permutations, rng, self.time_positions, self.lag,
                     self.grid_step, alpha)


def _contexts(sample: SelectionSample, tier: Tier, tiers: TierContexts | None):
    """(context of a step's tier, rank of a tier name in generality)."""
    default = _Context(tier, sample)
    built = {name: _Context(t, s) for name, (t, s) in (tiers or {}).items()}
    order = {name: i for i, name in enumerate(tiers or {})}
    return (lambda name: built.get(name, default)), (lambda name: order.get(name, -1))


def information_equivalence(
        selection: SelectionResult,
        sample: SelectionSample,
        tier: Tier,
        n_permutations: int,
        alpha: float,
        rng: np.random.Generator,
        tiers: TierContexts | None = None,
) -> list[dict]:
    """For every accepted round, the relation of its winner to each other
    candidate not selected before it (see the module docstring). Returns one
    record per accepted round: {"round", "selected", "comparisons": [{
    "candidate", "relation", "candidate_adds_beyond_selected": {p_value,
    effect, adds, untestable}, "selected_adds_beyond_candidate": {...}}],
    "flag_level"}. tiers: every tier of a cascade with its sample; each round
    is then compared with the tier that accepted it, otherwise with `tier`
    on `sample`."""
    context_of, _ = _contexts(sample, tier, tiers)
    records = []
    for i, step in enumerate(selection.trajectory):
        if not step.accepted:
            continue
        ctx = context_of(step.tier)
        candidates = ctx.candidates
        before = selection.selected[:i]
        winner = step.formulation
        others = [c for c in candidates if c != winner and c not in before]
        if not others:
            continue
        z = ctx.z(before)
        z_with_winner = ctx.z(before + [winner])
        # The round's comparisons are independent: they run in parallel, each with its own random stream,
        # split from the run's in a fixed order, so results do not depend on the number of workers (apart from rounding).
        def compare(job):
            c, stream = job
            forward = ctx.adds(c, candidates[c], z_with_winner, n_permutations, stream, alpha)
            c_conditioning = as2d(ctx.sample.conditioning_series[c])
            z_with_c = c_conditioning if z is None else np.hstack([z, c_conditioning])
            reverse = ctx.adds(winner, candidates[winner], z_with_c, n_permutations, stream, alpha)
            return {
                "candidate": c,
                "relation": _relation(forward, reverse),
                "candidate_adds_beyond_selected": forward,
                "selected_adds_beyond_candidate": reverse,
            }
        comparisons = parallel_map(compare, list(zip(others, rng.spawn(len(others)))))
        records.append({"round": i + 1, "selected": winner, "given": before,
                        "tier": ctx.tier.name, "alpha": alpha, "effect_size_floor": ctx.tier.effect_size_floor,
                        "flag_level": alpha / (2 * len(comparisons)), "comparisons": comparisons})
        summary = "; ".join(f"{cmp['candidate']}: {cmp['relation'].replace('_', ' ')}" for cmp in comparisons)
        logger.info(f"[i] Round {i + 1}, {winner} compared with the other candidates: {summary}.")
    return records


def _stack(candidates: dict, names: list[str]) -> np.ndarray:
    return np.hstack([as2d(candidates[nm]) for nm in names])


def equally_supported_sets(
        selection: SelectionResult,
        comparisons: list[dict],
        sample: SelectionSample,
        tier: Tier,
        n_permutations: int,
        alpha: float,
        rng: np.random.Generator,
        tiers: TierContexts | None = None,
) -> list[dict]:
    """The selected set and every set obtained by replacing selected
    formulations with candidates not distinguished from them, each verified
    as a whole (see the module docstring). Returns [{"set", "selected",
    "supported", "selected_adds_beyond_set": {p_value, effect, adds,
    untestable} or None, "members": {name: {...}}}], the selected set first.
    With tiers (see information_equivalence), each member is tested with the
    tier that accepted the formulation in its position, and the replaced
    formulations jointly with the most general of their tiers."""
    steps = [st for st in selection.trajectory if st.accepted]
    selected = [st.formulation for st in steps]
    if not selected:
        return []
    options = [[w] for w in selected]
    for rec in comparisons:
        i = selected.index(rec["selected"])
        for c in rec["comparisons"]:
            if c["relation"] == "not_distinguished":
                options[i].append(c["candidate"])
    context_of, rank = _contexts(sample, tier, tiers)
    seen, combos = set(), []
    for combo in itertools.product(*options):
        if len(set(combo)) < len(combo) or frozenset(combo) in seen:
            continue
        seen.add(frozenset(combo))
        combos.append(list(combo))

    # Each set is verified independently, in parallel, with its own random stream (see information_equivalence).
    def verify(job):
        members, stream = job
        missing_slots = [i for i, w in enumerate(selected) if w not in members]
        missing = [selected[i] for i in missing_slots]
        beyond = None
        if missing:
            general = max((steps[i].tier for i in missing_slots), key=rank)
            ctx = context_of(general)
            beyond = ctx.adds("+".join(missing), _stack(ctx.candidates, missing), ctx.z(members), n_permutations,
                              stream, alpha)
        member_tests = {}
        for slot, m in enumerate(members):
            ctx = context_of(steps[slot].tier)
            member_tests[m] = ctx.adds(m, ctx.candidates[m], ctx.z([o for o in members if o != m]), n_permutations,
                                       stream, alpha)
        supported = (beyond is None or not beyond["adds"]) and all(t["adds"] for t in member_tests.values())
        return {"set": members, "selected": not missing, "supported": bool(supported),
                "selected_adds_beyond_set": beyond, "members": member_tests}

    return parallel_map(verify, list(zip(combos, rng.spawn(len(combos))))) if combos else []


def _not_tested_reason(selection: SelectionResult) -> str | None:
    """Why selection ended before its last round tested every formulation not
    selected, given the selected set; None when that test was completed
    (docs/methods.md, Interchangeable formulations)."""
    if selection.untestable:
        return "the last round was untestable for too few clusters of events"
    if selection.ill_posed_reason:
        return f"the last round could not be tested ({selection.ill_posed_reason})"
    if selection.next_tier_unsupported is not None:
        return f"a needed escalation was unsupported ({selection.next_tier_unsupported})"
    last = selection.trajectory[-1] if selection.trajectory else None
    if last is None or last.accepted:
        return "selection ended without a round that accepted none"
    if last.outcome == "not_tested" or (last.cmi_calibration is not None and not last.cmi_calibration["calibrated"]):
        return "the free-form tier's acceptance was withheld (not calibrated on the sample)"
    return None


def formulation_classes(selection: SelectionResult, comparisons: list[dict], sets: list[dict],
                        formulations: list[str]) -> dict:
    """The class of every formulation (docs/methods.md, Interchangeable
    formulations, Classes):

    distinct: selected formulations that add information beyond every
        candidate compared with them;
    exchangeable: {selected formulation: candidates that replace it in a
        supported set};
    not_shown_distinct: {selected formulation: candidates it adds no
        information beyond}, for a selected formulation neither distinct nor
        exchangeable;
    redundant: formulations neither selected nor exchangeable, when the last
        round tested every formulation not selected and accepted none;
    not_tested: those formulations otherwise, with not_tested_reason.
    """
    selected = [st.formulation for st in selection.trajectory if st.accepted]
    exchangeable: dict[str, list[str]] = {}
    for s in sets:
        if s["supported"] and not s["selected"]:
            for w, c in zip(selected, s["set"]):
                if w != c and c not in exchangeable.setdefault(w, []):
                    exchangeable[w].append(c)
    exchangeable = {w: cs for w, cs in exchangeable.items() if cs}
    distinct, not_shown = [], {}
    by_selected = {rec["selected"]: rec["comparisons"] for rec in comparisons}
    for w in selected:
        if w in exchangeable:
            continue
        compared = by_selected.get(w, [])
        weaker = [c["candidate"] for c in compared if not c["selected_adds_beyond_candidate"]["adds"]]
        if weaker:
            not_shown[w] = weaker
        else:
            distinct.append(w)
    partners = {c for cs in exchangeable.values() for c in cs}
    others = [f for f in formulations if f not in selected and f not in partners]
    reason = _not_tested_reason(selection)
    return {"distinct": distinct, "exchangeable": exchangeable, "not_shown_distinct": not_shown,
            "redundant": [] if reason else others, "not_tested": others if reason else [],
            "not_tested_reason": reason if others else None}


def summarize_sets(selection: SelectionResult, comparisons: list[dict], sets: list[dict],
                   classes: dict | None = None) -> str:
    """A short statement of the outcome (docs/methods.md, Appendix B,
    Reporting of interchangeable formulations): the selected set; the class
    of each formulation, with each exchangeable formulation's candidates; the
    other sets the data support equally; and, for review, a candidate that
    encompasses a selected formulation."""
    selected = [st.formulation for st in selection.trajectory if st.accepted]
    if not selected:
        return ""
    fmt = lambda members: "{" + ", ".join(members) + "}"
    parts = [f"Selected {fmt(selected)}."]
    if classes is not None:
        if classes["exchangeable"]:
            parts.append("Exchangeable: " + "; ".join(
                f"{w} with {', '.join(cs)}" for w, cs in classes["exchangeable"].items()) + ".")
        by_selected = {rec["selected"]: rec["comparisons"] for rec in comparisons}
        for w in classes["distinct"]:
            compared = by_selected.get(w, [])
            if compared:
                p_max = max(c["selected_adds_beyond_candidate"]["p_value"] for c in compared)
                parts.append(f"Distinct: {w} adds information beyond each of "
                             f"{', '.join(c['candidate'] for c in compared)} (p <= {p_max:.2g}).")
            else:
                parts.append(f"Distinct: {w}.")
        for w, cs in classes["not_shown_distinct"].items():
            parts.append(f"Not shown to be distinct: {w} adds no information beyond {', '.join(cs)}.")
    for rec in comparisons:
        better = [c["candidate"] for c in rec["comparisons"] if c["relation"] == "encompasses_selected"]
        if better:
            parts.append(f"{', '.join(better)} adds information beyond {rec['selected']}, which adds none beyond it: "
                         "possibly the better choice.")
    chosen = next((s for s in sets if s["selected"]), None)
    if chosen is not None and not chosen["supported"]:
        weak = [m for m, t in chosen["members"].items() if not t["adds"]]
        parts.append(f"Given the other selected formulations, {', '.join(weak)} no longer adds detectably.")
    alternatives = [s for s in sets if s["supported"] and not s["selected"]]
    if alternatives:
        parts.append("Equally supported sets: " + ", ".join(fmt(s["set"]) for s in alternatives) + ".")
    if classes is not None:
        if classes["redundant"]:
            parts.append("Redundant: " + ", ".join(classes["redundant"]) + ".")
        if classes["not_tested"]:
            parts.append("Not tested: " + ", ".join(classes["not_tested"])
                         + f" ({classes['not_tested_reason']}).")
    return " ".join(parts)
