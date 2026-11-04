"""
Information equivalence of each selected formulation with the other
candidates, and the sets of formulations the data cannot distinguish from
the selected set.

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
adding means each carries information of its own. Neither adding is two
non-rejections, which are not evidence of no contribution. The pair is
equivalent (information equivalence; Statnikov et al., 2013, Definition 8)
only when both comparisons could have detected a contribution as large as
the margin (the meaningful-effect threshold when set, otherwise the
effect-size floor) with 80% power, the power fixed by the design (number of
samples, effective for dependence, and dimensions) rather than estimated
from the observed effect; otherwise the pair is not distinguishable with
these data. The effective sample size is the step's own.

The comparisons are reported and do not change the selection. The
selected_encompassed review flag, the one outcome suggesting the selection
may have picked the wrong formulation, requires its rejection to hold at
alpha divided by the number of tests in the round (Bonferroni).

Equally supported sets. Replacing selected formulations by candidates that
are equivalent or not distinguishable gives alternative sets of the same
size. Each is verified as a whole, as alternative Markov boundaries are
(Statnikov et al., 2013): it is supported if the selected set adds nothing
beyond it (the selected formulations it lacks, jointly, are not informative
given it) and every member adds information given the others. Equivalence
at one round is judged given the formulations selected before it, so a
combination of replacements is only a candidate until it is verified. A
supported set is equivalent to the selected set when all its replacements
are equivalent and the joint test had the power to detect the margin.

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
from model_selection.selection.greedy import SelectionResult, _stack_selected, escalated_from_text
from model_selection.selection.tiers import SelectionSample, Tier
from model_selection.statistics.power import format_share, minimum_detectable_r2
from model_selection.statistics.significance.base import permutation_p_value, stopping_exceedances
from model_selection.utils import as2d

logger = logging.getLogger(__name__)

def _detectable(n_eff: float, obs, tested, conditioning, alpha: float) -> float:
    """Smallest contribution detectable with 80% power for one comparison."""
    k_c = 0 if conditioning is None else as2d(conditioning).shape[1]
    return minimum_detectable_r2(n_eff, as2d(obs).shape[1], as2d(tested).shape[1], k_c, alpha)


def _relation(forward: dict, reverse: dict, margin: float) -> str:
    if forward["adds"] and reverse["adds"]:
        return "distinct"
    if forward["adds"]:
        return "encompasses_selected"
    if reverse["adds"]:
        return "encompassed_by_selected"
    powered = forward["detectable_r2"] <= margin and reverse["detectable_r2"] <= margin
    return "equivalent" if powered else "not_distinguishable"


def _adds(tier: Tier, name: str, values, obs, z, n_permutations, rng, time_positions, lag, grid_step,
          alpha, n_eff) -> dict:
    """Whether one candidate passes the tier's gates given z (its own null,
    no adjustment across candidates), and the contribution it could detect."""
    if hasattr(tier.test, "sequential_p_value"):
        observed, null, p_value = tier.test.sequential_p_value(
            values, obs, z, n_permutations, rng, time_positions, lag,
            stopping_exceedances(alpha, n_permutations), grid_step=grid_step)
    else:
        result = tier.test.run({name: values}, obs, z, n_permutations, rng, time_positions, lag,
                               grid_step=grid_step)
        observed, null = float(result.observed[0]), result.null[:, 0]
        p_value = permutation_p_value(observed, null)
    effect = float("nan")
    if p_value < alpha:  # the effect matters only for a significant candidate
        effect, _ = tier.test.magnitude(values, obs, z, time_positions, lag, observed, null, grid_step=grid_step)
    return {"p_value": float(p_value), "effect": float(effect),
            "adds": bool(p_value < alpha and effect >= tier.effect_size_floor),
            "detectable_r2": _detectable(n_eff, obs, values, z, alpha)}


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
        return _stack_selected(self.candidates, names, self.sample.covariates)

    def adds(self, name, values, z, n_permutations, rng, alpha, n_eff) -> dict:
        return _adds(self.tier, name, values, self.obs, z, n_permutations, rng, self.time_positions, self.lag,
                     self.grid_step, alpha, n_eff)


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
        margin: float,
        tiers: TierContexts | None = None,
) -> list[dict]:
    """For every accepted round, the relation of its winner to each other
    candidate not selected before it (see the module docstring). Returns one
    record per accepted round: {"round", "selected", "comparisons": [{
    "candidate", "relation", "candidate_adds_beyond_selected": {p_value,
    effect, adds, detectable_r2}, "selected_adds_beyond_candidate": {...}}],
    "flag_level"}. margin: the contribution an equivalence must be able to
    detect (see the module docstring). tiers: every tier of a cascade with
    its sample; each round is then compared with the tier that accepted it,
    otherwise with `tier` on `sample`."""
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
        n_eff = step.effective_sample_size or ctx.n
        z_with_winner = ctx.z(before + [winner])
        # The round's comparisons are independent: they run in parallel, each with its own random stream,
        # split from the run's in a fixed order, so results do not depend on the number of workers (apart from rounding).
        def compare(job):
            c, stream = job
            forward = ctx.adds(c, candidates[c], z_with_winner, n_permutations, stream, alpha, n_eff)
            z_with_c = as2d(candidates[c]) if z is None else np.hstack([z, as2d(candidates[c])])
            reverse = ctx.adds(winner, candidates[winner], z_with_c, n_permutations, stream, alpha, n_eff)
            return {
                "candidate": c,
                "relation": _relation(forward, reverse, margin),
                "candidate_adds_beyond_selected": forward,
                "selected_adds_beyond_candidate": reverse,
            }
        comparisons = parallel_map(compare, list(zip(others, rng.spawn(len(others)))))
        records.append({"round": i + 1, "selected": winner, "given": before, "margin": margin,
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
        margin: float,
        tiers: TierContexts | None = None,
) -> list[dict]:
    """The selected set and every set obtained by replacing selected
    formulations with candidates equivalent to them or not distinguishable
    from them, each verified as a whole (see the module docstring). Returns
    [{"set", "selected", "supported", "equivalent", "selected_adds_beyond_set":
    {p_value, effect, adds, detectable_r2} or None, "members": {name: {...}}}],
    the selected set first. With tiers (see information_equivalence), each
    member is tested with the tier that accepted the formulation in its
    position, and the replaced formulations jointly with the most general of
    their tiers."""
    steps = [st for st in selection.trajectory if st.accepted]
    selected = [st.formulation for st in steps]
    if not selected:
        return []
    options = [[w] for w in selected]
    relation_of = {}
    for rec in comparisons:
        i = selected.index(rec["selected"])
        for c in rec["comparisons"]:
            if c["relation"] in ("equivalent", "not_distinguishable"):
                options[i].append(c["candidate"])
                relation_of[(rec["selected"], c["candidate"])] = c["relation"]
    context_of, rank = _contexts(sample, tier, tiers)
    seen, combos = set(), []
    for combo in itertools.product(*options):
        if len(set(combo)) < len(combo) or frozenset(combo) in seen:
            continue
        seen.add(frozenset(combo))
        combos.append(list(combo))

    # Each set is verified independently, in parallel, with its own random stream (see information_equivalence).
    def verify(job):
        members, rng = job
        return _verify_set(members, rng)

    def _verify_set(members, rng):
        missing_slots = [i for i, w in enumerate(selected) if w not in members]
        missing = [selected[i] for i in missing_slots]
        beyond = None
        if missing:
            general = max((steps[i].tier for i in missing_slots), key=rank)
            ctx = context_of(general)
            n_eff = min((steps[i].effective_sample_size or ctx.n) for i in missing_slots if steps[i].tier == general)
            beyond = ctx.adds("+".join(missing), _stack(ctx.candidates, missing), ctx.z(members), n_permutations,
                              rng, alpha, n_eff)
        member_tests = {}
        for slot, m in enumerate(members):
            ctx = context_of(steps[slot].tier)
            n_eff = steps[slot].effective_sample_size or ctx.n
            member_tests[m] = ctx.adds(m, ctx.candidates[m], ctx.z([o for o in members if o != m]), n_permutations,
                                       rng, alpha, n_eff)
        supported = (beyond is None or not beyond["adds"]) and all(t["adds"] for t in member_tests.values())
        swaps = [relation_of[(w, c)] for w, c in zip(selected, members) if w != c]
        equivalent = bool(supported and all(r == "equivalent" for r in swaps)
                          and (beyond is None or beyond["detectable_r2"] <= margin))
        return {"set": members, "selected": not missing, "supported": bool(supported),
                "equivalent": equivalent, "selected_adds_beyond_set": beyond, "members": member_tests}

    return parallel_map(verify, list(zip(combos, rng.spawn(len(combos))))) if combos else []


def summarize_sets(selection: SelectionResult, comparisons: list[dict], sets: list[dict], margin: float,
                   alpha: float) -> str:
    """A short statement of the outcome: the selected set, which selected
    formulations another could replace and which none can, the other sets the
    data support equally, and whether a further formulation could add
    anything. Candidates that are selected, or that could replace another
    selected formulation, are left out of a formulation's statement."""
    selected = [st.formulation for st in selection.trajectory if st.accepted]
    if not selected:
        return ""
    fmt = lambda members: "{" + ", ".join(members) + "}"
    replaceable = {rec["selected"]: [c for c in rec["comparisons"]
                                     if c["relation"] in ("equivalent", "not_distinguishable")]
                   for rec in comparisons}
    parts = [f"Selected {fmt(selected)}."]
    swaps = []
    for rec in comparisons:
        w = rec["selected"]
        for c in replaceable[w]:
            if c["relation"] == "equivalent":
                swaps.append(f"{w} by {c['candidate']} (equivalent)")
            else:
                detect = max(c["candidate_adds_beyond_selected"]["detectable_r2"],
                             c["selected_adds_beyond_candidate"]["detectable_r2"])
                swaps.append(f"{w} by {c['candidate']} (not distinguishable; differences below "
                             f"R^2 = {format_share(detect)} cannot be excluded)")
    if swaps:
        parts.append("Replaceable: " + "; ".join(swaps) + ".")
    for rec in comparisons:
        w = rec["selected"]
        # The selected formulation adds information beyond these, so none of them can replace it.
        worse = [c for c in rec["comparisons"] if c["relation"] in ("encompassed_by_selected", "distinct")]
        better = [c["candidate"] for c in rec["comparisons"] if c["relation"] == "encompasses_selected"]
        if not replaceable[w] and worse:
            p_max = max(c["selected_adds_beyond_candidate"]["p_value"] for c in worse)
            parts.append(f"No other formulation can replace {w}: it adds information beyond each of "
                         f"{', '.join(c['candidate'] for c in worse)} (p <= {p_max:.2g}).")
        if better:
            parts.append(f"{', '.join(better)} adds information beyond {w}, which adds none beyond it: "
                         "possibly the better choice.")
    chosen = next((s for s in sets if s["selected"]), None)
    if chosen is not None and not chosen["supported"]:
        weak = [m for m, t in chosen["members"].items() if not t["adds"]]
        parts.append(f"Given the other selected formulations, {', '.join(weak)} no longer adds detectably.")
    alternatives = [s for s in sets if s["supported"] and not s["selected"]]
    if alternatives:
        parts.append("Equally supported sets: " + ", ".join(
            fmt(s["set"]) + ("" if s["equivalent"] else " (not distinguishable)") for s in alternatives) + ".")
    last = selection.trajectory[-1]
    if not last.accepted and last.detectable_r2 is not None:
        if last.p_value >= alpha:
            parts.append(f"No further formulation adds detectably (contributions below "
                         f"R^2 = {format_share(last.detectable_r2)} cannot be excluded).")
            prior = escalated_from_text(selection)
            if prior:
                parts.append(prior)
        else:
            parts.append("No further formulation adds a contribution as large as the effect-size floor.")
    return " ".join(parts)
