"""
The comparisons of each selected formulation with the formulations not
selected, the sets of formulations the data support equally, and the class
of every formulation after selection.

Greedy forward selection tests whether each round's winner is informative
about the observations given the formulations already selected, not whether
it is better than the other candidates. When two candidates carry the same
information, the winner can be decided by noise. The selected set is then
one of several equally valid sets, as with a distribution that has several
Markov boundaries (Statnikov et al., 2013).

The comparisons run on the final set S, after the backward pass
(selection.backward). Each selected formulation W is compared with every
formulation C not in S, including those the backward pass removed, given
the other members R of S, with the test and acceptance gates of W's tier
(the tier of its backward test), on that tier's sample, at level alpha:

    W adds beyond C:  is W informative about O given R and C?

W adding is encompassing (Mizon and Richard, 1986): W explains everything C
does and more. W not adding means the two are not distinguished. The other
direction, whether C adds beyond W given R, asks whether C adds given S.
Forward selection's stopping test answered that for every formulation not
selected, and the backward pass's removals cannot change the answer: by the
contraction property of conditional independence (Dawid, 1979; Pearl,
1988), C adds nothing given the forward selection and the removed
formulations add nothing given S imply that C adds nothing given S. So it
is not tested. When selection ended before its stopping test was completed,
the formulations not selected are reported as not tested.

Equally supported sets. Replacing selected formulations by candidates not
distinguished from them gives alternative sets of the same size. A set is
supported if the selected formulations it lacks, jointly, add no
information given it, and every member adds information given the others
(Statnikov et al., 2013). The selected set itself meets both conditions
through the backward pass. For a single replacement of W by C, the first
condition is the comparison of W with C, and C adding given the other
members R follows by contraction from W adding given R and W adding nothing
given R and C; each other member is tested, because without the
intersection property of conditional independence, which fails when
formulations are functions of one another, its membership is not implied.
A set with several replacements is verified by every test.

Classes (formulation_classes). Distinct: a selected formulation that adds
information beyond every formulation compared with it, an
intersection-union finding at level alpha (Berger, 1982). Exchangeable: a
selected formulation with the candidates that replace it in a supported set.
Redundant: not selected and not exchangeable, when the stopping round tested
every formulation not selected and accepted none. Not tested: not selected,
when selection ended before such a test was completed. A selected
formulation that is neither distinct nor exchangeable is not shown to be
distinct. Distinct rests on rejections; exchangeable and redundant rest on
non-rejections, whose chance of missing a real difference is not
controlled.

Single-candidate tests use sequential Monte Carlo p-values where the tier
provides them (Besag and Clifford, 1991): draws stop once the candidate can
no longer be significant at alpha, which gives the same decision as drawing
them all.

Limitations: the selected formulation was chosen for its large statistic,
which biases the test of W beyond C towards finding that W adds
(post-selection inference; Berk et al., 2013); and the tier's adequacy
checks were made for the selection's conditioning sets, not for the sets
with C in place of W.

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


def _relation(reverse: dict) -> str:
    """The relation of a selected formulation W to a formulation C not
    selected, from the test of W given the other selected formulations and C."""
    return "encompassed_by_selected" if reverse["adds"] else "not_distinguished"


def _adds(tier: Tier, name: str, values, obs, z, n_permutations, rng, time_positions, lag, grid_step,
          alpha, period_starts=None) -> dict:
    """Whether one candidate passes the tier's gates given z (its own null,
    no adjustment across candidates)."""
    untestable = False
    if hasattr(tier.test, "sequential_p_value"):
        observed, null, p_value = tier.test.sequential_p_value(
            values, obs, z, n_permutations, rng, time_positions, lag,
            stopping_exceedances(alpha, n_permutations), grid_step=grid_step)
    else:
        result = tier.test.run({name: values}, obs, z, n_permutations, rng, time_positions, lag,
                               grid_step=grid_step, period_starts=period_starts)
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
        effect, _ = tier.test.magnitude(values, obs, z, time_positions, lag, observed, null, grid_step=grid_step,
                                        period_starts=period_starts)
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
        self.period_starts = sample.period_starts

    def z(self, names: list[str]):
        return _stack_selected(self.sample.conditioning_series, names, self.sample.covariates)

    def adds(self, name, values, z, n_permutations, rng, alpha) -> dict:
        return _adds(self.tier, name, values, self.obs, z, n_permutations, rng, self.time_positions, self.lag,
                     self.grid_step, alpha, self.period_starts)


def _contexts(sample: SelectionSample, tier: Tier, tiers: TierContexts | None):
    """(context of a step's tier, rank of a tier name in generality)."""
    default = _Context(tier, sample)
    built = {name: _Context(t, s) for name, (t, s) in (tiers or {}).items()}
    order = {name: i for i, name in enumerate(tiers or {})}
    return (lambda name: built.get(name, default)), (lambda name: order.get(name, -1))


def _final_tiers(selection: SelectionResult) -> dict[str, str | None]:
    """The tier of each selected formulation's last test: its backward test
    when there was one (a validity departure can move it to a more general
    tier), otherwise the tier that accepted it."""
    tier_of = {st.formulation: st.tier for st in selection.trajectory if st.accepted}
    for rec in selection.backward:
        if rec.get("tier"):
            tier_of[rec["formulation"]] = rec["tier"]
    return tier_of


def information_equivalence(
        selection: SelectionResult,
        sample: SelectionSample,
        tier: Tier,
        n_permutations: int,
        alpha: float,
        rng: np.random.Generator,
        tiers: TierContexts | None = None,
) -> list[dict]:
    """For every formulation W of the final set (selection.selected), its
    relation to each formulation not in that set (see the module docstring).
    Returns one record per selected formulation: {"selected", "given", "tier",
    "alpha", "effect_size_floor", "comparisons": [{"candidate", "relation",
    "selected_adds_beyond_candidate": {p_value, effect, adds, untestable}}]}.
    tiers: every tier of a cascade with its sample; W is then compared with
    the tier of its last test, otherwise with `tier` on `sample`."""
    context_of, _ = _contexts(sample, tier, tiers)
    final = list(selection.selected)
    tier_of = _final_tiers(selection)
    records = []
    for w, stream in zip(final, rng.spawn(len(final))):
        ctx = context_of(tier_of.get(w))
        candidates = ctx.candidates
        rest = [m for m in final if m != w]
        others = [c for c in candidates if c not in final]
        if not others:
            continue
        z_rest = ctx.z(rest)

        # The comparisons are independent: they run in parallel, each with its own random stream, split from the
        # run's in a fixed order, so results do not depend on the number of workers (apart from rounding).
        def compare(job, ctx=ctx, w=w, z_rest=z_rest):
            c, sub = job
            c_conditioning = as2d(ctx.sample.conditioning_series[c])
            z_with_c = c_conditioning if z_rest is None else np.hstack([z_rest, c_conditioning])
            reverse = ctx.adds(w, ctx.candidates[w], z_with_c, n_permutations, sub, alpha)
            return {"candidate": c, "relation": _relation(reverse), "selected_adds_beyond_candidate": reverse}
        comparisons = parallel_map(compare, list(zip(others, stream.spawn(len(others)))))
        records.append({"selected": w, "given": rest, "tier": ctx.tier.name, "alpha": alpha,
                        "effect_size_floor": ctx.tier.effect_size_floor, "comparisons": comparisons})
        summary = "; ".join(f"{cmp['candidate']}: {cmp['relation'].replace('_', ' ')}" for cmp in comparisons)
        logger.info(f"[i] {w} compared with the formulations not selected: {summary}.")
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
    (see the module docstring). Returns [{"set", "selected", "supported",
    "selected_adds_beyond_set": {p_value, effect, adds, untestable} or None,
    "members": {name: {...}}}], the selected set first. A member test or a
    joint test that follows from other results is recorded as
    {"implied": True, "adds": ..., "by": reason} and not run. With tiers
    (see information_equivalence), each member is tested with the tier of
    the selected formulation in its position, and the replaced formulations
    jointly with the most general of their tiers."""
    selected = list(selection.selected)
    if not selected:
        return []
    tier_of = _final_tiers(selection)
    by_selected = {rec["selected"]: rec for rec in comparisons}
    options = [[w] for w in selected]
    for i, w in enumerate(selected):
        for c in by_selected.get(w, {}).get("comparisons", []):
            if c["relation"] == "not_distinguished":
                options[i].append(c["candidate"])
    context_of, rank = _contexts(sample, tier, tiers)
    backward_of = {r["formulation"]: r for r in selection.backward}
    seen, combos = set(), []
    for combo in itertools.product(*options):
        if len(set(combo)) < len(combo) or frozenset(combo) in seen:
            continue
        seen.add(frozenset(combo))
        combos.append(list(combo))

    def kept(w: str) -> bool:
        """W was tested by the backward pass and still adds given the final set's other members."""
        rec = backward_of.get(w)
        return rec is not None and rec["adds"] is True

    # Each set is verified independently, in parallel, with its own random stream (see information_equivalence).
    def verify(job):
        members, stream = job
        missing_slots = [i for i, w in enumerate(selected) if w not in members]
        missing = [selected[i] for i in missing_slots]
        beyond, member_tests = None, {}
        if not missing and all(kept(m) for m in members):
            member_tests = {m: {"implied": True, "adds": True, "by": "backward pass"} for m in members}
        elif len(missing) == 1 and kept(missing[0]):
            j, w = missing_slots[0], missing[0]
            c = members[j]
            cmp = next(x for x in by_selected[w]["comparisons"] if x["candidate"] == c)
            beyond = dict(cmp["selected_adds_beyond_candidate"], by=f"comparison of {w} with {c}")
            member_tests[c] = {"implied": True, "adds": True,
                               "by": f"contraction: {w} adds given the others, and nothing given them and {c}"}
            for slot, m in enumerate(members):
                if m == c:
                    continue
                ctx = context_of(tier_of.get(selected[slot]))
                member_tests[m] = ctx.adds(m, ctx.candidates[m], ctx.z([o for o in members if o != m]),
                                           n_permutations, stream, alpha)
        else:
            if missing:
                general = max((tier_of.get(selected[i]) for i in missing_slots), key=rank)
                ctx = context_of(general)
                beyond = ctx.adds("+".join(missing), _stack(ctx.candidates, missing), ctx.z(members),
                                  n_permutations, stream, alpha)
            for slot, m in enumerate(members):
                ctx = context_of(tier_of.get(selected[slot]))
                member_tests[m] = ctx.adds(m, ctx.candidates[m], ctx.z([o for o in members if o != m]),
                                           n_permutations, stream, alpha)
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
        formulation compared with them;
    exchangeable: {selected formulation: candidates that replace it in a
        supported set};
    not_shown_distinct: {selected formulation: candidates it adds no
        information beyond}, for a selected formulation neither distinct nor
        exchangeable;
    redundant: formulations neither selected nor exchangeable, when the
        stopping round tested every formulation not selected and accepted
        none;
    not_tested: those formulations otherwise, with not_tested_reason.

    Selected means the final set, after the backward pass; a formulation the
    backward pass removed is classed like any other formulation not
    selected.
    """
    selected = list(selection.selected)
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
    Reporting of interchangeable formulations): the selected set; the
    formulations the backward pass removed; the class of each formulation,
    with each exchangeable formulation's candidates; and the other sets the
    data support equally."""
    selected = list(selection.selected)
    if not selected:
        return ""
    fmt = lambda members: "{" + ", ".join(members) + "}"
    parts = [f"Selected {fmt(selected)}."]
    removed = [r for r in selection.backward if r["removed"]]
    if removed:
        parts.append("Removed by the backward pass: " + "; ".join(
            f"{r['formulation']} adds no information given {fmt(r['given'])}" for r in removed) + ".")
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
