"""Comparisons of each selected formulation with the other candidates, the
equally supported sets and the class of every formulation
(selection.equivalence)."""
import pytest
import numpy as np

from model_selection.selection import SelectionConfig, SelectionSample, build_tier, greedy_forward_selection
from model_selection.selection.equivalence import information_equivalence


def _run(candidates, obs, seed=0):
    n = len(obs)
    sample = SelectionSample(candidates, obs, np.arange(n), 1)
    tier = build_tier("linear", SelectionConfig(measure="linear", n_permutations=199, equivalence_checks=True))
    rng = np.random.default_rng(seed)
    result = greedy_forward_selection(sample, tier, 199, 0.05, rng=rng)
    return result, information_equivalence(result, sample, tier, 199, 0.05, np.random.default_rng(seed + 1))


def test_two_copies_of_the_same_information_are_not_distinguished():
    rng = np.random.default_rng(1)
    n = 600
    signal = rng.normal(size=n)
    obs = signal + rng.normal(size=n)
    a = signal + 0.01 * rng.normal(size=n)
    b = signal + 0.01 * rng.normal(size=n)
    result, records = _run({"a": a, "b": b, "noise": rng.normal(size=n)}, obs)
    (first,) = [r for r in records if r["round"] == 1]
    relations = {c["candidate"]: c["relation"] for c in first["comparisons"]}
    other = "b" if first["selected"] == "a" else "a"
    assert relations[other] == "not_distinguished"
    assert relations["noise"] == "encompassed_by_selected"


def test_a_noisier_copy_is_encompassed_and_a_distinct_signal_is_distinct():
    rng = np.random.default_rng(2)
    n = 600
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    obs = s1 + 0.7 * s2 + 0.5 * rng.normal(size=n)
    good = s1 + 0.05 * rng.normal(size=n)
    noisy = s1 + 1.0 * rng.normal(size=n)
    other = s2 + 0.05 * rng.normal(size=n)
    result, records = _run({"good": good, "noisy": noisy, "other": other}, obs)
    first = records[0]
    assert first["selected"] == "good"
    relations = {c["candidate"]: c["relation"] for c in first["comparisons"]}
    assert relations["noisy"] == "encompassed_by_selected"
    assert relations["other"] == "distinct"


@pytest.mark.integration
def test_comparisons_reach_the_result_file():
    import json
    from model_selection.pipeline import run_pooled_analysis
    from model_selection.pipeline.results import pooled_result_to_dict
    from tests.fixtures.basins import make_synthetic_basin
    result = run_pooled_analysis(make_synthetic_basin(seed=4), SelectionConfig(measure="linear", n_permutations=50),
                                 rng=np.random.default_rng(5))
    d = pooled_result_to_dict(result)
    assert d["pooled_equivalence"] and d["pooled_equivalence"][0]["comparisons"]
    json.dumps(d)


def test_equally_supported_sets_and_classes_for_a_copy_and_a_distinct_signal():
    from model_selection.selection.equivalence import equally_supported_sets, formulation_classes, summarize_sets
    rng = np.random.default_rng(3)
    n = 1000
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    obs = s1 + s2 + 0.5 * rng.normal(size=n)
    cands = {"a": s1 + 0.01 * rng.normal(size=n), "b": s1 + 0.01 * rng.normal(size=n),
             "c": s2 + 0.05 * rng.normal(size=n)}
    sample = SelectionSample(cands, obs, np.arange(n), 1)
    tier = build_tier("linear", SelectionConfig(measure="linear", n_permutations=199))
    result = greedy_forward_selection(sample, tier, 199, 0.05, rng=np.random.default_rng(4))
    assert sorted(result.selected) in (["a", "c"], ["b", "c"])
    records = information_equivalence(result, sample, tier, 199, 0.05, np.random.default_rng(5))
    sets = equally_supported_sets(result, records, sample, tier, 199, 0.05, np.random.default_rng(6))
    supported = {frozenset(s["set"]) for s in sets if s["supported"]}
    assert supported == {frozenset({"a", "c"}), frozenset({"b", "c"})}
    classes = formulation_classes(result, records, sets, list(cands))
    copy = "a" if "a" in result.selected else "b"
    assert classes["exchangeable"] == {copy: ["b" if copy == "a" else "a"]}
    assert classes["distinct"] == ["c"] and classes["redundant"] == [] and classes["not_tested"] == []
    line = summarize_sets(result, records, sets, classes)
    assert line.startswith("Selected {") and "Exchangeable:" in line and "Equally supported sets" in line
    assert "Distinct: c adds information beyond each of" in line



def _step(name, accepted, p=0.001, **kw):
    from model_selection.selection.greedy import SelectionStep
    return SelectionStep(formulation=name, association=0.5, p_value=p, accepted=accepted, n_selected_before=0,
                         n_candidates_tested=3, dependence_lag=0, **kw)


def _cmp(candidate, forward_adds, reverse_adds, relation, p=0.001):
    return {"candidate": candidate, "relation": relation,
            "candidate_adds_beyond_selected": {"p_value": p, "effect": 0.1, "adds": forward_adds,
                                               "untestable": False},
            "selected_adds_beyond_candidate": {"p_value": p, "effect": 0.1, "adds": reverse_adds,
                                               "untestable": False}}


def test_classes_follow_the_comparisons_sets_and_how_selection_ended():
    """Distinct when the selected formulation adds beyond every candidate;
    redundant when the last round tested and accepted none; not tested when
    selection ended before that test (docs/methods.md, Interchangeable
    formulations, Classes)."""
    from model_selection.selection.equivalence import formulation_classes, summarize_sets
    from model_selection.selection.greedy import SelectionResult
    records = [{"round": 1, "selected": "a", "flag_level": 0.0125,
                "comparisons": [_cmp("b", False, True, "encompassed_by_selected"),
                                _cmp("c", False, True, "encompassed_by_selected")]}]
    sets = [{"set": ["a"], "selected": True, "supported": True, "selected_adds_beyond_set": None, "members": {}}]
    stopped = SelectionResult(selected=["a"], trajectory=[_step("a", True), _step("b", False, p=0.4)])
    classes = formulation_classes(stopped, records, sets, ["a", "b", "c"])
    assert classes["distinct"] == ["a"] and classes["redundant"] == ["b", "c"] and not classes["not_tested"]
    line = summarize_sets(stopped, records, sets, classes)
    assert "Distinct: a adds information beyond each of b, c" in line and "Redundant: b, c." in line
    # A selection that ended at the cap on selected formulations never tested the rest.
    capped = SelectionResult(selected=["a"], trajectory=[_step("a", True)])
    classes = formulation_classes(capped, records, sets, ["a", "b", "c"])
    assert classes["not_tested"] == ["b", "c"] and not classes["redundant"] and classes["not_tested_reason"]
    # A round untestable for too few clusters of events.
    untestable = SelectionResult(selected=["a"], trajectory=[_step("a", True)], untestable=True,
                                 ill_posed_reason="5 clusters")
    assert formulation_classes(untestable, records, sets, ["a", "b", "c"])["not_tested"] == ["b", "c"]


def test_a_selected_formulation_a_candidate_encompasses_is_not_shown_to_be_distinct():
    from model_selection.selection.equivalence import formulation_classes, summarize_sets
    from model_selection.selection.greedy import SelectionResult
    records = [{"round": 1, "selected": "a", "flag_level": 0.025,
                "comparisons": [_cmp("b", True, False, "encompasses_selected")]}]
    sets = [{"set": ["a"], "selected": True, "supported": True, "selected_adds_beyond_set": None, "members": {}}]
    sel = SelectionResult(selected=["a"], trajectory=[_step("a", True), _step("b", False, p=0.3)])
    classes = formulation_classes(sel, records, sets, ["a", "b"])
    assert classes["not_shown_distinct"] == {"a": ["b"]} and not classes["distinct"]
    assert "possibly the better choice" in summarize_sets(sel, records, sets, classes)


def test_after_an_escalation_each_formulation_is_compared_with_the_tier_that_accepted_it():
    """A selection combining steps of two tiers compares each formulation with
    the test and sample of the tier that accepted it, not the committed tier's."""
    from model_selection.selection import Tier
    from model_selection.selection.equivalence import equally_supported_sets
    from model_selection.selection.greedy import SelectionResult, SelectionStep
    from tests.selection.test_greedy import StubTest
    rng = np.random.default_rng(0)
    big = SelectionSample({nm: rng.normal(size=60) for nm in "abc"}, rng.normal(size=60), np.arange(60), 0)
    small = SelectionSample({nm: rng.normal(size=12) for nm in "abc"}, rng.normal(size=12), np.arange(12), 0)
    t1 = Tier("t1", StubTest({"a": 0.5, "b": 0.5, "c": 0.0}), 0.02)
    t2 = Tier("t2", StubTest({"a": 0.5, "b": 0.5, "c": 0.0}), 0.02)
    step = lambda nm, tier: SelectionStep(formulation=nm, association=0.5, p_value=0.001, accepted=True,
                                          n_selected_before=0, n_candidates_tested=3, dependence_lag=0, tier=tier,
                                          effective_sample_size=60.0)
    sel = SelectionResult(selected=["a", "b"], trajectory=[step("a", "t1"), step("b", "t1")])
    records = information_equivalence(sel, small, t2, 99, 0.05, np.random.default_rng(1),
                                      {"t1": (t1, big), "t2": (t2, small)})
    assert [r["tier"] for r in records] == ["t1", "t1"]
    assert t1.test.calls and not t2.test.calls
    assert all(shape[0] == 60 for _, shape in t1.test.calls if shape is not None)
    equally_supported_sets(sel, records, small, t2, 99, 0.05, np.random.default_rng(2),
                           {"t1": (t1, big), "t2": (t2, small)})
    assert not t2.test.calls


def test_comparisons_do_not_depend_on_the_number_of_workers():
    """Each comparison and set check has its own random stream, split from the
    run's in a fixed order, so running them in parallel gives the same records:
    the same decisions, and numbers equal apart from rounding (the linear-algebra
    library sums in a different order with a different number of threads)."""
    from model_selection.parallel import set_workers
    from model_selection.selection.equivalence import equally_supported_sets
    rng = np.random.default_rng(0)
    n = 120
    base = rng.normal(size=n)
    obs = base + 0.5 * rng.normal(size=n)
    sample = SelectionSample({"a": base + 0.3 * rng.normal(size=n), "b": base + 0.3 * rng.normal(size=n),
                              "c": rng.normal(size=n)}, obs, np.arange(n), 0)
    tier = build_tier("linear", SelectionConfig())
    sel = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(1))
    out = []
    for workers in (1, 3):
        set_workers(workers)
        try:
            recs = information_equivalence(sel, sample, tier, 99, 0.05, np.random.default_rng(2))
            sets = equally_supported_sets(sel, recs, sample, tier, 99, 0.05, np.random.default_rng(3))
            out.append((recs, sets))
        finally:
            set_workers(None)
    def same(a, b):
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and not isinstance(a, bool):
            return isinstance(b, float) and (np.isnan(a) and np.isnan(b) or np.isclose(a, b, rtol=1e-10, atol=1e-12))
        return a == b
    assert same(out[0], out[1])


def test_a_comparison_on_too_few_clusters_is_untestable_and_never_adds():
    """With 5 clusters of events no p-value below alpha = 0.05 is attainable
    (significance.clusters), so a comparison detects nothing: it is marked
    untestable, and the pair can only be not distinguished."""
    from model_selection.selection.equivalence import _adds, _relation
    from model_selection.selection import build_tier
    from model_selection.statistics.significance.clusters import cluster_coordinates
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(5), 4)
    pos, lag = cluster_coordinates(labels)
    obs = rng.normal(size=labels.size)
    tier = build_tier("linear", SelectionConfig())
    fwd = _adds(tier, "a", obs + 0.1 * rng.normal(size=obs.size), obs, None, 99, rng, pos, lag, None, 0.05)
    rev = _adds(tier, "b", rng.normal(size=obs.size), obs, None, 99, rng, pos, lag, None, 0.05)
    assert fwd["untestable"] and not fwd["adds"]
    assert _relation(fwd, rev) == "not_distinguished"
