"""Information equivalence and encompassing of each selected formulation
with the other candidates (selection.equivalence)."""
import numpy as np

from model_selection.selection import SelectionConfig, SelectionSample, build_tier, greedy_forward_selection
from model_selection.selection.equivalence import information_equivalence


def _run(candidates, obs, seed=0):
    n = len(obs)
    sample = SelectionSample(candidates, obs, np.arange(n), 1)
    tier = build_tier("linear", SelectionConfig(measure="linear", n_permutations=199, equivalence_checks=True))
    rng = np.random.default_rng(seed)
    result = greedy_forward_selection(sample, tier, 199, 0.05, rng=rng)
    return result, information_equivalence(result, sample, tier, 199, 0.05, np.random.default_rng(seed + 1),
                                           margin=0.02)


def test_two_copies_of_the_same_information_are_equivalent():
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
    assert relations[other] == "equivalent"
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


def test_equally_supported_sets_replace_a_selected_formulation_by_its_equivalent():
    from model_selection.selection.equivalence import equally_supported_sets, summarize_sets
    rng = np.random.default_rng(3)
    n = 1000  # enough power that the detectable contribution is clearly below the 0.02 margin
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    obs = s1 + s2 + 0.5 * rng.normal(size=n)
    cands = {"a": s1 + 0.01 * rng.normal(size=n), "b": s1 + 0.01 * rng.normal(size=n),
             "c": s2 + 0.05 * rng.normal(size=n)}
    sample = SelectionSample(cands, obs, np.arange(n), 1)
    tier = build_tier("linear", SelectionConfig(measure="linear", n_permutations=199))
    result = greedy_forward_selection(sample, tier, 199, 0.05, rng=np.random.default_rng(4))
    assert sorted(result.selected) in (["a", "c"], ["b", "c"])
    records = information_equivalence(result, sample, tier, 199, 0.05, np.random.default_rng(5), margin=0.02)
    sets = equally_supported_sets(result, records, sample, tier, 199, 0.05, np.random.default_rng(6),
                                 margin=0.02)
    supported = {frozenset(s["set"]) for s in sets if s["supported"]}
    assert supported == {frozenset({"a", "c"}), frozenset({"b", "c"})}
    assert all(s["equivalent"] for s in sets if s["supported"] and not s["selected"])
    line = summarize_sets(result, records, sets, 0.02, 0.05)
    assert line.startswith("Selected {") and "Replaceable:" in line and "Equally supported sets" in line



def test_without_power_a_pair_is_not_distinguishable_rather_than_equivalent():
    rng = np.random.default_rng(7)
    n = 60
    signal = rng.normal(size=n)
    obs = signal + rng.normal(size=n)
    cands = {"a": signal + 0.01 * rng.normal(size=n), "b": signal + 0.01 * rng.normal(size=n)}
    result, records = _run(cands, obs)
    (cmp,) = records[0]["comparisons"]
    assert cmp["relation"] == "not_distinguishable"
    assert cmp["candidate_adds_beyond_selected"]["detectable_r2"] > 0.02


def test_the_equivalence_margin_is_on_the_r2_scale_in_every_tier():
    from model_selection.selection.stage import equivalence_margin
    for measure in ("linear", "nonlinear", "cmi"):
        assert equivalence_margin(SelectionConfig(measure=measure)) == 0.02
        assert equivalence_margin(SelectionConfig(measure=measure, meaningful_r2=0.1)) == 0.1


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
    records = information_equivalence(sel, small, t2, 99, 0.05, np.random.default_rng(1), 0.02,
                                      {"t1": (t1, big), "t2": (t2, small)})
    assert [r["tier"] for r in records] == ["t1", "t1"]
    assert t1.test.calls and not t2.test.calls
    assert all(shape[0] == 60 for _, shape in t1.test.calls if shape is not None)
    equally_supported_sets(sel, records, small, t2, 99, 0.05, np.random.default_rng(2), 0.02,
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
            recs = information_equivalence(sel, sample, tier, 99, 0.05, np.random.default_rng(2), 0.02)
            sets = equally_supported_sets(sel, recs, sample, tier, 99, 0.05, np.random.default_rng(3), 0.02)
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
