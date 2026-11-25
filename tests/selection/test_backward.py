"""The backward pass (selection.backward) and the comparisons on the final set
(selection.equivalence)."""
from types import SimpleNamespace

import numpy as np

import model_selection.selection.equivalence as eq
from model_selection.selection import SelectionConfig, SelectionSample, Tier
from model_selection.selection.backward import backward_pass
from model_selection.selection.config import AdequacyConfig
from model_selection.selection.greedy import SelectionResult, SelectionStep
from model_selection.selection.stage import run_selection_stage
from model_selection.selection.tiers import AdequacyCheck
from tests.selection.test_greedy import StubTest


def _stage(candidates, obs, seed=1, measure="linear"):
    n = len(obs)
    sample = SelectionSample(candidates, obs, np.arange(n), 1)
    return run_selection_stage(sample, SelectionConfig(measure=measure, n_permutations=199), AdequacyConfig(),
                               False, np.random.default_rng(seed)).selection


def test_a_composite_selected_first_is_removed_given_its_parts():
    """A composite selected first: 5 = 1 + 2 + noise is selected first, then 1 and 2;
    given 1 and 2, 5 adds nothing, and the backward pass removes it. It is
    then compared like any formulation not selected."""
    rng = np.random.default_rng(0)
    n = 1500
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    obs = s1 + 1.6 * s2 + 0.7 * rng.normal(size=n)
    f1, f2 = s1 + 0.3 * rng.normal(size=n), s2 + 0.3 * rng.normal(size=n)
    sel = _stage({"1": f1, "2": f2, "5": f1 + f2 + 0.5 * rng.normal(size=n), "3": rng.normal(size=n)}, obs)
    assert sel.forward_selected[0] == "5"
    assert sorted(sel.selected) == ["1", "2"]
    first = sel.backward[0]
    assert first["formulation"] == "5" and first["removed"] and sorted(first["given"]) == ["1", "2"]
    assert "5" in sel.classes["redundant"]
    assert "Removed by the backward pass: 5 adds no information given" in sel.equivalent_sets_summary
    # The comparisons condition on the final set and include the removed formulation.
    for rec in sel.equivalence:
        assert set(rec["given"]) == {"1", "2"} - {rec["selected"]}
        assert "5" in [c["candidate"] for c in rec["comparisons"]]


def test_near_copies_the_backward_pass_separates_are_exchangeable():
    """Of a composite and two near-copies, the composite is removed; the copy
    not selected is exchangeable with the one selected. The sets' tests that
    follow from the backward pass and the comparisons are not run."""
    rng = np.random.default_rng(3)
    n = 1000
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    obs = s1 + 1.6 * s2 + 0.5 * rng.normal(size=n)
    cands = {"a": s1 + 0.01 * rng.normal(size=n), "b": s1 + 0.01 * rng.normal(size=n),
             "c": s2 + 0.05 * rng.normal(size=n), "d": s1 + s2 + 0.3 * rng.normal(size=n)}
    sel = _stage(cands, obs, seed=4)
    assert sel.forward_selected[0] == "d" and "d" not in sel.selected
    copy = "a" if "a" in sel.selected else "b"
    other = "b" if copy == "a" else "a"
    assert sel.classes["exchangeable"] == {copy: [other]} and sel.classes["redundant"] == ["d"]
    chosen = next(s for s in sel.equivalent_sets if s["selected"])
    assert chosen["supported"] and all(t.get("implied") for t in chosen["members"].values())
    swapped = next(s for s in sel.equivalent_sets if not s["selected"])
    assert swapped["supported"] and swapped["members"][other]["implied"]
    assert not swapped["members"]["c"].get("implied")                   # tested: not implied for composites
    assert swapped["selected_adds_beyond_set"]["by"] == f"comparison of {copy} with {other}"


def _selection(names, tier="t1"):
    steps = [SelectionStep(formulation=nm, association=0.5, p_value=0.002, accepted=True, n_selected_before=i,
                           n_candidates_tested=len(names) - i, dependence_lag=0, tier=tier, effect=0.3)
             for i, nm in enumerate(names)]
    return SelectionResult(selected=list(names), trajectory=steps)


def _sample(names=("a", "b", "c", "x"), n=40):
    rng = np.random.default_rng(0)
    return SelectionSample({nm: rng.normal(size=n) for nm in names}, rng.normal(size=n), np.arange(n), 0)


def _fake_adds(monkeypatch, outcome):
    """Replace each backward test by outcome(name, tier name, conditioning width); records the calls."""
    calls = []

    def adds(self, name, values, z, n_permutations, rng, alpha):
        width = 0 if z is None else z.shape[1]
        calls.append((name, self.tier.name, width))
        return {"p_value": 0.5, "effect": 0.0, "untestable": False, **outcome(name, self.tier.name, width)}
    monkeypatch.setattr(eq._Context, "adds", adds)
    return calls


def test_nothing_removed_reuses_the_last_forward_test(monkeypatch):
    calls = _fake_adds(monkeypatch, lambda name, tier, width: {"adds": True, "p_value": 0.001})
    sel = _selection(["a", "b", "c"])
    tier = Tier("t1", StubTest({}), 0.02)
    backward_pass(sel, _sample(), tier, 99, 0.05, np.random.default_rng(1))
    assert [c[0] for c in calls] == ["a", "b"]                          # c faces exactly its forward test
    assert sel.selected == ["a", "b", "c"] and sel.forward_selected == ["a", "b", "c"]
    last = sel.backward[-1]
    assert last["reused_forward"] and last["p_value"] == 0.002 and last["given"] == ["a", "b"]


def test_a_removal_shrinks_later_conditioning_sets(monkeypatch):
    calls = _fake_adds(monkeypatch, lambda name, tier, width: {"adds": name != "a", "p_value": 0.001})
    sel = _selection(["a", "b", "c"])
    backward_pass(sel, _sample(), Tier("t1", StubTest({}), 0.02), 99, 0.05, np.random.default_rng(1))
    assert sel.selected == ["b", "c"]
    assert calls == [("a", "t1", 2), ("b", "t1", 1), ("c", "t1", 1)]  # c re-tested after a removal
    assert sel.backward[1]["given"] == ["c"] and not sel.backward[2]["reused_forward"]


def test_an_untestable_backward_test_keeps_the_formulation(monkeypatch):
    _fake_adds(monkeypatch, lambda name, tier, width: {"adds": False, "untestable": name == "a"})
    sel = _selection(["a", "b"])
    backward_pass(sel, _sample(), Tier("t1", StubTest({}), 0.02), 99, 0.05, np.random.default_rng(1))
    rec = sel.backward[0]
    assert sel.selected == ["a", "b"] and rec["adds"] is None and not rec["removed"]
    assert "clusters" in rec["not_tested_reason"]
    assert sel.backward[1]["reused_forward"]                            # nothing removed before b


def _departing_check():
    outcome = SimpleNamespace(tier_adequate=False, p_value=0.001, decision_reason="departs")
    return AdequacyCheck("nonlinear dependence on the selected formulations", lambda *a, **k: outcome,
                         needs_conditioning=True, stage="removal")


def test_a_validity_departure_moves_the_backward_test_to_the_next_tier(monkeypatch):
    calls = _fake_adds(monkeypatch, lambda name, tier, width: {"adds": True, "p_value": 0.001})
    t1 = Tier("t1", StubTest({}), 0.02, adequacy_checks=[_departing_check()])
    t2 = Tier("t2", StubTest({}), 0.02)
    sample = _sample()
    sel = _selection(["a", "b"])
    backward_pass(sel, sample, t2, 99, 0.05, np.random.default_rng(1), {"t1": (t1, sample), "t2": (t2, sample)})
    rec = sel.backward[0]
    assert calls == [("a", "t2", 1)] and rec["tier"] == "t2"
    assert rec["validity"] and rec["validity"][0]["tier"] == "t1" and not rec["validity"][0]["adequate"]
    # Without a more general tier, the formulation is kept, not tested.
    sel = _selection(["a", "b"])
    backward_pass(sel, sample, t1, 99, 0.05, np.random.default_rng(1), {"t1": (t1, sample)})
    assert sel.backward[0]["adds"] is None and "no more general tier" in sel.backward[0]["not_tested_reason"]
    assert sel.selected == ["a", "b"]


def test_the_comparisons_use_the_tier_of_the_backward_test(monkeypatch):
    calls = _fake_adds(monkeypatch, lambda name, tier, width: {"adds": True, "p_value": 0.001})
    t1 = Tier("t1", StubTest({}), 0.02, adequacy_checks=[_departing_check()])
    t2 = Tier("t2", StubTest({}), 0.02)
    sample = _sample()
    contexts = {"t1": (t1, sample), "t2": (t2, sample)}
    sel = _selection(["a", "b"])
    backward_pass(sel, sample, t2, 99, 0.05, np.random.default_rng(1), contexts)
    calls.clear()
    records = eq.information_equivalence(sel, sample, t2, 99, 0.05, np.random.default_rng(2), contexts)
    assert {r["selected"]: r["tier"] for r in records} == {"a": "t2", "b": "t1"}
    assert sorted(c[0] for c in calls) == ["a", "a", "b", "b"]          # each against c and x, one direction
