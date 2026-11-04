"""Tests of tier construction and the cascade's escalation."""
import numpy as np
import pytest

from model_selection.selection import (
    AdequacyConfig, SelectionConfig, SelectionSample, Tier, build_cascade, build_tier,
    effect_size_floor_for_tier, run_tiered_selection,
)
from model_selection.selection.tiers import AdequacyCheck
from model_selection.statistics.adequacy.base import AdequacyTestOutcome
from tests.selection.test_greedy import StubTest


def test_effect_size_floor_conversion():
    assert effect_size_floor_for_tier(0.02, "linear") == 0.02
    assert effect_size_floor_for_tier(0.02, "nonlinear") == 0.02
    assert effect_size_floor_for_tier(0.02, "cmi") == pytest.approx(-0.5 * np.log(0.98))
    with pytest.raises(ValueError):
        effect_size_floor_for_tier(0.02, "pdc")


def test_measure_validation():
    with pytest.raises(ValueError, match="measure"):
        SelectionConfig(measure="pdc")


def test_single_tier_has_no_adequacy_checks():
    for name in ("linear", "nonlinear", "cmi"):
        assert build_tier(name, SelectionConfig()).adequacy_checks == []


def test_cascade_order_checks_and_sample_override():
    override = SelectionSample({}, np.zeros(3), np.arange(3), 0)
    tiers = build_cascade(SelectionConfig(), AdequacyConfig(), multivariate=False, tier_samples={"nonlinear": override})
    assert [t.name for t in tiers] == ["linear", "nonlinear", "cmi"]
    assert [(c.name, c.stage) for c in tiers[0].adequacy_checks] == [
        ("nonlinear dependence on the selected formulations", "removal"), ("nonlinear candidate-observation relation", "relationship"),
        ("effect modification by the selected formulations", "relationship")]
    assert [(c.name, c.stage) for c in tiers[1].adequacy_checks] == [
        ("effect modification by the selected formulations", "relationship")]
    assert tiers[2].adequacy_checks == []
    assert tiers[1].sample is override and tiers[0].sample is None and tiers[2].sample is None
    assert tiers[0].structural_check is not None


def _sample(n=40):
    rng = np.random.default_rng(0)
    return SelectionSample({"a": rng.normal(size=n), "b": rng.normal(size=n)}, rng.normal(size=n), np.arange(n), 0)


def _always(adequate):
    return AdequacyCheck("check", lambda *a, **k: AdequacyTestOutcome(adequate, "rejected: x" if not adequate else "", 0.5))


def test_cascade_escalates_until_a_tier_is_adequate():
    effects = {"a": 0.5, "b": 0.0}
    tiers = [Tier("t1", StubTest(effects), 0.02, [_always(False)]),
             Tier("t2", StubTest(effects), 0.02, [_always(True)]),
             Tier("t3", StubTest(effects), 0.02)]
    res = run_tiered_selection(_sample(), tiers, 99, rng=np.random.default_rng(1))
    assert res.committed_tier == "t2" and res.selection.selected == ["a"]
    assert [(a.tier, a.committed) for a in res.attempted_tiers] == [("t1", False), ("t2", True)]


def test_cascade_raises_if_every_tier_rejects():
    tiers = [Tier("t1", StubTest({"a": 0.5, "b": 0.0}), 0.02, [_always(False)])]
    with pytest.raises(RuntimeError, match="Every tier"):
        run_tiered_selection(_sample(), tiers, 99)


def test_real_cascade_stays_linear_for_linear_data():
    rng = np.random.default_rng(2)
    n = 200
    z = rng.normal(size=n)
    sample = SelectionSample({"base": z + 0.3 * rng.normal(size=n), "extra": rng.normal(size=n)},
                             z + 0.3 * rng.normal(size=n), np.arange(n), 0)
    res = run_tiered_selection(sample, build_cascade(SelectionConfig(), AdequacyConfig(n_permutations=49), False),
                               99, rng=rng)
    assert res.committed_tier == "linear" and res.selection.selected == ["base"]


def test_real_cascade_escalates_on_nonlinear_main_effect():
    """The selected formulation acts nonlinearly on both the candidate and
    the observations, so linear removal is invalid and the cascade
    escalates."""
    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(size=n)
    obs = base + 0.5 * base ** 2 + 0.2 * rng.normal(size=n)
    near = base ** 2 + 0.3 * rng.normal(size=n)
    sample = SelectionSample({"base": base, "near": near}, obs, np.arange(n), 0)
    res = run_tiered_selection(sample, build_cascade(SelectionConfig(), AdequacyConfig(n_permutations=49), False),
                               99, rng=rng)
    assert res.attempted_tiers[0].tier == "linear" and not res.attempted_tiers[0].committed
    assert res.committed_tier in ("nonlinear", "cmi")


def test_check_a_keeps_the_linear_tier_when_only_one_side_is_nonlinear():
    """Linear partialling stays valid when either regression on Z is linear
    (the error is a product of the two regressions' errors), so the nonlinear-dependence check
    rejects only when the candidate and the observations both depart."""
    from model_selection.statistics.adequacy.specification import linear_specification_test
    rng = np.random.default_rng(4)
    n = 300
    z = rng.normal(size=(n, 1))
    linear_c = z[:, 0] + 0.3 * rng.normal(size=n)
    curved_o = z[:, 0] ** 2 + 0.3 * rng.normal(size=n)
    curved_c = z[:, 0] ** 2 + 0.3 * rng.normal(size=n)
    args = (z, 0.05, 0.02, 99, rng, np.arange(n), 0)
    one_side = linear_specification_test(np.column_stack([linear_c, curved_o]), *args, n_candidate_columns=1)
    both = linear_specification_test(np.column_stack([curved_c, curved_o]), *args, n_candidate_columns=1)
    assert one_side.tier_adequate and "other side" in one_side.decision_reason
    assert not both.tier_adequate


def test_adequacy_floor_follows_the_meaningful_threshold():
    assert AdequacyConfig().resolved_effect_size_floor(SelectionConfig()) == 0.02
    assert AdequacyConfig().resolved_effect_size_floor(SelectionConfig(meaningful_r2=0.05)) == 0.05
    assert AdequacyConfig(effect_size_floor=0.03).resolved_effect_size_floor(SelectionConfig(meaningful_r2=0.05)) == 0.03


def test_adequacy_checks_apply_the_resolved_floor():
    rng = np.random.default_rng(5)
    n = 200
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    o = c ** 2 + 0.3 * rng.normal(size=n)
    tiers = build_cascade(SelectionConfig(meaningful_r2=0.9), AdequacyConfig(n_permutations=49), False)
    residual_check = tiers[0].adequacy_checks[1]
    outcome = residual_check.test(c, o, z[:, None], n_permutations=49, rng=rng, time_positions=np.arange(n))
    assert "effect_size_floor=0.9" in outcome.decision_reason


def test_both_kernel_escalation_tests_use_the_shared_sample_cap():
    """The linear adjustment and effect-modification tests are built with
    SelectionConfig.nonlinear_max_n; the old AdequacyConfig cap only warns."""
    import warnings
    from model_selection.selection.config import AdequacyConfig, SelectionConfig
    from model_selection.selection.tiers import build_tier
    from model_selection.selection import tiers
    selection = SelectionConfig(nonlinear_max_n=1234)
    seen = {}
    real_spec = tiers.linear_specification_test
    tiers.linear_specification_test = lambda *a, **k: seen.setdefault("max_n", k["max_n"])
    try:
        removal = next(c for c in build_tier("linear", selection, AdequacyConfig()).adequacy_checks
                       if c.stage == "removal")
        removal.test(np.zeros(5), np.zeros(5), np.zeros((5, 1)))
    finally:
        tiers.linear_specification_test = real_spec
    assert seen["max_n"] == 1234
    from model_selection.statistics.adequacy import state_independence
    built = []
    real = state_independence.StateIndependenceTest.__init__

    def spy(self, *args, **kwargs):
        built.append(kwargs.get("max_n"))
        real(self, *args, **kwargs)
    state_independence.StateIndependenceTest.__init__ = spy
    try:
        build_tier("nonlinear", selection, AdequacyConfig())
    finally:
        state_independence.StateIndependenceTest.__init__ = real
    assert built == [1234]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AdequacyConfig(kernel_check_max_n=4000)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_real_cascade_escalates_from_the_linear_tier_on_effect_modification():
    """The candidate's slope on the observations changes sign with the selected
    formulation's state, so the linear tier's averaged statistic cancels and
    neither its adjustment nor its linearity test sees the departure; the
    candidate effect-modification test does, and the cascade leaves the
    linear tier."""
    rng = np.random.default_rng(6)
    n = 300
    base = rng.normal(size=n)
    flip = rng.normal(size=n)
    obs = base + 0.8 * flip * np.sign(base) + 0.3 * rng.normal(size=n)
    sample = SelectionSample({"base": base, "flip": flip}, obs, np.arange(n), 0)
    res = run_tiered_selection(sample, build_cascade(SelectionConfig(), AdequacyConfig(n_permutations=99), False),
                               99, rng=rng)
    first = res.attempted_tiers[0]
    assert first.tier == "linear" and not first.committed
    assert "effect modification" in first.reason


def test_real_cascade_escalates_for_a_remaining_candidate_that_is_not_the_winner():
    """At the stopping round the linear tier's winner is a weakly related
    candidate; another candidate acts only through its square, has no linear
    association and so does not win, but its linearity test departs, and the
    cascade leaves the linear tier."""
    rng = np.random.default_rng(7)
    n = 300
    base = rng.normal(size=n)
    curve = rng.normal(size=n)
    obs = base + 0.8 * (curve ** 2 - 1) + 0.3 * rng.normal(size=n)
    decoy = 0.11 * (obs - base) + rng.normal(size=n)
    sample = SelectionSample({"base": base, "curve": curve, "decoy": decoy}, obs, np.arange(n), 0)
    tiers = build_cascade(SelectionConfig(), AdequacyConfig(n_permutations=99), False)
    from model_selection.selection import greedy_forward_selection
    linear_trial = greedy_forward_selection(sample, tiers[0], 99, rng=np.random.default_rng(8))
    assert linear_trial.tier_rejected
    assert [s.formulation for s in linear_trial.trajectory] == ["base", "decoy"]
    assert "the remaining candidate 'curve'" in linear_trial.stopped_reason


def test_the_escalation_calibration_check_uses_the_cmi_tiers_own_sample():
    """With a separate free-form sample (a long pooled record), the check that
    guards escalation to the free-form tier rebuilds the round's candidate and
    conditioning set from that sample by name."""
    rng = np.random.default_rng(9)
    own = SelectionSample({"a": rng.normal(size=30), "b": rng.normal(size=30)}, rng.normal(size=30),
                          np.arange(0, 300, 10), 10, grid_step=10)
    tiers = build_cascade(SelectionConfig(), AdequacyConfig(), False, tier_samples={"cmi": own})
    seen = {}

    def calibration(candidate, obs, conditioning, time_positions, dependence_lag, rng, grid_step=None):
        seen.update(candidate=candidate, obs=obs, conditioning=conditioning, t=time_positions, step=grid_step)
        return {"calibrated": True}

    tiers[2].calibration = calibration
    from model_selection.selection.tiers import _guard_on_own_sample
    guard = _guard_on_own_sample(tiers[2])
    other = np.zeros(100)
    guard(other, other, other[:, None], np.arange(100), 1, rng, grid_step=1, names=("b", ["a"]))
    assert seen["candidate"] is own.candidates["b"] and seen["obs"] is own.obs and seen["t"] is own.time_positions
    assert np.array_equal(seen["conditioning"], own.candidates["a"][:, None]) and seen["step"] == 10


def _power_only_escalation(first_name, effects, guard=None):
    """A tier whose relationship check rejects for every candidate it does not accept."""
    tier = Tier(first_name, StubTest(effects), 0.02, [AdequacyCheck(
        "relation", lambda *a, **k: AdequacyTestOutcome(False, "rejected: x", 0.001), stage="relationship")])
    tier.escalation_guard = guard
    return tier


def test_a_power_only_escalation_continues_from_the_current_selection():
    """A relationship (power-only) departure leaves the tier's acceptances
    valid: the next tier starts from them and tests only the rest."""
    first = _power_only_escalation("t1", {"a": 0.5, "b": 0.0, "c": 0.0})
    second = Tier("t2", StubTest({"a": 0.5, "b": 0.5, "c": 0.0}), 0.02)
    sample = _sample()
    sample.candidates["c"] = np.random.default_rng(9).normal(size=len(sample.obs))
    res = run_tiered_selection(sample, [first, second], 99, rng=np.random.default_rng(3))
    assert res.committed_tier == "t2" and res.selection.selected == ["a", "b"]
    assert [(st.formulation, st.tier) for st in res.selection.trajectory if st.accepted] == [("a", "t1"), ("b", "t2")]
    assert sorted(second.test.calls[0][0]) == ["b", "c"]  # the carried formulation is conditioned on, not retested
    assert res.attempted_tiers[0].provisional_selection == []


def test_a_validity_escalation_restarts_from_an_empty_selection():
    effects = {"a": 0.5, "b": 0.0}
    first = Tier("t1", StubTest(effects), 0.02, [_always(False)])
    second = Tier("t2", StubTest(effects), 0.02)
    res = run_tiered_selection(_sample(), [first, second], 99, rng=np.random.default_rng(4))
    assert res.committed_tier == "t2" and res.selection.selected == ["a"]
    assert all(st.tier == "t2" for st in res.selection.trajectory)


def test_after_a_power_only_escalation_to_an_unusable_tier_the_current_selection_stands():
    """When the tier a power-only departure leads to is not calibrated on its
    sample, it is not run, and the current tier's valid selection stands."""
    guard = lambda *a, **k: {"calibrated": False, "false_acceptances": 9, "n_surrogates": 52, "p_value": 0.001}
    first = _power_only_escalation("t1", {"a": 0.5, "b": 0.0}, guard)
    second = Tier("t2", StubTest({"a": 0.5, "b": 0.0}), 0.02)
    sample = _sample()
    sample.covariates = np.zeros((len(sample.obs), 1))
    res = run_tiered_selection(sample, [first, second], 99, rng=np.random.default_rng(3))
    assert res.committed_tier == "t1" and res.selection.selected == ["a"] and not res.selection.tier_rejected
    attempts = [a.to_record() for a in res.attempted_tiers]
    assert [(a["tier"], a["committed"], a["run"]) for a in attempts] == [("t1", True, True), ("t2", False, False)]
    assert "selection stands" in attempts[1]["reason"] and "not calibrated" in attempts[1]["reason"]
    assert not second.test.calls


def test_review_flags_report_an_unsupported_escalation_and_a_powerless_escalated_tier():
    from model_selection.pipeline.results import PooledResult, review_flags
    from model_selection.selection.greedy import SelectionResult, SelectionStep
    attempts = [{"tier": "linear", "committed": False, "reason": "tier rejected", "run": True,
                 "provisional_selection": ["a", "b"]},
                {"tier": "nonlinear", "committed": True, "reason": "not run: ... no selection can be supported",
                 "run": False, "provisional_selection": []}]
    base = dict(gage="g", n_timesteps=10, n_complete_timesteps=10, measure="cascade", selection=SelectionResult(),
                zero_flow_fraction={}, zero_flow_flag=False, min_temporal_lag=1, committed_tier="nonlinear")
    flags = review_flags(PooledResult(**base, attempted_tiers=attempts))
    flag = next(f for f in flags if f["code"] == "escalation_unsupported")
    assert "linear tier: ['a', 'b']" in flag["message"]
    step = SelectionStep(formulation="c", association=0.1, p_value=0.5, accepted=False, n_selected_before=0,
                         n_candidates_tested=2, dependence_lag=1, detectable_r2=1.0, tier="nonlinear")
    attempts[1].update(run=True, reason="'c' not accepted")
    flags = review_flags(PooledResult(**dict(base, selection=SelectionResult(trajectory=[step])), attempted_tiers=attempts))
    assert "escalated_tier_undetectable" in [f["code"] for f in flags]


def test_the_test_that_led_to_the_next_tier_is_recorded_and_flagged():
    """The relationship test that rejected the first tier is kept with that
    tier's attempt, its resampling diagnostics are flagged although its step
    is not part of the committed selection, and the stopping round reports
    the first tier's smallest detectable R^2 alongside the second's."""
    from model_selection.pipeline.results import PooledResult, review_flags
    dep = {"estimated": True, "few_blocks": True, "n_blocks": 30}
    first = Tier("t1", StubTest({"a": 0.5, "b": 0.0}), 0.02, [AdequacyCheck(
        "effect modification", lambda *a, **k: AdequacyTestOutcome(False, "rejected: x", 0.001, dependence=dep),
        stage="relationship")])
    second = Tier("t2", StubTest({"a": 0.5, "b": 0.0}), 0.02)
    res = run_tiered_selection(_sample(), [first, second], 99, rng=np.random.default_rng(3))
    assert res.committed_tier == "t2" and res.selection.selected == ["a"]
    test = res.attempted_tiers[0].escalation_test
    assert (test["round"], test["candidate"], test["check"], test["p_value"]) == (2, "b", "effect modification", 0.001)
    assert res.selection.escalated_from["tier"] == "t1"
    assert "In the t1 tier, which tested this round before the escalation" in res.selection.stopped_reason
    base = dict(gage="g", n_timesteps=10, n_complete_timesteps=10, measure="cascade", selection=res.selection,
                zero_flow_fraction={}, zero_flow_flag=False, min_temporal_lag=1, committed_tier="t2")
    flags = review_flags(PooledResult(**base, attempted_tiers=[a.to_record() for a in res.attempted_tiers]))
    assert any(f["code"] == "few_blocks" and "t1 tier, round 2 (b), effect modification, the test that led to the "
               "next tier" in f["message"] for f in flags)


def test_a_powerless_escalated_tier_is_flagged_when_its_detectable_r2_is_reported_as_1():
    from model_selection.pipeline.results import PooledResult, review_flags
    from model_selection.selection.greedy import SelectionResult, SelectionStep
    step = SelectionStep(formulation="c", association=0.1, p_value=0.5, accepted=False, n_selected_before=0,
                         n_candidates_tested=2, dependence_lag=1, detectable_r2=0.997, tier="cmi")
    attempts = [{"tier": "nonlinear", "committed": False, "reason": "x", "run": True},
                {"tier": "cmi", "committed": True, "reason": "y", "run": True}]
    flags = review_flags(PooledResult(gage="g", n_timesteps=10, n_complete_timesteps=10, measure="cascade",
                                      selection=SelectionResult(trajectory=[step]), zero_flow_fraction={},
                                      zero_flow_flag=False, min_temporal_lag=1, committed_tier="cmi",
                                      attempted_tiers=attempts))
    assert "escalated_tier_undetectable" in [f["code"] for f in flags]


def test_the_powerless_escalated_tier_flag_uses_the_margin():
    from model_selection.pipeline.results import PooledResult, review_flags
    from model_selection.selection.greedy import SelectionResult, SelectionStep

    def flagged(detectable, meaningful=None):
        step = SelectionStep(formulation="c", association=0.1, p_value=0.5, accepted=False, n_selected_before=0,
                             n_candidates_tested=2, dependence_lag=1, detectable_r2=detectable, tier="cmi")
        result = PooledResult(gage="g", n_timesteps=10, n_complete_timesteps=10, measure="cascade",
                              selection=SelectionResult(trajectory=[step]), zero_flow_fraction={},
                              zero_flow_flag=False, min_temporal_lag=1, committed_tier="cmi", meaningful_r2=meaningful,
                              attempted_tiers=[{"tier": "nonlinear", "committed": False, "reason": "x", "run": True},
                                               {"tier": "cmi", "committed": True, "reason": "y", "run": True}])
        return "escalated_tier_undetectable" in [f["code"] for f in review_flags(result)]
    assert flagged(0.99) and flagged(0.05) and not flagged(0.015)
    assert not flagged(0.05, meaningful=0.1)


def test_the_powerless_escalated_tier_flag_looks_at_the_stopping_round():
    """A first round that accepted is not flagged for its own low power; the
    round at which the tier stopped is what the flag judges."""
    from model_selection.pipeline.results import PooledResult, review_flags
    from model_selection.selection.greedy import SelectionResult, SelectionStep

    def step(accepted, detectable):
        return SelectionStep(formulation="c", association=0.1, p_value=0.01 if accepted else 0.5, accepted=accepted,
                             n_selected_before=0, n_candidates_tested=2, dependence_lag=1, detectable_r2=detectable,
                             tier="nonlinear")

    def flagged(trajectory):
        result = PooledResult(gage="g", n_timesteps=10, n_complete_timesteps=10, measure="cascade",
                              selection=SelectionResult(trajectory=trajectory), zero_flow_fraction={},
                              zero_flow_flag=False, min_temporal_lag=1, committed_tier="nonlinear",
                              attempted_tiers=[{"tier": "linear", "committed": False, "reason": "x", "run": True},
                                               {"tier": "nonlinear", "committed": True, "reason": "y", "run": True}])
        return "escalated_tier_undetectable" in [f["code"] for f in review_flags(result)]
    assert not flagged([step(True, 0.75), step(False, 0.015)])
    assert flagged([step(True, 0.75), step(False, 0.19)])
    assert not flagged([step(True, 0.75)])
