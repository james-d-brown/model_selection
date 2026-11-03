"""
Tests of greedy forward selection: the round logic (winner, adequacy,
max-statistic calibration, both gates, stopping) with a stub test whose
outcomes are controlled, and end-to-end selection with each real tier.
"""
import numpy as np
import pytest

from model_selection.selection import SelectionConfig, SelectionSample, Tier, build_tier, greedy_forward_selection
from model_selection.selection.greedy import gate
from model_selection.selection.tiers import AdequacyCheck
from model_selection.statistics.adequacy.base import AdequacyTestOutcome
from model_selection.statistics.association.base import StructuralCheckOutcome
from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult


class StubTest(ConditionalIndependenceTest):
    """Effect size = a fixed value per candidate (0 once a candidate's 'cause'
    is selected); null draws are standard-normal noise scaled by `null_scale`."""
    name = "stub"

    def __init__(self, effects, null_scale=0.01):
        self.effects = effects
        self.null_scale = null_scale
        self.calls = []

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        names = list(candidates)
        self.calls.append((names, None if conditioning is None else conditioning.shape))
        observed = np.array([self.effects[nm] for nm in names])
        null = np.abs(rng.normal(scale=self.null_scale, size=(n_resamples, len(names))))
        return JointTestResult(names, observed, null, {nm: {"stub": True} for nm in names})


def _sample(names=("a", "b", "c"), n=50):
    rng = np.random.default_rng(0)
    return SelectionSample({nm: rng.normal(size=n) for nm in names}, rng.normal(size=n), np.arange(n), 0)


def _tier(effects, floor=0.02, **kw):
    return Tier("stub", StubTest(effects, **kw), floor)


def test_selects_in_order_of_effect_and_stops_at_first_rejection():
    res = greedy_forward_selection(_sample(), _tier({"a": 0.5, "b": 0.3, "c": 0.0}), 99, rng=np.random.default_rng(1))
    assert res.selected == ["a", "b"]
    assert [s.formulation for s in res.trajectory] == ["a", "b", "c"]
    assert res.trajectory[-1].accepted is False
    assert "'c' not accepted" in res.stopped_reason
    assert res.unconditional_association == {"a": 0.5, "b": 0.3, "c": 0.0}


def test_conditioning_grows_with_each_selection():
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(2))
    assert [c[1] for c in tier.test.calls] == [None, (50, 1), (50, 2)]
    assert [len(c[0]) for c in tier.test.calls] == [3, 2, 1]


def test_all_candidates_selected():
    res = greedy_forward_selection(_sample(), _tier({"a": 0.5, "b": 0.4, "c": 0.3}), 99, rng=np.random.default_rng(3))
    assert res.selected == ["a", "b", "c"] and res.stopped_reason == "all candidates selected"


def test_max_selected_stops_the_loop():
    res = greedy_forward_selection(_sample(), _tier({"a": 0.5, "b": 0.4, "c": 0.3}), 99, max_selected=1,
                                   rng=np.random.default_rng(4))
    assert res.selected == ["a"] and "max_selected=1" in res.stopped_reason


def test_empty_candidates():
    res = greedy_forward_selection(_sample(names=()), _tier({}), 99)
    assert res.selected == [] and res.stopped_reason == "no candidates remaining"


def test_time_positions_length_mismatch_raises():
    s = _sample()
    s.time_positions = np.arange(10)
    with pytest.raises(ValueError, match="time_positions"):
        greedy_forward_selection(s, _tier({"a": 1, "b": 1, "c": 1}), 9)


def test_magnitude_gate_rejects_significant_but_small_effects():
    res = greedy_forward_selection(_sample(), _tier({"a": 0.01, "b": 0.0, "c": 0.0}, null_scale=1e-4), 99,
                                   rng=np.random.default_rng(5))
    step = res.trajectory[0]
    assert step.p_value < 0.05 and not step.accepted and step.magnitude_unsubstantiated
    assert "magnitude unsubstantiated" in res.stopped_reason


def test_significance_gate_rejects_large_but_insignificant_effects():
    res = greedy_forward_selection(_sample(), _tier({"a": 0.5, "b": 0.0, "c": 0.0}, null_scale=1.0), 99,
                                   rng=np.random.default_rng(6))
    assert not res.trajectory[0].accepted and "not statistically significant" in res.stopped_reason


def test_gate_uses_maximum_for_p_value_and_own_column_for_bias():
    column = np.zeros(99)
    maxima = np.full(99, 0.5)
    accepted, p, corrected = gate(0.4, column, maxima, 0.05, 0.02)
    assert not accepted and p == 1.0 and corrected == pytest.approx(0.4)


def test_structural_check_stops_trial_as_ill_posed():
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    tier.structural_check = lambda n, p, q, k: StructuralCheckOutcome(False, "too few samples")
    res = greedy_forward_selection(_sample(), tier, 99)
    assert res.ill_posed and res.selected == [] and "too few samples" in res.ill_posed_reason


def test_a_later_round_failing_the_sample_size_requirement_keeps_the_selection():
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    tier.structural_check = lambda n, p, q, k: StructuralCheckOutcome(k == 0, "too few samples")
    res = greedy_forward_selection(_sample(), tier, 99)
    assert not res.ill_posed and res.selected == ["a"]
    assert "too few samples" in res.ill_posed_reason and "round 2 could not be tested" in res.stopped_reason


def test_adequacy_rejection_discards_trial_and_plots_are_recorded():
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    seen = []

    def reject_in_round_two(candidate, obs, conditioning, **kw):
        return AdequacyTestOutcome(conditioning is None, "rejected: test", 0.001)

    def plot(candidate, obs, conditioning, label, context=None):
        seen.append(label)
        return f"/tmp/{label}.html"
    tier.adequacy_checks = [AdequacyCheck("check", reject_in_round_two, plot)]
    res = greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(7))
    assert res.tier_rejected and res.selected == []
    assert seen == ["round1_a", "round2_b_given_a"]
    assert res.diagnostic_plot_paths == ["/tmp/round1_a.html", "/tmp/round2_b_given_a.html"]


def test_adequacy_checks_needing_conditioning_skip_round_one():
    tier = _tier({"a": 0.5, "b": 0.0, "c": 0.0})
    calls = []
    tier.adequacy_checks = [AdequacyCheck("needs z", lambda *a, **k: calls.append(1) or
                                          AdequacyTestOutcome(True, "", 1.0), needs_conditioning=True)]
    greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(8))
    assert len(calls) == 1  # round 2 only


def _real_sample(rng, n=150):
    driver = rng.normal(size=n)
    obs = driver + 0.3 * rng.normal(size=n)
    return SelectionSample({"informative": driver + 0.3 * rng.normal(size=n), "noise": rng.normal(size=n)},
                           obs, np.arange(n), 0)


@pytest.mark.parametrize("tier_name", ["linear", "nonlinear", "cmi"])
def test_each_tier_selects_the_informative_formulation(tier_name):
    """The informative formulation is selected first in every replicate; the
    uninformative one is accepted by chance in at most one of five (5% nominal)."""
    tier = build_tier(tier_name, SelectionConfig(measure=tier_name), workers=1)
    noise_accepted = 0
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        res = greedy_forward_selection(_real_sample(rng), tier, 99, rng=rng)
        assert res.selected[0] == "informative"
        assert res.trajectory[0].diagnostics
        noise_accepted += "noise" in res.selected
    assert noise_accepted <= 1


def test_low_samples_per_dimension_is_flagged_not_enforced():
    res = greedy_forward_selection(_sample(n=50), _tier({"a": 0.5, "b": 0.3, "c": 0.0}), 99,
                                   rng=np.random.default_rng(11), min_samples_per_dimension=20)
    steps = res.trajectory
    assert steps[0].samples_per_dimension == pytest.approx(25.0)  # 50 samples, candidate + obs
    assert steps[1].samples_per_dimension == pytest.approx(50 / 3)  # plus one selected column
    assert [s.low_samples_per_dimension for s in steps] == [False, True, True]
    assert res.selected == ["a", "b"]  # the flag never changes the decision


def _relationship_check(adequate, calls, only_round_one=False):
    def test(candidate, obs, conditioning, **kw):
        calls.append(1)
        ok = adequate or (only_round_one and conditioning is not None)
        return AdequacyTestOutcome(ok, "not rejected" if ok else "rejected: curved", 0.01)
    return AdequacyCheck("nonlinear candidate-observation relation", test, stage="relationship")


def test_relationship_check_after_acceptance_is_information_only():
    calls = []
    tier = _tier({"a": 0.5, "b": 0.0, "c": 0.0})
    tier.adequacy_checks = [_relationship_check(False, calls, only_round_one=True)]
    res = greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(12))
    assert res.selected == ["a"] and not res.tier_rejected      # accepted: no escalation
    first = res.trajectory[0].adequacy[0]
    assert first["role"] == "information" and first["adequate"] is False
    assert len(calls) == 3  # 'a', then 'b' (the stopping round's winner) and 'c' (the other remaining candidate)


def test_relationship_check_after_non_acceptance_escalates():
    calls = []
    tier = _tier({"a": 0.0, "b": 0.0, "c": 0.0})
    tier.adequacy_checks = [_relationship_check(False, calls)]
    res = greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(13))
    assert res.tier_rejected and "not accepted under the stub tier" in res.stopped_reason
    assert res.trajectory[0].adequacy[0]["role"] == "decision"


def test_relationship_check_passing_after_non_acceptance_lets_rejection_stand():
    tier = _tier({"a": 0.0, "b": 0.0, "c": 0.0})
    tier.adequacy_checks = [_relationship_check(True, [])]
    res = greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(14))
    assert not res.tier_rejected and res.selected == [] and "not accepted" in res.stopped_reason


def test_column_labels_name_series_and_feature():
    from model_selection.selection.greedy import _column_labels
    assert _column_labels("candidate 'x'", np.zeros(5), None) == ["candidate 'x'"]
    assert _column_labels("observations", np.zeros((5, 2)), ("PC1", "duration")) == [
        "observations, feature 1 (PC1)", "observations, feature 2 (duration)"]
    assert _column_labels("candidate 'x'", np.zeros((5, 2)), None) == [
        "candidate 'x', feature 1", "candidate 'x', feature 2"]


def _departing_check(departing, p_departs=0.001, calls=None):
    """Relationship check that departs (rejects) for the candidates whose
    arrays are in `departing`, with p-value p_departs, and passes otherwise."""
    def check(candidate, obs, conditioning, **kw):
        if calls is not None:
            calls.append(candidate)
        departs = any(candidate is c for c in departing)
        return AdequacyTestOutcome(not departs, "rejected" if departs else "not rejected",
                                   p_departs if departs else 0.9,
                                   bias_corrected_semipartial_effect_size=0.2 if departs else 0.0)
    return AdequacyCheck("nonlinear relation", check, stage="relationship")


def test_a_remaining_candidate_other_than_the_winner_can_escalate():
    """A candidate whose contribution the tier's statistic misses ranks low;
    its departure at alpha / m escalates, as the winner's would."""
    sample = _sample(names=("a", "b", "c", "d"))
    tier = _tier({"a": 0.5, "b": 0.01, "c": 0.0, "d": 0.0})
    tier.adequacy_checks = [_departing_check([sample.candidates["c"]])]
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(10))
    # a power-only escalation: the acceptance so far stays valid and is kept for the next tier
    assert res.tier_rejected and res.escalation_continues and res.selected == ["a"]
    assert "the remaining candidate 'c'" in res.stopped_reason and "each of the 3 remaining" in res.stopped_reason
    checked = {e["candidate"]: e for e in res.trajectory[-1].other_candidates_checked}
    assert checked["c"]["departs"] and checked["c"]["level"] == pytest.approx(0.05 / 3)


def test_the_remaining_candidates_are_one_family_at_alpha_over_m():
    """Every remaining candidate, the winner included, is tested at alpha / m
    (one relationship test here): a rejection at a p-value between alpha / m
    and alpha does not escalate."""
    sample = _sample(names=("a", "b", "c", "d"))
    tier = _tier({"a": 0.5, "b": 0.01, "c": 0.0, "d": 0.0})
    tier.adequacy_checks = [_departing_check([sample.candidates[nm] for nm in "bcd"], p_departs=0.03)]
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(10))
    assert not res.tier_rejected and res.selected == ["a"]
    step = res.trajectory[-1]
    winner_entry = [e for e in step.adequacy if e["check"] == "nonlinear relation"][0]
    assert winner_entry["role"] == "decision" and winner_entry["adequate"] and winner_entry["level"] == pytest.approx(0.05 / 3)
    assert {e["candidate"] for e in step.other_candidates_checked} == {"c", "d"}
    assert not any(e["departs"] for e in step.other_candidates_checked)


def test_the_winner_is_checked_first_and_a_departure_stops_further_checks():
    sample = _sample(names=("a", "b", "c", "d"))
    calls = []
    tier = _tier({"a": 0.5, "b": 0.01, "c": 0.0, "d": 0.0})
    tier.adequacy_checks = [_departing_check([sample.candidates["b"]], calls=calls)]
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(10))
    assert res.tier_rejected and "rejects for 'b'" in res.stopped_reason
    assert calls[-1] is sample.candidates["b"] and res.trajectory[-1].other_candidates_checked == []


def test_the_level_divides_alpha_by_every_relationship_test_of_every_candidate():
    """With two relationship tests per candidate the family has 2 m tests."""
    sample = _sample(names=("a", "b", "c", "d"))
    tier = _tier({"a": 0.5, "b": 0.01, "c": 0.0, "d": 0.0})
    second = _departing_check([])
    second.name = "second relation"
    tier.adequacy_checks = [_departing_check([]), second]
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(10))
    levels = [e["level"] for e in res.trajectory[-1].other_candidates_checked]
    assert len(levels) == 4 and all(lv == pytest.approx(0.05 / 6) for lv in levels)


def test_an_escalation_to_an_uncalibrated_tier_marks_the_next_tier_unsupported():
    """When a remaining candidate's departure leads to an uncalibrated
    free-form tier, the guard is applied to that candidate and the next tier
    is marked unsupported; the departure concerns power only, so the
    selection so far is kept."""
    sample = _sample(names=("a", "b", "c", "d"))
    guard_seen = []

    def guard(candidate, obs, conditioning, time_positions, dependence_lag, rng, grid_step=None, names=None):
        guard_seen.append(candidate)
        return _calibration(False, [])(candidate, obs, conditioning, time_positions, dependence_lag, rng)

    tier = _tier({"a": 0.5, "b": 0.01, "c": 0.0, "d": 0.0})
    tier.adequacy_checks = [_departing_check([sample.candidates["c"]])]
    tier.escalation_guard = guard
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(10))
    assert res.tier_rejected and res.escalation_continues and res.selected == ["a"]
    assert len(guard_seen) == 1 and guard_seen[0] is sample.candidates["c"]
    assert "rejects for the remaining candidate 'c'" in res.stopped_reason
    assert res.next_tier_unsupported is not None and "not calibrated" in res.next_tier_unsupported


def test_sample_covariates_are_conditioned_on_from_round_one():
    """Observation-free covariates of the sample (the event registration)
    enter the conditioning set in every round, before any formulation is
    selected."""
    sample = _sample()
    sample.covariates = np.random.default_rng(1).normal(size=(len(sample.obs), 2))
    stub = StubTest({"a": 0.5, "b": 0.0, "c": 0.0})
    greedy_forward_selection(sample, Tier("stub", stub, 0.02), 99, rng=np.random.default_rng(2))
    assert stub.calls[0] == (["a", "b", "c"], (len(sample.obs), 2))
    assert stub.calls[1][1] == (len(sample.obs), 3)


def _calibration(calibrated, calls):
    def calibrate(candidate, obs, conditioning, time_positions, dependence_lag, rng, grid_step=None, names=None):
        calls.append(1)
        return {"n_surrogates": 52, "false_acceptances": 1 if calibrated else 20,
                "false_acceptance_rate": 0.02 if calibrated else 0.38, "p_value": 0.9 if calibrated else 1e-9,
                "calibrated": calibrated}
    return calibrate


def test_an_acceptance_by_an_uncalibrated_tier_is_withheld_and_ends_selection():
    calls = []
    sample = _sample()
    sample.covariates = np.zeros((len(sample.obs), 1))
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    tier.calibration = _calibration(False, calls)
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(14))
    assert res.selected == [] and len(res.trajectory) == 1 and calls == [1]
    assert not res.trajectory[0].accepted and not res.trajectory[0].cmi_calibration["calibrated"]
    assert "not calibrated" in res.stopped_reason


def test_a_calibrated_tier_keeps_its_acceptances():
    calls = []
    sample = _sample()
    sample.covariates = np.zeros((len(sample.obs), 1))
    tier = _tier({"a": 0.5, "b": 0.3, "c": 0.0})
    tier.calibration = _calibration(True, calls)
    res = greedy_forward_selection(sample, tier, 99, rng=np.random.default_rng(15))
    assert res.selected == ["a", "b"] and len(calls) == 2


def test_escalation_to_an_uncalibrated_cmi_tier_marks_it_unsupported():
    """When the check that would escalate rejects but the CMI tier is not
    calibrated at that state, the CMI tier is marked unsupported."""
    calls, guard_calls = [], []
    tier = _tier({"a": 0.5, "b": 0.0, "c": 0.0})
    tier.adequacy_checks = [_relationship_check(False, calls, only_round_one=False)]
    tier.escalation_guard = _calibration(False, guard_calls)
    res = greedy_forward_selection(_sample(), tier, 99, rng=np.random.default_rng(16))
    assert res.tier_rejected and res.escalation_continues and res.selected == ["a"] and guard_calls == [1]
    assert "not calibrated" in res.next_tier_unsupported and not res.trajectory[-1].cmi_calibration["calibrated"]


def test_cmi_calibration_is_attached_only_when_requested():
    assert build_tier("cmi", SelectionConfig(measure="cmi", cmi_calibration=False)).calibration is None
    assert callable(build_tier("cmi", SelectionConfig(measure="cmi", cmi_calibration=True)).calibration)


class _EffectStub(StubTest):
    """A test whose effect size is a separate quantity from its statistic, as
    in the nonlinear tier."""

    def __init__(self, effects, sizes):
        super().__init__(effects)
        self.sizes = sizes

    def magnitude(self, candidate, obs, conditioning, time_positions, dependence_lag, observed, null_draws,
                  grid_step=None):
        name = next(nm for nm, v in self._values.items() if v is candidate)
        return self.sizes[name], None

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        self._values = candidates
        return super().run(candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag)


def test_the_winner_is_the_significant_candidate_with_the_largest_effect_size():
    """'a' has the larger statistic, 'b' the larger effect size; both are
    significant, so 'b' wins. 'c' has the largest effect size but is not
    significant, so it cannot win."""
    sample = _sample()
    test = _EffectStub({"a": 0.5, "b": 0.3, "c": 0.0}, {"a": 0.05, "b": 0.2, "c": 0.9})
    res = greedy_forward_selection(sample, Tier("stub", test, 0.02), 99, max_selected=1, rng=np.random.default_rng(1))
    assert res.trajectory[0].formulation == "b" and res.trajectory[0].accepted


def test_removed_covariates_count_as_conditioning_variables_in_the_power_calculation():
    """A sample whose covariates were removed by kernel regression counts that
    fit's effective degrees of freedom, so its detectable contribution is
    larger than the same sample's without them."""
    from dataclasses import replace
    sample = _sample()
    plain = greedy_forward_selection(sample, _tier({"a": 0.5, "b": 0.0, "c": 0.0}), 99, max_selected=1,
                                     rng=np.random.default_rng(1))
    removed = greedy_forward_selection(replace(sample, partialled_covariates=np.zeros((len(sample.obs), 2)),
                                               partialled_df=8.0),
                                       _tier({"a": 0.5, "b": 0.0, "c": 0.0}), 99, max_selected=1,
                                       rng=np.random.default_rng(1))
    assert removed.trajectory[0].detectable_r2 > plain.trajectory[0].detectable_r2
