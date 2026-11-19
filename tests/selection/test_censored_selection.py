"""
Selection with censored candidates and observations (docs/methods.md,
Censoring at high flow): candidates enter tests censored, but every
conditioning set holds the selected formulations' uncensored values, in the
selection rounds, the comparisons with other candidates and the CMI tier's
escalation guard; the CMI tier's surrogate candidates are built uncensored
and then censored at the candidate's threshold.
"""
import numpy as np
import pytest

from model_selection.preprocessing import CensoredSeries, Censoring
from model_selection.selection import SelectionSample, Tier, greedy_forward_selection
from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult

N = 60


class RecordingTest(ConditionalIndependenceTest):
    """Fixed effect per candidate; records the candidates and conditioning
    sets it is given."""
    name = "recording"

    def __init__(self, effects):
        self.effects, self.calls = effects, []

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None,
            period_starts=None):
        names = list(candidates)
        self.calls.append(({nm: np.array(v) for nm, v in candidates.items()},
                           None if conditioning is None else np.array(conditioning)))
        observed = np.array([self.effects[nm] for nm in names])
        null = np.abs(rng.normal(scale=0.01, size=(n_resamples, len(names))))
        return JointTestResult(names, observed, null, {nm: {} for nm in names})


def _censored_sample(names=("a", "b", "c"), q=0.8, seed=0):
    rng = np.random.default_rng(seed)
    unc = {nm: rng.normal(size=N) for nm in names}
    c = Censoring(q)
    thr = {nm: c.threshold(v) for nm, v in unc.items()}
    obs_unc = rng.normal(size=N)
    obs_thr = c.threshold(obs_unc)
    conditioning = {nm: rng.normal(size=N) for nm in names}       # distinct from the candidates' values
    phase = rng.normal(size=(N, 2))
    cs = CensoredSeries(unc, thr, obs_thr, obs_unc)
    return SelectionSample(cs.censored_candidates(), cs.censored_obs(),
                           np.arange(N), 0, covariates=phase, conditioning=conditioning, censored=cs)


def test_without_separate_conditioning_values_the_candidates_condition():
    rng = np.random.default_rng(1)
    s = SelectionSample({"a": rng.normal(size=5)}, rng.normal(size=5), np.arange(5), 0)
    assert s.conditioning_series is s.candidates
    values, censor = s.surrogate_source("a")
    assert values is s.candidates["a"] and censor is None


def test_conditioning_and_censored_records_must_cover_the_candidates():
    s = _censored_sample()
    with pytest.raises(ValueError, match="conditioning"):
        SelectionSample(s.candidates, s.obs, s.time_positions, 0, conditioning={"a": s.conditioning["a"]})
    with pytest.raises(ValueError, match="censored"):
        SelectionSample(s.candidates, s.obs, s.time_positions, 0,
                        censored=CensoredSeries({"a": s.censored.uncensored["a"]}, {"a": 0.0}, 0.0))


def test_selection_rounds_test_censored_candidates_given_uncensored_selected_formulations():
    s = _censored_sample()
    tier = Tier("recording", RecordingTest({"a": 0.5, "b": 0.3, "c": 0.0}), 0.02)
    res = greedy_forward_selection(s, tier, 99, rng=np.random.default_rng(2))
    assert res.selected == ["a", "b"]
    (cands1, z1), (cands2, z2), (cands3, z3) = tier.test.calls
    for cands in (cands1, cands2, cands3):
        for nm, v in cands.items():
            assert np.array_equal(v, s.candidates[nm])                    # censored as candidates
    assert np.array_equal(z1, s.covariates)
    assert np.array_equal(z2, np.column_stack([s.covariates, s.conditioning["a"]]))
    assert np.array_equal(z3, np.column_stack([s.covariates, s.conditioning["a"], s.conditioning["b"]]))


def test_subsampling_keeps_conditioning_values_censoring_records_and_thresholds():
    s = _censored_sample()
    idx = np.arange(0, N, 3)
    sub = s.subset(idx, idx * 10, 3.0)
    assert np.array_equal(sub.candidates["b"], s.candidates["b"][idx])
    assert np.array_equal(sub.conditioning["b"], s.conditioning["b"][idx])
    assert np.array_equal(sub.censored.uncensored["b"], s.censored.uncensored["b"][idx])
    assert sub.censored.thresholds == s.censored.thresholds
    assert np.array_equal(sub.covariates, s.covariates[idx]) and np.array_equal(sub.time_positions, idx * 10)
    assert sub.grid_step == 3.0


def test_calibration_receives_the_uncensored_candidate_and_its_censoring():
    s = _censored_sample()
    seen = {}

    def calibration(candidate, obs, conditioning, time_positions, lag, rng, grid_step=None, censor=None, period_starts=None):
        seen.update(candidate=np.array(candidate), conditioning=np.array(conditioning), censor=censor)
        return {"calibrated": True, "n_surrogates": 1, "false_acceptances": 0, "p_value": 1.0}

    tier = Tier("recording", RecordingTest({"a": 0.5, "b": 0.0, "c": 0.0}), 0.02, calibration=calibration)
    greedy_forward_selection(s, tier, 99, rng=np.random.default_rng(3))
    assert np.array_equal(seen["candidate"], s.censored.uncensored["a"])
    assert np.array_equal(seen["conditioning"], s.covariates)
    probe = np.linspace(-3, 3, N)
    assert np.array_equal(seen["censor"](probe), np.maximum(probe, s.censored.thresholds["a"]))


def test_escalation_guard_uses_the_cmi_tiers_own_uncensored_conditioning_and_surrogate_source():
    from model_selection.selection.tiers import _guard_on_own_sample
    own = _censored_sample(seed=5)
    seen = {}

    def calibration(candidate, obs, conditioning, time_positions, lag, rng, grid_step=None, censor=None, period_starts=None):
        seen.update(candidate=np.array(candidate), obs=np.array(obs), conditioning=np.array(conditioning),
                    censor=censor)
        return {"calibrated": True}

    cmi = Tier("cmi", RecordingTest({}), 0.02, calibration=calibration, sample=own)
    guard = _guard_on_own_sample(cmi)
    guard(np.zeros(3), np.zeros(3), np.zeros((3, 1)), np.arange(3), 0, np.random.default_rng(0),
          names=("b", ["a"]), censor=None)
    assert np.array_equal(seen["candidate"], own.censored.uncensored["b"])
    assert np.array_equal(seen["obs"], own.obs)
    assert np.array_equal(seen["conditioning"], np.column_stack([own.covariates, own.conditioning["a"]]))
    probe = np.linspace(-3, 3, N)
    assert np.array_equal(seen["censor"](probe), np.maximum(probe, own.censored.thresholds["b"]))


def test_comparisons_with_other_candidates_condition_on_uncensored_values(monkeypatch):
    import model_selection.selection.equivalence as eq
    s = _censored_sample()
    tier = Tier("recording", RecordingTest({"a": 0.5, "b": 0.0, "c": 0.0}), 0.02)
    selection = greedy_forward_selection(s, tier, 99, rng=np.random.default_rng(4))
    assert selection.selected == ["a"]
    calls = []

    def fake_adds(tier, name, values, obs, z, *args, **kwargs):
        calls.append((name, np.array(values), np.array(z)))
        return {"p_value": 1.0, "effect": float("nan"), "adds": False, "untestable": False}

    monkeypatch.setattr(eq, "_adds", fake_adds)
    eq.information_equivalence(selection, s, tier, 99, 0.05, np.random.default_rng(5))
    for name, values, z in calls:
        assert np.array_equal(values, s.candidates[name])                 # tested censored
        other = "a" if name != "a" else None
        if other:                                                         # "c beyond a": a conditions uncensored
            assert np.array_equal(z, np.column_stack([s.covariates, s.conditioning["a"]]))
        else:                                                             # "a beyond c": c conditions uncensored
            assert any(np.array_equal(z, np.column_stack([s.covariates, s.conditioning[c]])) for c in ("b", "c"))
    assert len(calls) == 4                                                # two comparisons, both directions


def test_calibration_surrogates_are_built_uncensored_and_censored_at_the_candidates_threshold():
    from model_selection.statistics.conditional_independence.calibration import cmi_calibration
    rng = np.random.default_rng(6)
    n = 400
    z = rng.normal(size=(n, 2))
    uncensored = z.sum(axis=1) + rng.normal(size=n)
    t = Censoring(0.9).threshold(uncensored)
    surrogates = []

    class Capture:
        def sequential_p_value(self, surrogate, *args, **kwargs):
            surrogates.append(np.array(surrogate).ravel())
            return 0.0, np.zeros(1), 1.0

    for censor in (lambda v: Censoring.apply(v, t), None):
        surrogates.clear()
        cmi_calibration(Capture(), uncensored, rng.normal(size=n), z, np.arange(n), 0, 0.05, 0.02,
                        np.random.default_rng(7), censor=censor)
        assert len(surrogates) > 10
        for sur in surrogates:
            if censor is not None:
                assert sur.min() == t and np.mean(sur == t) == pytest.approx(0.9, abs=0.06)
            else:
                assert np.mean(sur == sur.min()) < 0.01                   # no point mass without censoring
