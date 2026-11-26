"""
Selection with censored candidates and observations (docs/methods.md,
Censoring at high flow): candidates enter tests censored, but every
conditioning set holds the selected formulations' uncensored values, in the
selection rounds and the comparisons with other candidates.
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
        assert name == "a"                                                # only "a beyond c" is tested
        assert np.array_equal(values, s.candidates[name])                 # tested censored
        # c conditions uncensored, with the covariates
        assert any(np.array_equal(z, np.column_stack([s.covariates, s.conditioning[c]])) for c in ("b", "c"))
    assert len(calls) == 2                                                # two comparisons, one direction


