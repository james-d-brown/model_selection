"""
The pooled pathway with censoring at high flow (docs/methods.md, Samples,
Censoring at high flow): the sample it builds, the selection it runs on it,
and the results it writes.
"""
import json
import os

import numpy as np
import pytest

import model_selection.pipeline.pooled.pathway as pooled_pathway
from model_selection.pipeline import OutputConfig, run_pooled_analysis
from model_selection.pipeline.results import outcome_line, pooled_result_to_dict
from model_selection.pipeline.shared import common_basis
from model_selection.preprocessing import Censoring
from model_selection.preprocessing.seasonal import seasonal_phase
from model_selection.preprocessing.transforms import prepare_series
from model_selection.selection import SelectionConfig
from tests.fixtures.basins import make_synthetic_basin


def _captured_sample(monkeypatch, basin, censoring):
    """The sample the pathway passes to the selection stage."""
    seen = {}
    real = pooled_pathway.run_selection_stage

    def capture(sample, *args, **kwargs):
        seen["sample"] = sample
        seen["tier_samples"] = kwargs.get("tier_samples")
        return real(sample, *args, **kwargs)

    monkeypatch.setattr(pooled_pathway, "run_selection_stage", capture)
    run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=20), rng=np.random.default_rng(0),
                        censoring=censoring)
    return seen


def test_censored_sample_censors_candidates_and_observations_and_conditions_on_uncensored_anomalies(monkeypatch):
    basin = make_synthetic_basin(seed=2)
    plain = _captured_sample(monkeypatch, basin, None)["sample"]
    c = Censoring(0.9)
    s = _captured_sample(monkeypatch, basin, c)["sample"]

    common, _ = common_basis(basin, [])
    rows = np.flatnonzero(np.isfinite(common.obs.values.astype(float)))
    assert np.array_equal(s.time_positions, plain.time_positions)
    assert np.array_equal(s.time_positions, rows)
    # Candidates and observations: the transformed series, censored at each series' own quantile over the
    # complete rows.
    for name, series in common.formulations.items():
        scores = prepare_series(series.values.astype(float))[rows]
        t = np.quantile(scores, 0.9)
        assert s.censored.thresholds[name] == pytest.approx(t)
        assert np.array_equal(s.censored.uncensored[name], scores)
        assert np.array_equal(s.candidates[name], np.maximum(scores, t))
        assert np.mean(s.candidates[name] == t) == pytest.approx(0.9, abs=0.002)
    obs_scores = prepare_series(common.obs.values.astype(float))[rows]
    assert np.array_equal(s.obs, np.maximum(obs_scores, np.quantile(obs_scores, 0.9)))
    # The selected formulations condition uncensored, exactly as in the uncensored pathway, with the phase.
    for name in common.formulations:
        assert np.array_equal(s.conditioning[name], plain.candidates[name])
    assert np.array_equal(s.covariates, plain.covariates)
    assert np.allclose(s.covariates, seasonal_phase(common.obs.index)[rows])
    assert s.dependence_lag == plain.dependence_lag                      # from the uncensored formulations


def test_censored_run_reports_its_censoring_and_writes_its_own_result_file(tmp_path):
    basin = make_synthetic_basin(seed=3)
    out = OutputConfig(output_dir=str(tmp_path), assumption_plots=False, dependence_plot=False)
    sel = SelectionConfig(measure="linear", n_permutations=30, seed=11)
    plain = run_pooled_analysis(basin, sel, output=out)
    censored = run_pooled_analysis(basin, sel, output=out, censoring=Censoring(0.9))
    assert plain.censoring is None
    rec = censored.censoring
    assert rec["quantile"] == 0.9 and set(rec["thresholds"]) == set(basin.formulations)
    assert all(abs(v - 0.9) < 0.002 for v in rec["share_at_threshold"].values())
    assert any("Censoring at high flow" in n for n in censored.notes)
    files = sorted(os.listdir(tmp_path))
    assert f"{basin.gage}_pooled_result.json" in files and f"{basin.gage}_pooled_censored_result.json" in files
    with open(tmp_path / f"{basin.gage}_pooled_censored_result.json") as f:
        written = json.load(f)
    assert written["censoring"]["quantile"] == 0.9
    with open(tmp_path / f"{basin.gage}_pooled_result.json") as f:
        assert json.load(f)["censoring"] is None
    line = outcome_line(pooled=plain, pooled_censored=censored)
    assert "pooled {" in line and "pooled, censored {" in line and "all runs" not in line
    assert "both pathways" in line                                       # two runs
    assert outcome_line(pooled=censored).startswith(f"{basin.gage}: pooled, censored")
    with pytest.raises(ValueError, match="censoring"):
        outcome_line(pooled_censored=plain)


def test_censored_cascade_runs_its_subsamples_and_the_cmi_calibration_on_censored_data(monkeypatch):
    """The cascade on a long record: the nonlinear and CMI tiers get their own
    subsamples, which keep the censoring; the CMI calibration's surrogates
    have the candidate's point mass."""
    import model_selection.statistics.conditional_independence.calibration as cal
    seen = {"surrogate_shares": []}
    real_cal = cal.cmi_calibration

    def watch(test, candidate, obs, conditioning, *args, censor=None, **kwargs):
        seen["censor_given"] = censor is not None
        def censoring_watch(v):
            out = censor(v)
            seen["surrogate_shares"].append(float(np.mean(out == out.min())))
            return out
        return real_cal(test, candidate, obs, conditioning, *args, censor=censoring_watch if censor else None,
                        **kwargs)

    import model_selection.selection.tiers as tiers
    monkeypatch.setattr(tiers, "cmi_calibration", watch)
    basin = make_synthetic_basin(seed=4)
    seen_stage = {}
    real = pooled_pathway.run_selection_stage

    def capture(sample, *args, **kwargs):
        seen_stage["tier_samples"] = kwargs.get("tier_samples")
        return real(sample, *args, **kwargs)

    monkeypatch.setattr(pooled_pathway, "run_selection_stage", capture)
    result = run_pooled_analysis(basin, SelectionConfig(measure="cmi", n_permutations=50, cmi_max_n=400),
                                 rng=np.random.default_rng(5), censoring=Censoring(0.9))
    sub = seen_stage["tier_samples"]["cmi"]
    assert len(sub.obs) <= 400 and sub.censored is not None and sub.conditioning is not None
    for name in sub.candidates:
        t = sub.censored.thresholds[name]
        assert np.array_equal(sub.candidates[name], np.maximum(sub.censored.uncensored[name], t))
    assert result.censoring is not None and result.selection is not None
    # The informative formulation is accepted, so the calibration runs on its 52 censored surrogates.
    assert any(st.accepted and st.cmi_calibration for st in result.selection.trajectory)
    assert seen["censor_given"] and len(seen["surrogate_shares"]) == 52
    # A surrogate is the candidate's fit plus permuted residuals, censored at the candidate's threshold, so its
    # share at the threshold is near, not exactly, the quantile's 0.9; every one keeps a large point mass.
    assert min(seen["surrogate_shares"]) > 0.75


def test_run_basin_runs_the_censored_pooled_pathway_when_given_censoring():
    from model_selection.pipeline import basin_result_to_dict, run_basin
    basin = make_synthetic_basin(seed=3)
    sel = SelectionConfig(measure="linear", n_permutations=30, seed=11)
    result = run_basin(basin, sel, run_high_flow=False, censoring=Censoring(0.9))
    assert result.pooled.censoring is None and result.pooled_censored.censoring["quantile"] == 0.9
    alone = run_pooled_analysis(basin, sel, censoring=Censoring(0.9))
    assert result.pooled_censored.selection.selected == alone.selection.selected
    d = basin_result_to_dict(result)
    assert d["pooled_censored"]["censoring"]["quantile"] == 0.9
    assert all(f["pathway"] in ("pooled", "pooled_censored") for f in d["review_flags"])
    assert run_basin(basin, sel, run_high_flow=False).pooled_censored is None
