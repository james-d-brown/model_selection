"""
Information lost by thinning (docs/methods.md, Dependence between
samples): a tier run on a subsample of the record records, at each step,
the effective sample size and detectable R^2 the whole record would give,
and a stop that only the thinning makes undetectable is flagged.
"""
from dataclasses import replace

import numpy as np

from model_selection.pipeline import run_pooled_analysis
from model_selection.pipeline.results import pooled_result_to_dict, review_flags
from model_selection.selection import SelectionConfig
from tests.fixtures.basins import make_synthetic_basin


def _thinned_run():
    basin = make_synthetic_basin(seed=2)
    return run_pooled_analysis(basin, SelectionConfig(measure="nonlinear", nonlinear_max_n=800, n_permutations=60,
                                                      seed=4))


def test_a_thinned_tier_records_what_the_whole_record_would_give():
    result = _thinned_run()
    steps = result.selection.trajectory
    assert steps
    for st in steps:
        thin = st.thinning
        assert thin is not None and thin["n_samples"] == 800 and thin["n_samples_record"] > 20 * 800
        assert thin["effective_sample_size"] == st.effective_sample_size
        assert thin["effective_sample_size_record"] >= thin["effective_sample_size"]
        assert 0 < thin["information_retained"] <= 1
        assert thin["detectable_r2_record"] <= st.detectable_r2          # more effective samples, smaller effects
    written = pooled_result_to_dict(result)["pooled_trajectory"][0]["thinning"]
    assert written == steps[0].thinning


def test_unthinned_steps_have_no_thinning_record():
    basin = make_synthetic_basin(seed=2)
    result = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=60, seed=4))
    assert all(st.thinning is None for st in result.selection.trajectory)


def test_a_stop_that_only_thinning_makes_undetectable_is_flagged():
    result = _thinned_run()
    last = result.selection.trajectory[-1]
    thin = dict(last.thinning, detectable_r2_record=0.01, information_retained=0.1)
    stop = replace(last, accepted=False, detectable_r2=0.4, thinning=thin)
    sel = replace(result.selection, trajectory=result.selection.trajectory[:-1] + [stop])
    flagged = replace(result, selection=sel, meaningful_r2=0.05)
    (flag,) = [f for f in review_flags(flagged) if f["code"] == "thinning_limits_detection"]
    assert "800 of the record's" in flag["message"] and "about 10%" in flag["message"]
    assert "nonlinear_max_n" in flag["message"]
    # Undetectable on the whole record too: the thinning is not the reason, so no flag.
    whole = replace(stop, thinning=dict(thin, detectable_r2_record=0.2))
    sel = replace(result.selection, trajectory=result.selection.trajectory[:-1] + [whole])
    assert "thinning_limits_detection" not in [f["code"] for f in review_flags(replace(result, selection=sel,
                                                                                         meaningful_r2=0.05))]
