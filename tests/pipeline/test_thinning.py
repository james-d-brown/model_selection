"""
Information lost by thinning (docs/methods.md, Dependence between
samples): a tier run on a subsample of the record records, at each step,
the effective sample size the whole record would give and the share of it
the subsample keeps.
"""
import pytest

from model_selection.pipeline import run_pooled_analysis
from model_selection.pipeline.results import pooled_result_to_dict
from model_selection.selection import SelectionConfig
from tests.fixtures.basins import make_synthetic_basin


def _thinned_run():
    basin = make_synthetic_basin(seed=2)
    return run_pooled_analysis(basin, SelectionConfig(measure="nonlinear", nonlinear_max_n=800, n_permutations=60,
                                                      seed=4))


@pytest.mark.integration
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
        assert "detectable_r2_record" not in thin
    written = pooled_result_to_dict(result)["pooled_trajectory"][0]["thinning"]
    assert written == steps[0].thinning


@pytest.mark.integration
def test_unthinned_steps_have_no_thinning_record():
    basin = make_synthetic_basin(seed=2)
    result = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=60, seed=4))
    assert all(st.thinning is None for st in result.selection.trajectory)
