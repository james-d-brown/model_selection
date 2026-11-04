"""Tests of review flags in results and of the cross-basin summary."""
import json

from model_selection.io.summary import summarize_result_files, write_summary_csv
from model_selection.pipeline.results import PooledResult, pooled_result_to_dict, review_flags
from model_selection.selection.greedy import SelectionResult, SelectionStep


def _pooled(flagged: bool, gage="G1"):
    step = SelectionStep(formulation="m1", association=0.3, p_value=0.01, accepted=True, n_selected_before=1,
                         n_candidates_tested=2, dependence_lag=3, samples_per_dimension=5.0,
                         low_samples_per_dimension=flagged)
    sel = SelectionResult(selected=["m0", "m1"], trajectory=[step], stopped_reason="all candidates selected")
    return PooledResult(gage=gage, n_timesteps=100, zero_flow_fraction={}, zero_flow_flag=False, selection=sel,
                        min_temporal_lag=3, measure="cascade", committed_tier="cmi", cmi_calibration=True)


def test_flagged_step_produces_a_prominent_review_flag():
    d = pooled_result_to_dict(_pooled(True))
    assert list(d)[:2] == ["review_required", "review_flags"]
    assert d["review_required"] is True
    flag = d["review_flags"][0]
    assert flag["code"] == "low_samples_per_dimension"
    assert "m1" in flag["message"] and "5.0 samples per dimension" in flag["message"]
    assert d["pooled_selected"] == ["m0", "m1"]  # flags never change the selection


def test_no_flags_when_nothing_is_flagged():
    assert review_flags(_pooled(False)) == []
    assert pooled_result_to_dict(_pooled(False))["review_required"] is False


def test_summary_lists_flagged_results_first(tmp_path):
    for gage, flagged in (("A", False), ("B", True)):
        (tmp_path / f"{gage}_pooled_result.json").write_text(json.dumps(pooled_result_to_dict(_pooled(flagged, gage))))
    rows = summarize_result_files(str(tmp_path))
    assert [r["gage"] for r in rows] == ["B", "A"]
    assert rows[0]["review_required"] and rows[0]["flag_codes"] == "low_samples_per_dimension"
    out = write_summary_csv(rows, str(tmp_path / "summary.csv"))
    assert open(out).read().splitlines()[0].startswith("review_required,gage,pathway")


def test_low_samples_per_dimension_produces_a_labelled_heuristic_flag():
    result = _pooled(False)
    result.selection.trajectory[0].low_samples_per_dimension = True
    result.selection.trajectory[0].samples_per_dimension = 6.5
    flags = review_flags(result)
    assert [f["code"] for f in flags] == ["low_samples_per_dimension"]
    assert "6.5 samples per dimension" in flags[0]["message"] and "heuristic" in flags[0]["message"]


def test_non_decaying_dependence_produces_a_review_flag_from_test_or_check():
    result = _pooled(False)
    step = result.selection.trajectory[0]
    step.dependence = {"block_length": 500, "estimated": True, "decorrelation_lag": 900, "dependence_decays": False}
    step.adequacy = [{"check": "nonlinear candidate-observation relation",
                      "dependence": {"block_length": 400, "estimated": True, "decorrelation_lag": 800,
                                     "dependence_decays": False}}]
    flags = [f for f in review_flags(result) if f["code"] == "dependence_not_decaying"]
    assert len(flags) == 2
    assert "cmi tier test" in flags[0]["message"] and "nonlinear candidate-observation relation" in flags[1]["message"]


def test_no_dependence_flag_when_the_lag_is_not_estimated():
    # Event samples and spaced subsamples use the caller's lag: nothing to flag.
    result = _pooled(False)
    result.selection.trajectory[0].dependence = {"block_length": 3, "estimated": False, "dependence_decays": True}
    assert review_flags(result) == []


def test_dependent_samples_in_a_local_permutation_produce_a_review_flag():
    result = _pooled(False)
    result.selection.trajectory[0].dependence = {"estimated": True, "independent": False, "decorrelation_lag": 300}
    flags = review_flags(result)
    assert [f["code"] for f in flags] == ["dependent_samples_permuted"]
    assert "300" in flags[0]["message"]


def test_few_blocks_produces_a_labelled_heuristic_flag():
    result = _pooled(False)
    result.selection.trajectory[0].dependence = {"block_length": 40, "estimated": True, "decorrelation_lag": 30,
                                                 "dependence_decays": True, "n_blocks": 15.0, "few_blocks": True}
    flags = review_flags(result)
    assert [f["code"] for f in flags] == ["few_blocks"]
    assert "15 blocks" in flags[0]["message"] and "minimum of 50" in flags[0]["message"] and "% too small" in flags[0]["message"]


def test_a_pooled_cmi_selection_without_the_calibration_check_is_flagged():
    result = _pooled(False)
    result.cmi_calibration = False
    assert [f["code"] for f in review_flags(result)] == ["cmi_calibration_not_checked"]


def test_short_crossfit_blocks_and_capped_factors_produce_review_flags():
    result = _pooled(False)
    result.selection.trajectory[0].dependence = {"block_length": 3, "estimated": False, "dependence_decays": True,
                                                 "crossfit_blocks_short": True, "low_rank_material": True,
                                                 "low_rank_rank_change": 0.02}
    flags = review_flags(result)
    assert [f["code"] for f in flags] == ["crossfit_blocks_short", "low_rank_material"]
    result.selection.trajectory[0].dependence.update(crossfit_blocks_short=False, low_rank_material=False)
    assert review_flags(result) == []


def test_selected_encompassed_flag_requires_the_rounds_bonferroni_level():
    result = _pooled(False)

    def record(p):
        return [{"round": 1, "selected": "m1", "flag_level": 0.05 / 6, "comparisons": [{
            "candidate": "m2", "relation": "encompasses_selected",
            "candidate_adds_beyond_selected": {"p_value": p}, "selected_adds_beyond_candidate": {"p_value": 0.4}}]}]
    result.selection.equivalence = record(0.03)
    assert "selected_encompassed" not in [f["code"] for f in review_flags(result)]
    result.selection.equivalence = record(0.001)
    assert "selected_encompassed" in [f["code"] for f in review_flags(result)]


def test_outcome_line_states_each_pathway_and_the_union():
    from model_selection.pipeline.results import outcome_line
    pooled = _pooled(False)
    pooled.selection.equivalence = [{"round": 1, "selected": "m1", "flag_level": 0.025, "comparisons": [
        {"candidate": "m2", "relation": "encompassed_by_selected",
         "candidate_adds_beyond_selected": {"p_value": 0.4}, "selected_adds_beyond_candidate": {"p_value": 0.001}}]}]
    line = outcome_line(pooled=pooled)
    assert line.startswith("G1: pooled {m1}: none replaceable")
    pooled.selection.equivalence[0]["comparisons"][0]["relation"] = "equivalent"
    assert "replaceable m1 by m2 (equivalent)" in outcome_line(pooled=pooled)


def test_a_non_rejecting_check_on_too_few_samples_is_flagged():
    result = _pooled(False)
    result.selection.trajectory[0].adequacy = [{"check": "nonlinear dependence", "adequate": True, "dependence": {
        "estimated": False, "too_few_to_assess": True, "n_samples": 11}}]
    flags = [f for f in review_flags(result) if f["code"] == "adequacy_check_too_few_samples"]
    assert flags and "11 samples" in flags[0]["message"]
    result.selection.trajectory[0].adequacy[0]["adequate"] = False
    assert not [f for f in review_flags(result) if f["code"] == "adequacy_check_too_few_samples"]


def test_few_blocks_of_event_samples_produce_a_review_flag():
    step = SelectionStep(formulation="m1", association=0.3, p_value=0.01, accepted=True, n_selected_before=0,
                         n_candidates_tested=2, dependence_lag=730, samples_per_dimension=20.0,
                         low_samples_per_dimension=False,
                         dependence={"estimated": False, "n_blocks": 10.0, "few_blocks": True,
                                     "too_few_to_assess": False})
    sel = SelectionResult(selected=["m1"], trajectory=[step], stopped_reason="all candidates selected")
    result = PooledResult(gage="G", n_timesteps=1000, zero_flow_fraction={}, zero_flow_flag=False, selection=sel,
                          min_temporal_lag=730, measure="linear")
    flags = [f for f in review_flags(result) if f["code"] == "few_blocks"]
    assert len(flags) == 1 and "10.0" in flags[0]["message"] and "events" in flags[0]["message"]



def test_few_clusters_and_correlated_cluster_terms_produce_their_flags():
    result = _pooled(False)
    result.selection.trajectory[0].dependence = {
        "block_length": 2, "estimated": False, "n_blocks": 30.0, "few_blocks": True, "independent_units": True,
        "cluster_terms_correlation": 0.45, "cluster_terms_standard_error": 0.18, "n_clusters": 30,
        "cluster_terms_correlated": True}
    flags = {f["code"]: f["message"] for f in review_flags(result)}
    assert set(flags) == {"few_blocks", "event_terms_correlated"}
    assert "30 independent clusters" in flags["few_blocks"] and "MacKinnon and Webb" in flags["few_blocks"]
    assert "0.45" in flags["event_terms_correlated"] and "30 clusters" in flags["event_terms_correlated"]
    result.selection.trajectory[0].dependence.update(cluster_terms_correlated=False, n_blocks=80.0, few_blocks=False)
    assert review_flags(result) == []
