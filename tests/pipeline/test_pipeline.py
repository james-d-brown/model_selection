"""
End-to-end tests of both pathways and their orchestration on synthetic
basins (no parquet or USGS access): real event detection, FPCA, selection
and serialization. The selection mechanism itself is tested in
tests/selection.
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model_selection.event_detection import EventConfig, POTConfig
from model_selection.io.basin import BasinData
from model_selection.pipeline import (
    FPCAConfig,
    OutputConfig,
    basin_result_to_dict,
    high_flow_result_to_dict,
    pooled_result_to_dict,
    run_basin,
    run_high_flow_analysis,
    run_pooled_analysis,
)
from model_selection.pipeline.high_flow import outputs as hf_outputs
from model_selection.pipeline.high_flow import pathway as hf_pathway
from model_selection.pipeline.high_flow.diagnostics import basis_sensitivity_check
from model_selection.selection import (
    AdequacyConfig, SelectionConfig, SelectionSample, build_cascade, build_tier, effect_size_floor_for_tier,
    run_tiered_selection,
)
from model_selection.selection.greedy import SelectionResult, SelectionStep
from model_selection.statistics.association.canonical import partial_cca_structural_adequacy
from tests.fixtures.basins import make_synthetic_basin, make_synthetic_basin_with_gap

HF_EVENTS = EventConfig()


def test_pipeline_runs_end_to_end_and_selects_informative_formulation():
    basin = make_synthetic_basin()
    result = run_basin(
        basin, SelectionConfig(n_neighbors=10, n_permutations=150, alpha=0.05, effect_size_floor=0.02),
        events=EventConfig(high_flow_quantile=0.90), rng=np.random.default_rng(99),
    )
    assert result.gage == "TESTGAGE01"
    assert "formulation_informative" in result.pooled.selection.selected
    assert "formulation_noise" not in result.pooled.selection.selected
    assert result.high_flow is not None
    for pathway in (result.pooled, result.high_flow):
        r = pathway.pearson_correlation
        assert set(r) == {"formulation_informative", "formulation_noise"}
        assert r["formulation_informative"] > r["formulation_noise"]
        e = pathway.selection.unconditional_effect
        assert set(e) == set(r) and e["formulation_informative"] > e["formulation_noise"]
    t = result.pooled.pearson_correlation_transformed
    assert set(t) == set(result.pooled.pearson_correlation) and t["formulation_informative"] > t["formulation_noise"]
    json.dumps(basin_result_to_dict(result))


def test_pipeline_flags_high_zero_flow():
    basin = make_synthetic_basin(seed=1)
    idx = basin.obs.index
    rng = np.random.default_rng(2)
    basin.formulations["formulation_zero_heavy"] = pd.Series(
        np.where(rng.random(len(idx)) < 0.5, 0.0, rng.uniform(1, 5, len(idx))), index=idx)
    result = run_basin(basin, SelectionConfig(n_permutations=50), rng=np.random.default_rng(3))
    assert result.zero_flow_flag is True
    assert result.zero_flow_fraction["formulation_zero_heavy"] > 0.3


@pytest.mark.parametrize("measure", ["linear", "nonlinear", "cmi"])
def test_pipeline_runs_end_to_end_across_all_measures(measure):
    basin = make_synthetic_basin(seed=4)
    # The CMI calibration check is tested on its own (tests/selection); here only the plumbing of each measure.
    result = run_basin(basin, SelectionConfig(measure=measure, n_permutations=50, cmi_calibration=False),
                       events=HF_EVENTS, rng=np.random.default_rng(5))
    assert result.pooled.measure == measure
    assert result.high_flow is not None and result.high_flow.measure == measure
    as_dict = basin_result_to_dict(result)
    assert as_dict["pooled"]["measure"] == measure
    assert as_dict["high_flow"]["measure"] == measure
    json.dumps(as_dict)


def test_pipeline_run_high_flow_false_skips_high_flow_pathway():
    result = run_basin(make_synthetic_basin(seed=6), SelectionConfig(n_permutations=50), run_high_flow=False,
                       rng=np.random.default_rng(7))
    assert result.high_flow is None
    as_dict = basin_result_to_dict(result)
    assert as_dict["high_flow"] is None
    json.dumps(as_dict)


def test_config_objects_validate_their_values():
    with pytest.raises(ValueError, match="measure"):
        SelectionConfig(measure="bogus")
    with pytest.raises(ValueError, match="method"):
        EventConfig(method="bogus")
    with pytest.raises(ValueError, match="consensus"):
        EventConfig(consensus="bogus")


def test_pipeline_cascade_escalates_beyond_linear_through_real_api():
    """A U-shaped candidate effect, which a straight line cannot detect, makes
    the linear tier inadequate (nonlinear-relation check); the cascade escalates and writes
    adequacy plots for the rejected linear tier."""
    rng = np.random.default_rng(0)
    n = 600
    context = np.clip(rng.normal(scale=2, size=n) + 10, 0.01, None)
    x = np.clip(rng.normal(scale=2, size=n) + 10, 0.01, None)
    base = (x - 10) ** 2
    obs_vals = np.clip(1.5 * context + 2.0 * base + rng.normal(scale=0.5, size=n) + 20, 0.01, None)
    basin = BasinData(gage="CASCADETEST", obs=pd.Series(obs_vals),
                      formulations={"context_model": pd.Series(context), "kinked_model": pd.Series(x)})
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pooled_analysis(
            # The CMI calibration check is tested on its own; here only the escalation.
            basin, SelectionConfig(measure="cascade", n_permutations=100, min_temporal_lag=5, cmi_calibration=False),
            AdequacyConfig(n_permutations=100), OutputConfig(output_dir=tmpdir), rng=np.random.default_rng(1))
        first = result.attempted_tiers[0]
        assert (first["tier"], first["committed"]) == ("linear", False)
        assert "rejected:" in first["reason"]
        assert result.committed_tier in ("nonlinear", "cmi")
        assert first["diagnostic_plot_paths"]
        for path in first["diagnostic_plot_paths"]:
            assert os.path.getsize(path) > 0
        as_dict = pooled_result_to_dict(result)
        json.dumps(as_dict)
        assert as_dict["committed_tier"] == result.committed_tier


def test_pipeline_cascade_runs_on_both_pathways():
    basin = make_synthetic_basin(seed=8, n_hours=3000, n_events=15)
    result = run_basin(basin, SelectionConfig(measure="cascade", n_permutations=30), events=HF_EVENTS,
                       rng=np.random.default_rng(9))
    assert result.pooled.committed_tier in ("linear", "nonlinear", "cmi")
    assert result.high_flow.committed_tier in ("linear", "nonlinear", "cmi")
    assert len(result.high_flow.attempted_tiers) >= 1
    json.dumps(basin_result_to_dict(result))


def test_high_flow_selection_override():
    basin = make_synthetic_basin(seed=10, n_hours=1500, n_events=8)
    result = run_basin(basin, SelectionConfig(measure="cascade", n_permutations=30),
                       events=EventConfig(),
                       high_flow_selection=SelectionConfig(measure="cmi", n_permutations=30),
                       rng=np.random.default_rng(11))
    assert result.pooled.committed_tier in ("linear", "nonlinear", "cmi")
    assert result.high_flow.measure == "cmi"


def test_structural_check_belongs_to_the_linear_tier_only():
    assert build_tier("linear", SelectionConfig()).structural_check is partial_cca_structural_adequacy
    assert build_tier("nonlinear", SelectionConfig()).structural_check is None
    assert build_tier("cmi", SelectionConfig()).structural_check is None
    with pytest.raises(ValueError):
        build_tier("cascade", SelectionConfig())


def test_pooled_linear_reports_failed_magnitude_gate_in_result_and_json():
    rng = np.random.default_rng(0)
    n = 40
    basin = BasinData(gage="MAGNITUDETEST", obs=pd.Series(np.clip(rng.normal(scale=2, size=n) + 10, 0.01, None)),
                      formulations={"weak_model": pd.Series(np.clip(rng.normal(scale=2, size=n) + 10, 0.01, None))})
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pooled_analysis(
            basin, SelectionConfig(measure="linear", n_permutations=50, min_temporal_lag=1, effect_size_floor=0.02),
            output=OutputConfig(output_dir=tmpdir), rng=np.random.default_rng(1))
        assert result.selection.ill_posed is False
        assert result.selection.selected == []
        step = result.selection.trajectory[0]
        assert step.magnitude_unsubstantiated is True and step.accepted is False
        as_dict = pooled_result_to_dict(result)
        assert as_dict["ill_posed"] is False
        assert as_dict["pooled_trajectory"][0]["magnitude_unsubstantiated"] is True
        with open(result.result_json_path) as f:
            assert json.load(f)["pooled_trajectory"][0]["magnitude_unsubstantiated"] is True


def test_basis_sensitivity_check_skips_discarded_trials():
    """A trial discarded at round 2 still has round 1's accepted step last in
    its trajectory; the check must not mistake it for a rejection stop."""
    accepted_step = SelectionStep(formulation="model_a", association=0.5, p_value=0.001, accepted=True,
                                  n_selected_before=0, n_candidates_tested=1, dependence_lag=1)

    tier = build_tier("linear", SelectionConfig())
    for flags in ({"ill_posed": True}, {"tier_rejected": True}):
        selection = SelectionResult(selected=[], trajectory=[accepted_step], stopped_reason="test",
                                    unconditional_association={"model_a": 0.5}, **flags)
        assert basis_sensitivity_check(
            raw_formulations={}, obs=np.array([]), events=[], retained_event_ids=[],
            selection=selection, time_positions=np.array([]), tier=tier,
            n_permutations=10, alpha=0.05, dependence_lag=1, n_components=2,
            fpca_config=FPCAConfig(), reconstruction_by_series={}, rng=np.random.default_rng(0),
        ) is None


def test_cascade_converts_effect_size_floor_per_tier():
    tiers = {t.name: t for t in build_cascade(SelectionConfig(effect_size_floor=0.02), AdequacyConfig(),
                                              multivariate=False)}
    assert tiers["linear"].effect_size_floor == 0.02
    assert tiers["nonlinear"].effect_size_floor == 0.02
    assert tiers["cmi"].effect_size_floor == effect_size_floor_for_tier(0.02, "cmi") < 0.015


def test_independent_multivariate_candidate_rejected_by_linear_tier():
    """n=70, p=q=4 independent data: the linear tier is well posed, commits,
    and rejects the candidate through the ordinary gates."""
    rng = np.random.default_rng(0)
    n, p, q = 70, 4, 4
    sample = SelectionSample({"noise_model": rng.normal(size=(n, p))}, rng.normal(size=(n, q)), np.arange(n), 1)
    tiers = build_cascade(SelectionConfig(effect_size_floor=0.02), AdequacyConfig(), multivariate=True, workers=1)
    result = run_tiered_selection(sample, tiers, 99, rng=np.random.default_rng(1))
    assert result.committed_tier == "linear" and not result.ill_posed
    assert result.selection.selected == []
    step = result.selection.trajectory[0]
    assert not step.accepted
    assert abs(step.effect - (step.association - step.null_mean)) < 1e-9


def test_multivariate_adequacy_plots_written_through_real_cascade():
    from model_selection.pipeline.shared import adequacy_plot_factory
    with tempfile.TemporaryDirectory() as tmpdir:
        tiers = build_cascade(SelectionConfig(effect_size_floor=0.02), AdequacyConfig(n_permutations=50),
                              multivariate=True, workers=1,
                              plot_factory=adequacy_plot_factory(OutputConfig(output_dir=tmpdir), "PLOTTEST", "high_flow"))
        rng = np.random.default_rng(0)
        n, p, q = 100, 4, 4
        candidate = rng.normal(size=(n, p))
        obs = candidate @ rng.normal(size=(p, q)) * 0.5 + rng.normal(scale=0.3, size=(n, q))
        result = run_tiered_selection(SelectionSample({"good_model": candidate}, obs, np.arange(n), 1), tiers, 50,
                                      rng=np.random.default_rng(1))
        assert result.committed_tier == "linear"
        assert result.selection.diagnostic_plot_paths
        for path in result.selection.diagnostic_plot_paths:
            assert os.path.getsize(path) > 1000


def test_wasp_runs_end_to_end_and_selects_informative_formulation():
    """WASP finds about one event per storm, so the record needs enough storms
    for the test to have power: 45 here (the default fixture has 15)."""
    result = run_high_flow_analysis(
        make_synthetic_basin(n_hours=9000, n_events=45),
        SelectionConfig(measure="linear", n_permutations=200, effect_size_floor=0.02),
        events=EventConfig(method="wasp", high_flow_quantile=0.90),
        rng=np.random.default_rng(99))
    assert result.selection is not None
    assert "formulation_informative" in result.selection.selected
    assert "formulation_noise" not in result.selection.selected


def test_min_peak_separation_accepted_by_high_flow_pathway():
    result = run_high_flow_analysis(make_synthetic_basin(), SelectionConfig(measure="linear", n_permutations=20),
                                    events=EventConfig(pot=POTConfig(min_peak_separation=6)),
                                    rng=np.random.default_rng(1))
    assert result.n_raw_consensus_events >= 0


def _real_result(basin):
    return run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=10),
                                  rng=np.random.default_rng(1))


def test_per_model_diagnostics_write_one_file_per_model():
    import model_selection.visualization.html as html_module
    import model_selection.visualization.hydrograph as hydro_module
    from model_selection.event_detection.pot.detector import detect_model_events
    basin = make_synthetic_basin()
    raw = {name: s.values for name, s in basin.formulations.items()}
    own_windows = {name: detect_model_events(v, quantile_threshold=0.9, recession_ratio=0.25) for name, v in raw.items()}
    result = _real_result(basin)
    with patch.object(hydro_module, "plot_basin_diagnostics", return_value=MagicMock()) as mock_plot, \
            patch.object(html_module, "write_standalone_html") as mock_write, \
            tempfile.TemporaryDirectory() as tmpdir:
        _, _, per_model = hf_outputs.write_high_flow_outputs(
            basin, result, OutputConfig(output_dir=tmpdir, per_model_plots=True),
            own_windows=own_windows, raw_formulations=raw)
    assert mock_plot.call_count == len(own_windows)
    assert mock_write.call_count == len(own_windows)
    assert set(per_model) == set(own_windows)
    for model_name, path in per_model.items():
        assert path.endswith(f"TESTGAGE01_high_flow_diagnostic_{model_name}.html")


def test_per_model_diagnostics_off_by_default():
    import model_selection.visualization.hydrograph as hydro_module
    basin = make_synthetic_basin()
    result = _real_result(basin)
    with patch.object(hydro_module, "plot_basin_diagnostics", return_value=MagicMock()) as mock_plot, \
            tempfile.TemporaryDirectory() as tmpdir:
        _, _, per_model = hf_outputs.write_high_flow_outputs(basin, result, OutputConfig(output_dir=tmpdir))
    assert mock_plot.call_count == 0
    assert per_model == {}


def test_high_flow_pathway_passes_own_windows_to_output_writer():
    basin = make_synthetic_basin()
    with patch.object(hf_pathway, "write_high_flow_outputs", wraps=hf_pathway.write_high_flow_outputs) as mock_out, \
            tempfile.TemporaryDirectory() as tmpdir:
        run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=10),
                               output=OutputConfig(output_dir=tmpdir, per_model_plots=True, event_gallery=False,
                                                   assumption_plots=False),
                               rng=np.random.default_rng(1))
    assert mock_out.call_count == 1
    _, kwargs = mock_out.call_args
    assert set(kwargs["own_windows"]) == set(basin.formulations)
    assert kwargs["raw_formulations"] is not None


@pytest.mark.parametrize("method", ["pot", "wasp"])
def test_high_flow_pathway_handles_a_data_gap(method):
    result = run_high_flow_analysis(
        make_synthetic_basin_with_gap(), SelectionConfig(measure="linear", n_permutations=10, effect_size_floor=0.02),
        events=EventConfig(method=method, high_flow_quantile=0.90),
        rng=np.random.default_rng(1))
    assert result.selection is not None


def test_regularization_does_not_alter_association_input():
    """Interpolated values must never reach estimation: a gap-free basin and
    the same basin with a gap select the same formulation. WASP finds about one
    event per storm, so the record has 45 storms for the test to have power."""
    def run(basin):
        return run_high_flow_analysis(
            basin, SelectionConfig(measure="linear", n_permutations=200, effect_size_floor=0.02),
            events=EventConfig(method="wasp", high_flow_quantile=0.90),
            rng=np.random.default_rng(1))
    for result in (run(make_synthetic_basin(n_hours=9000, n_events=45)),
                   run(make_synthetic_basin_with_gap(n_hours=9000, n_events=45))):
        assert "formulation_informative" in result.selection.selected
        assert "formulation_noise" not in result.selection.selected


def test_high_flow_reports_detectability_and_flags_unresolvable_threshold():
    # Enough events for the detectable contribution to be below R^2 = 1.
    basin = make_synthetic_basin(n_hours=10000, n_events=40, seed=4)
    unset = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=200),
                                   rng=np.random.default_rng(1))
    d = high_flow_result_to_dict(unset)
    assert 0 < unset.detectable_r2 < 1 and unset.selection is not None
    assert "80% power" in d["detectability"]["statement"]
    assert d["detectability"]["unexplained_variation_pct"] == pytest.approx(100 * unset.detectable_r2)
    assert all(s["detectability"] is not None for s in d["high_flow_trajectory"])
    assert "cannot be excluded with 80% power" in unset.selection.stopped_reason
    assert "meaningful_effect_undetectable" not in [f["code"] for f in d["review_flags"]]
    # A threshold below what these data can detect.
    strict = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=200,
                                                           meaningful_r2=unset.detectable_r2 / 2),
                                    rng=np.random.default_rng(1))
    assert strict.selection is not None  # the threshold never stops selection
    codes = [f["code"] for f in high_flow_result_to_dict(strict)["review_flags"]]
    assert "meaningful_effect_undetectable" in codes
    json.dumps(high_flow_result_to_dict(strict))


def test_meaningful_r2_is_validated():
    with pytest.raises(ValueError, match="meaningful_r2"):
        SelectionConfig(meaningful_r2=1.5)


def test_pathways_use_the_largest_decorrelation_lag():
    from model_selection.pipeline.shared import max_decorrelation_lag
    from model_selection.preprocessing.transforms import prepare_series
    basin = make_synthetic_basin(seed=8)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    hf = run_high_flow_analysis(basin, sel, events=EventConfig(method="wasp", high_flow_quantile=0.90),
                                rng=np.random.default_rng(3))
    prepared = {k: prepare_series(s.values, rng=np.random.default_rng(0), index=s.index,
                                  seasonal_window_days=11) for k, s in basin.formulations.items()}
    lag_hf, _ = max_decorrelation_lag(prepared, sel.dependence_run_length)
    assert abs(hf.min_temporal_lag - lag_hf) <= max(2, 0.2 * lag_hf)  # jittered ranks differ slightly
    pooled = run_pooled_analysis(basin, sel, rng=np.random.default_rng(4))
    transformed = {k: prepare_series(s.values, rng=np.random.default_rng(0), index=s.index,
                                     seasonal_window_days=11)
                   for k, s in basin.formulations.items()}
    lag, _ = max_decorrelation_lag(transformed, sel.dependence_run_length)
    assert abs(pooled.min_temporal_lag - lag) <= max(2, 0.2 * lag)  # jittered ranks differ slightly


def test_pooled_seasonal_conditioning_is_on_by_default_and_can_be_disabled():
    from model_selection.pipeline import SeasonalConfig
    from model_selection.pipeline.results import pooled_result_to_dict
    basin = make_synthetic_basin(seed=9)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    on = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1))
    off = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1), seasonal=SeasonalConfig(enabled=False))
    assert pooled_result_to_dict(on)["seasonal_conditioning"] == {"enabled": True, "window_days": 11, "record_too_short": False}
    assert pooled_result_to_dict(off)["seasonal_conditioning"] == {"enabled": False, "window_days": None, "record_too_short": False}
    assert any("Conditioning on the time of year:" in n for n in on.notes)
    assert any("disabled" in n for n in off.notes)


def test_seasonal_conditioning_is_disabled_on_a_record_shorter_than_three_years():
    """On a 125-day record the calendar day does not recur, so neither pathway
    conditions on it; both say why and raise a review flag."""
    from model_selection.pipeline.results import review_flags
    from model_selection.pipeline.high_flow.pathway import run_high_flow_analysis
    basin = make_synthetic_basin(n_hours=3000, seed=2)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    pooled = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1))
    high_flow = run_high_flow_analysis(basin, sel, rng=np.random.default_rng(1))
    assert pooled_result_to_dict(pooled)["seasonal_conditioning"] == {"enabled": False, "window_days": None,
                                                                      "record_too_short": True}
    assert high_flow_result_to_dict(high_flow)["seasonal_conditioning"]["enabled"] is False
    for result in (pooled, high_flow):
        assert any("fewer than the 3 needed to estimate the seasonal cycle" in n for n in result.notes)
        assert "seasonal_record_too_short" in [f["code"] for f in review_flags(result)]


def test_pooled_tests_condition_on_the_seasonal_phase():
    """Every pooled test conditions on (sin, cos) of the day of year when the
    seasonal conditioning is on, and on nothing extra when it is off."""
    from model_selection.pipeline import SeasonalConfig
    from model_selection.pipeline.pooled import pathway as pooled_pathway
    from model_selection.preprocessing.seasonal import seasonal_phase
    basin = make_synthetic_basin(seed=12)
    captured = []
    real = pooled_pathway.run_selection_stage

    def spy(sample, *args, **kwargs):
        captured.append(sample)
        return real(sample, *args, **kwargs)

    sel = SelectionConfig(measure="linear", n_permutations=20)
    with patch.object(pooled_pathway, "run_selection_stage", side_effect=spy):
        run_pooled_analysis(basin, sel, rng=np.random.default_rng(1))
        run_pooled_analysis(basin, sel, rng=np.random.default_rng(1), seasonal=SeasonalConfig(enabled=False))
    on, off = captured
    assert np.allclose(on.covariates, seasonal_phase(basin.obs.index)[on.time_positions])
    assert off.covariates is None


def test_high_flow_dependence_lag_removes_the_calendar_day_mean_by_default_and_can_be_disabled():
    from model_selection.pipeline import SeasonalConfig
    from model_selection.pipeline.results import high_flow_result_to_dict
    basin = make_synthetic_basin(seed=10)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    ev = EventConfig(method="wasp", high_flow_quantile=0.90)
    on = run_high_flow_analysis(basin, sel, events=ev, rng=np.random.default_rng(1))
    off = run_high_flow_analysis(basin, sel, events=ev, rng=np.random.default_rng(1),
                                 seasonal=SeasonalConfig(enabled=False))
    assert high_flow_result_to_dict(on)["seasonal_conditioning"]["enabled"] is True
    assert high_flow_result_to_dict(off)["seasonal_conditioning"]["enabled"] is False
    assert any("calendar-day means removed" in n for n in on.notes)
    assert any("with their seasonal cycles" in n for n in off.notes)


def test_basis_sensitivity_check_repeats_the_stopping_round_with_the_main_features():
    """Each alternative basis describes every series exactly as the main
    sample does (duration feature included), and the K bases are tested at
    alpha / K."""
    from model_selection.pipeline.high_flow import diagnostics as hf_diagnostics
    basin = make_synthetic_basin(seed=3)
    shapes = []
    real = hf_diagnostics.event_feature_matrices

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        shapes.append(out[0].shape[1])
        return out

    with patch.object(hf_diagnostics, "event_feature_matrices", side_effect=spy):
        result = run_high_flow_analysis(
            basin, SelectionConfig(measure="linear", n_permutations=50), rng=np.random.default_rng(2))
    check = result.basis_sensitivity_check
    assert check is not None
    remaining = set(basin.formulations) - set(check["selected"])
    assert {b["basis"] for b in check["per_basis"]} == remaining
    assert check["level"] == pytest.approx(0.05 / len(remaining))
    assert shapes and all(d == result.n_components for d in shapes)


def test_missing_values_are_explicit_and_excluded_on_the_same_basis_for_every_series():
    """A missing value in one series removes that timestep from every series
    in the pooled pathway; nothing is dropped from the basin itself."""
    from model_selection.io.basin import on_regular_grid
    basin = make_synthetic_basin_with_gap(seed=5)
    gridded = on_regular_grid(basin)
    assert len(gridded.obs) > len(basin.obs) and gridded.obs.isna().sum() == len(gridded.obs) - len(basin.obs)
    name = next(iter(basin.formulations))
    gridded.formulations[name].iloc[100:130] = np.nan
    gridded.obs.iloc[700:710] = np.nan
    result = run_pooled_analysis(gridded, SelectionConfig(measure="linear", n_permutations=50),
                                 rng=np.random.default_rng(3))
    expected = int(np.logical_and.reduce(
        [np.isfinite(gridded.obs.values)] + [np.isfinite(s.values) for s in gridded.formulations.values()]).sum())
    assert result.n_timesteps == len(gridded.obs)
    assert result.n_complete_timesteps == expected


def test_an_observation_gap_keeps_the_event_and_a_formulation_gap_removes_it():
    basin = make_synthetic_basin(seed=6)
    first = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20),
                                   rng=np.random.default_rng(4))
    event = first.high_flow_events[0]
    gapped = make_synthetic_basin(seed=6)
    gapped.obs.iloc[event.peak_index] = np.nan
    second = run_high_flow_analysis(gapped, SelectionConfig(measure="linear", n_permutations=20),
                                    rng=np.random.default_rng(4))
    assert second.high_flow_power["n_events_with_observation_gaps"] >= 1
    assert event.start_time in {ev.start_time for ev in second.high_flow_events}
    sim_gap = make_synthetic_basin(seed=6)
    next(iter(sim_gap.formulations.values())).iloc[event.peak_index] = np.nan
    third = run_high_flow_analysis(sim_gap, SelectionConfig(measure="linear", n_permutations=20),
                                   rng=np.random.default_rng(4))
    assert third.high_flow_power["n_events_with_missing_values"] >= 1
    assert all(not (ev.start <= event.peak_index <= ev.end) for ev in third.high_flow_events)


def test_both_pathways_use_one_common_basis_and_flag_a_formulation_with_missing_values():
    """Missing values in a formulation where the observations exist remove
    those timesteps from every series in both pathways, and are flagged."""
    from model_selection.pipeline.results import review_flags
    basin = make_synthetic_basin(seed=7)
    name = next(iter(basin.formulations))
    basin.formulations[name].iloc[200:260] = np.nan
    basin.obs.iloc[900:950] = np.nan
    result = run_basin(basin, SelectionConfig(measure="linear", n_permutations=20), rng=np.random.default_rng(5))
    for pathway in (result.pooled, result.high_flow):
        mv = pathway.missing_values
        assert mv["n_common_timesteps"] == len(basin.obs) - 110
        assert mv["excluded_by_formulation"][name] == 60
        assert "formulation_missing_values" in {f["code"] for f in review_flags(pathway)}
    assert result.pooled.n_complete_timesteps == len(basin.obs) - 110
    for ev in result.high_flow.high_flow_events:  # a formulation's gap removes the events it touches
        assert not (ev.start <= 259 and ev.end >= 200)


def test_event_detection_uses_each_formulations_own_values_where_observations_are_missing():
    """Masking a formulation where the observations are missing would replace
    its real values with interpolated ones; detection uses its own values,
    and an observation gap is handled when the observed curve is scored."""
    captured = {}
    real = hf_pathway.detect_events_for_series

    def spy(series, config):
        captured.setdefault(series.name, series)
        return real(series, config)

    basin = make_synthetic_basin(seed=9)
    for name, s in basin.formulations.items():
        s.name = name
    basin.obs.iloc[1000:1100] = np.nan
    with patch.object(hf_pathway, "detect_events_for_series", side_effect=spy):
        result = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20),
                                        rng=np.random.default_rng(6))
    for name in basin.formulations:
        assert captured[name].iloc[1000:1100].notna().all()
    assert result.selection is not None


def test_phase_grid_defaults_to_the_longest_event_and_keeps_peaks():
    """No event curve is resampled onto fewer points than it has, so peaks
    are not lost between grid points."""
    from model_selection.statistics.fpca.fpca import phase_normalize
    basin = make_synthetic_basin(seed=10)
    result = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20),
                                    rng=np.random.default_rng(7))
    longest = max(ev.duration_timesteps for ev in result.high_flow_events)
    assert result.fpca_n_grid >= longest
    for ev in result.high_flow_events:
        curve = basin.obs.values[ev.start:ev.end + 1]
        assert phase_normalize(curve, result.fpca_n_grid).max() >= 0.9 * curve.max()


def test_roughness_penalty_is_the_same_functional_on_any_grid():
    """The penalty discretizes the integrated squared second derivative
    relative to the integrated square, so a given lambda means the same
    smoothing on coarse and fine grids: for sin(2 pi t) the ratio is (2 pi)^4."""
    from model_selection.statistics.fpca.fpca import roughness_penalty_matrix
    for n in (60, 240, 2000):
        t = np.linspace(0, 1, n)
        v = np.sin(2 * np.pi * t)
        ratio = (v @ roughness_penalty_matrix(n) @ v) / (v @ v)
        assert ratio == pytest.approx((2 * np.pi) ** 4, rel=0.06)


def test_seed_makes_a_run_repeatable_and_an_unseeded_run_records_its_seed():
    from model_selection.pipeline.pooled.pathway import run_pooled_analysis
    basin = make_synthetic_basin(seed=8)
    a = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=50, seed=123))
    b = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=50, seed=123))
    assert a.seed == b.seed == 123
    assert [s.p_value for s in a.selection.trajectory] == [s.p_value for s in b.selection.trajectory]
    c = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=50))
    assert isinstance(c.seed, int)
    d = run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=50, seed=c.seed))
    assert [s.p_value for s in c.selection.trajectory] == [s.p_value for s in d.selection.trajectory]


def test_pooled_result_flags_a_meaningful_threshold_below_the_detectable_contribution():
    from model_selection.pipeline.results import review_flags
    result = run_pooled_analysis(make_synthetic_basin(seed=9),
                                 SelectionConfig(measure="linear", n_permutations=50, meaningful_r2=1e-6, seed=1))
    assert result.meaningful_r2 == 1e-6
    assert "meaningful_effect_undetectable" in [f["code"] for f in review_flags(result)]


def test_high_flow_run_without_usable_events_is_flagged_untestable():
    """Formulations in which the detector finds no peaks give no consensus
    events; selection is not run, and the result says so with the untestable
    flag. White noise is too fast for WASP to resolve any peak."""
    from model_selection.pipeline.results import review_flags
    basin = make_synthetic_basin(seed=10)
    a = basin.formulations["formulation_noise"]
    basin.formulations = {"a": a, "b": a.iloc[::-1].set_axis(a.index)}
    result = run_high_flow_analysis(basin, SelectionConfig(n_permutations=30, seed=2), events=HF_EVENTS)
    assert result.selection is None
    assert "untestable" in [f["code"] for f in review_flags(result)]


def test_pooled_result_reports_subsampling_only_for_the_committed_tier():
    basin = make_synthetic_basin(n_hours=2600, seed=11)
    result = run_pooled_analysis(basin, SelectionConfig(n_permutations=30, seed=3))
    if result.committed_tier == "linear":
        assert result.n_after_subsampling is None and not result.dependence_aware_subsampling_used
    else:
        assert result.n_after_subsampling is not None and result.n_after_subsampling <= 2000


def test_registration_covariates_are_the_log_window_length_and_the_peak_position():
    from types import SimpleNamespace
    from model_selection.pipeline.high_flow.event_sample import registration_covariates
    events = [SimpleNamespace(event_id=3, start=10, end=29, peak_index=15),
              SimpleNamespace(event_id=7, start=100, end=100 + 199, peak_index=100 + 199)]
    reg = registration_covariates(events, [7, 3])
    assert reg.shape == (2, 2)
    assert reg[0] == pytest.approx([np.log(200), 1.0]) and reg[1] == pytest.approx([np.log(20), 5 / 19])


def test_events_with_observation_gaps_stay_in_the_sample():
    basin = make_synthetic_basin(n_hours=20000, n_events=80, seed=4)
    complete = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20),
                                      rng=np.random.default_rng(1))
    peaks = [ev.peak_index for ev in complete.high_flow_events if ev.duration_timesteps >= 20][:10]
    obs = basin.obs.copy()
    for p in peaks:  # gauge failures at the peaks of ten events
        obs.iloc[p - 2:p + 3] = np.nan
    gapped_basin = type(basin)(gage=basin.gage, formulations=basin.formulations, obs=obs)
    gapped = run_high_flow_analysis(gapped_basin, SelectionConfig(measure="linear", n_permutations=20),
                                    rng=np.random.default_rng(1))
    assert len(gapped.high_flow_events) == len(complete.high_flow_events)
    d = high_flow_result_to_dict(gapped)
    assert any("gaps in the observations" in n for n in d["notes"])


def test_cmi_calibration_check_runs_through_the_pipeline_and_is_recorded():
    # Pooled pathway, CMI tier, the calibration check on (the default): it runs on the
    # acceptance, and its result reaches the step and the result file.
    basin = make_synthetic_basin(seed=4)
    result = run_pooled_analysis(basin, SelectionConfig(measure="cmi", n_permutations=50),
                                 rng=np.random.default_rng(5))
    checked = [s for s in result.selection.trajectory if s.cmi_calibration is not None]
    assert checked and checked[0].cmi_calibration["n_surrogates"] == 52
    d = pooled_result_to_dict(result)
    assert d["cmi_calibration"] is True
    assert any(s.get("cmi_calibration") for s in d["pooled_trajectory"])
    json.dumps(d)


def test_an_uncalibrated_cmi_tier_withholds_the_acceptance_through_the_pipeline():
    import model_selection.selection.tiers as tiers
    uncalibrated = {"n_surrogates": 52, "false_acceptances": 12, "false_acceptance_rate": 12 / 52,
                    "p_value": 1e-5, "calibrated": False}
    basin = make_synthetic_basin(seed=4)
    with patch.object(tiers, "cmi_calibration", return_value=uncalibrated):
        result = run_pooled_analysis(basin, SelectionConfig(measure="cmi", n_permutations=50),
                                     rng=np.random.default_rng(5))
    assert not any(s.accepted and s.cmi_calibration for s in result.selection.trajectory)
    assert "cmi_not_calibrated" in [f["code"] for f in pooled_result_to_dict(result)["review_flags"]]


def test_wasp_flags_a_series_whose_scale_range_is_too_short():
    """The noise formulation's e-folding time is under one step, so WASP
    cannot resolve its events; the result says which series."""
    from model_selection.pipeline.results import review_flags
    basin = make_synthetic_basin(seed=6)
    result = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20, seed=1),
                                    events=EventConfig(method="wasp"))
    flags = {f["code"]: f["message"] for f in review_flags(result)}
    assert "wasp_scale_range_too_short" in flags and "formulation_noise" in flags["wasp_scale_range_too_short"]
    assert "formulation_informative" not in flags["wasp_scale_range_too_short"]


def test_crowded_events_shorten_only_the_theiler_window(monkeypatch):
    """Where the dependence lag would leave an event fewer than five
    permutation partners, the high-flow pathway shortens the free-form tier's
    Theiler window; the selection sample keeps the full dependence lag, which
    every other test uses."""
    import model_selection.pipeline.high_flow.pathway as pathway_module
    captured = {}
    real = pathway_module.run_selection_stage

    def spy(sample, selection, *args, **kwargs):
        captured["lag"], captured["window"] = sample.dependence_lag, selection.theiler_window
        return real(sample, selection, *args, **kwargs)

    monkeypatch.setattr(pathway_module, "run_selection_stage", spy)
    # 15 events in 3000 hours: a 1200-hour lag leaves each event few partners outside it.
    result = run_high_flow_analysis(make_synthetic_basin(n_hours=3000, n_events=15),
                                    SelectionConfig(measure="linear", n_permutations=20, min_temporal_lag=1200),
                                    rng=np.random.default_rng(3))
    assert captured["lag"] == 1200
    assert captured["window"] is not None and captured["window"] < 1200
    assert any("Theiler window reduced from 1200" in note for note in result.notes)


def test_long_pooled_records_give_the_nonlinear_tier_the_full_cap(monkeypatch):
    """The nonlinear tier gets as many evenly spaced points as its cap allows,
    whatever the dependence lag; the free-form tier gets points at least one
    lag apart; both are drawn from the full record."""
    import model_selection.pipeline.pooled.pathway as pooled_module
    captured = {}
    real = pooled_module.run_selection_stage

    def spy(sample, selection, *args, tier_samples=None, **kwargs):
        captured["full"], captured["samples"] = sample, tier_samples
        return real(sample, selection, *args, tier_samples=tier_samples, **kwargs)

    monkeypatch.setattr(pooled_module, "run_selection_stage", spy)
    basin = make_synthetic_basin(n_hours=3000, seed=12)
    run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=20, min_temporal_lag=40,
                                               nonlinear_max_n=500, cmi_max_n=500),
                        rng=np.random.default_rng(1))
    # measure="linear" draws no tier samples; the cascade does
    run_pooled_analysis(basin, SelectionConfig(n_permutations=20, min_temporal_lag=40, nonlinear_max_n=500,
                                               cmi_max_n=500, cmi_calibration=False, equivalence_checks=False),
                        rng=np.random.default_rng(1))
    full, samples = captured["full"], captured["samples"]
    nonlinear, cmi = samples["nonlinear"], samples["cmi"]
    assert len(nonlinear.obs) == 500 and nonlinear.grid_step < 40
    assert np.diff(cmi.time_positions).min() >= 40 and len(cmi.obs) < 500
    for sub in (nonlinear, cmi):  # drawn from the full record, each within half a time step of its own time
        rows = np.searchsorted(full.time_positions, np.rint(sub.time_positions))
        assert np.all(np.abs(full.time_positions[rows] - sub.time_positions) <= 0.5)
        assert np.array_equal(full.obs[rows], sub.obs)
