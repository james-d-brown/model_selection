"""
End-to-end tests of the pooled pathway and its orchestration on synthetic
basins (no parquet or USGS access): selection and serialization. The
experimental high-flow event pathway is tested in
tests/experimental/test_high_flow_pathway.py, and the selection mechanism
itself in tests/selection.
"""
import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from model_selection.io.basin import BasinData
from model_selection.pipeline import (
    OutputConfig,
    basin_result_to_dict,
    pooled_result_to_dict,
    run_basin,
    run_pooled_analysis,
)
from model_selection.selection import (
    AdequacyConfig, SelectionConfig, SelectionSample, build_cascade, build_tier, effect_size_floor_for_tier,
    run_tiered_selection,
)
from model_selection.statistics.association.canonical import partial_cca_structural_adequacy
from tests.fixtures.basins import make_synthetic_basin, make_synthetic_basin_with_gap


@pytest.mark.integration
def test_pipeline_runs_end_to_end_and_selects_informative_formulation():
    basin = make_synthetic_basin()
    result = run_basin(
        basin, SelectionConfig(n_neighbors=10, n_permutations=150, alpha=0.05, effect_size_floor=0.02),
        rng=np.random.default_rng(99),
    )
    assert result.gage == "TESTGAGE01"
    assert "formulation_informative" in result.pooled.selection.selected
    assert "formulation_noise" not in result.pooled.selection.selected
    assert result.high_flow is None
    for pathway in (result.pooled,):
        r = pathway.pearson_correlation
        assert set(r) == {"formulation_informative", "formulation_noise"}
        assert r["formulation_informative"] > r["formulation_noise"]
        e = pathway.selection.unconditional_effect
        assert set(e) == set(r) and e["formulation_informative"] > e["formulation_noise"]
    t = result.pooled.pearson_correlation_transformed
    assert set(t) == set(result.pooled.pearson_correlation) and t["formulation_informative"] > t["formulation_noise"]
    json.dumps(basin_result_to_dict(result))


@pytest.mark.integration
def test_pipeline_flags_high_zero_flow():
    basin = make_synthetic_basin(seed=1)
    idx = basin.obs.index
    rng = np.random.default_rng(2)
    basin.formulations["formulation_zero_heavy"] = pd.Series(
        np.where(rng.random(len(idx)) < 0.5, 0.0, rng.uniform(1, 5, len(idx))), index=idx)
    result = run_basin(basin, SelectionConfig(n_permutations=50), rng=np.random.default_rng(3))
    assert result.zero_flow_flag is True
    assert result.zero_flow_fraction["formulation_zero_heavy"] > 0.3


@pytest.mark.integration
@pytest.mark.parametrize("measure", ["linear", "nonlinear", "cmi"])
def test_pipeline_runs_end_to_end_across_all_measures(measure):
    basin = make_synthetic_basin(seed=4)
    result = run_basin(basin, SelectionConfig(measure=measure, n_permutations=50),
                       rng=np.random.default_rng(5))
    assert result.pooled.measure == measure
    as_dict = basin_result_to_dict(result)
    assert as_dict["pooled"]["measure"] == measure
    json.dumps(as_dict)


@pytest.mark.integration
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


@pytest.mark.integration
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
            basin, SelectionConfig(measure="cascade", n_permutations=100, min_temporal_lag=5),
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


def test_structural_check_belongs_to_the_linear_tier_only():
    assert build_tier("linear", SelectionConfig()).structural_check is partial_cca_structural_adequacy
    assert build_tier("nonlinear", SelectionConfig()).structural_check is None
    assert build_tier("cmi", SelectionConfig()).structural_check is None
    with pytest.raises(ValueError):
        build_tier("cascade", SelectionConfig())


@pytest.mark.integration
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
    # The effect is 1 - Wilks' lambda less its chance level from the null mean of Pillai's trace, the mean of
    # the width draws (corrected for the bandwidth's bias, not divided by the fixed-b draws).
    from model_selection.statistics.conditional_independence.linear import chance_level
    pc = step.diagnostics["partial_canonical_correlation"]
    assert abs(step.effect - (pc - chance_level(np.array([step.null_width_mean]), q, p))) < 1e-9


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


def test_meaningful_r2_is_validated():
    with pytest.raises(ValueError, match="meaningful_r2"):
        SelectionConfig(meaningful_r2=1.5)


@pytest.mark.integration
def test_pooled_pathway_uses_the_largest_decorrelation_lag():
    from model_selection.pipeline.shared import max_decorrelation_lag
    from model_selection.preprocessing.transforms import dependence_series
    basin = make_synthetic_basin(seed=8)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    prepared = {k: dependence_series(s.values, index=s.index, seasonal=True) for k, s in basin.formulations.items()}
    prepared["observations"] = dependence_series(basin.obs.values, index=basin.obs.index, seasonal=True)
    pooled = run_pooled_analysis(basin, sel, rng=np.random.default_rng(4))
    lag, _ = max_decorrelation_lag(prepared, sel.dependence_run_length)
    assert pooled.min_temporal_lag == lag


@pytest.mark.integration
def test_pooled_seasonal_conditioning_is_on_by_default_and_can_be_disabled():
    from model_selection.pipeline import SeasonalConfig
    from model_selection.pipeline.results import pooled_result_to_dict
    basin = make_synthetic_basin(seed=9)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    on = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1))
    off = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1), seasonal=SeasonalConfig(enabled=False))
    assert pooled_result_to_dict(on)["seasonal_conditioning"] == {"enabled": True, "record_too_short": False}
    assert pooled_result_to_dict(off)["seasonal_conditioning"] == {"enabled": False, "record_too_short": False}
    assert any("Conditioning on the time of year:" in n for n in on.notes)
    assert any("disabled" in n for n in off.notes)


@pytest.mark.integration
def test_seasonal_conditioning_is_disabled_on_a_record_shorter_than_three_years():
    """On a 125-day record the calendar day does not recur, so the pooled pathway
    does not condition on it; it says why and raises a review flag."""
    from model_selection.pipeline.results import review_flags
    basin = make_synthetic_basin(n_hours=3000, seed=2)
    sel = SelectionConfig(measure="linear", n_permutations=20)
    pooled = run_pooled_analysis(basin, sel, rng=np.random.default_rng(1))
    assert pooled_result_to_dict(pooled)["seasonal_conditioning"] == {"enabled": False, "record_too_short": True}
    for result in (pooled,):
        assert any("fewer than the 3 needed to describe the seasonal cycle" in n for n in result.notes)
        assert "seasonal_record_too_short" in [f["code"] for f in review_flags(result)]


@pytest.mark.integration
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


@pytest.mark.integration
def test_pooled_series_keep_their_seasonal_cycles():
    """The pooled tests analyse each series' standardized signed square root,
    with its seasonal cycle, and condition on the seasonal phase."""
    from model_selection.pipeline.pooled import pathway as pooled_pathway
    from model_selection.preprocessing.transforms import prepare_series
    basin = make_synthetic_basin(seed=12)
    captured = []
    real = pooled_pathway.run_selection_stage

    def spy(sample, *args, **kwargs):
        captured.append(sample)
        return real(sample, *args, **kwargs)

    with patch.object(pooled_pathway, "run_selection_stage", side_effect=spy):
        run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=20), rng=np.random.default_rng(1))
    sample = captured[0]
    rows = sample.time_positions
    assert np.allclose(sample.obs, prepare_series(basin.obs.values.astype(float))[rows])
    for name, series in basin.formulations.items():
        assert np.allclose(sample.candidates[name], prepare_series(series.values.astype(float))[rows])


@pytest.mark.integration
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


@pytest.mark.integration
def test_pooled_pathway_uses_one_common_basis_and_flag_a_formulation_with_missing_values():
    """Missing values in a formulation where the observations exist remove
    those timesteps from every series in the pooled pathway, and are flagged."""
    from model_selection.pipeline.results import review_flags
    basin = make_synthetic_basin(seed=7)
    name = next(iter(basin.formulations))
    basin.formulations[name].iloc[200:260] = np.nan
    basin.obs.iloc[900:950] = np.nan
    result = run_basin(basin, SelectionConfig(measure="linear", n_permutations=20), rng=np.random.default_rng(5))
    for pathway in (result.pooled,):
        mv = pathway.missing_values
        assert mv["n_common_timesteps"] == len(basin.obs) - 110
        assert mv["excluded_by_formulation"][name] == 60
        assert "formulation_missing_values" in {f["code"] for f in review_flags(pathway)}
    assert result.pooled.n_complete_timesteps == len(basin.obs) - 110


@pytest.mark.integration
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


@pytest.mark.integration
def test_pooled_result_reports_subsampling_only_for_the_committed_tier():
    basin = make_synthetic_basin(n_hours=2600, seed=11)
    result = run_pooled_analysis(basin, SelectionConfig(n_permutations=30, seed=3))
    if result.committed_tier == "linear":
        assert result.n_after_subsampling is None and not result.dependence_aware_subsampling_used
    else:
        assert result.n_after_subsampling is not None and result.n_after_subsampling <= 2000


@pytest.mark.integration
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
                                               cmi_max_n=500, equivalence_checks=False),
                        rng=np.random.default_rng(1))
    full, samples = captured["full"], captured["samples"]
    nonlinear, cmi = samples["nonlinear"], samples["cmi"]
    assert len(nonlinear.obs) == 500 and nonlinear.grid_step < 40
    assert np.diff(cmi.time_positions).min() >= 40 and len(cmi.obs) < 500
    for sub in (nonlinear, cmi):  # drawn from the full record, each within half a time step of its own time
        rows = np.searchsorted(full.time_positions, np.rint(sub.time_positions))
        assert np.all(np.abs(full.time_positions[rows] - sub.time_positions) <= 0.5)
        assert np.array_equal(full.obs[rows], sub.obs)
