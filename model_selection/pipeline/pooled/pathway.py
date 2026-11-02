"""
The pooled pathway: model selection on the whole record, one sample per
timestep.

0. Restrict every series to the common basis: the timesteps at which the
   observations and every formulation have values (pipeline.shared.common_basis).
1. Transform every series (log, removal of its calendar-day mean by default,
   rank-normal score with random tie-breaking); every test then conditions
   on the seasonal phase. Records too short to estimate the seasonal cycle
   (preprocessing.seasonal.MIN_YEARS) are analysed with their seasonal
   cycles.
2. Estimate the dependence lag (largest Politis decorrelation lag of the
   formulations).
3. Run the selection stage (one tier or the cascade). Records longer than
   the nonlinear tier's cap (SelectionConfig.nonlinear_max_n) give that tier
   an evenly spaced subsample of the cap's size, whose dependence its
   bootstrap accounts for. The CMI tier's local permutation treats samples
   as exchangeable, so it runs on samples at least one dependence lag apart,
   at most SelectionConfig.cmi_max_n of them. Both are drawn from the full
   record.

See docs/methods.md, Samples.
"""
from __future__ import annotations


import logging

import numpy as np

from model_selection.io.basin import BasinData
from model_selection.io.results import write_json
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.results import (
    POOLED_RESULT_JSON_FILENAME_TEMPLATE,
    PooledResult,
    pooled_result_to_dict,
)
from model_selection.pipeline.shared import (
    adequacy_plot_factory, common_basis, log_review_flags, max_decorrelation_lag, resolve_rng,
    write_dependence_plot, zero_flow_check,
)
from model_selection.preprocessing.seasonal import SeasonalConfig, resolve_seasonal, seasonal_phase
from model_selection.preprocessing.transforms import prepare_series
from model_selection.selection.tiers import SelectionSample
from model_selection.selection.config import CASCADE, AdequacyConfig, SelectionConfig
from model_selection.selection.stage import run_selection_stage
from model_selection.statistics.significance.resampling import grid_subsample

logger = logging.getLogger(__name__)


def _spaced_subsample(sample: SelectionSample, max_n: int, rng: np.random.Generator, at_lag_spacing: bool):
    """(subsampled sample or None if unchanged, natural_independent_n); see
    grid_subsample for the two spacing rules."""
    n = len(sample.obs)
    idx, positions, step, natural_n = grid_subsample(sample.time_positions, sample.grid_step, sample.dependence_lag,
                                                     max_n, rng, at_lag_spacing=at_lag_spacing)
    if idx.size >= n:
        return None, natural_n
    return SelectionSample(
        candidates={name: arr[idx] for name, arr in sample.candidates.items()},
        obs=sample.obs[idx], time_positions=positions, dependence_lag=sample.dependence_lag,
        grid_step=step, covariates=None if sample.covariates is None else sample.covariates[idx],
        covariate_label=sample.covariate_label,
    ), natural_n


def run_pooled_analysis(
        basin: BasinData,
        selection: SelectionConfig | None = None,
        adequacy: AdequacyConfig | None = None,
        output: OutputConfig | None = None,
        rng: np.random.Generator | None = None,
        seasonal: SeasonalConfig | None = None,
) -> PooledResult:
    """Run the pooled pathway for one basin (see module docstring).

    seasonal : conditioning on the time of year (default: enabled, 11-day
        window for the calendar-day mean); SeasonalConfig(enabled=False)
        analyses the series with their seasonal cycles, as happens
        automatically on records too short to estimate the seasonal cycle
        (docs/methods.md, Seasonality).
    Writes {gage}_pooled_result.json when output.output_dir is set.
    """
    selection = selection or SelectionConfig()
    adequacy = adequacy or AdequacyConfig()
    output = output or OutputConfig()
    rng, seed = resolve_rng(rng, selection.seed, basin.gage, "pooled")
    notes: list[str] = []
    basin, missing_values = common_basis(basin, notes)
    gage = basin.gage
    n_grid = len(basin.obs)
    rows = np.flatnonzero(np.isfinite(basin.obs.values.astype(float)))
    n = rows.size

    zero_flow, zero_flow_flag = zero_flow_check(basin, notes)

    seasonal, too_short = resolve_seasonal(seasonal, basin.obs.index[rows])
    window = seasonal.window_days if seasonal.enabled else None
    obs_t = prepare_series(basin.obs.values.astype(float), rng=rng, index=basin.obs.index,
                           seasonal_window_days=window)
    candidates_t = {name: prepare_series(series.values.astype(float), rng=rng, index=series.index,
                                         seasonal_window_days=window)
                    for name, series in basin.formulations.items()}
    if seasonal.enabled:
        notes.append(f"Conditioning on the time of year: each series' calendar-day mean ({seasonal.window_days}-day "
                     "triangular window) removed before the rank-normal transform, and every test conditions on "
                     "the seasonal phase (sin and cos of the day of year).")
    elif too_short:
        notes.append(too_short)
    else:
        notes.append("Conditioning on the time of year disabled: series analysed with their seasonal cycles.")

    min_temporal_lag = selection.min_temporal_lag
    if min_temporal_lag is None:
        min_temporal_lag, per_formulation = max_decorrelation_lag(candidates_t, selection.dependence_run_length)
        notes.append(
            f"min_temporal_lag (dependence lag) auto-estimated as {min_temporal_lag} timesteps: the largest "
            "Politis decorrelation lag across the simulated formulations, transformed as above (obs-free; "
            + ", ".join(f"{k}={v}" for k, v in per_formulation.items())
            + "); pass min_temporal_lag explicitly to override."
        )
    else:
        notes.append(f"min_temporal_lag={min_temporal_lag} (caller-specified).")
    write_dependence_plot(output, gage, candidates_t, obs_t, basin.obs.index, selection.dependence_run_length, window)

    phase = seasonal_phase(basin.obs.index)[rows] if seasonal.enabled else None
    sample = SelectionSample({name: v[rows] for name, v in candidates_t.items()}, obs_t[rows], rows,
                             min_temporal_lag, grid_step=1, covariates=phase,
                             covariate_label="the seasonal phase" if phase is not None else None)
    # Each costly tier gets its own subsample of the full record. The nonlinear
    # tier (kernel regression is O(n^3)) uses as many evenly spaced points as its
    # cap allows, and its bootstrap accounts for their dependence. The CMI
    # tier's local permutation treats samples as exchangeable, so it uses
    # samples at least one dependence lag apart, at most cmi_max_n.
    caps = {"nonlinear": selection.nonlinear_max_n, "cmi": selection.cmi_max_n}
    lag_spaced = {"nonlinear": False, "cmi": True}
    tiers_used = list(caps) if selection.measure == CASCADE else [selection.measure]
    tier_samples = {}
    natural_by_tier = {}
    for tier_name in tiers_used:
        if tier_name not in caps:
            continue
        sub, natural_by_tier[tier_name] = _spaced_subsample(sample, caps[tier_name], rng, lag_spaced[tier_name])
        if sub is not None:
            tier_samples[tier_name] = sub
            how = ("points at least one dependence lag apart, whose local permutation treats them as exchangeable"
                   if lag_spaced[tier_name] else
                   f"evenly spaced points, as many as its cap of {caps[tier_name]} allows; its bootstrap accounts "
                   "for their dependence")
            notes.append(f"the {tier_name} tier runs on {len(sub.obs)} of the record's {n} points: {how}.")
    outcome = run_selection_stage(
        sample, selection, adequacy, multivariate=False,
        rng=rng, tier_samples=tier_samples,
        plot_factory=adequacy_plot_factory(output, gage, "pooled"), label=f"pooled, basin {gage}",
    )
    # Report the sample of the tier whose selection is the result.
    used = outcome.committed_tier or selection.measure
    n_after_subsampling = len(tier_samples[used].obs) if used in tier_samples else None
    natural_independent_n = natural_by_tier.get(used) if used in tier_samples else None
    if used in tier_samples:
        min_temporal_lag = tier_samples[used].dependence_lag

    result = PooledResult(
        seed=seed, meaningful_r2=selection.meaningful_r2, cmi_calibration=selection.cmi_calibration,
        gage=gage, n_timesteps=n_grid, n_complete_timesteps=n, missing_values=missing_values,
        zero_flow_fraction=zero_flow, zero_flow_flag=zero_flow_flag,
        selection=outcome.selection, min_temporal_lag=min_temporal_lag,
        measure=selection.measure,
        dependence_aware_subsampling_used=n_after_subsampling is not None,
        n_after_subsampling=n_after_subsampling,
        natural_independent_sample_size=natural_independent_n,
        notes=notes,
        committed_tier=outcome.committed_tier,
        adequacy_floor=adequacy.resolved_effect_size_floor(selection) if selection.measure == CASCADE else None,
        attempted_tiers=outcome.attempted_tiers,
        seasonal_conditioning={"enabled": seasonal.enabled,
                                  "window_days": seasonal.window_days if seasonal.enabled else None,
                                  "record_too_short": too_short is not None},
    )
    if output.output_dir is not None:
        result.result_json_path = write_json(
            pooled_result_to_dict(result), output.output_dir, POOLED_RESULT_JSON_FILENAME_TEMPLATE.format(gage=gage))
        logger.info(f"[+] Wrote pooled outputs for {gage} to {output.output_dir}")
    log_review_flags(result, "pooled")
    return result
