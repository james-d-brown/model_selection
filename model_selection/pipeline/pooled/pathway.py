"""
The pooled pathway: model selection on the whole record, one sample per
timestep.

0. Restrict every series to the common basis: the timesteps at which the
   observations and every formulation have values (pipeline.shared.common_basis).
1. Transform every series (standardized signed square root of flow). Every test conditions on the seasonal phase, except on records too short to
   describe the seasonal cycle (preprocessing.seasonal.MIN_YEARS).
2. Estimate the dependence lag (largest Politis decorrelation lag of the
   formulations, each as analysed minus its fit on the seasonal phase;
   preprocessing.transforms.dependence_series).
3. Run the selection stage (one tier or the cascade). Records longer than
   the nonlinear tier's cap (SelectionConfig.nonlinear_max_n) give that tier
   an evenly spaced subsample of the cap's size, whose dependence its
   bootstrap accounts for. The CMI tier's local permutation treats samples
   as exchangeable, so it runs on samples at least one dependence lag apart,
   at most SelectionConfig.cmi_max_n of them. Both are drawn from the full
   record.

With censoring (a preprocessing.censoring.Censoring), the candidates and the
observations are the transformed series of step 1, each censored at its own
quantile; the selected
formulations enter the conditioning set uncensored, as in step 1, with the
seasonal phase. The dependence lag is that of step 2. Results are written
to {gage}_pooled_censored_result.json.

See docs/methods.md, Samples; and Censoring at high flow.
"""
from __future__ import annotations


import logging

import numpy as np

from model_selection.io.basin import BasinData
from model_selection.io.results import write_json
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.results import (
    POOLED_CENSORED_RESULT_JSON_FILENAME_TEMPLATE,
    POOLED_RESULT_JSON_FILENAME_TEMPLATE,
    PooledResult,
    pooled_result_to_dict,
)
from model_selection.pipeline.shared import (
    adequacy_plot_factory, common_basis, log_pearson, log_review_flags, max_decorrelation_lag,
    pearson_with_observations, resolve_rng, write_dependence_plot, zero_flow_check,
)
from model_selection.preprocessing.censoring import CensoredSeries, Censoring
from model_selection.preprocessing.seasonal import SeasonalConfig, resolve_seasonal, seasonal_phase
from model_selection.preprocessing.transforms import dependence_series, prepare_series
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
    return sample.subset(idx, positions, step), natural_n


def _censored_series(basin: BasinData, rows: np.ndarray, censoring: Censoring):
    """(censored candidates, censored observations, CensoredSeries): each
    series as transformed for the pooled pathway at the complete rows,
    censored at its own quantile over those rows."""
    uncensored = {name: prepare_series(series.values.astype(float))[rows]
                  for name, series in basin.formulations.items()}
    obs_uncensored = prepare_series(basin.obs.values.astype(float))[rows]
    thresholds = {name: censoring.threshold(v) for name, v in uncensored.items()}
    obs_threshold = censoring.threshold(obs_uncensored)
    censored = CensoredSeries(uncensored, thresholds, obs_threshold, obs_uncensored)
    return censored.censored_candidates(), censored.censored_obs(), censored


def run_pooled_analysis(
        basin: BasinData,
        selection: SelectionConfig | None = None,
        adequacy: AdequacyConfig | None = None,
        output: OutputConfig | None = None,
        rng: np.random.Generator | None = None,
        seasonal: SeasonalConfig | None = None,
        censoring: Censoring | None = None,
) -> PooledResult:
    """Run the pooled pathway for one basin (see module docstring).

    censoring : censor the candidates and the observations at a quantile of
        each series' own record (docs/methods.md, Censoring at high flow),
        e.g. Censoring.at_high_flow(events) for the high-flow events'
        quantile; None (default) for the uncensored pathway.

    seasonal : conditioning on the time of year (default: enabled: every
        test conditions on the seasonal phase, and the dependence lag is
        estimated from the series adjusted for it); SeasonalConfig(enabled=False)
        turns it off, as happens automatically on records too short to
        describe the seasonal cycle (docs/methods.md, Seasonality).
    Writes {gage}_pooled_result.json ({gage}_pooled_censored_result.json
    with censoring) when output.output_dir is set.
    """
    selection = selection or SelectionConfig()
    adequacy = adequacy or AdequacyConfig()
    output = output or OutputConfig()
    pathway = "pooled" if censoring is None else "pooled_censored"
    label = "pooled" if censoring is None else "pooled, censored at high flow"
    rng, seed = resolve_rng(rng, selection.seed, basin.gage, label)
    notes: list[str] = []
    basin, missing_values = common_basis(basin, notes)
    gage = basin.gage
    n_grid = len(basin.obs)
    rows = np.flatnonzero(np.isfinite(basin.obs.values.astype(float)))
    n = rows.size

    zero_flow, zero_flow_flag = zero_flow_check(basin, notes)

    seasonal, too_short = resolve_seasonal(seasonal, basin.obs.index[rows])
    obs_t = prepare_series(basin.obs.values.astype(float))
    candidates_t = {name: prepare_series(series.values.astype(float)) for name, series in basin.formulations.items()}
    if seasonal.enabled:
        notes.append("Conditioning on the time of year: every test conditions on the seasonal phase (sin and cos "
                     "of the day of year).")
    elif too_short:
        notes.append(too_short)
    else:
        notes.append("Conditioning on the time of year disabled: the tests do not condition on the seasonal phase.")

    lag_series = {name: dependence_series(s.values.astype(float), index=s.index, seasonal=seasonal.enabled)
                  for name, s in basin.formulations.items()}
    min_temporal_lag = selection.min_temporal_lag
    if min_temporal_lag is None:
        min_temporal_lag, per_formulation = max_decorrelation_lag(lag_series, selection.dependence_run_length)
        notes.append(
            f"min_temporal_lag (dependence lag) auto-estimated as {min_temporal_lag} timesteps: the largest "
            "Politis decorrelation lag across the simulated formulations, each as analysed (standardized signed "
            "square root of flow)"
            + (" minus its fit on the seasonal phase" if seasonal.enabled else "")
            + " (obs-free; " + ", ".join(f"{k}={v}" for k, v in per_formulation.items())
            + "); pass min_temporal_lag explicitly to override."
        )
    else:
        notes.append(f"min_temporal_lag={min_temporal_lag} (caller-specified).")
    write_dependence_plot(output, gage, lag_series,
                          dependence_series(basin.obs.values.astype(float), index=basin.obs.index,
                                            seasonal=seasonal.enabled),
                          basin.obs.index, selection.dependence_run_length, seasonal.enabled,
                          raw={name: s.values.astype(float) for name, s in basin.formulations.items()},
                          raw_observations=basin.obs.values.astype(float))

    phase = seasonal_phase(basin.obs.index)[rows] if seasonal.enabled else None
    covariate_label = "the seasonal phase" if phase is not None else None
    censoring_record = None
    if censoring is None:
        sample = SelectionSample({name: v[rows] for name, v in candidates_t.items()}, obs_t[rows], rows,
                                 min_temporal_lag, grid_step=1, covariates=phase, covariate_label=covariate_label)
    else:
        # The candidate and the observations are censored; the selected
        # formulations enter the conditioning set uncensored, as prepared above,
        # with the seasonal phase (docs/methods.md, Censoring at high flow).
        censored_candidates, censored_obs, censored = _censored_series(basin, rows, censoring)
        sample = SelectionSample(censored_candidates, censored_obs, rows, min_temporal_lag, grid_step=1,
                                 covariates=phase, covariate_label=covariate_label,
                                 conditioning={name: v[rows] for name, v in candidates_t.items()},
                                 censored=censored)
        shares = censored.shares_at_threshold(censored_candidates, censored_obs)
        censoring_record = {"quantile": censoring.quantile, "thresholds": dict(censored.thresholds),
                            "obs_threshold": censored.obs_threshold, "share_at_threshold": shares}
        notes.append(
            f"Censoring at high flow: the candidates and the observations are the standardized signed square "
            f"root of flow, "
            f"each censored at its own {censoring.quantile:g} quantile "
            "(every value below it set to it); the selected formulations enter the conditioning set uncensored, "
            "as prepared above" + (", with the seasonal phase" if phase is not None else "") + ". Share of each "
            "series at its threshold: " + ", ".join(f"{k} {v:.3f}" for k, v in shares.items()) + ".")
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
        plot_factory=adequacy_plot_factory(output, gage, pathway), label=f"{label}, basin {gage}",
    )
    # Report the sample of the tier whose selection is the result.
    used = outcome.committed_tier or selection.measure
    n_after_subsampling = len(tier_samples[used].obs) if used in tier_samples else None
    natural_independent_n = natural_by_tier.get(used) if used in tier_samples else None
    if used in tier_samples:
        min_temporal_lag = tier_samples[used].dependence_lag

    pearson = pearson_with_observations(basin.obs.values, {k: v.values for k, v in basin.formulations.items()})
    pearson_transformed = pearson_with_observations(obs_t[rows], {k: v[rows] for k, v in candidates_t.items()})
    log_pearson(gage, label, "every complete time step", pearson, pearson_transformed)
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
        pearson_correlation=pearson,
        pearson_correlation_transformed=pearson_transformed,
        seasonal_conditioning={"enabled": seasonal.enabled, "record_too_short": too_short is not None},
        censoring=censoring_record,
    )
    if output.output_dir is not None:
        template = (POOLED_RESULT_JSON_FILENAME_TEMPLATE if censoring is None
                    else POOLED_CENSORED_RESULT_JSON_FILENAME_TEMPLATE)
        result.result_json_path = write_json(
            pooled_result_to_dict(result), output.output_dir, template.format(gage=gage))
        logger.info(f"[+] Wrote {label} outputs for {gage} to {output.output_dir}")
    log_review_flags(result, label)
    return result
