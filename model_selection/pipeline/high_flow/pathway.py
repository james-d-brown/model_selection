"""
The high-flow pathway: model selection on high-flow events, one sample per
event.

1. Detect each series' own events (EventConfig.method) on all of that
   series' own values, obs-free for the models; obs's own events are used
   only for its duration feature.
2. Form consensus events from the models (event_detection.consensus) and
   widen them to the models' own events (event_detection.windows). Keep only
   the events in which every formulation has a value throughout and the
   observations have at least one (docs/methods.md, Identifying high-flow events).
3. Build the event sample: shared FPCA basis, FPC scores (+ own duration),
   standardized jointly (event_sample).
4. Run the selection stage on the event sample, with the neighbour counts
   and the dependence lag limited by the number and spacing of events.
5. Diagnostics: reconstruction fidelity, ensemble representation, peak-phase
   spread and the basis sensitivity check (diagnostics).

Timesteps between events never enter the sample, and one event contributes
one sample however long it lasts, which removes the pseudo-replication of
timestep-level high-flow samples.

See docs/methods.md, Samples.
"""
from __future__ import annotations

import dataclasses
import logging
import os

import numpy as np

from model_selection.event_detection.config import EventConfig
from model_selection.event_detection.consensus import (
    consensus_rank,
    group_runs_by_own_events,
    independent_event_summary,
    rank_consensus_regime,
    summarize_events,
)
from model_selection.event_detection.detect import detect_events_for_series
from model_selection.event_detection.windows import (
    ExpandedEvent,
    attach_timestamps,
    build_expanded_events,
)
from model_selection.io.basin import OBS_KEY, BasinData, complete_mask, on_regular_grid
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.high_flow import diagnostics
from model_selection.pipeline.high_flow.config import FPCAConfig
from model_selection.pipeline.high_flow.event_sample import (
    analyze_peak_phase_spread,
    build_fpc_score_table,
    event_feature_matrices,
    registration_covariates,
    summarize_peak_phase_spread,
    with_basis_source,
)
from model_selection.pipeline.high_flow.outputs import write_high_flow_outputs
from model_selection.pipeline.results import HighFlowResult
from model_selection.pipeline.shared import (
    adequacy_plot_factory, common_basis, log_pearson, log_review_flags, max_decorrelation_lag,
    pearson_with_observations, resolve_rng, write_dependence_plot, zero_flow_check,
)
from model_selection.preprocessing.seasonal import SeasonalConfig, resolve_seasonal
from model_selection.preprocessing.transforms import prepare_series
from model_selection.selection.tiers import SelectionSample
from model_selection.selection.config import CASCADE, AdequacyConfig, SelectionConfig
from model_selection.selection.stage import run_selection_stage
from model_selection.statistics.power import describe_detectable, minimum_detectable_r2, rao_degrees

logger = logging.getLogger(__name__)


def _min_valid_neighbor_count(time_positions: np.ndarray, lag: int) -> int:
    """Smallest number of other samples at least `lag` away, over all samples."""
    positions = np.asarray(time_positions, dtype=np.float64)
    if positions.size == 0:
        return 0
    diffs = np.abs(positions[:, None] - positions[None, :])
    return int((diffs >= lag).sum(axis=1).min())


def _max_lag_for_min_neighbors(time_positions: np.ndarray, floor: int, max_lag: int) -> int:
    """Largest lag <= max_lag leaving every sample at least `floor` valid
    partners (binary search; the count is non-increasing in lag)."""
    if max_lag <= 0:
        return 0
    lo, hi, best = 0, max_lag, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _min_valid_neighbor_count(time_positions, mid) >= floor:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


@dataclasses.dataclass
class _EventStage:
    """Outputs of steps 1-2."""
    own_windows: dict
    own_peaks: dict | None
    metadata: dict
    consensus_events: list
    power_summary: dict
    consensus_rank_proxy: np.ndarray


def _detect_consensus_events(basin: BasinData, events: EventConfig,
                             raw_formulations: dict[str, np.ndarray]) -> _EventStage:
    high_flow_mask = rank_consensus_regime(
        raw_formulations, quantile_threshold=events.high_flow_quantile, consensus=events.consensus)

    own_windows, own_peaks, metadata = {}, {}, {}
    for name, series in basin.formulations.items():
        r = detect_events_for_series(series, events)
        own_windows[name] = r.decluster_result
        if r.peaks is not None:
            own_peaks[name] = r.peaks
        if r.metadata is not None:
            metadata[name] = r.metadata
    obs_result = detect_events_for_series(basin.obs, events)
    if obs_result.metadata is not None:
        metadata[OBS_KEY] = obs_result.metadata

    # obs-free proxy used to locate each event's peak (NaN where a formulation is missing)
    consensus_rank_proxy = consensus_rank(raw_formulations, "median")

    decluster_result = group_runs_by_own_events(high_flow_mask, own_windows)
    power_summary = independent_event_summary(decluster_result)
    power_summary["n_high_flow_timesteps"] = int(high_flow_mask.sum())
    consensus_events = summarize_events(decluster_result, peak_series=consensus_rank_proxy, index=basin.obs.index)
    return _EventStage(
        own_windows=own_windows,
        own_peaks=own_peaks if len(own_peaks) == len(own_windows) else None,
        metadata=metadata, consensus_events=consensus_events,
        power_summary=power_summary, consensus_rank_proxy=consensus_rank_proxy,
    )


def run_high_flow_analysis(
        basin: BasinData,
        selection: SelectionConfig | None = None,
        adequacy: AdequacyConfig | None = None,
        events: EventConfig | None = None,
        fpca: FPCAConfig | None = None,
        output: OutputConfig | None = None,
        rng: np.random.Generator | None = None,
        seasonal: SeasonalConfig | None = None,
) -> HighFlowResult:
    """Run the high-flow pathway for one basin (see module docstring).

    seasonal : the series preparation used to estimate the dependence lag
        (default: calendar-day mean removed, as in the pooled pathway); the
        events and their features are not adjusted for season
        (docs/methods.md, Seasonality).

    Writes {gage}_high_flow_result.json and the requested plots when
    output.output_dir is set. When fewer than two usable events remain,
    selection is skipped and result.selection is None.
    """
    selection = selection or SelectionConfig()
    adequacy = adequacy or AdequacyConfig()
    events = events or EventConfig()
    fpca = fpca or FPCAConfig()
    output = output or OutputConfig()
    rng, seed = resolve_rng(rng, selection.seed, basin.gage, "high-flow")
    notes: list[str] = []
    # Detection uses each series' own values; the analysis uses the common basis.
    gridded = on_regular_grid(basin)
    basin, missing_values = common_basis(gridded, notes)
    seasonal, too_short = resolve_seasonal(
        seasonal, basin.obs.index[np.isfinite(basin.obs.values.astype(float))])
    if too_short:
        notes.append(too_short)
    gage = basin.gage
    output_dir = output.output_dir

    zero_flow, zero_flow_flag = zero_flow_check(basin, notes)
    # Event curves use the formulations' own values: an observation gap within an event is handled when the
    # observed curve is scored, and does not remove the formulations' values (docs/methods.md, Missing values).
    raw_formulations = {name: series.values.astype(float) for name, series in gridded.formulations.items()}

    min_temporal_lag = selection.min_temporal_lag
    if min_temporal_lag is None:
        window = seasonal.window_days if seasonal.enabled else None
        prepared = {name: prepare_series(s.values.astype(float), rng=rng,
                                         index=s.index, seasonal_window_days=window)
                    for name, s in basin.formulations.items()}
        min_temporal_lag, per_formulation = max_decorrelation_lag(prepared, selection.dependence_run_length)
        prep = (f"with their calendar-day means removed ({seasonal.window_days}-day window)"
                if seasonal.enabled else "with their seasonal cycles")
        notes.append(
            f"min_temporal_lag (dependence lag) auto-estimated as {min_temporal_lag} timesteps: the largest "
            f"Politis decorrelation lag across the simulated formulations, prepared as in the pooled pathway and {prep} "
            "(obs-free; " + ", ".join(f"{k}={v}" for k, v in per_formulation.items())
            + "); pass min_temporal_lag explicitly to override."
        )
    else:
        notes.append(f"min_temporal_lag={min_temporal_lag} (caller-specified).")
    if output.dependence_plot and output.output_dir is not None:
        # A separate random stream for the plot's preparation, so the analysis' draws are unchanged.
        plot_rng = np.random.default_rng(0)
        window = seasonal.window_days if seasonal.enabled else None
        plotted = {name: prepare_series(s.values.astype(float), rng=plot_rng, index=s.index, seasonal_window_days=window)
                   for name, s in basin.formulations.items()}
        plotted_obs = prepare_series(basin.obs.values.astype(float), rng=plot_rng, index=basin.obs.index,
                                     seasonal_window_days=window)
        write_dependence_plot(output, gage, plotted, plotted_obs, basin.obs.index, selection.dependence_run_length,
                              window)
    notes.append(f"Per-series event detection: method={events.method!r}"
                 + (f", {events.pot}" if events.method == "pot" else f", {events.wasp}") + ".")
    notes.append("Consensus events: high-flow runs joined when at least half of the formulations' own events span "
                 "both; widened windows separated at the minimum between peaks.")

    own_formulations = {name: s.values.astype(float) for name, s in gridded.formulations.items()}
    stage = _detect_consensus_events(gridded, events, own_formulations)
    if events.method == "wasp":
        notes.append("WASP peak/noise split significance per series is reported in wasp_metadata; an "
                     "insignificant split still yields events (best available split), flagged there.")
    power_summary = stage.power_summary
    n_raw_consensus_events = len(stage.consensus_events)

    result_selection = None
    high_flow_events: list[ExpandedEvent] = []
    committed_tier, attempted_tiers = None, []
    fpca_fit_source, fpca_cumulative = "", float("nan")
    fpca_orthogonality_error, fpca_explained = 0.0, []
    fpca_n_grid = None
    fidelity_by_series, fidelity_by_event = {}, []
    peak_phase_spread, ensemble = {}, {}
    sensitivity = None
    detectable_r2 = None
    power_stop_reason = None
    n_components = 0
    event_gallery_path = overview_diagnostic_path = None

    # Events were detected on each series' own values.
    usable_events = []
    if n_raw_consensus_events >= 2:
        all_events = attach_timestamps(build_expanded_events(
            stage.consensus_events, stage.own_windows, peak_series=stage.consensus_rank_proxy,
            own_peaks=stage.own_peaks), basin.obs.index)
        n_without_peak = n_raw_consensus_events - len(all_events)
        power_summary["n_consensus_events_without_peak"] = n_without_peak
        if n_without_peak:
            notes.append(
                f"{n_without_peak} of {n_raw_consensus_events} consensus high-flow periods contain no peak detected "
                "by any formulation's own event detector and are not events; where they adjoin an event, their "
                "timesteps return to it.")
        obs_values = basin.obs.values.astype(float)
        # An event needs every formulation complete over its window and at least one observed value.
        usable_events = [ev for ev in all_events
                         if complete_mask([v[ev.start:ev.end + 1] for v in raw_formulations.values()]).all()
                         and np.isfinite(obs_values[ev.start:ev.end + 1]).any()]
        n_obs_gaps = sum(not np.isfinite(obs_values[ev.start:ev.end + 1]).all() for ev in usable_events)
        if len(usable_events) < len(all_events):
            notes.append(
                f"{len(all_events) - len(usable_events)} of {len(all_events)} events excluded because a formulation "
                "lacks values within the event or the observations have none.")
        if n_obs_gaps:
            notes.append(
                f"{n_obs_gaps} of {len(usable_events)} events have gaps in the observations; their observed curves are "
                "scored from the observed points (conditional expectation under the fitted basis).")
        power_summary["n_events_with_missing_values"] = len(all_events) - len(usable_events)
        power_summary["n_events_with_observation_gaps"] = n_obs_gaps

    if len(usable_events) < 2:
        power_stop_reason = (f"{len(usable_events)} usable consensus high-flow event(s); at least 2 are needed "
                             "for any test.")
        notes.append(f"High-flow selection not run: {power_stop_reason}")
        power_summary["n_fpc_events"] = 0
    else:
        expanded_events = usable_events
        table, fpca_result, event_curves = build_fpc_score_table(
            raw_formulations, expanded_events, obs=obs_values, config=fpca)
        fpca_fit_source = fpca_result.fit_source
        fpca_cumulative = float(fpca_result.cumulative_explained_variance_ratio[-1])
        fpca_orthogonality_error = fpca_result.component_orthogonality_error
        fpca_n_grid = fpca_result.n_grid
        fpca_explained = [float(v) for v in fpca_result.explained_variance_ratio]
        n_components = fpca_result.components.shape[0]
        notes.append(
            f"FPCA basis: {fpca_result.fit_source}, n_curves_fit={fpca_result.n_curves_fit}, "
            f"n_components={n_components} ({fpca_result.selection_note}), "
            f"cumulative_explained_variance={fpca_cumulative:.3f}."
        )

        fidelity_by_series, fidelity_by_event = diagnostics.reconstruction_fidelity(table)
        obs_stats = fidelity_by_series.get(OBS_KEY, {})
        notes.append(
            f"Reconstruction fidelity (obs): median RMSE={obs_stats.get('rmse_median', float('nan')):.4f}, "
            f"median R2={obs_stats.get('r2_median', float('nan')):.3f} (see fpca_reconstruction_fidelity "
            "and reconstruction_fidelity_by_event)."
        )
        peak_phase_df = analyze_peak_phase_spread(event_curves, n_grid=fpca_result.n_grid)
        if len(peak_phase_df) > 0:
            peak_phase_spread = summarize_peak_phase_spread(peak_phase_df)

        # Every series must retain the same events (curves share windows).
        retained_by_series = {key: set(table.loc[table["series"] == key, "event_id"])
                              for key in list(raw_formulations) + [OBS_KEY]}
        retained_event_ids = sorted(set.intersection(*retained_by_series.values()))
        if any(ids != set(retained_event_ids) for ids in retained_by_series.values()):
            raise ValueError("FPC score table retained different events per series; every series is "
                             "sliced on the same windows, so this indicates an upstream bug.")
        n_fpc_events = len(retained_event_ids)
        power_summary["n_fpc_events"] = n_fpc_events

        # Whether the formulations' shapes cover the observed ones is judged on a basis fitted to the
        # formulations only: a basis that includes the observations reconstructs them better by construction.
        if fpca.basis_fit_source == "pooled":
            ensemble_by_event = fidelity_by_event
        else:
            ensemble_table, _, _ = build_fpc_score_table(
                raw_formulations, expanded_events, obs=obs_values, config=with_basis_source(fpca, "pooled"))
            ensemble_by_event = diagnostics.reconstruction_fidelity(ensemble_table)[1]
        ensemble = diagnostics.ensemble_representation(ensemble_by_event, fpca)
        if ensemble.get("low_confidence"):
            notes.append(
                f"LOW CONFIDENCE: {ensemble['fraction_events_poorly_represented']:.0%} of events have obs "
                f"reconstruction R2 below {fpca.ensemble_r2_threshold} (threshold: "
                f"{fpca.ensemble_fraction_threshold:.0%} of events); the model ensemble may not represent "
                "some observed behaviour, which would make high-flow results overstate model quality."
            )


        if output_dir is not None and n_fpc_events >= 2:
            from model_selection.visualization.fpca import plot_full_event_gallery, plot_overview_diagnostics
            from model_selection.visualization.html import write_multi_figure_html, write_standalone_html
            if output.event_gallery:
                event_gallery_path = os.path.join(output_dir, f"{gage}_fpca_event_gallery.html")
                write_standalone_html(plot_full_event_gallery(event_curves, table, fpca_result),
                                      event_gallery_path, f"{gage} FPCA event gallery")
                logger.info(f"[+] Wrote high-flow event gallery for {gage} to {event_gallery_path}")
            overview_diagnostic_path = os.path.join(output_dir, f"{gage}_fpca_overview.html")
            write_multi_figure_html(plot_overview_diagnostics(fpca_result),
                                    overview_diagnostic_path, f"{gage} FPCA overview diagnostics")
            logger.info(f"[+] Wrote high-flow overview diagnostics for {gage} to {overview_diagnostic_path}")

        if n_fpc_events < 2:
            power_stop_reason = (f"{n_fpc_events} event(s) survived min_event_length={fpca.min_event_length}; "
                                 "at least 2 are needed for any test.")
            notes.append(f"High-flow selection not run: {power_stop_reason}")
        else:
            event_by_id = {ev.event_id: ev for ev in expanded_events}
            obs_features, candidate_events = event_feature_matrices(
                table, retained_event_ids, list(raw_formulations), n_components, fpca)
            # Every test conditions on the parameters of the events' shared registration.
            registration = registration_covariates(expanded_events, retained_event_ids)

            # Can the event sample support the tests at all?
            # See docs/methods.md, Detectable effect and the meaningful-effect threshold.
            n_dims = obs_features.shape[1]
            n_reg = registration.shape[1]
            detectable_r2 = minimum_detectable_r2(n_fpc_events, n_dims, n_dims, n_reg, selection.alpha)
            if rao_degrees(n_fpc_events, n_dims, n_dims, n_reg).v <= 0:
                power_stop_reason = (
                    f"{n_fpc_events} events cannot support a test with {n_dims}-dimensional event features "
                    "(no denominator degrees of freedom remain)."
                )
            detectability = describe_detectable(detectable_r2, n_dims)
            notes.append(f"Detectability (round 1, {n_fpc_events} events, {n_dims} dimensions): "
                         f"{detectability['statement']}")
            logger.info(f"[i] {gage} high-flow: {detectability['statement']}")
            if power_stop_reason:
                notes.append(f"High-flow selection not run: {power_stop_reason}")
            else:
                time_positions = np.array([
                    event_by_id[eid].peak_index if event_by_id[eid].peak_index is not None else event_by_id[eid].start
                    for eid in retained_event_ids
                ])

                hf_n_neighbors = min(selection.n_neighbors, max(2, n_fpc_events - 1))
                if hf_n_neighbors < selection.n_neighbors:
                    notes.append(f"high-flow permutation neighbourhood clamped to {hf_n_neighbors} from "
                                 f"{selection.n_neighbors}: only {n_fpc_events} event samples.")

                # The free-form tier's Theiler window must leave every event enough
                # permutation partners. Only the local permutation needs the shorter
                # window; every other test keeps the full dependence lag.
                partner_floor = min(hf_n_neighbors, 5)
                min_valid = _min_valid_neighbor_count(time_positions, min_temporal_lag)
                theiler_window = None
                if min_valid < partner_floor:
                    theiler_window = _max_lag_for_min_neighbors(time_positions, partner_floor, min_temporal_lag)
                    notes.append(
                        f"high-flow free-form tier's Theiler window reduced from {min_temporal_lag} to "
                        f"{theiler_window}: at {min_temporal_lag} the worst-placed event had only {min_valid} other "
                        f"events that far away (need >= {partner_floor}); every other test keeps the dependence "
                        f"lag of {min_temporal_lag}."
                    )

                feature_names = tuple(f"PC{c + 1}" for c in range(n_components))
                sample = SelectionSample(candidate_events, obs_features, time_positions, min_temporal_lag,
                                         feature_names=feature_names, covariates=registration,
                                         covariate_label="the registration parameters (window length and peak "
                                                         "position)")
                outcome = run_selection_stage(
                    sample, dataclasses.replace(selection, n_neighbors=hf_n_neighbors, theiler_window=theiler_window),
                    adequacy,
                    multivariate=n_dims > 1, rng=rng,
                    plot_factory=adequacy_plot_factory(output, gage, "high_flow", feature_names),
                    label=f"high-flow, basin {gage}",
                )
                result_selection = outcome.selection
                committed_tier, attempted_tiers = outcome.committed_tier, outcome.attempted_tiers

                sensitivity = diagnostics.basis_sensitivity_check(
                    raw_formulations, obs_values, expanded_events, retained_event_ids,
                    result_selection, time_positions, tier=outcome.tier,
                    n_permutations=selection.resolved_n_permutations, alpha=selection.alpha,
                    dependence_lag=min_temporal_lag, n_components=n_components, fpca_config=fpca,
                    reconstruction_by_series=fidelity_by_series, rng=rng,
                    registration=registration,
                )
                if sensitivity is not None:
                    logger.info(f"[i] Basis sensitivity check for {gage}: "
                                + ("a candidate was accepted on another basis; review required."
                                   if sensitivity["requires_closer_inspection"]
                                   else "no candidate accepted on any alternative basis."))
                    accepted_on = [b["basis"] for b in sensitivity["per_basis"] if b["accepted"]]
                    notes.append(
                        f"Basis sensitivity check: the stopping round was repeated on {len(sensitivity['per_basis'])} "
                        f"alternative basis/bases (each remaining candidate's own curves), at level "
                        f"{sensitivity['level']:.3g} each. A candidate was accepted on: {accepted_on or 'none'} "
                        "(see basis_sensitivity_check)."
                    )
                high_flow_events = [ev for ev in expanded_events if ev.event_id in retained_event_ids]

    in_events = np.zeros(len(basin.obs), dtype=bool)
    for ev in usable_events:
        in_events[ev.start:ev.end + 1] = True
    pearson = (pearson_with_observations(basin.obs.values, raw_formulations, in_events)
               if usable_events else {})
    log_pearson(gage, "high-flow", "the time steps inside the high-flow events", pearson)
    result = HighFlowResult(
        seed=seed, cmi_calibration=selection.cmi_calibration, pearson_correlation=pearson,
        gage=gage, n_timesteps=len(basin.obs),
        zero_flow_fraction=zero_flow, zero_flow_flag=zero_flow_flag,
        selection=result_selection, high_flow_power=power_summary,
        high_flow_events=high_flow_events, n_raw_consensus_events=n_raw_consensus_events,
        n_components=n_components, fpca_fit_source=fpca_fit_source,
        fpca_cumulative_explained_variance=fpca_cumulative,
        fpca_component_orthogonality_error=fpca_orthogonality_error, fpca_n_grid=fpca_n_grid,
        fpca_regularization_lambda=fpca.regularization_lambda,
        fpca_explained_variance_ratio=fpca_explained,
        fpca_reconstruction_fidelity={"by_series": fidelity_by_series} if fidelity_by_series else {},
        peak_phase_spread=peak_phase_spread,
        reconstruction_fidelity_by_event=fidelity_by_event,
        basis_sensitivity_check=sensitivity,
        requires_closer_inspection=sensitivity["requires_closer_inspection"] if sensitivity else False,
        ensemble_representation=ensemble,
        min_temporal_lag=min_temporal_lag,
        missing_values=missing_values,
        seasonal_conditioning={"enabled": seasonal.enabled,
                                  "window_days": seasonal.window_days if seasonal.enabled else None,
                                  "applies_to": "dependence lag estimate",
                                  "record_too_short": too_short is not None},
        pot_ratio=events.pot.pot_ratio, recession_ratio=events.pot.recession_ratio, notes=notes,
        wasp_metadata=stage.metadata or None,
        event_gallery_path=event_gallery_path, overview_diagnostic_path=overview_diagnostic_path,
        measure=selection.measure, committed_tier=committed_tier, attempted_tiers=attempted_tiers,
        adequacy_floor=adequacy.resolved_effect_size_floor(selection) if selection.measure == CASCADE else None,
        detectable_r2=detectable_r2, meaningful_r2=selection.meaningful_r2, power_stop_reason=power_stop_reason,
    )
    if output_dir is not None:
        # Plots show each formulation's own values and the observations with their gaps.
        json_path, plot_path, _ = write_high_flow_outputs(
            BasinData(gage=gage, formulations=gridded.formulations, obs=basin.obs), result, output, own_windows=stage.own_windows, raw_formulations=raw_formulations,
            high_flow_quantile=events.high_flow_quantile)
        result.result_json_path, result.diagnostic_plot_path = json_path, plot_path
    log_review_flags(result, "high-flow")
    return result
