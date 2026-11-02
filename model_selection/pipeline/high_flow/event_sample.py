"""
Building the high-flow sample: one feature vector per event and series.

Every series is sliced on the same final event windows, each curve is
resampled onto a common phase grid, one FPCA basis is fitted (by default
to the formulations' and the observations' complete curves) and every
series, obs included, is scored on it, so PC1 means
the same shape mode for every series. The formulations' curves are complete
and are projected; the observed curves may have gauge gaps and are scored
by conditional expectation given their observed points (fpca.conditional_scores),
every observed curve alike. Features are then standardized
jointly across series, so that every series' features are on one scale; no
test depends on that scale (the free-form tier's estimator works on ranks).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from model_selection.io.basin import OBS_KEY
from model_selection.pipeline.high_flow.config import FPCAConfig
from model_selection.statistics.fpca.fpca import (
    FPCAResult,
    fit_regularized_functional_pca,
    n_components_for_variance,
    phase_normalize,
    conditional_scores,
    project,
    reconstruct,
    reconstruction_error,
)


def extract_event_curves(
        raw_formulations: dict[str, np.ndarray],
        events: list,
        obs: np.ndarray | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """{event_id: {series name: raw curve}} with every series sliced on the same
    [start, end] window; obs is included under OBS_KEY if given.
    """
    out: dict[int, dict[str, np.ndarray]] = {}
    for ev in events:
        sl = slice(ev.start, ev.end + 1)
        curves = {name: np.asarray(series[sl], dtype=np.float64) for name, series in raw_formulations.items()}
        if obs is not None:
            curves[OBS_KEY] = np.asarray(obs[sl], dtype=np.float64)
        out[ev.event_id] = curves
    return out


def build_phase_normalized_matrix(
        event_curves: dict[int, dict[str, np.ndarray]],
        series_key: str,
        n_grid: int = 50,
        min_event_length: int = 1,
) -> tuple[np.ndarray, list[int]]:
    """Phase-normalized curves of one series stacked over the events that have a
    curve of at least min_event_length samples. Returns (matrix, event_ids).
    """
    rows = []
    ids = []
    for event_id, curves in event_curves.items():
        if series_key not in curves:
            continue
        curve = curves[series_key]
        if curve.shape[0] < min_event_length:
            continue
        rows.append(phase_normalize(curve, n_grid=n_grid))
        ids.append(event_id)
    if not rows:
        return np.empty((0, n_grid)), []
    return np.vstack(rows), ids


def compute_peak_phase(curve: np.ndarray, n_grid: int = 50) -> float:
    """Relative phase in [0, 1] of the curve's maximum after phase normalization."""
    normalized = phase_normalize(curve, n_grid=n_grid)
    if not np.isfinite(normalized).any():
        return float("nan")
    return float(np.nanargmax(normalized) / (n_grid - 1))


def analyze_peak_phase_spread(
        event_curves: dict[int, dict[str, np.ndarray]],
        n_grid: int = 50,
) -> pd.DataFrame:
    """One row per (event, series): the phase of that curve's peak."""
    rows = []
    for event_id, curves in event_curves.items():
        for series, curve in curves.items():
            if curve.shape[0] < 2 or not np.isfinite(curve).any():
                continue
            rows.append({
                "event_id": event_id,
                "series": series,
                "peak_phase": compute_peak_phase(curve, n_grid=n_grid),
            })
    return pd.DataFrame(rows)


def summarize_peak_phase_spread(peak_phase_df: pd.DataFrame) -> dict:
    """Split peak-phase variability into within-event (across series for one
    storm) and across-event (storm to storm) standard deviations. Peak curves
    are deliberately not registered (aligned) before FPCA: that would erase
    the model-vs-obs timing differences selection is meant to detect.
    """
    overall_std = float(peak_phase_df["peak_phase"].std(ddof=1))
    per_event_mean = peak_phase_df.groupby("event_id")["peak_phase"].transform("mean")
    within_event_std = float((peak_phase_df["peak_phase"] - per_event_mean).std(ddof=1))
    event_means = peak_phase_df.groupby("event_id")["peak_phase"].mean()
    across_event_std = float(event_means.std(ddof=1))
    return {
        "overall_std": overall_std,
        "within_event_std_across_models": within_event_std,
        "across_event_std_storm_to_storm": across_event_std,
        "n_events": int(peak_phase_df["event_id"].nunique()),
        "n_rows": int(len(peak_phase_df)),
    }


def resolved_grid_length(events: list, min_event_length: int = 1) -> int:
    """Phase-grid length that resamples no event curve onto fewer points than
    it has: the longest event's length in timesteps (at least 2)."""
    lengths = [ev.duration_timesteps for ev in events if ev.duration_timesteps >= min_event_length]
    return max(2, max(lengths, default=2))


def build_fpc_score_table(
        raw_formulations: dict[str, np.ndarray],
        events: list,
        obs: np.ndarray | None = None,
        config: FPCAConfig | None = None,
) -> tuple[pd.DataFrame, FPCAResult, dict[int, dict[str, np.ndarray]]]:
    config = config or FPCAConfig()
    n_components = config.n_components
    variance_threshold = config.variance_threshold
    max_components = config.max_components
    min_event_length = config.min_event_length
    n_grid = config.n_grid or resolved_grid_length(events, min_event_length)
    basis_fit_source = config.basis_fit_source
    regularization_lambda = config.regularization_lambda
    """Extract curves, fit one shared FPCA basis and project every series onto it.

    Returns (table, fpca, event_curves). table has one row per (event, series)
    with event_id, series, pc1..pcK, duration_timesteps (the window length),
    reconstruction_rmse and reconstruction_r2.
    """
    obs_inclusive_sources = {"obs", "pooled_with_obs"}
    if basis_fit_source in obs_inclusive_sources and obs is None:
        raise ValueError(f"basis_fit_source={basis_fit_source!r} requires obs to be provided")

    event_curves = extract_event_curves(raw_formulations, events, obs=obs)
    series_keys = list(raw_formulations.keys()) + ([OBS_KEY] if obs is not None else [])
    model_keys = list(raw_formulations.keys())

    if basis_fit_source == "pooled":
        fit_source_keys = model_keys
        fit_label = "pooled (models only, obs-free)"
    elif basis_fit_source == "pooled_with_obs":
        fit_source_keys = series_keys
        fit_label = "pooled (models + obs)"
    elif basis_fit_source == "obs":
        fit_source_keys = [OBS_KEY]
        fit_label = "obs only"
    else:
        if basis_fit_source not in series_keys:
            raise ValueError(f"basis_fit_source={basis_fit_source!r} not found among {series_keys}")
        fit_source_keys = [basis_fit_source]
        fit_label = basis_fit_source

    mats = []
    for key in fit_source_keys:
        m, _ids = build_phase_normalized_matrix(event_curves, key, n_grid=n_grid, min_event_length=min_event_length)
        m = m[np.isfinite(m).all(axis=1)]  # the basis is fitted on complete curves only
        if m.shape[0] > 0:
            mats.append(m)
    fit_matrix = np.vstack(mats)

    if n_components is not None:
        chosen_n_components = n_components
        selection_note = f"fixed n_components={n_components}"
    else:
        chosen_n_components, capped = n_components_for_variance(fit_matrix, variance_threshold, max_components)
        selection_note = f"variance_threshold={variance_threshold}, chose n_components={chosen_n_components}"
        if capped:
            selection_note += f" (capped at {max_components} components, target not fully reached)"

    fpca = fit_regularized_functional_pca(fit_matrix, n_components=chosen_n_components, lam=regularization_lambda,
                                           fit_source=fit_label, selection_note=selection_note)

    duration_by_event = {ev.event_id: ev.duration_timesteps for ev in events}

    rows = []
    for key in series_keys:
        matrix, ids = build_phase_normalized_matrix(event_curves, key, n_grid=n_grid, min_event_length=min_event_length)
        if matrix.shape[0] == 0:
            continue
        scores = conditional_scores(matrix, fpca) if key == OBS_KEY else project(matrix, fpca)
        recon = reconstruct(scores, fpca)
        err = reconstruction_error(matrix, recon)
        for row_i, event_id in enumerate(ids):
            row = {
                "event_id": event_id,
                "series": key,
                "duration_timesteps": duration_by_event.get(event_id),
                "reconstruction_rmse": err["rmse"].iloc[row_i],
                "reconstruction_r2": err["r2"].iloc[row_i],
            }
            for c in range(chosen_n_components):
                row[f"pc{c + 1}"] = scores[row_i, c]
            rows.append(row)

    table = pd.DataFrame(rows)
    return table, fpca, event_curves


def standardize_features(matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Z-score every series' feature matrix with one per-column mean and std
    computed over the formulations pooled, observations excluded, so that
    the features stay observation-free (separate standardization would erase
    the between-series differences being measured). Zero-variance columns are
    left unscaled.
    """
    reference = [m for key, m in matrices.items() if key != OBS_KEY] or list(matrices.values())
    pooled = np.vstack(reference)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0, ddof=1)
    std_safe = np.where(std > 0, std, 1.0)
    return {name: (matrix - mean) / std_safe for name, matrix in matrices.items()}


def series_matrix(table: pd.DataFrame, series_key: str, event_ids: list[int], pc_cols: list[str]) -> np.ndarray:
    """(n_events, n_pcs) scores of one series, rows ordered by event_ids."""
    sub = table[table["series"] == series_key].set_index("event_id").reindex(event_ids)
    return sub[pc_cols].to_numpy()


def event_feature_matrices(
        table: pd.DataFrame,
        retained_event_ids: list[int],
        model_names: list[str],
        n_components: int,
        config: FPCAConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The event sample's feature matrices: each series' FPC scores,
    standardized jointly across all series (standardize_features).

    Returns (obs_features, {model: features}).
    """
    pc_cols = [f"pc{c + 1}" for c in range(n_components)]
    keys = [OBS_KEY] + list(model_names)
    matrices = {key: series_matrix(table, key, retained_event_ids, pc_cols) for key in keys}
    standardized = standardize_features(matrices)
    return standardized[OBS_KEY], {name: standardized[name] for name in model_names}


def registration_covariates(events: list, event_ids: list[int]) -> np.ndarray:
    """The parameters of each event's registration, one row per event in
    event_ids. Phase normalization maps the event window linearly onto [0, 1]
    for every series, a warp with two parameters: here the log of the window
    length (timesteps) and the position of the consensus peak within the
    window (0 at its start, 1 at its end). Both are computed from the
    formulations only (docs/methods.md, Event representation)."""
    by_id = {ev.event_id: ev for ev in events}
    rows = []
    for eid in event_ids:
        ev = by_id[eid]
        length = ev.end - ev.start + 1
        peak = ev.peak_index if ev.peak_index is not None else ev.start
        rows.append((np.log(length), (peak - ev.start) / max(length - 1, 1)))
    return np.array(rows, dtype=float)


def with_basis_source(config: FPCAConfig, source: str, **changes) -> FPCAConfig:
    """Copy of config fitting the basis on `source` instead."""
    return dataclasses.replace(config, basis_fit_source=source, **changes)
