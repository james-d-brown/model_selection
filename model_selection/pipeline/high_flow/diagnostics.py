"""
Checks on how far the high-flow result can be trusted: reconstruction
fidelity of the FPCA basis, whether the model ensemble represents observed
event shapes at all, and the basis sensitivity check run when selection
stops on a candidate that was not accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model_selection.io.basin import OBS_KEY
from model_selection.pipeline.high_flow.config import FPCAConfig
from model_selection.pipeline.high_flow.event_sample import (
    build_fpc_score_table,
    event_feature_matrices,
    with_basis_source,
)
from model_selection.selection.greedy import SelectionResult, gate
from model_selection.selection.tiers import Tier


def reconstruction_fidelity(table: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Per-series summary and per-(event, series) records of reconstruction
    RMSE and R^2 under the fitted basis."""
    by_series = {}
    for series_name, sub in table.groupby("series"):
        rmse = sub["reconstruction_rmse"]
        r2 = sub["reconstruction_r2"]
        by_series[series_name] = {
            "n_events": int(len(sub)),
            "rmse_median": float(rmse.median()),
            "rmse_p90": float(rmse.quantile(0.9)),
            "r2_median": float(r2.median()),
            "r2_min": float(r2.min()),
            "r2_below_neg10_count": int((r2 < -10).sum()),
        }
    by_event = [
        {
            "event_id": int(row.event_id),
            "series": row.series,
            "duration_timesteps": int(row.duration_timesteps) if row.duration_timesteps is not None else None,
            "reconstruction_rmse": float(row.reconstruction_rmse),
            "reconstruction_r2": float(row.reconstruction_r2),
        }
        for row in table.itertuples()
    ]
    return by_series, by_event


def ensemble_representation(by_event: list[dict], config: FPCAConfig) -> dict:
    """Low-confidence flag: too many events whose observed shape the
    ensemble-derived basis reconstructs poorly. Empty if obs has no events."""
    obs_r2 = np.array([r["reconstruction_r2"] for r in by_event if r["series"] == OBS_KEY])
    if len(obs_r2) == 0:
        return {}
    frac_poor = float((obs_r2 < config.ensemble_r2_threshold).mean())
    return {
        "n_events_checked": int(len(obs_r2)),
        "obs_r2_median": float(np.median(obs_r2)),
        "r2_threshold": config.ensemble_r2_threshold,
        "fraction_events_poorly_represented": frac_poor,
        "fraction_threshold": config.ensemble_fraction_threshold,
        "low_confidence": bool(frac_poor > config.ensemble_fraction_threshold),
    }


def basis_sensitivity_check(
        raw_formulations: dict[str, np.ndarray],
        obs: np.ndarray,
        events: list,
        retained_event_ids: list[int],
        selection: SelectionResult,
        time_positions: np.ndarray,
        tier: Tier,
        n_permutations: int,
        alpha: float,
        dependence_lag: int,
        n_components: int,
        fpca_config: FPCAConfig,
        reconstruction_by_series: dict,
        rng: np.random.Generator,
        registration: np.ndarray | None = None,
) -> dict | None:
    """Whether the decision that stopped selection depends on the FPCA basis.

    Runs only when selection stopped because its last tested winner was not
    accepted. The shared basis is fitted to all series' curves together; a
    remaining candidate whose distinctive behaviour lies in shapes that basis
    does not represent could be missed. For each remaining candidate, a basis
    is fitted to that candidate's own curves (still obs-free), every series
    is described on it exactly as in the main sample (same number of
    components, joint standardization, the same registration covariates in
    the conditioning set), and the
    stopping round is repeated under the committed tier: the same joint test
    of all remaining candidates given the selected formulations, with the
    same gate. With K alternative bases each is tested at level alpha / K
    (Bonferroni), so the chance that any basis accepts a candidate when none
    contributes is at most alpha. The observations' reconstruction R^2 under
    each basis is reported for information. Nothing found here changes the
    selection. Returns None when the check does not apply.
    """
    if (selection.ill_posed or selection.tier_rejected or not selection.trajectory
            or selection.trajectory[-1].accepted):
        return None
    already_selected = list(selection.selected)
    remaining = sorted(set(raw_formulations) - set(already_selected))
    if not remaining:
        return None
    level = alpha / len(remaining)
    obs_r2_pooled = reconstruction_by_series.get(OBS_KEY, {}).get("r2_median")

    per_basis = []
    for basis in remaining:
        config = with_basis_source(fpca_config, basis, n_components=n_components)
        table, _, _ = build_fpc_score_table(raw_formulations, events, obs=obs, config=config)
        obs_features, candidate_events = event_feature_matrices(
            table, retained_event_ids, list(raw_formulations), n_components, config)
        parts = ([] if registration is None else [registration]) + [candidate_events[s] for s in already_selected]
        z = np.hstack(parts) if parts else None
        if tier.name == "linear" and registration is not None:
            # As in the selection: the linear tier removes the registration from every series first.
            from model_selection.selection.tiers import SelectionSample, partial_out_covariates
            residual = partial_out_covariates(SelectionSample(candidate_events, obs_features, time_positions,
                                                              dependence_lag, covariates=registration))
            candidate_events, obs_features = residual.candidates, residual.obs
            z = np.hstack([candidate_events[s] for s in already_selected]) if already_selected else None
        result = tier.test.run({c: candidate_events[c] for c in remaining}, obs_features, z, n_permutations,
                               rng, time_positions, dependence_lag)
        w = result.index(result.winner)
        effect, _ = tier.test.magnitude(candidate_events[result.winner], obs_features, z, time_positions,
                                        dependence_lag, float(result.observed[w]), result.null[:, w])
        accepted, p_value, bias_corrected = gate(float(result.observed[w]), result.null[:, w],
                                                 result.null.max(axis=1), level, tier.effect_size_floor,
                                                 effect=effect)
        obs_rows = table[table["series"] == OBS_KEY]
        per_basis.append({
            "basis": basis,
            "winner": result.winner,
            "association": float(result.observed[w]),
            "p_value": p_value,
            "effect": bias_corrected,
            "accepted": accepted,
            "obs_reconstruction_r2_median": float(obs_rows["reconstruction_r2"].median()) if len(obs_rows) else None,
        })

    return {
        "triggered": True,
        "stopping_candidate": selection.trajectory[-1].formulation,
        "selected": already_selected,
        "level": level,
        "obs_reconstruction_r2_median_shared_basis": obs_r2_pooled,
        "per_basis": per_basis,
        "requires_closer_inspection": any(b["accepted"] for b in per_basis),
    }
