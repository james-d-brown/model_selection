"""The basin data container and the full ingestion of one basin: model
parquets plus USGS observations, aligned on one regular time grid.

Missing values are explicit: every series is placed on the same regular grid
(the series' common modal time step, spanning all their timestamps), with NaN
wherever a series has no value. No timestep is dropped here; each analysis
selects the timesteps it can use, on the same basis for every formulation
(docs/methods.md, Missing values)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_selection.io.parquet import load_basin_parquets
from model_selection.io.units import convert_streamflow_units
from model_selection.io.usgs import pull_usgs_obs

logger = logging.getLogger(__name__)

OBS_KEY = "__obs__"
"""Key under which observations appear alongside model names in per-series dicts."""


@dataclass
class BasinData:
    """Model formulations and observations for one gage, on a shared index
    (see on_regular_grid); NaN marks a missing value."""
    gage: str
    formulations: dict[str, pd.Series]
    obs: pd.Series


def regular_index(indexes: list[pd.Index]) -> pd.Index:
    """The regular grid spanning every timestamp of `indexes` at their modal
    step. Raises if a timestamp is not on that grid."""
    union = indexes[0]
    for idx in indexes[1:]:
        union = union.union(idx)
    union = union.sort_values()
    if len(union) < 2:
        return union
    step = union.to_series().diff().dropna().mode().iloc[0]
    if isinstance(union, pd.DatetimeIndex):
        grid = pd.date_range(union[0], union[-1], freq=step)
    else:
        grid = pd.Index(np.arange(union[0], union[-1] + step, step))
    if grid.get_indexer(union).min() < 0:
        raise ValueError("timestamps are not aligned to a regular grid at their modal step "
                         f"({step}); sub-step timestamps are not supported")
    return grid


def on_regular_grid(basin: "BasinData") -> "BasinData":
    """The basin with every series reindexed onto one regular grid (NaN where
    a series has no value). Idempotent."""
    series = [basin.obs, *basin.formulations.values()]
    grid = regular_index([s.index for s in series])
    if all(s.index.equals(grid) for s in series):
        return basin
    return BasinData(gage=basin.gage,
                     formulations={k: s.reindex(grid) for k, s in basin.formulations.items()},
                     obs=basin.obs.reindex(grid))


def complete_mask(series: list[np.ndarray]) -> np.ndarray:
    """True at timesteps where every series has a finite value."""
    return np.logical_and.reduce([np.isfinite(np.asarray(s, dtype=float)) for s in series])


def align_formulations(formulations: dict[str, pd.Series]) -> pd.DataFrame:
    """Outer-join all formulations' series onto a common DatetimeIndex (NaN
    where a formulation has no value)."""
    df = pd.concat(formulations, axis=1, join="outer")
    if df.empty:
        raise ValueError("no timestamps in the formulations")
    return df


def align_obs_to_sim(obs: pd.Series, sim_index: pd.DatetimeIndex, resample_rule: str = "nearest") -> pd.Series:
    """Bring instantaneous obs onto the hourly simulation index, by hourly mean
    or by nearest sample within 30 minutes.
    """
    if resample_rule == "mean":
        hourly = obs.resample("1h").mean()
    elif resample_rule == "nearest":
        hourly = obs.reindex(sim_index, method="nearest", tolerance=pd.Timedelta("30min"))
    else:
        raise ValueError("resample_rule must be 'mean' or 'nearest'")

    return hourly.reindex(sim_index)


def load_basin(
        basin_dir: str,
        gage: str,
        start_date: str,
        end_date: str,
        resample_rule: str = "nearest",
        api_token_env_var: str = "API_USGS_PAT",
) -> BasinData:
    """Load every formulation for `gage` from basin_dir, pull USGS observations
    for [start_date, end_date], convert them to the simulation units and
    align everything on one regular grid, keeping missing values as NaN.
    """
    formulations, sim_unit = load_basin_parquets(basin_dir, gage)
    sim_df = align_formulations(formulations)

    obs_raw = pull_usgs_obs(gage, start_date, end_date, api_token_env_var=api_token_env_var)
    obs_converted = convert_streamflow_units(obs_raw, from_unit="cfs", to_unit=sim_unit)
    obs_aligned = align_obs_to_sim(obs_converted, sim_df.index, resample_rule=resample_rule)

    combined = sim_df.copy()
    combined[OBS_KEY] = obs_aligned
    if not complete_mask([combined[c].to_numpy() for c in combined.columns]).any():
        raise ValueError(f"no timestep of gage={gage} has values for every series")

    logger.info("[*] Aligned observations and simulations for " + gage)

    obs_final = combined.pop(OBS_KEY)
    formulations_final = {col: combined[col] for col in combined.columns}
    return on_regular_grid(BasinData(gage=gage, formulations=formulations_final, obs=obs_final))
