"""USGS observed streamflow via dataretrieval.waterdata.get_continuous.

The API token is read from the API_USGS_PAT environment variable.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

logger = logging.getLogger(__name__)


def _build_chunk_envelopes(
        start_dt: pd.Timestamp, end_dt: pd.Timestamp, chunk_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start_dt, end_dt] into consecutive, non-overlapping windows."""
    chunk_edges = pd.date_range(start=start_dt, end=end_dt, freq=f"{chunk_days}D", tz="UTC")

    if chunk_edges.empty or chunk_edges[0] != start_dt:
        chunk_edges = chunk_edges.insert(0, start_dt)

    if chunk_edges[-1] < end_dt:
        chunk_edges = chunk_edges.append(pd.DatetimeIndex([end_dt]))

    return [(chunk_edges[i], chunk_edges[i + 1]) for i in range(len(chunk_edges) - 1)]


def _fetch_usgs_chunk(
        waterdata_module,
        location_id: str,
        parameter_code: str,
        c_start: pd.Timestamp,
        c_end: pd.Timestamp,
) -> pd.Series | None:
    """Fetch and normalize a single chunk. Runs inside a worker thread."""
    time_envelope = f"{c_start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{c_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    try:
        obs_df, _metadata = waterdata_module.get_continuous(
            monitoring_location_id=location_id,
            parameter_code=parameter_code,
            time=time_envelope,
        )
    except Exception as e:
        raise RuntimeError(f"API call failed for chunk {time_envelope}: {e}") from e

    if obs_df is None or obs_df.empty:
        return None

    if isinstance(obs_df.index, pd.MultiIndex):
        obs_df = obs_df.reset_index()

    if "time" not in obs_df.columns or "value" not in obs_df.columns:
        raise ValueError(f"Unexpected schema shape. Found: {list(obs_df.columns)}")

    working_df = obs_df.dropna(subset=["time"]).copy()

    working_df["time"] = pd.to_datetime(working_df["time"], errors="coerce")
    if working_df["time"].dt.tz is None:
        working_df["time"] = working_df["time"].dt.tz_localize("UTC")
    else:
        working_df["time"] = working_df["time"].dt.tz_convert("UTC")

    working_df = working_df.dropna(subset=["time"])

    if working_df.empty:
        return None

    return working_df.set_index("time")["value"].astype("float64")


def pull_usgs_obs(
        gage: str,
        start_date: str,
        end_date: str,
        api_token_env_var: str = "API_USGS_PAT",
        parameter_code: str = "00060",
        chunk_days: int = 1095,  # WDFN continuous-data API caps requests at ~3 years
        max_workers: int = 8,
) -> pd.Series:
    """Pull observed streamflow for `gage` between start_date and end_date.

    The record is split into chunks of at most chunk_days (the continuous-data
    API caps requests at about three years) fetched concurrently on a thread
    pool. api_token_env_var names an alternative variable to copy into
    API_USGS_PAT.
    """
    try:
        from dataretrieval import waterdata
    except ImportError as e:
        raise ImportError("dataretrieval is required") from e

    if api_token_env_var != "API_USGS_PAT":
        token = os.environ.get(api_token_env_var)
        if token:
            os.environ["API_USGS_PAT"] = token
    if not os.environ.get("API_USGS_PAT"):
        raise EnvironmentError("expected a USGS API token")

    location_id = gage if gage.upper().startswith("USGS-") else f"USGS-{gage}"

    logger.info("[*] Reading USGS observations for " + location_id + "...")

    start_dt = pd.Timestamp(start_date).tz_convert("UTC") if pd.Timestamp(start_date).tz else pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date).tz_convert("UTC") if pd.Timestamp(end_date).tz else pd.Timestamp(end_date, tz="UTC")

    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date")

    time_envelopes = _build_chunk_envelopes(start_dt, end_dt, chunk_days)

    all_chunks_series: list[pd.Series] = []
    errors: list[tuple[tuple[pd.Timestamp, pd.Timestamp], Exception]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(time_envelopes))) as pool:
        futures = {
            pool.submit(_fetch_usgs_chunk, waterdata, location_id, parameter_code, c_start, c_end): (c_start, c_end)
            for c_start, c_end in time_envelopes
        }
        for future in as_completed(futures):
            envelope = futures[future]
            try:
                result = future.result()
            except Exception as e:
                errors.append((envelope, e))
                continue
            if result is not None:
                all_chunks_series.append(result)

    if errors:
        first_envelope, first_err = errors[0]
        raise RuntimeError(
            f"{len(errors)} of {len(time_envelopes)} chunk(s) failed for gage={gage} "
            f"(first failure {first_envelope[0]}–{first_envelope[1]}): {first_err}"
        ) from first_err

    if not all_chunks_series:
        raise ValueError(f"No rows found for gage={gage}")

    full_series = pd.concat(all_chunks_series)
    full_series = full_series[~full_series.index.duplicated(keep="first")]

    logger.info("[+]...finished reading USGS observations for " + location_id)

    return full_series.sort_index()
