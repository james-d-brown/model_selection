"""Reading model output: one parquet file per formulation per gage, with
feature_id, formulation and measurement_unit in the file metadata."""
from __future__ import annotations

import glob
import os

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = {"sim_flow"}

REQUIRED_METADATA_KEYS = {"feature_id", "formulation", "measurement_unit"}


def load_basin_parquets(basin_dir: str, gage: str) -> tuple[dict[str, pd.Series], str]:
    """Load every formulation's simulated flow for `gage` from basin_dir.
    Returns ({formulation: series indexed in UTC}, measurement unit).
    """
    paths = sorted(glob.glob(os.path.join(basin_dir, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no parquet files found in {basin_dir}")

    formulations: dict[str, pd.Series] = {}
    target_unit: str | None = None

    for path in paths:
        try:
            meta = pq.read_metadata(path)
            custom_meta = meta.metadata or {}
        except Exception as e:
            raise RuntimeError(f"Failed to parse parquet metadata for {path}: {e}")

        decoded_meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in custom_meta.items()}

        missing = REQUIRED_METADATA_KEYS - set(decoded_meta.keys())
        if missing:
            raise ValueError(f"{path} missing required structural metadata keys: {missing}")

        file_gage = decoded_meta["feature_id"]
        file_formulation = decoded_meta["formulation"]
        file_unit = decoded_meta["measurement_unit"]

        if file_gage != gage:
            continue

        if target_unit is not None and file_unit != target_unit:
            raise ValueError(
                f"Mismatched measurement units for gage {gage} across files. Expected '{target_unit}', found '{file_unit}'"
            )
        target_unit = file_unit

        df = pd.read_parquet(path)

        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(f"{path} missing required columns: {missing_cols}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{path} does not have a DatetimeIndex")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        if file_formulation in formulations:
            raise ValueError(f"duplicate formulation '{file_formulation}'")

        formulations[file_formulation] = df["sim_flow"].astype("float32").sort_index()

    if not formulations:
        raise FileNotFoundError(f"no rows found for gage={gage} in {basin_dir}")

    return formulations, target_unit
