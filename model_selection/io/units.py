"""Streamflow unit conversion."""
from __future__ import annotations

import pandas as pd


def convert_streamflow_units(series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    """Convert a streamflow pandas Series between cfs and metric volumetric units."""
    src = from_unit.strip().lower()
    dst = to_unit.strip().lower()

    if src == dst:
        return series

    metric_aliases = {"m3/s", "cms", "cm/s"}
    CFS_TO_CMS = 0.028316846592

    if src == "cfs" and dst in metric_aliases:
        return series * CFS_TO_CMS
    elif src in metric_aliases and dst == "cfs":
        return series / CFS_TO_CMS
    else:
        raise ValueError(
            f"Unsupported or ambiguous unit conversion requested: from '{from_unit}' to '{to_unit}'"
        )
