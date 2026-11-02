"""
Characteristic response timescale of a flow series: the ACF e-folding time,
the lag at which the autocorrelation first falls to 1/e (von Storch and
Zwiers, 1999). It is used only as the scale ceiling of the WASP event
detector, as a characteristic timescale of the series' response, not as a
decorrelation length; the significance tests use the Politis–White
estimates (statistics.significance.block_length).

Always compute it on the continuous record; when only separate segments are
available, pass segment_ids so no lag spans a segment boundary.

The series must carry a DatetimeIndex. max_lag and lag_step are calendar
durations, converted to native steps using the index's median spacing, so
the search is resolution-independent. The 365-day max_lag is a ceiling (the
search normally stops far sooner); going further risks matching seasonal
re-correlation. Lags are tested at 1-hour spacing whatever the native
resolution. The returned timescale is in native index steps.

See docs/methods.md, Event detection methods.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BasinTimescaleResult:
    """timescale is in native index steps (a censored lower bound when
    e_folding_found is False). acf, lags and n_pairs_by_lag cover only the
    lags evaluated before the search stopped; lag 0 is not included."""
    timescale: float
    acf: np.ndarray
    lags: np.ndarray
    e_folding_found: bool
    n_pairs_by_lag: np.ndarray


def _infer_native_timestep(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Median spacing of the index (robust to occasional gaps, unlike pd.infer_freq)."""
    if len(index) < 2:
        raise ValueError("series must have at least 2 timestamps to infer a native timestep")
    diffs = index.to_series().diff().dropna()
    return diffs.median()


def _segment_acf_at_lags_early_stopping(
    values: np.ndarray, segment_ids: np.ndarray, lags: np.ndarray, min_pairs: int, threshold: float,
) -> tuple[np.ndarray, np.ndarray, bool, int]:
    """ACF at each lag in `lags`, using only within-segment pairs, stopping as soon
    as fewer than min_pairs pairs remain or the ACF reaches `threshold`.
    Returns (acf, n_pairs, crossed_threshold, n_evaluated).
    """
    n = values.shape[0]
    mu = np.mean(values)
    var = np.var(values)
    centered = values - mu

    acf_vals, pair_vals = [], []
    for h in lags:
        h = int(h)
        valid = segment_ids[: n - h] == segment_ids[h:]
        n_pairs_h = int(np.sum(valid))
        pair_vals.append(n_pairs_h)
        if n_pairs_h < 2 or var <= 0:
            acf_h = 0.0
        else:
            acf_h = float(np.sum(centered[: n - h][valid] * centered[h:][valid]) / (n_pairs_h * var))
        acf_vals.append(acf_h)

        if n_pairs_h < min_pairs:
            return np.array(acf_vals), np.array(pair_vals, dtype=np.int64), False, len(acf_vals)
        if acf_h <= threshold:
            return np.array(acf_vals), np.array(pair_vals, dtype=np.int64), True, len(acf_vals)

    return np.array(acf_vals), np.array(pair_vals, dtype=np.int64), False, len(acf_vals)


def estimate_basin_timescale(
    series: pd.Series,
    segment_ids: np.ndarray | None = None,
    max_lag: pd.Timedelta = pd.Timedelta(days=365),
    lag_step: pd.Timedelta = pd.Timedelta(hours=1),
    min_pairs: int = 20,
) -> BasinTimescaleResult:
    """ACF e-folding time of `series` in native index steps, linearly interpolated
    between the tested lags either side of the 1/e crossing.

    series : time-indexed pandas Series (TypeError otherwise), the continuous
        record in time order.
    segment_ids : optional segment label per point; lags never cross segments.
    max_lag, lag_step : calendar search ceiling and lag spacing.
    min_pairs : the search stops once a lag has fewer within-segment pairs.
    """
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            "estimate_basin_timescale requires a pandas Series with a DatetimeIndex "
            "(see module docstring) -- pass the actual time-indexed series, not a "
            "raw .values array, so max_lag/lag_step can be expressed in real "
            "calendar time rather than an ambiguous, resolution-dependent number "
            "of index positions."
        )

    n = len(series)
    values = series.to_numpy(dtype=np.float64)

    if segment_ids is None:
        segment_ids_arr = np.zeros(n, dtype=np.int64)
    else:
        segment_ids_arr = np.asarray(segment_ids)
        if segment_ids_arr.shape[0] != n:
            raise ValueError("segment_ids must match series length")

    native_timestep = _infer_native_timestep(series.index)

    lag_step_native = max(1, int(round(lag_step / native_timestep)))
    max_lag_native = int(round(max_lag / native_timestep))
    max_lag_native = max(lag_step_native, max_lag_native)
    max_lag_native = min(max_lag_native, n - 1)

    lags_to_test = np.arange(lag_step_native, max_lag_native + 1, lag_step_native, dtype=np.int64)
    if lags_to_test.size == 0:
        # series shorter than one lag_step: test the largest available lag
        lags_to_test = np.array([max(1, n - 1)], dtype=np.int64)

    threshold = 1.0 / np.e
    acf, n_pairs, stopped_on_crossing, n_evaluated = _segment_acf_at_lags_early_stopping(
        values, segment_ids_arr, lags_to_test, min_pairs, threshold,
    )
    lags_evaluated = lags_to_test[:n_evaluated]

    e_folding_found = False
    timescale = float(lags_evaluated[-1]) if len(lags_evaluated) else 0.0
    if stopped_on_crossing:
        h = int(lags_evaluated[-1])
        prev_lag = int(lags_evaluated[-2]) if len(lags_evaluated) > 1 else 0
        prev_acf = float(acf[-2]) if len(lags_evaluated) > 1 else 1.0
        if prev_acf == acf[-1]:
            timescale = float(h)
        else:
            frac = (prev_acf - threshold) / (prev_acf - acf[-1])
            frac = float(np.clip(frac, 0.0, 1.0))
            timescale = float(prev_lag) + frac * float(h - prev_lag)
        e_folding_found = True

    return BasinTimescaleResult(
        timescale=timescale,
        acf=acf,
        lags=lags_evaluated,
        e_folding_found=e_folding_found,
        n_pairs_by_lag=n_pairs,
    )
