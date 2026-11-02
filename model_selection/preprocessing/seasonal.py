"""
Conditioning on the time of year (pooled pathway).

Every pooled test conditions on the calendar day d of each sample (Zhang et
al., 2017; Huang et al., 2020). Two parts implement it:

- calendar_day_anomalies: each series' calendar-day mean mu(d) is removed,
  w(t) = z(t) - mu(d). In the linear tier this is the conditioning itself
  (Frisch and Waugh, 1933; Lovell, 1963); in the other tiers, which also
  condition on d directly, subtracting a function of d from each series
  leaves the hypothesis unchanged. With a few decades of record each
  calendar day holds only a few tens of values, so mu(d) pools the days
  within a short window centred on d, with triangular weights that decrease
  to zero beyond the window, as daily climatologies do (a 5-day window for
  the ETCCDI percentile indices, Zhang et al., 2005). A short window keeps
  abrupt seasonal transitions, such as the onset of snowmelt, sharp.
- seasonal_phase: the calendar day as a point on the unit circle,
  (sin, cos) of 2 pi d / 365, the conditioning variable the nonlinear and
  CMI tiers use; the year has no start or end on a circle.

The calendar wraps at the year's end, and 29 February is pooled with 28
February.

The seasonal cycle can be estimated only when each calendar day recurs over
several years. resolve_seasonal therefore disables the conditioning when
the median calendar day has data in fewer than MIN_YEARS years, and the
pathway raises a review flag. On a shorter record the phase acts as a clock,
and conditioning on it would remove the record's own events.

See docs/methods.md, Seasonality.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

N_DAYS = 365

MIN_YEARS = 3
"""Fewest years in which the median calendar day must have data for the
seasonal cycle to be estimated (a convention)."""


@dataclass
class SeasonalConfig:
    """Conditioning on the time of year in the pooled pathway.

    enabled : remove each series' calendar-day mean before the rank-normal
        transform, and condition every test on the seasonal phase.
    window_days : full width of the calendar-day window of the mean (odd;
        the day itself and (window_days - 1) / 2 days on each side),
        triangular weights.
    """
    enabled: bool = True
    window_days: int = 11

    def __post_init__(self):
        if self.window_days < 1 or self.window_days % 2 == 0:
            raise ValueError(f"window_days must be a positive odd number -- got {self.window_days}")
        if self.window_days > N_DAYS:
            raise ValueError(f"window_days must be at most {N_DAYS} -- got {self.window_days}")


def calendar_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Calendar day 0..364 of each timestamp; 29 February maps to 28 February."""
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy() - 1
    leap_after_feb = index.is_leap_year & (doy >= 59)
    return np.where(leap_after_feb, doy - 1, doy).astype(int)


def seasonal_mean(values: np.ndarray, index: pd.DatetimeIndex, window_days: int) -> np.ndarray:
    """mu, of length 365: the weighted mean of `values` for every calendar
    day, pooling the calendar days within the window with triangular
    weights. NaN values are ignored."""
    values = np.asarray(values, dtype=float)
    day = calendar_day(index)
    ok = np.isfinite(values)
    count = np.bincount(day[ok], minlength=N_DAYS).astype(float)
    total = np.bincount(day[ok], weights=values[ok], minlength=N_DAYS)
    half = (window_days - 1) // 2
    offsets = np.arange(-half, half + 1)
    weights = 1.0 - np.abs(offsets) / (half + 1)
    n_w = np.zeros(N_DAYS)
    s_w = np.zeros(N_DAYS)
    for k, w in zip(offsets, weights):
        n_w += w * np.roll(count, -k)
        s_w += w * np.roll(total, -k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s_w / n_w


def calendar_day_anomalies(values: np.ndarray, index: pd.DatetimeIndex, window_days: int = 11) -> np.ndarray:
    """values minus their own calendar-day mean (seasonal_mean)."""
    values = np.asarray(values, dtype=float)
    return values - seasonal_mean(values, index, window_days)[calendar_day(index)]


def seasonal_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """(n, 2): sin and cos of 2 pi d / 365 for the calendar day d of each
    timestamp."""
    angle = 2.0 * np.pi * calendar_day(index) / N_DAYS
    return np.column_stack([np.sin(angle), np.cos(angle)])


def years_per_calendar_day(index: pd.DatetimeIndex) -> float:
    """Median over the 365 calendar days of the number of distinct years in
    which the day is observed (0 for days never observed)."""
    index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return 0.0
    pairs = np.unique(np.column_stack([calendar_day(index), index.year.to_numpy()]), axis=0)
    return float(np.median(np.bincount(pairs[:, 0], minlength=N_DAYS)))


def resolve_seasonal(config: SeasonalConfig | None, index: pd.DatetimeIndex) -> tuple[SeasonalConfig, str | None]:
    """The seasonal configuration to use on a record with timestamps `index`,
    and a note when the conditioning had to be disabled because the record
    is too short to estimate the seasonal cycle (MIN_YEARS). The pathways
    record the disabling in seasonal_conditioning["record_too_short"], which
    raises the seasonal_record_too_short review flag."""
    config = config or SeasonalConfig()
    if not config.enabled:
        return config, None
    years = years_per_calendar_day(index)
    if years >= MIN_YEARS:
        return config, None
    return replace(config, enabled=False), (
        f"Conditioning on the time of year disabled: the median calendar day has data in {years:g} year(s), "
        f"fewer than the {MIN_YEARS} needed to estimate the seasonal cycle; series analysed with their seasonal "
        "cycles.")
