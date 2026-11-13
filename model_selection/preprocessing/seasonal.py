"""
Conditioning on the time of year (pooled pathway).

Every pooled test conditions on the calendar day d of each sample (Zhang et
al., 2017; Huang et al., 2020), through the seasonal phase: the calendar day
as a point on the unit circle, (sin, cos) of 2 pi d / 365, so that the year
has no start or end. The series are analysed with their seasonal cycles. The calendar wraps at the year's end, and
29 February is pooled with 28 February.

The dependence lag is estimated from each series after an adjustment for
the time of year (phase_residuals): the series minus a kernel ridge
regression on the phase, fitted to the whole record, so that the seasonal
cycle does not make its autocorrelation oscillate.

The seasonal cycle can be described only when each calendar day recurs over
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
seasonal cycle to be described (a convention)."""

@dataclass
class SeasonalConfig:
    """Conditioning on the time of year in the pooled pathway.

    enabled : condition every test on the seasonal phase, and estimate the
        dependence lag from the series adjusted for it (phase_residuals).
    """
    enabled: bool = True


def calendar_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Calendar day 0..364 of each timestamp; 29 February maps to 28 February."""
    index = pd.DatetimeIndex(index)
    doy = index.dayofyear.to_numpy() - 1
    leap_after_feb = index.is_leap_year & (doy >= 59)
    return np.where(leap_after_feb, doy - 1, doy).astype(int)


def seasonal_phase(index: pd.DatetimeIndex) -> np.ndarray:
    """(n, 2): sin and cos of 2 pi d / 365 for the calendar day d of each
    timestamp."""
    angle = 2.0 * np.pi * calendar_day(index) / N_DAYS
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _weighted_ridge(K: np.ndarray, count: np.ndarray, mean: np.ndarray, penalties) -> list[np.ndarray]:
    """Kernel ridge on the distinct calendar days, each weighted by its
    sample count: coefficients for every penalty, from one
    eigendecomposition."""
    w = np.sqrt(count)
    e, Q = np.linalg.eigh(w[:, None] * K * w[None, :])
    g = Q.T @ (w * mean)
    return [w * (Q @ (g / (e + lam))) for lam in penalties]


def phase_residuals(values: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """`values` minus their kernel ridge regression on the seasonal phase,
    fitted to the whole record. The kernel, its bandwidths (multiples of the
    median distance between the standardized phases) and the penalties are
    those of the tests' regressions (statistics.regression.kernel_residuals);
    the bandwidth and penalty are chosen by cross-validation leaving out one
    calendar year at a time, so that every fit sees the whole seasonal cycle.
    With 365 distinct phases, kernel ridge on the samples equals kernel ridge
    on the observed calendar days with each day's sample count as its
    weight. The final fit is not cross-fitted: fits to different parts of
    the record differ slightly, and their differences would add steps
    between the parts, which look like slow dependence. NaN stays NaN."""
    from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth
    from model_selection.statistics.regression.kernel_residuals import BANDWIDTH_MULTIPLIERS, RIDGE_PENALTIES

    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    ok = np.isfinite(values)
    if not ok.any():
        return out
    index = pd.DatetimeIndex(index)
    v, day, year = values[ok], calendar_day(index)[ok], index.year.to_numpy()[ok]
    phase = seasonal_phase(index)[ok]
    centre, scale = phase.mean(axis=0), phase.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    sample = (phase - centre) / scale
    sample = sample[np.linspace(0, len(sample) - 1, min(len(sample), 2000)).round().astype(int)]
    base = median_heuristic_bandwidth(sample)
    u = 2.0 * np.pi * np.arange(N_DAYS) / N_DAYS
    points = (np.column_stack([np.sin(u), np.cos(u)]) - centre) / scale
    sq = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)

    def fit(K, rows, penalties):
        count = np.bincount(day[rows], minlength=N_DAYS).astype(float)
        total = np.bincount(day[rows], weights=v[rows], minlength=N_DAYS)
        seen = count > 0
        level = total.sum() / count.sum()
        coefs = _weighted_ridge(K[np.ix_(seen, seen)], count[seen], total[seen] / count[seen] - level, penalties)
        return [level + K[:, seen] @ c for c in coefs]

    years = np.unique(year)
    choice = (BANDWIDTH_MULTIPLIERS[-1], RIDGE_PENALTIES[-1])
    if years.size >= 2:
        best = np.inf
        for m in BANDWIDTH_MULTIPLIERS:
            K = np.exp(-sq / (2.0 * (base * m) ** 2))
            err = np.zeros(len(RIDGE_PENALTIES))
            for y in years:
                held = year == y
                for j, curve in enumerate(fit(K, ~held, RIDGE_PENALTIES)):
                    err[j] += float(np.sum((v[held] - curve[day[held]]) ** 2))
            j = int(np.argmin(err))
            if err[j] < best:
                best, choice = err[j], (m, RIDGE_PENALTIES[j])
    m, lam = choice
    curve = fit(np.exp(-sq / (2.0 * (base * m) ** 2)), np.ones(v.size, dtype=bool), [lam])[0]
    out[ok] = v - curve[day]
    return out


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
    is too short to describe the seasonal cycle (MIN_YEARS). The pathways
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
        f"fewer than the {MIN_YEARS} needed to describe the seasonal cycle; the tests do not condition on the "
        "seasonal phase.")
