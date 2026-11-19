"""Conditioning on the time of year (docs/methods.md, Seasonality)."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from model_selection.preprocessing.seasonal import (
    MIN_YEARS, N_DAYS, SeasonalConfig, calendar_day, phase_residuals, resolve_seasonal, seasonal_phase,
    water_year, water_year_starts, years_per_calendar_day,
)


def _hourly(years=10, start="2001-01-01"):
    return pd.date_range(start, periods=int(years * 365.25 * 24), freq="h")


def _phase(index):
    return 2 * np.pi * calendar_day(index) / N_DAYS


def test_calendar_day_pools_29_february_with_28_february_and_aligns_years():
    idx = pd.DatetimeIndex(["2001-02-28", "2004-02-28", "2004-02-29", "2001-03-01", "2004-03-01",
                            "2001-12-31", "2004-12-31", "2004-01-01"])
    assert calendar_day(idx).tolist() == [58, 58, 58, 59, 59, 364, 364, 0]


def test_seasonal_phase_places_the_calendar_day_on_the_unit_circle():
    idx = pd.DatetimeIndex(["2001-01-01", "2001-04-02", "2001-12-31", "2004-02-29"])
    phase = seasonal_phase(idx)
    assert phase.shape == (4, 2) and np.allclose(np.hypot(phase[:, 0], phase[:, 1]), 1.0)
    angle = 2 * np.pi * calendar_day(idx) / N_DAYS
    assert np.allclose(phase, np.column_stack([np.sin(angle), np.cos(angle)]))


def test_phase_residuals_remove_a_function_of_the_calendar_day_only():
    # The fitted seasonal mean is one value per calendar day, the same in
    # every year, and the spread around it is not altered.
    rng = np.random.default_rng(0)
    idx = _hourly()
    phase = _phase(idx)
    sd = 1.0 + 0.8 * np.sin(phase)            # spread varies through the year
    x = 3.0 * np.sin(phase) + sd * rng.standard_normal(idx.size)
    r = phase_residuals(x, idx)
    fit = x - r
    day = calendar_day(idx)
    for d in (0, 100, 250):
        assert np.ptp(fit[day == d]) < 1e-9
    assert abs(np.corrcoef(r, np.sin(phase))[0, 1]) < 0.02
    high, low = np.sin(phase) > 0.7, np.sin(phase) < -0.7
    assert np.std(r[high]) / np.std(r[low]) > 3


def test_an_abrupt_seasonal_onset_is_removed_away_from_the_step():
    # A step in the seasonal mean (like snowmelt onset) leaves a residual
    # seasonal mean of more than 5% of the step only within 20 days of it,
    # about the narrowest bandwidth on the tests' grid.
    rng = np.random.default_rng(3)
    idx = _hourly(years=20)
    day = calendar_day(idx)
    x = np.where((day >= 120) & (day < 200), 5.0, 0.0) + rng.standard_normal(idx.size)
    r = phase_residuals(x, idx)
    mean_by_day = np.bincount(day, weights=r, minlength=N_DAYS) / np.bincount(day, minlength=N_DAYS)
    far = (np.abs(np.arange(N_DAYS) - 120) > 20) & (np.abs(np.arange(N_DAYS) - 200) > 20)
    assert np.all(np.abs(mean_by_day[far]) < 0.05 * 5.0)


def test_phase_residuals_ignore_and_keep_missing_values():
    rng = np.random.default_rng(4)
    idx = _hourly(years=3)
    x = rng.standard_normal(idx.size)
    x[1000:1500] = np.nan
    r = phase_residuals(x, idx)
    assert np.all(np.isnan(r[1000:1500])) and np.all(np.isfinite(np.delete(r, np.s_[1000:1500])))


def test_adjusting_for_the_phase_lets_the_decorrelation_lag_reflect_the_noise():
    # With the annual cycle present, the autocorrelation never decays as the
    # method assumes; after the adjustment it decays as the noise does.
    from model_selection.statistics.significance.block_length import estimate_dependence
    rng = np.random.default_rng(5)
    idx = _hourly(years=6)
    phi = np.exp(-1 / 24)
    noise = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal(idx.size))
    x = 2.0 * np.sin(_phase(idx)) + noise
    raw = estimate_dependence(x)
    adjusted = estimate_dependence(phase_residuals(x, idx))
    noise_only = estimate_dependence(noise)
    assert raw.decorrelation_lag > 10 * noise_only.decorrelation_lag
    assert adjusted.decorrelation_lag == pytest.approx(noise_only.decorrelation_lag, rel=0.5)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_lag_of_a_persistent_series_survives_a_seasonal_mean_and_onset(seed):
    # A persistent daily series (AR(1), phi = 0.95) with a seasonal mean and a
    # sharp onset: after the adjustment its lag is that of the series without
    # them, within the sampling variation of the lag itself.
    from model_selection.statistics.significance.block_length import estimate_dependence
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2001-01-01", periods=int(20 * 365.25), freq="D")
    day = calendar_day(idx)
    x = lfilter([1], [1, -0.95], rng.standard_normal(idx.size))
    x = (x - x.mean()) / x.std()
    k = (day - 120) % N_DAYS
    y = x + 2.0 * np.sin(_phase(idx)) + 3.0 * (1 - np.exp(-k / 2.0)) * np.exp(-k / 30.0)
    ratio = estimate_dependence(phase_residuals(y, idx)).decorrelation_lag / estimate_dependence(x).decorrelation_lag
    assert 0.8 <= ratio <= 1.25


def test_config_is_enabled_by_default():
    assert SeasonalConfig().enabled and not SeasonalConfig(enabled=False).enabled


def test_years_per_calendar_day_counts_distinct_years_per_day():
    assert years_per_calendar_day(_hourly(years=3)) == 3
    assert years_per_calendar_day(pd.date_range("2015-01-01", periods=3000, freq="h")) == 0  # 125 days
    # a gap of two summers: the median calendar day is still seen in every year
    idx = _hourly(years=5)
    kept = idx[~((idx.month.isin([7, 8])) & (idx.year.isin([2002, 2003])))]
    assert years_per_calendar_day(kept) == 5
    assert years_per_calendar_day(pd.DatetimeIndex([])) == 0


def test_seasonal_conditioning_is_kept_on_records_of_at_least_three_years():
    config, note = resolve_seasonal(None, _hourly(years=MIN_YEARS))
    assert config.enabled and note is None


def test_seasonal_conditioning_is_disabled_on_records_too_short_to_describe_the_cycle():
    config, note = resolve_seasonal(SeasonalConfig(), _hourly(years=2))
    assert not config.enabled
    assert "fewer than the 3 needed" in note


def test_disabled_seasonal_conditioning_stays_disabled_without_a_note():
    config, note = resolve_seasonal(SeasonalConfig(enabled=False), _hourly(years=10))
    assert not config.enabled and note is None


def test_water_years_run_from_1_october_to_30_september():
    index = pd.DatetimeIndex(["2000-09-30 23:00", "2000-10-01 00:00", "2001-09-30 23:00", "2001-10-01 00:00"])
    assert water_year(index).tolist() == [2000, 2001, 2001, 2002]
    assert water_year_starts(index).tolist() == [1, 3]
    assert water_year_starts(pd.date_range("2000-10-01", periods=48, freq="h")).size == 0


def test_the_seasonal_tuning_leaves_the_neighbouring_water_years_out_of_each_fit():
    from model_selection.preprocessing.seasonal import _tuning_fit_rows
    year = np.repeat(np.arange(2001, 2011), 3)
    assert sorted(set(year[_tuning_fit_rows(year, 2005)])) == [2001, 2002, 2003, 2007, 2008, 2009, 2010]
    assert sorted(set(year[_tuning_fit_rows(year, 2001)])) == list(range(2003, 2011))
    short = np.repeat([2001, 2002, 2003], 3)                 # the neighbours would leave no year to fit
    assert sorted(set(short[_tuning_fit_rows(short, 2002)])) == [2001, 2003]
