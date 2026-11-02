"""Conditioning on the time of year (docs/methods.md, Seasonality)."""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from model_selection.preprocessing.seasonal import (
    MIN_YEARS, N_DAYS, SeasonalConfig, calendar_day, calendar_day_anomalies, resolve_seasonal, seasonal_mean,
    seasonal_phase, years_per_calendar_day,
)


def _hourly(years=10, start="2001-01-01"):
    return pd.date_range(start, periods=int(years * 365.25 * 24), freq="h")


def _phase(index):
    return 2 * np.pi * calendar_day(index) / N_DAYS


def test_calendar_day_pools_29_february_with_28_february_and_aligns_years():
    idx = pd.DatetimeIndex(["2001-02-28", "2004-02-28", "2004-02-29", "2001-03-01", "2004-03-01",
                            "2001-12-31", "2004-12-31", "2004-01-01"])
    assert calendar_day(idx).tolist() == [58, 58, 58, 59, 59, 364, 364, 0]


def test_anomalies_remove_the_seasonal_mean_only():
    rng = np.random.default_rng(0)
    idx = _hourly()
    phase = _phase(idx)
    sd = 1.0 + 0.8 * np.sin(phase)            # spread varies through the year
    x = 3.0 * np.sin(phase) + sd * rng.standard_normal(idx.size)
    w = calendar_day_anomalies(x, idx, window_days=11)
    assert abs(np.corrcoef(w, np.sin(phase))[0, 1]) < 0.02
    high, low = np.sin(phase) > 0.7, np.sin(phase) < -0.7
    assert np.std(w[high]) / np.std(w[low]) > 3        # the spread is not altered


def test_seasonal_phase_places_the_calendar_day_on_the_unit_circle():
    idx = pd.DatetimeIndex(["2001-01-01", "2001-04-02", "2001-12-31", "2004-02-29"])
    phase = seasonal_phase(idx)
    assert phase.shape == (4, 2) and np.allclose(np.hypot(phase[:, 0], phase[:, 1]), 1.0)
    angle = 2 * np.pi * calendar_day(idx) / N_DAYS
    assert np.allclose(phase, np.column_stack([np.sin(angle), np.cos(angle)]))


def test_removing_the_calendar_day_mean_is_conditioning_on_the_day_in_least_squares():
    """Frisch-Waugh-Lovell: with a one-day window, the residual covariance of
    two anomaly series equals that of the series regressed on calendar-day
    indicators."""
    rng = np.random.default_rng(9)
    idx = pd.date_range("2001-01-01", periods=6 * 365, freq="D")
    day = calendar_day(idx)
    a = np.sin(_phase(idx)) + rng.standard_normal(idx.size)
    b = np.cos(_phase(idx)) ** 3 + rng.standard_normal(idx.size)
    D = np.eye(N_DAYS)[day]
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    wa, wb = calendar_day_anomalies(a, idx, 1), calendar_day_anomalies(b, idx, 1)
    assert np.allclose(wa, ra) and np.allclose(wb, rb)


def test_one_day_window_centres_every_calendar_day_exactly():
    rng = np.random.default_rng(1)
    idx = _hourly(years=4)
    x = rng.gamma(2.0, size=idx.size) + np.sin(_phase(idx))
    w = calendar_day_anomalies(x, idx, window_days=1)
    day = calendar_day(idx)
    means = np.bincount(day, weights=w, minlength=N_DAYS) / np.bincount(day, minlength=N_DAYS)
    assert np.allclose(means, 0.0, atol=1e-9)


def test_window_wraps_around_the_year_with_triangular_weights():
    rng = np.random.default_rng(2)
    idx = _hourly(years=3)
    x = rng.standard_normal(idx.size)
    mu = seasonal_mean(x, idx, window_days=5)
    day = calendar_day(idx)
    weights = {363: 1 / 3, 364: 2 / 3, 0: 1.0, 1: 2 / 3, 2: 1 / 3}
    num = sum(w * x[day == d].sum() for d, w in weights.items())
    den = sum(w * np.sum(day == d) for d, w in weights.items())
    assert mu[0] == pytest.approx(num / den)


def test_an_abrupt_seasonal_onset_stays_sharp():
    # A step in the seasonal mean (like snowmelt onset) biases the anomalies
    # only within half a window of the step.
    rng = np.random.default_rng(3)
    idx = _hourly(years=20)
    day = calendar_day(idx)
    x = np.where((day >= 120) & (day < 200), 5.0, 0.0) + rng.standard_normal(idx.size)
    w = calendar_day_anomalies(x, idx, window_days=11)
    mean_by_day = np.bincount(day, weights=w, minlength=N_DAYS) / np.bincount(day, minlength=N_DAYS)
    far = (np.abs(np.arange(N_DAYS) - 120) > 5) & (np.abs(np.arange(N_DAYS) - 200) > 5)
    assert np.all(np.abs(mean_by_day[far]) < 0.15)


def test_missing_values_are_ignored_and_kept_missing():
    rng = np.random.default_rng(4)
    idx = _hourly(years=3)
    x = rng.standard_normal(idx.size)
    x[1000:1500] = np.nan
    w = calendar_day_anomalies(x, idx, window_days=11)
    assert np.all(np.isnan(w[1000:1500])) and np.all(np.isfinite(np.delete(w, np.s_[1000:1500])))


def test_removing_the_seasonal_cycle_lets_the_decorrelation_lag_reflect_the_noise():
    # With the annual cycle present, the autocorrelation never decays as the
    # method assumes; its anomalies decay as the short-memory noise does.
    from model_selection.statistics.significance.block_length import estimate_dependence
    rng = np.random.default_rng(5)
    idx = _hourly(years=6)
    phi = np.exp(-1 / 24)
    noise = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal(idx.size))
    x = 2.0 * np.sin(_phase(idx)) + noise
    raw = estimate_dependence(x)
    w = calendar_day_anomalies(x, idx, window_days=11)
    anomalies = estimate_dependence(w)
    noise_only = estimate_dependence(noise)
    assert raw.decorrelation_lag > 10 * noise_only.decorrelation_lag
    assert anomalies.decorrelation_lag == pytest.approx(noise_only.decorrelation_lag, rel=0.5)


def test_config_requires_an_odd_window():
    assert SeasonalConfig().enabled and SeasonalConfig().window_days == 11
    with pytest.raises(ValueError):
        SeasonalConfig(window_days=10)


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


def test_seasonal_conditioning_is_disabled_on_records_too_short_to_estimate_the_cycle():
    config, note = resolve_seasonal(SeasonalConfig(window_days=7), _hourly(years=2))
    assert not config.enabled and config.window_days == 7
    assert "fewer than the 3 needed" in note


def test_disabled_seasonal_conditioning_stays_disabled_without_a_note():
    config, note = resolve_seasonal(SeasonalConfig(enabled=False), _hourly(years=10))
    assert not config.enabled and note is None
