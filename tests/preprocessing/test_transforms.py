"""Series preparation: log, optional seasonal anomalies, rank-normal score
with ties sharing their average rank (docs/methods.md, Samples)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from model_selection.preprocessing.transforms import log_transform, prepare_series, rank_normal_transform


def test_rank_normal_scores_are_the_normal_quantiles_of_the_ranks():
    x = np.array([3.0, 1.0, 2.0, 10.0])
    expected = norm.ppf((np.array([3, 1, 2, 4]) - 0.5) / 4)
    assert np.allclose(rank_normal_transform(x), expected)


def test_tied_values_share_one_normal_score():
    """Zero flows share their average rank, so a point mass stays one value;
    untied values keep their order and are unchanged by the ties' presence
    beyond their ranks."""
    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(1000), rng.gamma(2.0, size=9000)])
    a = rank_normal_transform(x)
    assert np.unique(a[:1000]).size == 1 and a[0] == pytest.approx(norm.ppf((500.5 - 0.5) / x.size))
    assert np.unique(a).size == 9001
    assert np.max(a[:1000]) < np.min(a[1000:])              # zeros stay below every positive flow
    assert np.array_equal(np.argsort(a[1000:]), np.argsort(x[1000:]))  # untied order unchanged
    assert np.array_equal(a, rank_normal_transform(x))      # deterministic


def test_missing_values_stay_missing_and_do_not_count_in_the_ranks():
    x = np.array([1.0, np.nan, 3.0, 2.0])
    out = rank_normal_transform(x)
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 3, 2]], norm.ppf((np.arange(1, 4) - 0.5) / 3))


def test_log_matters_only_through_seasonal_conditioning():
    # Without seasonal standardization the output depends only on the order of
    # values, which the log does not change.
    rng = np.random.default_rng(4)
    x = rng.gamma(2.0, size=5000)
    assert np.allclose(prepare_series(x), rank_normal_transform(x))


def test_seasonal_conditioning_needs_timestamps():
    with pytest.raises(ValueError):
        prepare_series(np.ones(10) + np.arange(10), seasonal_window_days=11)


def test_prepared_series_is_near_standard_normal_and_keeps_zero_flow_ties():
    """After the calendar-day mean is removed, a zero flow's value depends on
    its calendar day, so zero flows tie with zero flows of the same calendar
    day in other years and keep one shared score; the marginal stays near
    standard normal."""
    rng = np.random.default_rng(6)
    idx = pd.date_range("2001-01-01", periods=3 * 8760, freq="h")
    x = np.where(rng.random(idx.size) < 0.1, 0.0, rng.gamma(2.0, size=idx.size))
    out = prepare_series(x, index=idx, seasonal_window_days=11)
    zero = x == 0
    assert np.unique(out[zero]).size < zero.sum()
    assert np.unique(out[~zero]).size == (~zero).sum()
    assert abs(out.mean()) < 0.01 and out.std() == pytest.approx(1.0, abs=0.02)


def test_log_offset_is_one_percent_of_the_smallest_positive_value():
    assert np.allclose(log_transform(np.array([0.0, 2.0, 4.0])), np.log(np.array([0.0, 2.0, 4.0]) + 0.02))
