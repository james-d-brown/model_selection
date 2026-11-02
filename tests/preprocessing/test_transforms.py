"""Series preparation: log, optional seasonal anomalies, rank-normal score
with random tie-breaking (docs/methods.md, Samples)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from model_selection.preprocessing.transforms import log_transform, prepare_series, rank_normal_transform


def test_rank_normal_scores_are_the_normal_quantiles_of_the_ranks():
    x = np.array([3.0, 1.0, 2.0, 10.0])
    expected = norm.ppf((np.array([3, 1, 2, 4]) - 0.5) / 4)
    assert np.allclose(rank_normal_transform(x, np.random.default_rng(0)), expected)


def test_ties_are_broken_at_random_and_none_remain():
    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(1000), rng.gamma(2.0, size=9000)])
    a = rank_normal_transform(x, np.random.default_rng(2))
    b = rank_normal_transform(x, np.random.default_rng(3))
    assert np.unique(a).size == x.size                      # no ties left
    assert not np.array_equal(a[:1000], b[:1000])           # order among zeros is random
    assert np.max(a[:1000]) < np.min(a[1000:])              # zeros stay below every positive flow
    assert np.array_equal(np.argsort(a[1000:]), np.argsort(x[1000:]))  # untied order unchanged


def test_missing_values_stay_missing_and_do_not_count_in_the_ranks():
    x = np.array([1.0, np.nan, 3.0, 2.0])
    out = rank_normal_transform(x, np.random.default_rng(0))
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 3, 2]], norm.ppf((np.arange(1, 4) - 0.5) / 3))


def test_log_matters_only_through_seasonal_conditioning():
    # Without seasonal standardization the output depends only on the order of
    # values, which the log does not change.
    rng = np.random.default_rng(4)
    x = rng.gamma(2.0, size=5000)
    assert np.allclose(prepare_series(x, np.random.default_rng(5)), rank_normal_transform(x, np.random.default_rng(5)))


def test_seasonal_conditioning_needs_timestamps():
    with pytest.raises(ValueError):
        prepare_series(np.ones(10) + np.arange(10), np.random.default_rng(0), seasonal_window_days=11)


def test_prepared_series_is_standard_normal_and_tie_free():
    rng = np.random.default_rng(6)
    idx = pd.date_range("2001-01-01", periods=3 * 8760, freq="h")
    x = np.where(rng.random(idx.size) < 0.1, 0.0, rng.gamma(2.0, size=idx.size))
    out = prepare_series(x, np.random.default_rng(7), index=idx, seasonal_window_days=11)
    assert np.unique(out).size == x.size
    assert abs(out.mean()) < 1e-6 and out.std() == pytest.approx(1.0, abs=0.01)


def test_log_offset_is_one_percent_of_the_smallest_positive_value():
    assert np.allclose(log_transform(np.array([0.0, 2.0, 4.0])), np.log(np.array([0.0, 2.0, 4.0]) + 0.02))
