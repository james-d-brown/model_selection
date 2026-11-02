"""
Unit tests for preprocessing.regularize -- the shared, method-agnostic gap-
regularization utility used by pipeline.run_high_flow_analysis before
EITHER event-detection method (see that module's own docstring for why
this lives here, not inside event_detection.wasp or event_detection.consensus).
"""
import numpy as np
import pandas as pd

from model_selection.preprocessing.regularize import regularize_series


def _make_series_with_gap(n_before=10, gap_hours=3, n_after=5, freq="h"):
    """n hours before a gap, a gap of gap_hours (i.e. gap_hours-1 missing
    samples), then n_after more hours."""
    idx1 = pd.date_range("2020-01-01", periods=n_before, freq=freq)
    idx2 = pd.date_range(idx1[-1] + pd.Timedelta(hours=gap_hours), periods=n_after, freq=freq)
    idx = idx1.append(idx2)
    values = np.arange(len(idx), dtype=np.float64) * 10
    return pd.Series(values, index=idx)


def test_regularize_series_already_uniform_is_unchanged():
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    series = pd.Series(np.arange(10, dtype=np.float64), index=idx)
    r = regularize_series(series)
    assert len(r.values) == len(series)
    assert np.array_equal(r.values, series.values)
    assert np.array_equal(r.orig_to_regularized, np.arange(10))
    assert np.array_equal(r.regularized_to_orig, np.arange(10))


def test_regularize_series_fills_gap_with_correct_length():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    # gap_hours=3 means 2 missing samples between the two runs
    assert len(r.values) == len(series) + 2


def test_regularize_series_preserves_every_original_value():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    for i in range(len(series)):
        assert r.values[r.orig_to_regularized[i]] == series.values[i]


def test_regularize_series_interpolated_values_are_between_neighbors():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    left_val = series.values[9]
    right_val = series.values[10]
    interpolated_positions = [j for j in range(len(r.values))
                               if r.regularized_to_orig[j] == 9 and r.values[j] not in series.values]
    for j in interpolated_positions:
        assert min(left_val, right_val) <= r.values[j] <= max(left_val, right_val)


def test_regularize_series_result_index_is_genuinely_uniform():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    diffs = r.regularized_index.to_series().diff().dropna()
    assert (diffs == diffs.iloc[0]).all()


def test_position_to_original_exact_for_genuine_samples():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    for i in range(len(series)):
        mapped_back = r.position_to_original(int(r.orig_to_regularized[i]))
        assert mapped_back == i


def test_event_id_to_original_correct_length_and_values():
    series = _make_series_with_gap(n_before=10, gap_hours=3, n_after=5)
    r = regularize_series(series)
    # a fake event spanning the whole regularized series with a single id
    fake_event_id = np.zeros(len(r.values), dtype=np.int64)
    mapped = r.event_id_to_original(fake_event_id)
    assert len(mapped) == len(series)
    assert (mapped == 0).all()


def test_no_artificial_spike_from_a_real_gap_in_derivative_terms():
    """Regression test for the actual, real-data failure this module
    fixes: a real, multi-hour gap, if
    treated as a single adjacent timestep, compresses whatever genuine
    change happened during it into an artificially sharp jump. After
    regularization, the maximum single-step change across the
    (formerly gap-spanning) region must be small relative to the raw,
    naive one-step jump the original, un-regularized series would show.
    """
    n_before, n_after = 50, 50
    idx1 = pd.date_range("2020-01-01", periods=n_before, freq="h")
    # a 100-hour gap (99 missing samples) -- flow drifts slowly during
    # this genuinely long gap, from 3.0 up to 30.0
    idx2 = pd.date_range(idx1[-1] + pd.Timedelta(hours=100), periods=n_after, freq="h")
    idx = idx1.append(idx2)
    values = np.concatenate([np.full(n_before, 3.0), np.full(n_after, 30.0)])
    series = pd.Series(values, index=idx)

    naive_max_step = np.abs(np.diff(series.values)).max()  # the artificial, un-regularized jump

    r = regularize_series(series)
    regularized_max_step = np.abs(np.diff(r.values)).max()

    assert regularized_max_step < naive_max_step
    # the interpolated step size should be the total change spread over
    # the number of equal INTERVALS between the two genuine samples --
    # 99 missing samples means 100 equal intervals, not 99 (a step
    # divides the gap into n_missing+1 pieces, not n_missing)
    expected_step = (30.0 - 3.0) / (99 + 1)
    assert abs(regularized_max_step - expected_step) < 1e-9


def test_multiple_gaps_all_handled_correctly():
    idx1 = pd.date_range("2020-01-01", periods=10, freq="h")
    idx2 = pd.date_range(idx1[-1] + pd.Timedelta(hours=3), periods=10, freq="h")
    idx3 = pd.date_range(idx2[-1] + pd.Timedelta(hours=5), periods=10, freq="h")
    idx = idx1.append(idx2).append(idx3)
    values = np.arange(len(idx), dtype=np.float64)
    series = pd.Series(values, index=idx)

    r = regularize_series(series)
    # gap 1: 2 missing, gap 2: 4 missing
    assert len(r.values) == len(series) + 2 + 4
    for i in range(len(series)):
        assert r.values[r.orig_to_regularized[i]] == series.values[i]
