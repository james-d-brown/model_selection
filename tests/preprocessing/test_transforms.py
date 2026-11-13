"""Series preparation: the standardized signed square root of flow, and the
series for the dependence lag, adjusted for the time of year
(docs/methods.md, Samples)."""
import numpy as np
import pandas as pd
import pytest

from model_selection.preprocessing.transforms import (
    EXPONENT, dependence_series, prepare_series, signed_power, standardize_series,
)


def test_the_exponent_is_the_square_root():
    assert EXPONENT == 0.5
    assert np.allclose(signed_power(np.array([4.0, 0.0, -9.0])), [2.0, 0.0, -3.0])


def test_prepared_series_is_the_standardized_signed_square_root():
    rng = np.random.default_rng(4)
    x = rng.gamma(2.0, size=5000)
    out = prepare_series(x)
    expected = np.sqrt(x)
    assert np.allclose(out, (expected - expected.mean()) / expected.std())
    assert abs(out.mean()) < 1e-12 and out.std() == pytest.approx(1.0)


def test_negative_flow_keeps_its_sign_and_order():
    x = np.array([-4.0, -1.0, 0.0, 1.0, 9.0])
    out = prepare_series(x)
    assert np.all(np.diff(out) > 0)
    root = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    assert np.allclose(out, (root - root.mean()) / root.std())


def test_zero_flows_are_a_point_mass_with_no_offset():
    rng = np.random.default_rng(6)
    x = np.where(rng.random(20000) < 0.1, 0.0, rng.gamma(2.0, size=20000))
    out = prepare_series(x)
    zero = x == 0
    assert np.unique(out[zero]).size == 1
    assert np.max(out[zero]) < np.min(out[~zero])


def test_prepared_series_do_not_depend_on_the_flow_units():
    rng = np.random.default_rng(8)
    x = rng.gamma(2.0, size=1000) - 0.2                 # some negative flow too
    assert np.allclose(prepare_series(x), prepare_series(35.3147 * x))   # m3/s to ft3/s


def test_missing_values_stay_missing_and_do_not_count_in_the_standardization():
    x = np.array([1.0, np.nan, 4.0, 9.0])
    out = prepare_series(x)
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 2, 3]], prepare_series(np.array([1.0, 4.0, 9.0])))
    assert np.allclose(standardize_series(np.array([2.0, 2.0])), [0.0, 0.0])     # constant: centred only


def test_dependence_series_is_the_analysed_series_less_its_fit_on_the_phase():
    from model_selection.preprocessing.seasonal import phase_residuals
    rng = np.random.default_rng(7)
    idx = pd.date_range("2001-01-01", periods=5 * 365, freq="D")
    x = np.exp(np.sin(2 * np.pi * idx.dayofyear.values / 365.25) + rng.standard_normal(idx.size))
    assert np.allclose(dependence_series(x, idx, seasonal=True), phase_residuals(prepare_series(x), idx))
    assert np.allclose(dependence_series(x, idx, seasonal=False), prepare_series(x))


def test_the_seasonal_adjustment_needs_timestamps():
    with pytest.raises(ValueError):
        dependence_series(np.ones(10) + np.arange(10), seasonal=True)
