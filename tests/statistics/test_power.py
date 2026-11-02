"""Tests of the set-correlation power calculation against Cohen (1988,
equations 10.1.6 to 10.1.9, 10.2.2 and 10.3.1)."""
import numpy as np
import pytest
from scipy import stats

from model_selection.statistics.power import (
    f2_to_r2, minimum_detectable_r2, rao_degrees, set_correlation_power,
)


def test_degrees_of_freedom_follow_cohen_10_1_6_to_10_1_9():
    d = rao_degrees(n=50, k_y=3, k_x=4, k_c=2)
    s = np.sqrt((9 * 16 - 4) / (9 + 16 - 5))            # 10.1.9
    m = 50 - 2 - (3 + 4 + 3) / 2                         # 10.1.8, partial (k_A = k_C), Model 1
    assert d.u == 12                                     # 10.1.6
    assert np.isclose(d.s, s)
    assert np.isclose(d.v, m * s + 1 - 12 / 2)           # 10.1.7


def test_s_is_one_when_ky2_kx2_equals_4():
    assert rao_degrees(40, 1, 2).s == 1.0 and rao_degrees(40, 2, 1).s == 1.0 and rao_degrees(40, 1, 1).s == 1.0


def test_scalar_case_reduces_to_multiple_regression():
    for n in (20, 60, 200):
        ref = stats.ncf.sf(stats.f.ppf(0.95, 1, n - 2), 1, n - 2, 0.15 * n)
        assert np.isclose(set_correlation_power(n, 1, 1, 0, 0.15, 0.05), ref)


def test_effect_size_inverts_10_2_2():
    for k in (1, 2, 4):
        r2 = f2_to_r2(0.35, k, k)
        s = rao_degrees(10_000, k, k).s
        assert np.isclose((1 - r2) ** (-1 / s) - 1, 0.35)


def test_undefined_test_has_no_power():
    assert set_correlation_power(20, 12, 12, 0, 0.35, 0.05) == 0.0
    assert minimum_detectable_r2(20, 12, 12, 0, 0.05) == 1.0


def test_minimum_detectable_effect_shrinks_with_n_and_grows_with_dimension():
    assert minimum_detectable_r2(120, 3, 3, 0, 0.05) < minimum_detectable_r2(60, 3, 3, 0, 0.05)
    assert minimum_detectable_r2(60, 5, 5, 0, 0.05) > minimum_detectable_r2(60, 2, 2, 0, 0.05)


def test_plain_language_description():
    from model_selection.statistics.power import describe_detectable
    d = describe_detectable(0.24)
    assert d["unexplained_variation_pct"] == pytest.approx(24.0)
    assert d["rmse_reduction_pct"] == pytest.approx(100 * (1 - np.sqrt(0.76)))
    assert "24%" in d["statement"] and "root-mean-square error 13% smaller" in d["statement"]
    assert "cannot support the test" in describe_detectable(1.0)["statement"]
    several = describe_detectable(0.24, n_features=3)
    assert several["rmse_reduction_pct"] is None and "root-mean-square" not in several["statement"]


def test_small_detectable_effects_are_not_rounded_to_zero():
    from model_selection.statistics.power import describe_detectable, format_share
    assert format_share(0.0000896) == "0.00009"
    assert format_share(0.0123) == "0.012"
    assert format_share(24.3) == "24"
    assert "0.009% of" in describe_detectable(0.00009)["statement"]


def test_effective_sample_size_recovers_n_for_independent_nulls():
    # Null draws of a squared partial correlation from independent samples:
    # Beta(1/2, (n - k_c - 2) / 2) for one dimension each side.
    from model_selection.statistics.power import effective_sample_size
    rng = np.random.default_rng(0)
    n, k_c = 400, 1
    draws = rng.beta(0.5, (n - k_c - 2) / 2, size=20000)
    assert effective_sample_size(n, draws, 1, 1, k_c) == pytest.approx(n, rel=0.03)


def test_effective_sample_size_shrinks_with_a_wider_null_and_is_capped():
    from model_selection.statistics.power import effective_sample_size
    rng = np.random.default_rng(1)
    draws = rng.beta(0.5, 198 / 2, size=20000)        # null of about 200 independent samples
    assert effective_sample_size(180_000, draws, 1, 1, 1) == pytest.approx(200, rel=0.05)
    assert effective_sample_size(50, draws, 1, 1, 1) == 50
    assert effective_sample_size(50, np.zeros(10), 1, 1, 1) == 50
