"""Dependence lengths (Politis–White decorrelation lag and block length) and
their effect on the false-positive rate of the block-resampled tests
(docs/methods.md, Dependence between samples)."""
from unittest import mock

import numpy as np
import pytest
from scipy.signal import lfilter

from model_selection.statistics.adequacy.linearity import LinearityTest
from model_selection.statistics.adequacy.specification import linear_specification_test
from model_selection.statistics.conditional_independence import LinearResidualTest
from model_selection.statistics.significance.base import permutation_p_value
from model_selection.statistics.significance.block_length import (
    C_THRESHOLD, DEFAULT_RUN_LENGTH, ResolvedBlock, estimate_dependence, product_dependence, resolve_block,
    sampling_stride,
)
from model_selection.statistics.significance.resampling import time_block_permutation


def politis_white_block_length(x):
    return estimate_dependence(x).block_length


def product_block_length(a, b):
    return max(e.block_length for e in product_dependence(a, b))


def is_evenly_spaced(t):
    return sampling_stride(t) is not None


def ar1(n, tau, rng):
    phi = np.exp(-1.0 / tau)
    x = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal(n + int(10 * tau)))
    return x[int(10 * tau):]


def theoretical_cbb_block(rho, n):
    """Optimal circular-block length for an AR(1) with coefficient rho:
    (3/2)^(1/3) (G / g)^(2/3) n^(1/3), G / g = 2 rho / (1 - rho^2)."""
    return 1.5 ** (1 / 3) * (2 * rho / (1 - rho ** 2)) ** (2 / 3) * n ** (1 / 3)


def band(n):
    return C_THRESHOLD * np.sqrt(np.log10(n) / n)


# --- The block length ---------------------------------------------------

def test_white_noise_gets_unit_blocks():
    rng = np.random.default_rng(0)
    est = estimate_dependence(rng.standard_normal(5000))
    assert est.block_length <= 2 and est.decays


def test_ar1_block_length_matches_theory_on_average():
    rng = np.random.default_rng(1)
    n, tau = 20000, 10
    est = np.mean([politis_white_block_length(ar1(n, tau, rng)) for _ in range(20)])
    assert est == pytest.approx(theoretical_cbb_block(np.exp(-1 / tau), n), rel=0.3)


def test_block_length_is_not_capped_at_three_root_n():
    # The reference code caps b at 3 sqrt(n); the method's theory does not.
    # For an AR(1) with e-folding time 200 and n = 5000 the theoretical
    # length (about 670) is three times that cap (212).
    rng = np.random.default_rng(2)
    n, tau = 5000, 200
    b = np.mean([politis_white_block_length(ar1(n, tau, rng)) for _ in range(10)])
    assert b > 3 * np.sqrt(n)
    assert b == pytest.approx(theoretical_cbb_block(np.exp(-1 / tau), n), rel=0.5)


def test_block_length_grows_with_persistence():
    rng = np.random.default_rng(3)
    b = [politis_white_block_length(ar1(5000, tau, rng)) for tau in (2, 10, 50)]
    assert b[0] < b[1] < b[2]


def test_agrees_with_independent_implementation_where_its_caps_do_not_bind():
    arch = pytest.importorskip("arch.bootstrap")
    rng = np.random.default_rng(4)
    for tau in (3, 10):
        x = ar1(4000, tau, rng)
        ref = float(arch.optimal_block_length(x)["circular"].iloc[0])
        assert abs(politis_white_block_length(x) - ref) <= 1.0 + 0.1 * ref


def test_products_of_independent_series_need_shorter_blocks_than_either_series():
    # Under independence the product's autocorrelation is rho_a(k) rho_b(k).
    rng = np.random.default_rng(5)
    a, b = ar1(8000, 20, rng), ar1(8000, 20, rng)
    assert product_block_length(a, b) < min(politis_white_block_length(a), politis_white_block_length(b))


# --- The decorrelation lag ----------------------------------------------

def test_decorrelation_lag_tracks_where_the_autocorrelation_falls_below_the_band():
    # For an AR(1), rho(k) = exp(-k / tau) falls below the band at
    # tau ln(1 / band); estimation noise can only lengthen the estimate.
    rng = np.random.default_rng(6)
    n, tau = 20000, 10
    expected = tau * np.log(1 / band(n))
    m = np.mean([estimate_dependence(ar1(n, tau, rng)).decorrelation_lag for _ in range(20)])
    assert expected * 0.8 <= m <= expected * 1.6


def test_lag_search_is_not_capped_at_root_n():
    # Long memory (groundwater-like, e-folding time 300): the reference
    # code's search stops at sqrt(n) + 5 = 205 lags; without that cap the
    # estimate reaches well beyond it.
    rng = np.random.default_rng(7)
    n, tau = 40000, 300
    m = [estimate_dependence(ar1(n, tau, rng)).decorrelation_lag for _ in range(10)]
    assert np.median(m) > 2 * (np.sqrt(n) + DEFAULT_RUN_LENGTH)


def test_decorrelation_lag_is_often_too_short_when_memory_is_long_relative_to_the_record():
    # Documented limitation: the significance band is sized for estimation
    # noise of nearly independent data; with long memory the estimated
    # autocorrelation dips into it early. With n / tau = 133 the median
    # estimate falls short of the lag where the true autocorrelation reaches
    # the band.
    rng = np.random.default_rng(16)
    n, tau = 40000, 300
    true_lag = tau * np.log(1 / band(n))
    m = [estimate_dependence(ar1(n, tau, rng)).decorrelation_lag for _ in range(20)]
    assert np.median(m) < true_lag


def test_lag_search_extends_to_a_quarter_of_the_record():
    # Box and Jenkins (1976): autocorrelations are usefully estimated to n / 4.
    rng = np.random.default_rng(8)
    est = estimate_dependence(ar1(4000, 5, rng))
    assert est.max_lag_searched == 1000


def _trend(n, rng):
    """A deterministic trend: its sample autocorrelation stays high at every
    lag up to n / 4, so it never decays within the record."""
    return np.linspace(0.0, 10.0, n) + 0.1 * rng.standard_normal(n)


def test_non_decaying_dependence_is_reported():
    est = estimate_dependence(_trend(4000, np.random.default_rng(9)))
    assert not est.decays
    assert est.decorrelation_lag == est.max_lag_searched


def test_block_length_stays_below_half_the_record():
    # With the lag window limited to n / 4, the block length cannot reach
    # n / 2 when the long-run variance is positive, so two blocks always fit.
    rng = np.random.default_rng(10)
    n = 400
    t = np.arange(n)
    series = [_trend(n, rng), np.cumsum(rng.standard_normal(n)), ar1(n, 200, rng)] + [
        np.sin(2 * np.pi * t / p) + 0.01 * rng.standard_normal(n) for p in (100, 200, 400, 600)]
    for x in series:
        assert estimate_dependence(x).block_length < n / 2


def test_longer_run_length_cannot_shorten_the_decorrelation_lag():
    rng = np.random.default_rng(11)
    x = ar1(20000, 20, rng)
    assert estimate_dependence(x, 50).decorrelation_lag >= estimate_dependence(x, 5).decorrelation_lag


def test_decorrelation_lag_can_stop_at_a_cycle_zero_crossing():
    # Documented limitation: a cycle's autocorrelation is briefly
    # insignificant near each zero crossing, so the run of five lags can end
    # the search a quarter period out although the correlation returns.
    rng = np.random.default_rng(12)
    n, period = 40000, 2000
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / period) + 0.5 * rng.standard_normal(n)
    est = estimate_dependence(x)
    assert est.decays
    assert est.decorrelation_lag == pytest.approx(period / 4, rel=0.2)


# --- How the block length is applied --------------------------------------

def test_uneven_samples_keep_the_callers_lag():
    rng = np.random.default_rng(13)
    a, b = ar1(500, 20, rng), ar1(500, 20, rng)
    events = np.sort(rng.choice(100000, size=500, replace=False))
    assert not is_evenly_spaced(events)
    resolved = resolve_block(events, 37, [(a, b)])
    assert resolved.length == 37 and not resolved.estimated
    assert is_evenly_spaced(np.arange(500))
    assert resolve_block(np.arange(500), 37, [(a, b)]).estimated


def test_resolved_block_serves_every_candidate_with_the_longest_bandwidth():
    # One set of multipliers serves every candidate: the longest bandwidth, the
    # longest decorrelation lag, and non-decay if any factor fails to decay.
    rng = np.random.default_rng(31)
    t = np.arange(4000)
    quick = (ar1(4000, 3, rng), ar1(4000, 3, rng))
    slow = (ar1(4000, 40, rng), ar1(4000, 40, rng))
    alone = resolve_block(t, 0, [slow])
    both = resolve_block(t, 0, [quick, slow])
    assert both.length == alone.length > resolve_block(t, 0, [quick]).length
    assert both.decorrelation_lag == alone.decorrelation_lag


def test_long_blocks_are_never_replaced_by_a_free_permutation():
    # A block longer than half the record is reduced to half: the two halves
    # are exchanged intact, so each output is a rotation, never a shuffle.
    rng = np.random.default_rng(15)
    n = 100
    for _ in range(20):
        perm = time_block_permutation(n, np.arange(n), 80, rng)
        assert sorted(perm) == list(range(n))
        assert np.count_nonzero(np.diff(perm) != 1) <= 2


def test_run_length_setting_reaches_every_resampling_test():
    from model_selection.selection.config import AdequacyConfig, SelectionConfig
    from model_selection.selection.tiers import build_tier
    sel = SelectionConfig(dependence_run_length=37)
    assert sel.dependence_run_length == 37
    assert SelectionConfig().dependence_run_length == DEFAULT_RUN_LENGTH == 5
    for name in ("linear", "nonlinear", "cmi"):
        assert build_tier(name, sel, AdequacyConfig()).test.run_length == 37


# False-positive rates under H0 with autocorrelated, evenly spaced samples,
# in a record long relative to its memory (n / tau = 500), where the
# Politis–White length should bring the rate to alpha. With blocks one
# e-folding time long the nulls are too narrow. 150 datasets give a standard
# error of about 0.018 at alpha = 0.05. In short, persistent records the rate
# stays above alpha whatever the block length (docs/methods.md, Temporal
# dependence); that regime is not asserted here.
TAU, N, REPS, PERMS = 6, 3000, 150, 99


def _old_blocks(module):
    """Force blocks of the caller's lag (the e-folding time), as before."""
    return mock.patch(f"{module}.resolve_block",
                      lambda t, lag, *args: ResolvedBlock(int(round(lag)), estimated=False))


def _rate(trial, seed, patch=None):
    rng = np.random.default_rng(seed)
    if patch is None:
        return np.mean([trial(rng) < 0.05 for _ in range(REPS)])
    with patch:
        return np.mean([trial(rng) < 0.05 for _ in range(REPS)])


def _null_data(rng):
    z = ar1(N, TAU, rng)
    c = 0.7 * z + 0.7 * ar1(N, TAU, rng)
    o = 0.7 * z + 0.7 * ar1(N, TAU, rng)
    return z, c, o


def test_linear_tier_false_positive_rate():
    def trial(rng):
        z, c, o = _null_data(rng)
        res = LinearResidualTest().run({"c": c}, o, z[:, None], PERMS, rng, np.arange(N), TAU)
        return permutation_p_value(float(res.observed[0]), res.null[:, 0])
    new = _rate(trial, 10)
    old = _rate(trial, 10, _old_blocks("model_selection.statistics.conditional_independence.linear"))
    assert new <= 0.09
    assert old > new


def test_check_b_false_positive_rate():
    def trial(rng):
        z, c, _ = _null_data(rng)
        o = 0.5 * z + 0.5 * c + 0.7 * ar1(N, TAU, rng)
        return LinearityTest(effect_size_floor=0.0).run(
            c, o, z[:, None], time_positions=np.arange(N), min_temporal_lag=TAU,
            n_permutations=PERMS, rng=rng).permutation_p_value
    new = _rate(trial, 11)
    old = _rate(trial, 11, _old_blocks("model_selection.statistics.adequacy.base"))
    assert new <= 0.09
    assert old > new


def test_check_a_false_positive_rate():
    # A shorter record than the other checks: the kernel check's eigendecomposition is O(n^3).
    n = 800

    def trial(rng):
        z = ar1(n, TAU, rng)
        c = 0.7 * z + 0.7 * ar1(n, TAU, rng)
        o = 0.7 * z + 0.7 * ar1(n, TAU, rng)
        out = linear_specification_test(np.column_stack([c, o]), z[:, None], 0.05, 0.0, PERMS, rng,
                                         np.arange(n), TAU)
        return min(1.0, 2 * out.p_value)
    new = _rate(trial, 12)
    assert new <= 0.09


# --- Step 2: the dependence lag and constant-stride subsamples ------------

def test_dependence_lag_is_the_largest_decorrelation_lag_across_formulations():
    # Generous choice (Hegger et al., 1999): the most persistent formulation sets it.
    from model_selection.pipeline.shared import max_decorrelation_lag
    rng = np.random.default_rng(20)
    fast, slow = ar1(20000, 5, rng), ar1(20000, 50, rng)
    lag, per = max_decorrelation_lag({"fast": fast, "slow": slow}, DEFAULT_RUN_LENGTH)
    assert per["slow"] > per["fast"]
    assert lag == per["slow"] == estimate_dependence(slow).decorrelation_lag


def test_strided_subsample_block_length_is_counted_in_strides_and_returned_in_time_units():
    rng = np.random.default_rng(21)
    x = ar1(4000, 3, rng)
    y = ar1(4000, 3, rng)
    stride = 250
    positions = np.arange(4000) * stride
    resolved = resolve_block(positions, 0, [(x, y)])
    assert resolved.estimated
    assert resolved.length == resolve_block(np.arange(4000), 0, [(x, y)]).length * stride


def test_independent_strided_subsample_has_a_one_step_decorrelation_lag_and_short_blocks():
    # Independent samples: decorrelation lag of one stride; the block length
    # rests on noisy autocovariances at the first lags and is rounded up, so it
    # is a few strides at most.
    rng = np.random.default_rng(22)
    stride = 175
    resolved = resolve_block(np.arange(1000) * stride, 0, [(rng.standard_normal(1000), rng.standard_normal(1000))])
    assert resolved.decorrelation_lag == stride
    assert stride <= resolved.length <= 3 * stride


def test_sample_independence_is_reported_for_evenly_spaced_samples_only():
    from model_selection.statistics.significance.block_length import samples_independent
    rng = np.random.default_rng(23)
    iid = samples_independent(np.arange(2000) * 100, [rng.standard_normal(2000)])
    assert iid["independent"] and iid["decorrelation_lag"] <= 100
    persistent = samples_independent(np.arange(2000), [ar1(2000, 20, rng)])
    assert not persistent["independent"] and persistent["decorrelation_lag"] > 1
    events = np.sort(rng.choice(100000, size=500, replace=False))
    assert samples_independent(events, [rng.standard_normal(500)]) is None


def test_cmi_records_whether_the_permuted_samples_are_independent():
    from model_selection.statistics.conditional_independence import CMITest
    rng = np.random.default_rng(24)
    n = 600
    z = ar1(n, 20, rng)
    res = CMITest(workers=1).run({"c": z + 0.5 * ar1(n, 20, rng)}, z + 0.5 * ar1(n, 20, rng), z[:, None],
                                 19, rng, np.arange(n), 20)
    assert res.dependence["estimated"] and res.dependence["independent"] is False


def test_block_count_is_the_record_divided_by_the_block_length():
    from model_selection.statistics.significance.block_length import MIN_BLOCKS
    rng = np.random.default_rng(30)
    short = resolve_block(np.arange(600), 0, [(ar1(600, 40, rng), ar1(600, 40, rng))])
    assert short.n_blocks == pytest.approx(600 / (short.length / 1.0))
    assert short.few_blocks and short.n_blocks < MIN_BLOCKS
    long = resolve_block(np.arange(20000), 0, [(ar1(20000, 5, rng), ar1(20000, 5, rng))])
    assert not long.few_blocks
    strided = resolve_block(np.arange(1000) * 175, 0, [(rng.standard_normal(1000), rng.standard_normal(1000))])
    assert strided.n_blocks == pytest.approx(1000 / (strided.length / 175))
    events = np.sort(rng.choice(10**5, 300, replace=False))
    assert resolve_block(events, 24, [(np.ones(300), np.ones(300))]).n_blocks == (events[-1] - events[0]) / 24


def test_autocovariance_with_nothing_missing_is_the_standard_estimator():
    from model_selection.statistics.significance.block_length import _autocovariances
    rng = np.random.default_rng(30)
    x = ar1(500, 5, rng)
    e = x - x.mean()
    direct = np.array([np.sum(e[: 500 - k] * e[k:]) / 500 for k in range(20)])
    assert np.allclose(_autocovariances(x, 19), direct)


def test_missing_values_leave_the_dependence_estimate_close_to_the_complete_record():
    """Missing points are skipped, not spliced over: removing a fifth of an
    AR(1) record at random, or a long block, changes the estimate little."""
    rng = np.random.default_rng(31)
    x = ar1(20000, 10, rng)
    full = estimate_dependence(x)
    scattered = x.copy()
    scattered[rng.random(x.size) < 0.2] = np.nan
    block = x.copy()
    block[5000:9000] = np.nan
    for gappy in (scattered, block):
        est = estimate_dependence(gappy)
        assert abs(est.decorrelation_lag - full.decorrelation_lag) <= 0.3 * full.decorrelation_lag
        assert abs(est.block_length - full.block_length) <= 0.3 * full.block_length


def test_samples_on_a_declared_grid_with_gaps_are_estimated():
    rng = np.random.default_rng(32)
    keep = np.sort(rng.choice(5000, 4000, replace=False))
    a, b = ar1(5000, 8, rng)[keep], ar1(5000, 8, rng)[keep]
    assert not resolve_block(keep, 37, [(a, b)]).estimated  # not evenly spaced, no grid declared
    on_grid = resolve_block(keep * 3, 37, [(a, b)], grid_step=3)
    assert on_grid.estimated and on_grid.length % 3 == 0


def test_trace_autocovariance_of_scalar_terms_is_their_autocovariance():
    """For scalar per-sample terms the trace autocovariance is the ordinary
    autocovariance, so the tier-2 bandwidth reduces to the Politis–White
    estimate of the terms themselves; gaps are skipped as in
    _autocovariances."""
    from model_selection.statistics.significance.block_length import _autocovariances, trace_autocovariance
    rng = np.random.default_rng(50)
    x = ar1(600, 10, rng)
    x = x - x.mean()
    assert np.allclose(trace_autocovariance(np.outer(x, x), np.arange(600), 600), _autocovariances(x, 150))
    keep = np.sort(rng.choice(600, 450, replace=False))
    gappy = np.full(600, np.nan)
    gappy[keep] = x[keep] - x[keep].mean()
    xs = gappy[keep]
    assert np.allclose(trace_autocovariance(np.outer(xs, xs), keep, 600), _autocovariances(gappy, 150))


def test_too_few_samples_are_not_assessed_and_fall_back_to_the_callers_lag():
    from model_selection.statistics.significance.block_length import MIN_ACF_N, resolve_block, samples_independent
    rng = np.random.default_rng(0)
    n = MIN_ACF_N - 39  # e.g. 11 points spaced at a long dependence lag
    t = np.arange(n) * 100.0
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    resolved = resolve_block(t, 100.0, [(x, y)], grid_step=100.0)
    assert not resolved.estimated and resolved.too_few_to_assess and resolved.length == 100
    assert resolved.n_blocks is None and not resolved.few_blocks
    assert samples_independent(t, [x], grid_step=100.0)["estimated"] is False


def test_event_samples_record_their_span_in_blocks_of_the_callers_lag():
    events = np.array([0, 40, 900, 1500, 7300, 7310])
    resolved = resolve_block(events, 730, [(np.arange(6.0), np.arange(6.0)[::-1])])
    assert not resolved.estimated and resolved.n_blocks == 7310 / 730 and resolved.few_blocks
    assert resolved.to_record()["n_blocks"] == resolved.n_blocks
    assert resolve_block(events, 0, [(np.arange(6.0), np.arange(6.0))]).n_blocks is None  # independent



# --- The multipliers' bandwidth for testing --------------------------------

def test_the_non_increasing_fit_pools_adjacent_violators():
    from model_selection.statistics.significance.block_length import _non_increasing_fit
    assert np.allclose(_non_increasing_fit([3.0, 1.0, 2.0]), [3.0, 1.5, 1.5])
    assert np.allclose(_non_increasing_fit([1.0, 2.0, 3.0]), [2.0, 2.0, 2.0])
    y = np.array([5.0, 4.0, 4.0, 1.0])
    assert np.allclose(_non_increasing_fit(y), y)


def test_independent_terms_have_zero_persistence_and_independent_multipliers():
    rng = np.random.default_rng(32)
    resolved = resolve_block(np.arange(5000), 0, [(rng.standard_normal(5000), rng.standard_normal(5000))])
    assert resolved.persistence == 0.0 and resolved.length == 1


def test_the_persistence_of_products_of_two_autoregressions_is_recovered():
    """For independent AR(1) series with coefficients r1 and r2, the products
    are autocorrelated as (r1 r2)^k, whose persistence a = G / J is
    2 r / (1 - r^2), r = r1 r2. The window stops where the factors become
    insignificant, which leaves out a little of the tail."""
    from model_selection.statistics.significance.block_length import testing_bandwidth
    rng = np.random.default_rng(33)
    n, r1, r2 = 100000, 0.98, 0.95
    x = np.empty(n); y = np.empty(n)
    ex, ey = rng.standard_normal(n), rng.standard_normal(n)
    x[0], y[0] = ex[0], ey[0]
    for t in range(1, n):
        x[t] = r1 * x[t - 1] + ex[t]
        y[t] = r2 * y[t - 1] + ey[t]
    r = r1 * r2
    truth = 2 * r / (1 - r * r)
    resolved = resolve_block(np.arange(n), 0, [(x, y)])
    assert 0.75 * truth < resolved.persistence < 1.1 * truth
    assert resolved.length == testing_bandwidth(resolved.persistence, n)


def test_a_weak_slow_tail_in_the_products_is_kept():
    """Each factor has a fast component and a weak slow one. The products'
    tail lies below the per-lag significance band, but the factors' own
    autocorrelations reach well past it, so the window keeps the tail and
    the persistence is close to its value from the known autocorrelation."""
    rng = np.random.default_rng(34)
    n = 200000

    def series():
        e1, e2 = rng.standard_normal(n), rng.standard_normal(n)
        fast, slow = np.empty(n), np.empty(n)
        fast[0], slow[0] = e1[0], e2[0]
        for t in range(1, n):
            fast[t] = 0.9 * fast[t - 1] + e1[t]
            slow[t] = 0.998 * slow[t - 1] + 0.06 * e2[t]
        return fast + slow

    x, y = series(), series()
    # Known autocorrelation of each factor and of their products.
    k = np.arange(20000)
    var_f, var_s = 1 / (1 - 0.81), 0.0036 / (1 - 0.998 ** 2)
    rho = (var_f * 0.9 ** k + var_s * 0.998 ** k) / (var_f + var_s)
    prod = rho ** 2
    truth = 2 * np.sum(k[1:] * prod[1:]) / (1 + 2 * np.sum(prod[1:]))
    resolved = resolve_block(np.arange(n), 0, [(x, y)])
    assert 0.5 * truth < resolved.persistence < 1.5 * truth


def test_window_extends_to_the_products_own_lag_for_lagged_dependence():
    """H0 concerns matched time steps. A white candidate c and o_t = c_{t-1} +
    c_{t+1} are uncorrelated at lag 0, but their products are autocorrelated
    at lag 1 (the cross-covariances at +1 and -1). The factors' window is empty;
    the products' own lag (Bartlett band) extends it, so the bandwidth exceeds 1."""
    from model_selection.statistics.significance.block_length import resolve_block
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.gamma(0.6, 1.0, size=n + 2) - 0.6
    c, o = x[1:-1], x[:-2] + x[2:]
    r = resolve_block(np.arange(n, dtype=float), 1, [(c, o)], grid_step=1.0)
    rec = r.to_record()
    assert rec["factor_window"] == 0 and rec["window_extended_by_products"]
    assert r.length > 1


def test_window_not_extended_for_series_independent_at_every_lag():
    """Two independent, persistent series: the products have no autocorrelation
    beyond the factors' window, and the Bartlett band does not extend it."""
    from model_selection.statistics.significance.block_length import resolve_block
    rng = np.random.default_rng(1)
    n = 16000
    extended = 0
    for _ in range(10):
        e1, e2 = rng.normal(size=n), rng.normal(size=n)
        c, o = lfilter([1.0], [1.0, -0.99], e1), lfilter([1.0], [1.0, -0.98], e2)
        rec = resolve_block(np.arange(n, dtype=float), 1, [(c - c.mean(), o - o.mean())], grid_step=1.0).to_record()
        extended += bool(rec["window_extended_by_products"])
    assert extended == 0
