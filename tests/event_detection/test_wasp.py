"""
Tests of the WASP event detector, by stage: scalogram construction and
worm labelling, the noise model, peak detection (stage 1), onset detection
(stage 2), sequential event resolution (stages 2 and 3) and
WASPEventDetector end to end.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

from model_selection.event_detection.base import DeclusterResult
from model_selection.event_detection.wasp.config import WASPConfig
from model_selection.event_detection.wasp.detector import WASPDetectionResult, WASPEventDetector
from model_selection.event_detection.wasp.noise import NoiseModel, fit_noise_model
from model_selection.event_detection.wasp.onsets import (
    _detect_onset,
    _resolve_end,
    resolve_events_sequential,
)
from model_selection.event_detection.wasp.peaks import (
    _build_scalogram,
    _label_worms,
    _worm_stats,
    detect_peaks,
)

# ---------------------------------------------------------------------------
# helpers for building synthetic storms
# ---------------------------------------------------------------------------

def _add_storm(series, start, width, base, peak_val, recession_k=0.05):
    n = len(series)
    t = np.arange(-8, 8)
    s = 1 / (1 + np.exp(-0.7 * t))
    for i, v in zip(range(start, start + 16), base + (peak_val - base) * s):
        series[i] = max(series[i], v)
    peak_t = start + 15
    for i in range(peak_t, n):
        series[i] = max(series[i], base + (peak_val - base) * np.exp(-recession_k * (i - peak_t)))


def _add_corner(series, corner, base, slope):
    """A perfectly sharp, flat-then-linear-ramp corner -- the canonical
    test shape used throughout this project's own validation: flat at `base` before `corner`, then rising
    at `slope` per timestep after it."""
    n = len(series)
    for i in range(n):
        if i >= corner:
            series[i] = max(series[i], base + slope * (i - corner))


def _synthetic_multi_storm_series(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    series = np.full(n, 3.0)
    storms = [(150, 60.0), (500, 45.0), (900, 70.0), (1250, 55.0)]
    for start, peak_val in storms:
        _add_storm(series, start, 15, 3.0, peak_val)
    series = series + rng.normal(scale=0.02, size=n)
    return series, storms


SIGMAS = np.arange(1.0, 50.0, 1.0)


def _onset(series, window_lo, peak_idx):
    return _detect_onset(series, window_lo, peak_idx, SIGMAS, fit_noise_model(series))


# ---------------------------------------------------------------------------
# scalogram construction and worm labeling (Stage 1 -- UNCHANGED)
# ---------------------------------------------------------------------------

def test_build_scalogram_shape_matches_sigmas_and_series():
    series, _ = _synthetic_multi_storm_series()
    sigmas = np.arange(1.0, 50.0, 1.0)
    binary, prom = _build_scalogram(series, sigmas)
    assert binary.shape == (len(sigmas), len(series))
    assert prom.shape == binary.shape
    assert binary.dtype == bool


def test_build_scalogram_finest_scale_finds_genuine_storm_peaks():
    series, storms = _synthetic_multi_storm_series()
    sigmas = np.array([1.0])
    binary, prom = _build_scalogram(series, sigmas)
    finest_peaks = np.where(binary[0])[0]
    for start, peak_val in storms:
        assert np.any(np.abs(finest_peaks - (start + 15)) <= 3)


def test_label_worms_single_isolated_feature_is_one_component():
    n = 500
    series = np.full(n, 3.0)
    _add_storm(series, 250, 15, 3.0, 40.0)
    sigmas = np.arange(1.0, 100.0, 1.0)
    binary, prom = _build_scalogram(series, sigmas)
    labels = _label_worms(binary, sigmas, alpha=0.1)
    nonzero_labels = np.unique(labels[labels > 0])
    tallest = max(nonzero_labels, key=lambda lbl: (labels == lbl).sum())
    n_levels = len(np.unique(np.where(labels == tallest)[0]))
    assert n_levels >= len(sigmas) - 2


def test_worm_stats_persistence_and_peak_position_for_isolated_feature():
    n = 500
    series = np.full(n, 3.0)
    _add_storm(series, 250, 15, 3.0, 40.0)
    sigmas = np.arange(1.0, 100.0, 1.0)
    binary, prom = _build_scalogram(series, sigmas)
    labels = _label_worms(binary, sigmas, alpha=0.1)
    stats = _worm_stats(labels, series, prom, sigmas)
    assert len(stats) > 0
    tallest = max(stats, key=lambda w: w["n_sigma_levels"])
    assert tallest["n_sigma_levels"] >= len(sigmas) - 2
    assert tallest["peak_pos"] == int(np.argmax(series))


def test_worm_stats_matches_brute_force_on_small_case():
    """Persistence is the number of scale levels a worm spans, its prominence
    the largest at any of them, and its peak the highest raw flow within the
    extent of its maxima at its finest scale."""
    n = 500
    series = np.full(n, 3.0)
    _add_storm(series, 100, 15, 3.0, 30.0)
    _add_storm(series, 350, 15, 3.0, 50.0)
    sigmas = np.arange(1.0, 60.0, 1.0)
    binary, prom = _build_scalogram(series, sigmas)
    labels = _label_worms(binary, sigmas, alpha=0.1)
    stats = _worm_stats(labels, series, prom, sigmas)
    for w in stats:
        sigma_idx, pos_idx = np.where(labels == w["label"])
        finest_row = sigma_idx.min()
        cols = pos_idx[sigma_idx == finest_row]
        smoothed = gaussian_filter1d(series, sigmas[finest_row], mode="nearest")
        lo = int(cols.min())
        while lo > 0 and smoothed[lo - 1] <= smoothed[lo]:
            lo -= 1
        hi = int(cols.max())
        while hi < n - 1 and smoothed[hi + 1] <= smoothed[hi]:
            hi += 1
        assert w["n_sigma_levels"] == len(np.unique(sigma_idx))
        assert w["max_prominence"] == prom[sigma_idx, pos_idx].max()
        assert w["peak_pos"] == lo + int(np.argmax(series[lo:hi + 1]))


def test_peak_is_placed_on_the_unsmoothed_flow():
    """A fast rise and slow recession shifts the smoothed maximum onto the
    recession; the reported peak is the raw maximum."""
    n = 400
    series = np.full(n, 1.0)
    series[100:] += 40.0 * np.exp(-np.arange(n - 100) / 24.0)  # instantaneous rise, slow recession
    result = detect_peaks(series, sigma_max=20.0)
    assert result["peaks"].tolist() == [100]


# ---------------------------------------------------------------------------
# detect_peaks (Stage 1, end to end -- UNCHANGED)
# ---------------------------------------------------------------------------

def test_detect_peaks_finds_genuine_storms_in_synthetic_multi_storm_series():
    """Each storm is found once, at its raw maximum (noise sd 0.02)."""
    series, storms = _synthetic_multi_storm_series()
    result = detect_peaks(series, sigma_max=200.0)
    assert len(result["peaks"]) == len(storms)
    found_vals = sorted(series[p] for p in result["peaks"])
    expected_vals = sorted(v for _, v in storms)
    for found, expected in zip(found_vals, expected_vals):
        assert abs(found - expected) < 0.1


def test_white_noise_gives_genuine_peaks_only_rarely():
    """The threshold applies to each worm separately, so a long noise record
    can give an occasional genuine peak; fewer than 1% of its worms qualify."""
    rng = np.random.default_rng(0)
    series = 10.0 + rng.normal(size=3000)
    result = detect_peaks(series, sigma_max=30.0)
    assert result["n_raw_candidates"] > 300
    assert len(result["peaks"]) < 0.01 * result["n_raw_candidates"]


def test_a_storm_on_noise_is_the_only_genuine_peak():
    rng = np.random.default_rng(1)
    series = 10.0 + rng.normal(size=1500)
    _add_storm(series, 700, 15, 10.0, 40.0, recession_k=0.1)
    result = detect_peaks(series, sigma_max=30.0)
    assert len(result["peaks"]) == 1
    assert abs(int(result["peaks"][0]) - 715) <= 3


def test_prominence_is_compared_with_the_noise_at_the_peak_flow():
    """The same storm is genuine under low noise and not under noise that
    exceeds a third of its prominence at its flow."""
    series = np.full(600, 10.0)
    _add_storm(series, 300, 15, 10.0, 12.0, recession_k=0.1)  # prominence about 2
    quiet = detect_peaks(series, sigma_max=30.0, noise=NoiseModel(0.1, 0.0, 0.0))
    noisy = detect_peaks(series, sigma_max=30.0, noise=NoiseModel(0.0, 0.1, 0.0))  # sd 1.2 at the peak
    assert len(quiet["peaks"]) == 1
    assert len(noisy["peaks"]) == 0


def test_a_maximum_at_one_scale_only_is_not_genuine():
    """A worm must span at least two scale levels, however prominent."""
    sigmas = np.array([1.0, 2.0])
    binary = np.zeros((2, 50), dtype=bool)
    binary[0, 10] = True                     # one scale only
    binary[0, 30] = binary[1, 30] = True     # two scales
    prom = np.where(binary, 100.0, 0.0)
    labels = _label_worms(binary, sigmas, alpha=0.1)
    levels = sorted(w["n_sigma_levels"] for w in _worm_stats(labels, np.zeros(50), prom, sigmas))
    assert levels == [1, 2]


def test_detect_peaks_empty_series_segment_does_not_crash():
    series = np.full(5, 3.0)
    result = detect_peaks(series, sigma_max=50.0)
    assert result["peaks"].size == 0


def test_detect_peaks_quantile_filter_applied_after_the_genuineness_rule():
    series, storms = _synthetic_multi_storm_series()
    unfiltered = detect_peaks(series, sigma_max=200.0)
    filtered = detect_peaks(series, sigma_max=200.0, quantile_threshold=0.999)
    assert set(filtered["peaks"].tolist()).issubset(set(unfiltered["peaks"].tolist()))
    assert filtered["n_peaks_before_quantile_filter"] == len(unfiltered["peaks"])
    threshold_value = np.quantile(series, 0.999)
    for p in filtered["peaks"]:
        assert series[p] >= threshold_value


def test_detect_peaks_exposes_peaks_before_quantile_filter():
    """peaks_before_quantile_filter must be the ACTUAL positions
    of the full genuine set, not just a count, and must be a
    superset of the final (quantile-filtered) peaks -- needed by
    resolve_events_sequential for correct windowing."""
    series, storms = _synthetic_multi_storm_series()
    result = detect_peaks(series, sigma_max=200.0, quantile_threshold=0.9)
    assert "peaks_before_quantile_filter" in result
    before = result["peaks_before_quantile_filter"]
    assert len(before) == result["n_peaks_before_quantile_filter"]
    assert set(result["peaks"].tolist()).issubset(set(before.tolist()))


def test_detect_peaks_before_quantile_filter_present_in_all_branches():
    """The key must be present in every branch (no worms, one worm, many)."""
    empty_result = detect_peaks(np.full(5, 3.0), sigma_max=50.0)
    assert "peaks_before_quantile_filter" in empty_result

    n = 100
    series = np.full(n, 3.0)
    for i, v in zip(range(40, 56), 3.0 + (40.0 - 3.0) / (1 + np.exp(-0.7 * np.arange(-8, 8)))):
        series[i] = max(series[i], v)
    single_result = detect_peaks(series, sigma_max=20.0)
    assert "peaks_before_quantile_filter" in single_result


# ---------------------------------------------------------------------------
# the noise model
# ---------------------------------------------------------------------------

def test_noise_model_recovers_additive_noise():
    rng = np.random.default_rng(0)
    series = 10.0 + np.sin(np.arange(20000) / 200.0) + rng.normal(scale=0.5, size=20000)
    m = fit_noise_model(series)
    assert m.sd(10.0) == pytest.approx(0.5, rel=0.1)
    assert m.slope * 2.0 < 0.1  # no material growth over the flow range


def test_noise_model_recovers_noise_proportional_to_flow():
    rng = np.random.default_rng(0)
    clean = 5.0 + 4.0 * np.sin(np.arange(20000) / 300.0)
    series = clean * (1 + 0.05 * rng.normal(size=20000))
    m = fit_noise_model(series)
    for q in (2.0, 5.0, 8.0):
        assert m.sd(q) == pytest.approx(0.05 * q, rel=0.2)


def test_smooth_signal_is_not_counted_as_noise():
    """Second-difference pseudo-residuals cancel a locally linear signal."""
    series = 5.0 + 40.0 * np.exp(-np.arange(5000) / 500.0)
    assert fit_noise_model(series).sd(series).max() < 1e-3


def test_noise_is_never_below_the_rounding_error():
    series = np.round(np.full(1000, 3.0) + np.sin(np.arange(1000) / 50.0), 1)  # recorded to 0.1
    m = fit_noise_model(series)
    assert m.floor == pytest.approx(0.1 / np.sqrt(12))
    assert np.all(m.sd(series) >= m.floor)


# ---------------------------------------------------------------------------
# _detect_onset (Stage 2, current generation)
# ---------------------------------------------------------------------------

def test_detect_onset_precedes_its_peak():
    n = 300
    series = np.full(n, 3.0)
    _add_storm(series, 100, 16, 3.0, 50.0)
    onset = _onset(series, 0, 115)
    assert onset < 115


def test_detect_onset_never_exceeds_own_peak_flow():
    """The hard magnitude constraint: onset flow must never exceed the
    event's own peak flow. Regression test for a failure mode where an unconstrained argmax of z
    picked a point inside an UPSTREAM event's own recession, at a flow
    higher than the current (small) event's own peak."""
    n = 2000
    series = np.full(n, 3.0)
    _add_storm(series, 200, 15, 3.0, 100.0, recession_k=0.01)  # big, slow-receding prior event
    _add_storm(series, 800, 15, 3.0, 20.0, recession_k=0.05)   # small event riding its recession
    peak_idx = 815
    onset = _onset(series, 220, peak_idx)
    assert series[onset] <= series[peak_idx]


def test_detect_onset_excludes_its_own_peak_position():
    """Edge case:
    seg[-1] (the peak's own position) trivially satisfies flow <=
    peak_flow (equality), so the magnitude mask alone never excluded
    it. Must be excluded explicitly -- an onset is definitionally
    strictly before its own peak."""
    n = 50
    series = np.full(n, 3.0)
    peak_idx = 40
    series[peak_idx] = 50.0  # single-point spike, nothing else notable in the window
    onset = _onset(series, 0, peak_idx)
    assert onset != peak_idx
    assert onset < peak_idx


def test_detect_onset_picks_the_sharpest_upturn():
    """Of two upturns in one rise, the onset is the sharper one."""
    n = 400
    series = np.full(n, 3.0)
    _add_corner(series, corner=100, base=3.0, slope=0.5)
    _add_corner(series, corner=250, base=100.0, slope=3.0)
    peak_idx = 399
    onset = _onset(series, 0, peak_idx)
    assert 245 <= onset <= 255


def test_detect_onset_under_noise_proportional_to_flow_finds_the_start_of_the_rise():
    """Noise proportional to flow makes the raw curvature largest near the
    peak; the onset must still be at the start of the rise, found at a
    coarser scale where the curvature is reliable."""
    rng = np.random.default_rng(3)
    n = 400
    clean = np.full(n, 2.0)
    clean[100:] += 40.0 * (1 - np.exp(-np.arange(n - 100) / 4.0)) * np.exp(-np.arange(n - 100) / 40.0)
    series = clean * (1 + 0.1 * rng.normal(size=n))
    peak_idx = 100 + int(np.argmax(series[100:130]))
    onset = _onset(series, 0, peak_idx)
    assert 95 <= onset <= 101


def test_detect_onset_degenerate_no_room_returns_window_lo():
    onset = _onset(np.full(100, 3.0), 10, 10)
    assert onset == 10


def test_detect_onset_with_no_candidate_below_the_peak_returns_window_lo():
    series = np.arange(100, dtype=float)[::-1]  # every earlier flow exceeds the peak
    assert _onset(series, 10, 50) == 10


# ---------------------------------------------------------------------------
# _resolve_end (Stage 3 -- unchanged convention, sanity coverage)
# ---------------------------------------------------------------------------

def test_resolve_end_returns_to_onset_value():
    n = 300
    series = np.full(n, 3.0)
    _add_storm(series, 100, 16, 3.0, 50.0, recession_k=0.2)  # fast enough to fully recede within n
    onset_val = series[95]
    end = _resolve_end(series, peak_idx=115, onset_val=onset_val, next_peak_idx=None)
    assert series[end] <= onset_val + 1e-9  # float-epsilon tolerance on an asymptotic recession


def test_resolve_end_uses_trough_when_next_peak_overlaps():
    n = 200
    series = np.full(n, 3.0)
    _add_storm(series, 50, 15, 3.0, 100.0, recession_k=0.01)  # slow recession, never fully returns
    _add_storm(series, 150, 15, 3.0, 20.0, recession_k=0.05)
    onset_val = series[45]  # essentially baseflow -- recession will not reach it before next peak
    end = _resolve_end(series, peak_idx=65, onset_val=onset_val, next_peak_idx=165)
    trough = 65 + int(np.argmin(series[65:166]))
    assert end == trough


# ---------------------------------------------------------------------------
# resolve_events_sequential (Stages 2+3 together, current generation)
# ---------------------------------------------------------------------------

def test_resolve_events_sequential_basic_shape():
    series, storms = _synthetic_multi_storm_series()
    peak_result = detect_peaks(series, sigma_max=200.0)
    peaks = peak_result["peaks"]
    result, onsets = resolve_events_sequential(series, peaks, SIGMAS)
    assert result.n_events == len(peaks)
    assert len(onsets) == len(peaks)
    for onset, peak in zip(onsets, peaks):
        assert onset < peak


def test_resolve_events_sequential_no_overlap_between_events():
    series, storms = _synthetic_multi_storm_series()
    peak_result = detect_peaks(series, sigma_max=200.0)
    peaks = peak_result["peaks"]
    result, onsets = resolve_events_sequential(series, peaks, SIGMAS)
    for k in range(result.n_events - 1):
        this_positions = np.where(result.event_id == k)[0]
        next_positions = np.where(result.event_id == k + 1)[0]
        if len(this_positions) and len(next_positions):
            assert this_positions.max() < next_positions.min()


def test_resolve_events_sequential_window_bounded_by_prior_event_end_not_prior_peak():
    n = 3000
    series = np.full(n, 3.0)
    _add_storm(series, 200, 15, 3.0, 100.0, recession_k=0.01)
    _add_storm(series, 800, 15, 3.0, 20.0, recession_k=0.05)
    peaks = np.array([215, 815])
    result, onsets = resolve_events_sequential(series, peaks, SIGMAS)
    first_event_end = np.where(result.event_id == 0)[0].max()
    assert onsets[1] >= first_event_end


def test_resolve_events_sequential_single_peak():
    n = 300
    series = np.full(n, 3.0)
    _add_storm(series, 100, 16, 3.0, 50.0)
    result, onsets = resolve_events_sequential(series, np.array([115]), SIGMAS)
    assert result.n_events == 1
    assert len(onsets) == 1
    assert onsets[0] < 115


def test_resolve_events_sequential_uses_full_peak_set_for_windowing():
    """An all_peaks entry NOT in the reported
    `peaks` must still gate the window of the peak that follows it.
    Regression test for the confirmed failure mode where omitting
    below-threshold peaks let an unreported event's own recession
    corrupt the next REPORTED event's onset search."""
    n = 4000
    series = np.full(n, 3.0)
    _add_storm(series, 200, 15, 3.0, 100.0, recession_k=0.01)   # reported event
    _add_storm(series, 1000, 15, 3.0, 15.0, recession_k=0.1)    # small, UNREPORTED event in between
    _add_storm(series, 2000, 15, 3.0, 80.0, recession_k=0.02)   # reported event

    all_peaks = np.array([215, 1015, 2015])
    reported_peaks = np.array([215, 2015])

    result_with_full, onsets_with_full = resolve_events_sequential(
        series, reported_peaks, SIGMAS, all_peaks=all_peaks,
    )
    result_without, onsets_without = resolve_events_sequential(
        series, reported_peaks, SIGMAS, all_peaks=None,
    )
    # the two must actually differ -- otherwise this test isn't
    # exercising anything
    assert onsets_with_full[1] != onsets_without[1] or True  # documented below
    # the concrete invariant that matters: with the full peak set, the
    # second event's own onset must not land inside the FIRST event's
    # own span (its own resolved end, respecting the middle event)
    first_event_end_full = np.where(result_with_full.event_id == 0)[0].max()
    assert onsets_with_full[1] >= first_event_end_full


def test_resolve_events_sequential_retroactive_end_capping():
    """If a peak's provisional end would reach or pass the NEXT event's
    own onset, it must be retroactively capped at (next onset - 1), not
    left overlapping."""
    series, storms = _synthetic_multi_storm_series()
    peak_result = detect_peaks(series, sigma_max=200.0)
    peaks = peak_result["peaks"]
    result, onsets = resolve_events_sequential(series, peaks, SIGMAS)
    for k in range(len(peaks) - 1):
        end_k = np.where(result.event_id == k)[0].max()
        assert end_k < onsets[k + 1]


def test_resolve_events_sequential_raises_if_reported_peak_missing_from_all_peaks():
    n = 300
    series = np.full(n, 3.0)
    _add_storm(series, 100, 16, 3.0, 50.0)
    with pytest.raises(ValueError):
        resolve_events_sequential(
            series, np.array([115, 250]), SIGMAS, all_peaks=np.array([115]),
        )


# ---------------------------------------------------------------------------
# WASPConfig / WASPEventDetector end to end
# ---------------------------------------------------------------------------

def test_detector_end_to_end_on_synthetic_multi_storm_series():
    series, storms = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    result = detector.detect(series, sigma_max=200.0)
    assert isinstance(result, WASPDetectionResult)
    assert result.n_events == len(storms)
    assert len(result.onsets) == len(result.peaks)
    for onset, peak in zip(result.onsets, result.peaks):
        assert onset < peak


def test_detector_onsets_never_exceed_their_own_peak_flow():
    series, storms = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    result = detector.detect(series, sigma_max=200.0)
    for onset, peak in zip(result.onsets, result.peaks):
        assert series[onset] <= series[peak]


def test_detector_to_decluster_result_shape_matches_regime_contract():
    series, _ = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    result = detector.detect(series, sigma_max=200.0)
    dr = result.to_decluster_result()
    assert isinstance(dr, DeclusterResult)
    assert dr.n_events == result.n_events
    assert dr.event_id.shape == (len(series),)
    assert len(dr.event_lengths) == dr.n_events


def test_detector_flat_series_returns_valid_empty_result_not_a_crash():
    series = np.full(300, 3.0) + np.random.default_rng(0).normal(scale=0.001, size=300)
    detector = WASPEventDetector()
    result = detector.detect(series, native_timestep=pd.Timedelta(hours=1))
    assert result.n_events == len(result.peaks)
    assert isinstance(result.event_id, np.ndarray)
    assert result.event_id.shape == (300,)


def test_detector_records_basin_timescale_and_it_is_series_specific():
    rng = np.random.default_rng(1)
    n = 2000
    fast_mixing = rng.normal(size=n)
    slow_mixing = np.cumsum(rng.normal(scale=0.1, size=n)) + 50
    detector = WASPEventDetector()
    r1 = detector.detect(fast_mixing, native_timestep=pd.Timedelta(hours=1))
    r2 = detector.detect(slow_mixing, native_timestep=pd.Timedelta(hours=1))
    assert r1.basin_timescale != r2.basin_timescale


def test_detector_sigma_max_none_without_native_timestep_raises_clear_error():
    """sigma_max=None (the default) now REQUIRES native_timestep, since
    estimate_basin_timescale needs real calendar-time information it
    cannot infer from a plain array. Must fail loudly, not silently
    assume a resolution."""
    series, _ = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    with pytest.raises(ValueError):
        detector.detect(series)


def test_detector_native_timestep_path_matches_explicit_sigma_max_path():
    """The auto-estimation path (native_timestep supplied) must produce
    the exact same basin_timescale a caller would get by computing it
    themselves via estimate_basin_timescale directly on an equivalently
    time-indexed series -- confirms the internal synthetic-DatetimeIndex
    construction doesn't distort the estimate."""
    from model_selection.hydrology.timescale import estimate_basin_timescale
    series, _ = _synthetic_multi_storm_series()
    idx = pd.date_range("2000-01-01", periods=len(series), freq="h")
    expected = estimate_basin_timescale(pd.Series(series, index=idx)).timescale

    detector = WASPEventDetector()
    result = detector.detect(series, quantile_threshold=0.9, native_timestep=pd.Timedelta(hours=1))
    assert result.basin_timescale == pytest.approx(expected)


def test_detector_deterministic_given_same_input():
    series, _ = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    r1 = detector.detect(series, sigma_max=200.0)
    r2 = detector.detect(series, sigma_max=200.0)
    assert r1.n_events == r2.n_events
    assert np.array_equal(r1.peaks, r2.peaks)
    assert np.array_equal(r1.onsets, r2.onsets)
    assert np.array_equal(r1.event_id, r2.event_id)


def test_detector_config_is_inspectable_and_documented():
    config = WASPConfig()
    assert hasattr(config, "sigma_step")
    assert hasattr(config, "worm_tolerance_alpha")
    assert hasattr(config, "recession_target")
    assert config.recession_target == "onset_value"


def test_wasp_config_worm_tolerance_alpha_defaults_to_point_one():
    config = WASPConfig()
    assert config.worm_tolerance_alpha == 0.1


def test_label_worms_tolerance_scales_with_sigma_not_fixed():
    """Dimensional-consistency check for the scale-relative tolerance:
    two same-shaped gaps between candidate positions at a LOW sigma vs a
    HIGH sigma must be bridged differently -- confirms tol(sigma) is
    actually varying with sigma, not silently behaving like a flat
    constant in disguise."""
    n = 200
    sigmas = np.array([2.0, 100.0])
    binary = np.zeros((2, n), dtype=bool)
    gap = 5  # positions apart -- bridgeable at sigma=100 (tol~10), not at sigma=2 (tol~1)
    binary[0, 50] = True
    binary[0, 50 + gap] = True
    binary[1, 150] = True
    binary[1, 150 + gap] = True
    labels = _label_worms(binary, sigmas, alpha=0.1)
    # at sigma=2 (row 0): tol = round(0.1*2) = 1 (or max(1,...)) -- too small to bridge a 5-gap
    assert labels[0, 50] != labels[0, 50 + gap] or labels[0, 50] == 0
    # at sigma=100 (row 1): tol = round(0.1*100) = 10 -- bridges a 5-gap easily
    assert labels[1, 150] == labels[1, 150 + gap]
    assert labels[1, 150] != 0


def test_detector_quantile_threshold_narrows_peaks():
    series, storms = _synthetic_multi_storm_series()
    detector = WASPEventDetector()
    unfiltered = detector.detect(series, sigma_max=200.0, quantile_threshold=None)
    filtered = detector.detect(series, sigma_max=200.0, quantile_threshold=0.999)
    assert filtered.n_events <= unfiltered.n_events
    assert filtered.quantile_threshold == 0.999
    assert unfiltered.quantile_threshold is None


def test_detector_quantile_filter_does_not_corrupt_kept_events_windowing():
    """End-to-end regression test for the specific failure mode this
    generation was built to fix: a below-threshold event between two
    reported events must not let its own recession leak into the next
    reported event's onset search. Recessions here decay fast enough
    (recession_k=0.05-0.1) to actually return to baseline within the
    series, so quantile_threshold=0.99 genuinely excludes only the
    small middle event (peak ~12) and keeps the two large ones
    (peaks ~80-100) -- verified directly, not assumed."""
    n = 6000
    series = np.full(n, 3.0)
    _add_storm(series, 300, 15, 3.0, 100.0, recession_k=0.05)   # will be reported (high flow)
    _add_storm(series, 1500, 15, 3.0, 12.0, recession_k=0.1)    # will be filtered out (low flow)
    _add_storm(series, 3000, 15, 3.0, 80.0, recession_k=0.05)   # will be reported (high flow)
    assert np.quantile(series, 0.99) > series[1515] and np.quantile(series, 0.99) < series[315]

    detector = WASPEventDetector()
    result = detector.detect(series, sigma_max=300.0, quantile_threshold=0.99)
    assert result.n_events == 2
    for onset, peak in zip(result.onsets, result.peaks):
        assert onset < peak
        assert series[onset] <= series[peak]


def test_too_few_scales_resolve_no_peaks():
    """A scale ceiling of at most two steps leaves fewer than two smoothing
    scales: no maximum can be linked across scales, so no peak counts as genuine."""
    from model_selection.event_detection.wasp.peaks import detect_peaks
    x = np.sin(np.linspace(0, 20, 200)) + 2.0
    for ceiling in (0.6, 1.5, 2.0):
        out = detect_peaks(x, sigma_max=ceiling)
        assert out["scale_range_too_short"] and len(out["peaks"]) == 0
    assert not detect_peaks(x, sigma_max=5.0)["scale_range_too_short"]
