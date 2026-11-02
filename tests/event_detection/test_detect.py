"""
Tests of the single per-series detection entry point
(event_detection.detect): method dispatch via EventConfig, the
DeclusterResult contract for every method, metadata, parameter wiring, and
regularization of gapped series. The methods themselves are tested in
test_pot_and_consensus.py and test_wasp.py.
"""
import numpy as np
import pandas as pd
import pytest

from model_selection.event_detection import (
    DeclusterResult,
    EventConfig,
    POTConfig,
    POTDetector,
    WASPDetector,
    detect_events_for_series,
    detect_events_per_series,
    make_detector,
)
from model_selection.event_detection.base import DetectionResult, EventDetector


def _synthetic_values(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    series = np.full(n, 3.0)

    def add_storm(vals, start, peak_val, k=0.05):
        t = np.arange(-8, 8)
        s = 1 / (1 + np.exp(-0.7 * t))
        for i, v in zip(range(start, start + 16), 3.0 + (peak_val - 3.0) * s):
            vals[i] = max(vals[i], v)
        for i in range(start + 16, len(vals)):
            vals[i] = max(vals[i], 3.0 + (peak_val - 3.0) * np.exp(-k * (i - (start + 16))))

    add_storm(series, 150, 60.0)
    add_storm(series, 900, 45.0)
    return series + rng.normal(scale=0.02, size=n)


def _as_series(values, freq="h"):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq=freq, tz="UTC"))


def test_default_method_is_wasp():
    assert EventConfig().method == "wasp"
    assert isinstance(make_detector(EventConfig()), WASPDetector)
    assert isinstance(make_detector(EventConfig(method="pot")), POTDetector)


def test_invalid_method_raises_clearly():
    with pytest.raises(ValueError, match="method"):
        EventConfig(method="bogus")


@pytest.mark.parametrize("method", ["pot", "wasp"])
def test_every_method_returns_the_decluster_contract(method):
    series = _as_series(_synthetic_values())
    result = detect_events_for_series(series, EventConfig(method=method))
    dr = result.decluster_result
    assert isinstance(dr, DeclusterResult)
    assert dr.event_id.shape == (len(series),)
    assert dr.n_events >= 1
    assert dr.event_lengths.shape == (dr.n_events,)
    assert set(np.unique(dr.event_id)) <= set(range(-1, dr.n_events))


def test_pot_has_no_metadata_and_wasp_reports_its_diagnostics():
    series = _as_series(_synthetic_values())
    assert detect_events_for_series(series, EventConfig(method="pot")).metadata is None
    meta = detect_events_for_series(series, EventConfig(method="wasp")).metadata
    for key in ("noise_intercept", "noise_slope", "basin_timescale",
                "n_raw_peak_candidates", "quantile_threshold",
                "n_peaks_before_quantile_filter", "n_peaks_after_quantile_filter"):
        assert key in meta


def test_quantile_threshold_reaches_wasp():
    series = _as_series(_synthetic_values())
    loose = detect_events_for_series(series, EventConfig(method="wasp", high_flow_quantile=0.0))
    strict = detect_events_for_series(series, EventConfig(method="wasp", high_flow_quantile=0.999))
    assert strict.metadata["quantile_threshold"] == 0.999
    assert strict.decluster_result.n_events <= loose.decluster_result.n_events


def test_min_peak_separation_reaches_pot():
    """A nearby secondary peak is its own event by default and rejected as a
    candidate with min_peak_separation=8."""
    values = np.full(100, 1.0)
    values[40:60] = [1, 2, 5, 10, 20, 30, 25, 18, 12, 8, 6, 9, 14, 10, 6, 4, 3, 2, 1.5, 1]
    series = _as_series(values)
    base = dict(method="pot", high_flow_quantile=0.0)
    default = detect_events_for_series(series, EventConfig(pot=POTConfig(recession_ratio=0.9), **base))
    filtered = detect_events_for_series(
        series, EventConfig(pot=POTConfig(recession_ratio=0.9, min_peak_separation=8), **base))
    assert default.decluster_result.n_events == 2
    assert filtered.decluster_result.n_events == 1


def test_gapped_series_maps_back_to_original_positions():
    values = _synthetic_values()
    full = _as_series(values)
    gapped = full.drop(full.index[1000:1005])
    result = detect_events_for_series(gapped, EventConfig(method="wasp"))
    assert result.decluster_result.event_id.shape == (len(gapped),)


def test_per_series_dict_and_custom_detector():
    class Nothing(EventDetector):
        name = "nothing"

        def detect(self, values, timestep, quantile_threshold):
            return DetectionResult(events=DeclusterResult.empty(len(values)))

    series = _as_series(_synthetic_values())
    results = detect_events_per_series({"a": series, "b": series}, detector=Nothing())
    assert set(results) == {"a", "b"}
    assert all(r.decluster_result.n_events == 0 for r in results.values())


def test_distinct_series_get_distinct_events():
    a = _as_series(_synthetic_values(seed=0))
    vals_b = _synthetic_values(seed=1)
    vals_b[400:420] += 50.0
    results = detect_events_per_series({"a": a, "b": _as_series(vals_b)}, EventConfig(method="wasp"))
    assert not np.array_equal(results["a"].decluster_result.event_id, results["b"].decluster_result.event_id)



def test_detectors_report_peaks_on_original_positions():
    from model_selection.event_detection import EventConfig
    from model_selection.event_detection.detect import detect_events_for_series
    t = np.arange(2000, dtype=float)
    flow = 1.0 + 5 * np.exp(-0.5 * ((t - 500) / 20) ** 2) + 4 * np.exp(-0.5 * ((t - 1400) / 20) ** 2)
    series = pd.Series(flow, index=pd.date_range("2000-01-01", periods=2000, freq="h"))
    for method in ("wasp", "pot"):
        r = detect_events_for_series(series, EventConfig(method=method, high_flow_quantile=0.9))
        assert r.peaks is not None
        assert any(abs(p - 500) <= 2 for p in r.peaks) and any(abs(p - 1400) <= 2 for p in r.peaks)
