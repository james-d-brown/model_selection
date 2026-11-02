import numpy as np
import pandas as pd
import pytest

from model_selection.event_detection.base import DeclusterResult
from model_selection.event_detection.pot.detector import detect_model_events
from model_selection.event_detection.windows import (
    ExpandedEvent,
    attach_timestamps,
    own_window_to_expanded_events,
)


def _ramp_series(control_x, control_y, n):
    return np.interp(np.arange(n), control_x, control_y)


def test_own_window_to_expanded_events_matches_known_ab_c_scenario():
    """Direct validation against the precisely-known A/B/C scenario
    (A absorbs B; C stays independent): event 0's span/peak must
    correspond to the merged A/B group (peak at A, the taller of the
    two), event 1's to C."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    result = detect_model_events(series, quantile_threshold=0.0, recession_ratio=0.9)

    events = own_window_to_expanded_events(series, result)
    assert len(events) == 2
    assert events[0].event_id == 0
    assert events[0].peak_index == 10  # A, the tallest of the merged A/B group
    assert events[1].event_id == 1
    assert events[1].peak_index == 30  # C


def test_own_window_to_expanded_events_span_matches_decluster_result_exactly():
    """Each ExpandedEvent's [start, end] must exactly match the
    corresponding positions in decluster_result.event_id -- no
    truncation, widening, or off-by-one drift introduced by the
    conversion."""
    rng = np.random.default_rng(2)
    n = 2000
    series = np.full(n, 3.0)
    for c in rng.choice(np.arange(100, n - 100), size=5, replace=False):
        width = rng.integers(15, 60)
        height = rng.uniform(10, 50)
        t = np.arange(min(width, n - c))
        end = min(c + width, n)
        series[c:end] = np.maximum(
            series[c:end], series[c] + height * np.exp(-((t[:end - c] - width / 3) ** 2) / (2 * max(width / 6, 1) ** 2))
        )
    result = detect_model_events(series, quantile_threshold=0.85, recession_ratio=0.25)

    events = own_window_to_expanded_events(series, result)
    assert len(events) == result.n_events
    for ev in events:
        positions = np.where(result.event_id == ev.event_id)[0]
        assert ev.start == positions.min()
        assert ev.end == positions.max()
        assert ev.duration_timesteps == positions.max() - positions.min() + 1


def test_own_window_to_expanded_events_peak_index_is_this_models_own_argmax():
    """peak_index must be THIS series' own argmax within its own span --
    verified directly against a known, hand-computed peak location."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 6, 4, 12, 4.5, 8, 1]  # B (idx 20) is the tallest peak here
    series = _ramp_series(control_x, control_y, 41)
    result = detect_model_events(series, quantile_threshold=0.0, recession_ratio=0.9)
    events = own_window_to_expanded_events(series, result)
    # the event containing idx 20 must have its OWN peak_index at 20
    # (this series' own tallest point there), not at A (idx 10) or C (idx 30)
    matching = [ev for ev in events if ev.start <= 20 <= ev.end]
    assert len(matching) == 1
    assert matching[0].peak_index == 20
    assert series[matching[0].peak_index] == series[20]


def test_own_window_to_expanded_events_empty_result_returns_empty_list():
    n = 100
    empty = DeclusterResult(event_id=np.full(n, -1, dtype=np.int64), n_events=0,
                              event_lengths=np.array([], dtype=np.int64))
    series = np.full(n, 3.0)
    events = own_window_to_expanded_events(series, empty)
    assert events == []


def test_own_window_to_expanded_events_output_is_duck_type_compatible_with_attach_timestamps():
    """The whole point of reusing ExpandedEvent: attach_timestamps (built
    for the consensus/build_expanded_events pathway) must work
    identically on this function's own output, with no special-casing."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    result = detect_model_events(series, quantile_threshold=0.0, recession_ratio=0.9)
    events = own_window_to_expanded_events(series, result)

    idx = pd.date_range("2020-01-01", periods=41, freq="h")
    events_ts = attach_timestamps(events, idx)
    assert all(isinstance(ev, ExpandedEvent) for ev in events_ts)
    assert events_ts[0].start_time == idx[events_ts[0].start]
    assert events_ts[0].end_time == idx[events_ts[0].end]
    assert events_ts[0].peak_time == idx[events_ts[0].peak_index]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# --- Widening consensus events and separating overlaps --------------------

def _own_result(n, spans):
    event_id = np.full(n, -1, dtype=np.int64)
    for k, (start, end) in enumerate(spans):
        event_id[start:end + 1] = k
    return DeclusterResult(event_id=event_id, n_events=len(spans),
                           event_lengths=np.array([e - s + 1 for s, e in spans], dtype=np.int64))


def _consensus(spans, peak_series):
    from model_selection.event_detection.consensus import HighFlowEvent
    events = []
    for k, (start, end) in enumerate(spans):
        peak = start + int(np.argmax(peak_series[start:end + 1]))
        events.append(HighFlowEvent(event_id=k, start=start, end=end, duration_timesteps=end - start + 1,
                                    n_high_flow_timesteps=end - start + 1, peak_index=peak))
    return events


def _two_peaks(n=80):
    t = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((t - 20) / 4) ** 2) + 0.8 * np.exp(-0.5 * ((t - 50) / 4) ** 2)


def test_windows_widen_to_the_models_own_events():
    from model_selection.event_detection.windows import build_expanded_events
    proxy = _two_peaks()
    consensus = _consensus([(18, 22)], proxy)
    own = {"a": _own_result(80, [(12, 30)]), "b": _own_result(80, [(15, 26)])}
    (ev,) = build_expanded_events(consensus, own, peak_series=proxy)
    assert (ev.start, ev.end) == (12, 30)
    assert ev.peak_index == 20


def test_overlapping_windows_are_separated_at_the_minimum_between_peaks_not_merged():
    # One model's own event spans both floods: the widened windows overlap.
    # They stay two events, separated at the trough between the peaks, the
    # trough going to the earlier event (as the POT detector does).
    from model_selection.event_detection.windows import build_expanded_events
    proxy = _two_peaks()
    consensus = _consensus([(18, 22), (48, 52)], proxy)
    own = {"lumping": _own_result(80, [(10, 65)]), "b": _own_result(80, [(14, 28), (42, 58)])}
    events = build_expanded_events(consensus, own, peak_series=proxy)
    assert len(events) == 2
    trough = 20 + int(np.argmin(proxy[20:50]))
    assert (events[0].start, events[0].end) == (10, trough)
    assert (events[1].start, events[1].end) == (trough + 1, 65)
    assert events[0].peak_index == 20 and events[1].peak_index == 50
    assert [ev.source_consensus_event_ids for ev in events] == [[0], [1]]


def test_separated_windows_never_share_timesteps():
    from model_selection.event_detection.windows import build_expanded_events
    proxy = _two_peaks(120) + 0.6 * np.exp(-0.5 * ((np.arange(120) - 85) / 4) ** 2)
    consensus = _consensus([(18, 22), (48, 52), (83, 87)], proxy)
    own = {"lumping": _own_result(120, [(5, 110)]), "b": _own_result(120, [(15, 25), (45, 55), (80, 90)])}
    events = build_expanded_events(consensus, own, peak_series=proxy)
    assert len(events) == 3
    for left, right in zip(events[:-1], events[1:]):
        assert left.end < right.start



def test_a_window_without_a_detected_peak_is_not_an_event_and_returns_to_its_neighbour():
    # A flood peaking at 20 and a brief high-flow excursion on its recession
    # (40-42) whose widened window overlaps the flood's. No formulation's
    # detector found a peak in the excursion, so it is not an event and the
    # flood keeps its full window.
    from model_selection.event_detection.windows import build_expanded_events
    t = np.arange(80, dtype=float)
    proxy = np.exp(-0.5 * ((t - 20) / 4) ** 2) + 0.05 * np.exp(-0.5 * ((t - 41) / 1.0) ** 2)
    consensus = _consensus([(18, 22), (40, 42)], proxy)
    own = {m: _own_result(80, [(10, 60)]) for m in ("a", "b", "c")}
    own_peaks = {"a": np.array([20]), "b": np.array([21]), "c": np.array([19])}
    removed = []
    events = build_expanded_events(consensus, own, peak_series=proxy, own_peaks=own_peaks, removed_windows=removed)
    assert len(events) == 1 and (events[0].start, events[0].end) == (10, 60)
    assert events[0].n_formulations_with_peak == 3 and [r[0] for r in removed] == [1]
    # Without the peak requirement the excursion becomes a separate window cut from the flood.
    assert len(build_expanded_events(consensus, own, peak_series=proxy)) == 2


def test_one_formulations_peak_keeps_a_window():
    from model_selection.event_detection.windows import build_expanded_events
    proxy = _two_peaks()
    consensus = _consensus([(18, 22), (48, 52)], proxy)
    own = {"lumping": _own_result(80, [(10, 65)]), "b": _own_result(80, [(14, 28), (42, 58)])}
    own_peaks = {"lumping": np.array([20]), "b": np.array([20, 50])}
    events = build_expanded_events(consensus, own, peak_series=proxy, own_peaks=own_peaks)
    assert [ev.n_formulations_with_peak for ev in events] == [2, 1]
