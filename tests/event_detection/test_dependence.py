"""Tests of the rules that group high-flow events into clusters of dependent
events (event_detection.dependence) and of the widened windows they read."""
import numpy as np
import pytest

from model_selection.event_detection import EventConfig
from model_selection.event_detection.dependence import (
    arrival_clusters, intervals_run_length, recession_clusters, subset_labels,
)


def test_recession_clusters_chain_overlapping_windows_in_peak_order():
    # windows (start, end) with peaks; the third starts before the first's end (chained through the long first
    # window although it does not overlap the second); the fourth is separate; the fifth overlaps the fourth.
    starts = [0, 12, 30, 100, 118]
    ends = [40, 20, 60, 120, 140]
    peaks = [5, 15, 35, 105, 125]
    assert list(recession_clusters(starts, ends, peaks)) == [0, 0, 0, 1, 1]
    # A window that starts one step after the latest end begins a new cluster; touching is not overlapping.
    assert list(recession_clusters([0, 11], [10, 20], [5, 15])) == [0, 1]
    assert list(recession_clusters([0, 10], [10, 20], [5, 15])) == [0, 0]
    # Labels follow the input order, whatever it is.
    assert list(recession_clusters([100, 0], [120, 10], [110, 5])) == [1, 0]


def test_the_intervals_estimator_and_run_length_follow_ferro_and_segers():
    # Clusters of three peaks one step apart, the clusters 100 steps apart: about a third as many clusters as
    # peaks, and the run length separates the clusters.
    peaks = np.concatenate([np.arange(3) + 100 * k for k in range(30)])
    rl = intervals_run_length(peaks)
    T = np.diff(np.sort(peaks)).astype(float)
    N = peaks.size
    expected = 2 * np.sum(T - 1) ** 2 / ((N - 1) * np.sum((T - 1) * (T - 2)))
    assert rl.extremal_index == pytest.approx(min(1.0, expected))
    assert rl.n_clusters == 30 and rl.run_length == 1.0
    labels, _ = arrival_clusters(peaks)
    assert np.array_equal(labels, np.repeat(np.arange(30), 3))
    # Peaks arriving at random (Poisson) do not cluster: the extremal index is near one.
    rng = np.random.default_rng(0)
    random_peaks = np.cumsum(rng.exponential(200.0, size=400)).astype(int)
    assert intervals_run_length(random_peaks).extremal_index > 0.9


def test_subset_labels_keep_chains_through_removed_events():
    labels = np.array([0, 0, 0, 1, 2, 2])
    # Drop the middle event of the first chain: the other two stay together; labels renumbered in order.
    assert list(subset_labels(labels, np.array([0, 2, 3, 5]))) == [0, 0, 1, 2]


def test_expanded_events_keep_their_widened_windows():
    from model_selection.event_detection.base import DeclusterResult
    from model_selection.event_detection.consensus import HighFlowEvent
    from model_selection.event_detection.windows import build_expanded_events
    n = 60
    ids = np.full(n, -1)
    ids[5:30] = 0      # one formulation's own event spans both consensus events
    ids[40:50] = 1
    own = {"m": DeclusterResult(event_id=ids, n_events=2, event_lengths=np.array([25, 10]))}
    consensus = [HighFlowEvent(event_id=0, start=8, end=12, duration_timesteps=5, n_high_flow_timesteps=5, peak_index=10),
                 HighFlowEvent(event_id=1, start=20, end=24, duration_timesteps=5, n_high_flow_timesteps=5, peak_index=22),
                 HighFlowEvent(event_id=2, start=42, end=46, duration_timesteps=5, n_high_flow_timesteps=5, peak_index=44)]
    proxy = np.zeros(n)
    proxy[[10, 22, 44]] = 1.0
    events = build_expanded_events(consensus, own, peak_series=proxy)
    assert [(e.widened_start, e.widened_end) for e in events] == [(5, 29), (5, 29), (40, 49)]
    assert events[0].end < events[1].start          # separated windows do not overlap ...
    labels = recession_clusters([e.widened_start for e in events], [e.widened_end for e in events],
                                [e.peak_index for e in events])
    assert list(labels) == [0, 0, 1]                 # ... but the first two were one response


def test_event_config_validates_the_dependence_rule():
    assert EventConfig().dependence == "recession"
    for rule in ("recession", "arrivals", "lag"):
        assert EventConfig(dependence=rule).dependence == rule
    with pytest.raises(ValueError, match="dependence"):
        EventConfig(dependence="blocks")
