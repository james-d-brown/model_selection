"""
Which high-flow events are dependent: rules that group events into clusters.

Each rule returns one cluster label per event. Events in the same cluster
are treated as dependent, and events in different clusters as independent.
How the tests use the clusters is separate from the rule that forms them
(statistics.significance.clusters), so a rule can be replaced without
changing the tests. See docs/methods.md, Dependence between samples.

- Recession overlap (the default): an event depends on the one before it
  when it begins before that event has receded, that is, when their
  widened windows overlap before they are separated. Overlapping events
  are chained. This is the hydrological notion of independent floods
  (Cunnane, 1979). A recession ends where the formulations' own detectors
  end their events.
- Clustering of arrivals: events are chained when their peaks are no
  further apart than the run length of the automatic declustering of
  Ferro and Segers (2003), applied to the times between peaks. Their result
  concerns the times between exceedances of a high threshold; applied to
  event peaks it describes only whether events arrive in bunches.

The third rule, the decorrelation length, uses the dependence lag directly
and forms no clusters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RULES = ("recession", "arrivals", "lag")


@dataclass(frozen=True)
class RunLength:
    """Ferro and Segers' (2003) intervals estimate of the extremal index and
    the run length of their automatic declustering, in time steps."""
    extremal_index: float
    n_clusters: int
    run_length: float


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Labels renumbered 0, 1, ... in order of first appearance."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse]


def recession_clusters(starts, ends, peaks) -> np.ndarray:
    """Cluster label of each event: events in peak order join the current
    cluster when their widened window starts no later than the latest end
    of the cluster's windows (the earlier recession had not finished).
    starts, ends, peaks : widened window bounds and peak positions, in time
    steps, one per event in any order; labels are returned in that order."""
    starts, ends, peaks = (np.asarray(a, dtype=float) for a in (starts, ends, peaks))
    n = peaks.size
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return labels
    order = np.argsort(peaks, kind="stable")
    label, latest_end = 0, ends[order[0]]
    for k in order[1:]:
        if starts[k] > latest_end:
            label += 1
            latest_end = ends[k]
        else:
            latest_end = max(latest_end, ends[k])
        labels[k] = label
    return labels


def intervals_run_length(peaks) -> RunLength:
    """The intervals estimator of the extremal index (Ferro and Segers,
    2003, eq. 4) on the times between consecutive peaks, and the run length
    of their automatic declustering: the C-th largest inter-peak time, with
    C = floor(theta N) + 1 clusters for N peaks, lowered past ties."""
    t = np.sort(np.asarray(peaks, dtype=float))
    T = np.diff(t)
    N = t.size
    if N < 3 or T.size == 0:
        return RunLength(1.0, N, 0.0)
    if T.max() > 2:
        theta = 2.0 * np.sum(T - 1) ** 2 / ((N - 1) * np.sum((T - 1) * (T - 2)))
    else:
        theta = 2.0 * np.sum(T) ** 2 / ((N - 1) * np.sum(T ** 2))
    theta = float(min(1.0, theta))
    C = min(int(np.floor(theta * N)) + 1, T.size)
    s = np.sort(T)[::-1]
    while C > 1 and s[C - 2] == s[C - 1]:
        C -= 1
    return RunLength(theta, C, float(s[C - 1]))


def arrival_clusters(peaks) -> tuple[np.ndarray, RunLength]:
    """Cluster label of each event (in the given order) and the run length:
    consecutive peaks no further apart than the run length share a cluster."""
    peaks = np.asarray(peaks, dtype=float)
    labels = np.zeros(peaks.size, dtype=int)
    rl = intervals_run_length(peaks)
    if peaks.size < 2:
        return labels, rl
    order = np.argsort(peaks, kind="stable")
    gaps = np.diff(peaks[order])
    labels[order] = np.concatenate([[0], np.cumsum(gaps > rl.run_length)])
    return labels, rl


def subset_labels(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Labels of the kept events, renumbered 0, 1, ... in their order. Events
    left out break no chains: events chained through a removed event stay
    in one cluster."""
    return _relabel(np.asarray(labels)[np.asarray(keep)])
