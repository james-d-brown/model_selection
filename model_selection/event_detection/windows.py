"""
Final event windows for curve extraction.

A consensus window can clip a model's own rising limb or recession.
build_expanded_events therefore widens each consensus event to the union
of the best-overlapping own event of every model. Where widened windows
overlap, they are separated at the minimum of the obs-free consensus proxy
between their peaks, the boundary going to the earlier event, as the POT
detector separates overlapping neighbours; two events never claim the same
timesteps and are never merged here.

An event must contain a detected peak: a window counts as an event only if
at least one formulation's own detector found a peak within it (for WASP, a
genuine peak, a maximum persisting across scales, of any height). The
consensus high-flow threshold, the widening and the separation are
constructions after detection, and a window they produce without a
detected peak, such as a brief excursion above the threshold on a
recession, is not an event. It is removed and the remaining windows are
separated again without it, so where it adjoined an event its timesteps
return to that event; this repeats until every window contains a peak.

Each event keeps its widened window before separation (widened_start,
widened_end). Where two widened windows overlap, the later event began
before the earlier one had receded; the recession-overlap rule of the
dependence between events uses this (event_detection.dependence).

The per-model own windows are computed once per basin run and passed in
everywhere, so every consumer sees the same events.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from model_selection.event_detection.base import DeclusterResult


@dataclass
class ExpandedEvent:
    """A final event window (attribute-compatible with HighFlowEvent).
    source_consensus_event_ids holds the consensus event it was built from.
    widened_start, widened_end : the window widened to the formulations'
    own events, before overlapping windows were separated (None when not
    built by build_expanded_events).
    """
    event_id: int
    start: int
    end: int
    duration_timesteps: int
    source_consensus_event_ids: list[int] = field(default_factory=list)
    peak_index: int | None = None
    n_formulations_with_peak: int | None = None  # None when the detectors report no peaks
    start_time: "object | None" = None  # pd.Timestamp, set by attach_timestamps
    end_time: "object | None" = None
    peak_time: "object | None" = None
    widened_start: int | None = None
    widened_end: int | None = None


def expand_event_to_model_union(
        consensus_start: int,
        consensus_end: int,
        own_windows: dict[str, DeclusterResult],
) -> tuple[int, int]:
    """Union of [consensus_start, consensus_end] with each model's own event that
    overlaps it most.
    """
    start, end = consensus_start, consensus_end
    c_range = np.arange(consensus_start, consensus_end + 1)
    for dr in own_windows.values():
        overlapping_ids = dr.event_id[c_range]
        overlapping_ids = overlapping_ids[overlapping_ids != -1]
        if overlapping_ids.size == 0:
            continue
        best_id = np.bincount(overlapping_ids).argmax()
        own_idx = np.where(dr.event_id == best_id)[0]
        start = min(start, int(own_idx.min()))
        end = max(end, int(own_idx.max()))
    return start, end


def expand_all_events(
        consensus_events: list,
        own_windows: dict[str, DeclusterResult],
) -> list[tuple[int, int, int]]:
    """[(event_id, expanded_start, expanded_end)] for every consensus event."""
    out = []
    for ev in consensus_events:
        start, end = expand_event_to_model_union(ev.start, ev.end, own_windows)
        out.append((ev.event_id, start, end))
    return out


def separate_overlapping_windows(
        windows: list[tuple[int, int, int]],
        peaks: dict[int, int],
        peak_series: np.ndarray | None,
) -> list[tuple[int, int, int]]:
    """Separate overlapping windows [(event_id, start, end)], one per consensus
    event. Between consecutive events (ordered by peak) the separating point
    is the minimum of peak_series between their peaks, assigned to the
    earlier event (without peak_series, the earlier peak). A window is
    limited by the separating point on a side only where its span overlaps a
    neighbour's on that side, so a window that overlaps nothing keeps its full
    span. Returns [(event_id, start, end)] in time order, without overlaps.
    """
    ordered = sorted(windows, key=lambda w: peaks[w[0]])
    m = len(ordered)
    cuts = []
    for k in range(m - 1):
        pa, pb = peaks[ordered[k][0]], peaks[ordered[k + 1][0]]
        if peak_series is not None and pb > pa and np.isfinite(peak_series[pa:pb]).any():
            cuts.append(pa + int(np.nanargmin(peak_series[pa:pb])))
        else:
            cuts.append(pa)
    out = []
    prev_end = -1
    for k, (eid, start, end) in enumerate(ordered):
        if k > 0 and start <= max(w[2] for w in ordered[:k]):
            start = max(start, cuts[k - 1] + 1)
        if k < m - 1 and end >= min(w[1] for w in ordered[k + 1:]):
            end = min(end, cuts[k])
        start = max(start, prev_end + 1)
        out.append((eid, start, end))
        prev_end = end
    return out


def formulations_with_peak(start: int, end: int, own_peaks: dict[str, np.ndarray]) -> int:
    """Number of formulations with at least one detected peak in [start, end]."""
    return sum(bool(np.any((p >= start) & (p <= end))) for p in own_peaks.values())


def windows_without_peak(windows: list[tuple[int, int, int]], own_peaks: dict[str, np.ndarray]) -> set[int]:
    """Event ids of the windows [(event_id, start, end)] in which no
    formulation has a detected peak."""
    return {eid for eid, s, e in windows if formulations_with_peak(s, e, own_peaks) == 0}


def build_expanded_events(
        consensus_events: list,
        own_windows: dict[str, DeclusterResult],
        peak_series: np.ndarray | None = None,
        own_peaks: dict[str, np.ndarray] | None = None,
        removed_windows: list | None = None,
) -> list[ExpandedEvent]:
    """Widen every consensus event to the union of the models' own events,
    separate overlapping windows at the minimum between peaks, and return
    one ExpandedEvent per consensus event that contains a detected peak.
    peak_series (obs-free) locates peaks and the separating minimum.
    own_peaks ({formulation: peak positions} by each detector's own
    criterion), when given, removes the windows in which no formulation has
    a detected peak, separating the rest again until every window contains
    one (see the module docstring). removed_windows, when given, receives
    (consensus event id, start, end) for every removed window, as judged.
    """
    expanded = expand_all_events(consensus_events, own_windows)
    peaks = {ev.event_id: (ev.peak_index if ev.peak_index is not None else ev.start) for ev in consensus_events}
    separated = separate_overlapping_windows(expanded, peaks, peak_series)
    kept = expanded
    if own_peaks:
        while separated and (removed := windows_without_peak(separated, own_peaks)):
            if removed_windows is not None:
                removed_windows.extend((eid, s, e) for eid, s, e in separated if eid in removed)
            kept = [w for w in kept if w[0] not in removed]
            separated = separate_overlapping_windows(kept, peaks, peak_series)

    widened = {eid: (s, e) for eid, s, e in kept}
    events = []
    for i, (source_id, start, end) in enumerate(separated):
        peak_i = None
        if peak_series is not None and np.isfinite(peak_series[start:end + 1]).any():
            idx_range = np.arange(start, end + 1)
            peak_i = int(idx_range[np.nanargmax(peak_series[idx_range])])
        events.append(ExpandedEvent(
            event_id=i,
            start=start,
            end=end,
            duration_timesteps=end - start + 1,
            source_consensus_event_ids=[source_id],
            peak_index=peak_i,
            n_formulations_with_peak=(formulations_with_peak(start, end, own_peaks) if own_peaks else None),
            widened_start=widened[source_id][0],
            widened_end=widened[source_id][1],
        ))
    return events


def attach_timestamps(events: list[ExpandedEvent], index) -> list[ExpandedEvent]:
    """Copies of `events` with start/end/peak times taken from `index`."""
    out = []
    for ev in events:
        out.append(dataclasses.replace(
            ev,
            start_time=index[ev.start] if index is not None else None,
            end_time=index[ev.end] if index is not None else None,
            peak_time=(index[ev.peak_index] if (index is not None and ev.peak_index is not None) else None),
        ))
    return out


def own_window_to_expanded_events(series: np.ndarray, decluster_result: DeclusterResult) -> list[ExpandedEvent]:
    """One model's own events as ExpandedEvents (peak located on that model's own
    series), e.g. for per-model hydrograph plots.
    """
    series = np.asarray(series, dtype=np.float64)
    events = []
    for k in range(decluster_result.n_events):
        positions = np.where(decluster_result.event_id == k)[0]
        if positions.size == 0:
            continue
        start, end = int(positions.min()), int(positions.max())
        peak_i = start + int(np.argmax(series[start:end + 1]))
        events.append(ExpandedEvent(
            event_id=k,
            start=start,
            end=end,
            duration_timesteps=end - start + 1,
            source_consensus_event_ids=[],
            peak_index=peak_i,
        ))
    return events
