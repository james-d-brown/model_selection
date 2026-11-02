"""
How consensus high-flow events are built for one basin: a stage-by-stage
account of the high-flow pathway's event construction, without running
selection.

Stages (docs/methods.md, Samples):
  raw runs        contiguous runs of consensus high-flow timesteps
  events          adjacent runs joined when at least half of the formulations have a
                  single own event spanning both
  windows         each event widened to the formulations' own events;
                  overlapping windows separated at the minimum between peaks;
                  windows containing no peak detected by any formulation's
                  own detector are not events
  event samples   windows of at least FPCAConfig.min_event_length timesteps

For every window, the report gives how many raw runs it contains, which
formulations' own events spanned each join, whether it was separated
from its predecessor at an overlap, how many formulations have an own
event overlapping its consensus high-flow period, and how many have a
detected peak in it. Removed windows are listed, as the peak rule judged
them, with no window number and event_sample False.

Usage (for example at the end of evaluate.py, after `basin` is loaded):

    from scripts.diagnostics.event_merging import event_merging_report
    summary, windows = event_merging_report(basin, EventConfig(method="wasp", high_flow_quantile=0.9),
                                            output_dir=output_dir)
    print(summary)

Writes {gage}_event_merging_windows.csv and {gage}_event_merging_summary.json
to output_dir when given.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from model_selection.event_detection.config import EventConfig
from model_selection.event_detection.consensus import (
    consensus_rank, group_runs_by_own_events, rank_consensus_regime, raw_events_from_mask, summarize_events,
)
from model_selection.event_detection.detect import detect_events_for_series
from model_selection.event_detection.windows import build_expanded_events, expand_all_events
from model_selection.pipeline.high_flow.config import FPCAConfig


def _spanning(own_windows: dict, left: np.ndarray, right: np.ndarray) -> list[str]:
    """Formulations with a single own event present in both position sets."""
    names = []
    for name, dr in own_windows.items():
        if (set(dr.event_id[left].tolist()) - {-1}) & (set(dr.event_id[right].tolist()) - {-1}):
            names.append(name)
    return names


def event_merging_report(basin, events: EventConfig | None = None, output_dir: str | None = None,
                         fpca: FPCAConfig | None = None) -> tuple[dict, pd.DataFrame]:
    """(summary, per-window table) of the event construction stages."""
    events = events or EventConfig()
    fpca = fpca or FPCAConfig()
    raw = {name: s.values for name, s in basin.formulations.items()}
    index = basin.obs.index

    mask = rank_consensus_regime(raw, quantile_threshold=events.high_flow_quantile, consensus=events.consensus)
    detected = {name: detect_events_for_series(s, events) for name, s in basin.formulations.items()}
    own_windows = {name: d.decluster_result for name, d in detected.items()}
    own_peaks = ({name: d.peaks for name, d in detected.items()}
                 if all(d.peaks is not None for d in detected.values()) else None)
    proxy = consensus_rank(raw, "median")  # as in the high-flow pathway

    runs = raw_events_from_mask(mask)
    grouped = group_runs_by_own_events(mask, own_windows)
    consensus = summarize_events(grouped, peak_series=proxy)
    widened = {eid: (s, e) for eid, s, e in expand_all_events(consensus, own_windows)}
    removed = []
    windows = build_expanded_events(consensus, own_windows, peak_series=proxy, own_peaks=own_peaks,
                                    removed_windows=removed)

    def n_own_events(positions):
        return sum(bool(np.any(dr.event_id[positions] >= 0)) for dr in own_windows.values())

    rows = []
    prev_widened_end = None
    for w in windows:
        (source,) = w.source_consensus_event_ids
        positions = np.flatnonzero(grouped.event_id == source)
        run_ids = sorted(set(runs.event_id[positions].tolist()))
        joins = []
        for left, right in zip(run_ids[:-1], run_ids[1:]):
            names = _spanning(own_windows, np.flatnonzero(runs.event_id == left), np.flatnonzero(runs.event_id == right))
            joins.append(";".join(names))
        w_start, w_end = widened[source]
        separated = prev_widened_end is not None and w_start <= prev_widened_end
        prev_widened_end = w_end if prev_widened_end is None else max(prev_widened_end, w_end)
        rows.append({
            "window": w.event_id,
            "start": index[w.start], "end": index[w.end], "peak": index[w.peak_index],
            "duration_timesteps": w.duration_timesteps,
            "raw_runs": len(run_ids),
            "formulations_spanning_each_join": " | ".join(joins),
            "separated_from_previous_at_overlap": bool(separated),
            "formulations_with_own_event": n_own_events(positions),
            "formulations_with_peak": w.n_formulations_with_peak,
            "event_sample": w.duration_timesteps >= fpca.min_event_length,
        })
    peak_of = {ev.event_id: ev.peak_index for ev in consensus}
    for source, start, end in removed:
        positions = np.flatnonzero(grouped.event_id == source)
        rows.append({
            "window": None, "start": index[start], "end": index[end], "peak": index[peak_of[source]],
            "duration_timesteps": end - start + 1,
            "raw_runs": len(set(runs.event_id[positions].tolist())),
            "formulations_spanning_each_join": "", "separated_from_previous_at_overlap": None,
            "formulations_with_own_event": n_own_events(positions), "formulations_with_peak": 0,
            "event_sample": False,
        })
    table = pd.DataFrame(rows).sort_values("start", kind="stable").reset_index(drop=True)

    summary = {
        "gage": basin.gage,
        "raw_runs": int(runs.n_events),
        "events_after_joining": int(grouped.n_events),
        "joins": int(runs.n_events - grouped.n_events),
        "consensus_events_without_peak": int(len(consensus) - len(windows)),
        "formulations": len(own_windows),
        "own_events_per_formulation": {name: int(dr.n_events) for name, dr in own_windows.items()},
        "peaks_per_formulation": (None if own_peaks is None else
                                  {name: int(len(p)) for name, p in own_peaks.items()}),
        "windows_separated_at_overlaps": int(table["separated_from_previous_at_overlap"].fillna(False).astype(bool).sum())
        if len(table) else 0,
        "event_samples": int(table["event_sample"].sum()) if len(table) else 0,
        "event_samples_with_several_raw_runs": (
            int(((table["raw_runs"] > 1) & table["event_sample"]).sum()) if len(table) else 0),
    }
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        table.to_csv(os.path.join(output_dir, f"{basin.gage}_event_merging_windows.csv"), index=False)
        with open(os.path.join(output_dir, f"{basin.gage}_event_merging_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary, table
