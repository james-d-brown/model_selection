"""
Consensus high-flow events across model formulations (never using obs).

Events are built from the formulations' simulations, never from the
observations: evaluating predictions only where an extreme was observed
rewards predictions that place extremes too often (the "forecaster's
dilemma" of Lerch et al., 2017, which applies to simulations as to
forecasts), and simulation-defined events keep each formulation's false
alarms in the sample (docs/methods.md, Two pathways):

  1. rank_consensus_regime: each model's flow becomes its own percentile
     rank; ranks are combined per timestep ("median", "mean" or "any" = max)
     and thresholded. Ranks keep large models from dominating and keep a
     peak caught sharply by one model from being averaged away. "median"
     is the upper median, so a timestep is high flow when at least half of
     the models are at or above their own threshold (two of four, two of
     three).
  2. group_runs_by_own_events: contiguous runs of high-flow timesteps are
     joined into one event when at least half of the models have a single
     own event (from event_detection.detect) spanning both runs. The same
     principle as the median in step 1; no separation length is
     involved.

Each event is then widened to the models' own events and overlaps are
separated at the minimum between peaks (event_detection.windows).

See docs/methods.md, Samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from model_selection.event_detection.base import DeclusterResult


def consensus_rank(model_series: dict[str, np.ndarray], consensus: str = "median") -> np.ndarray:
    """Per-timestep combination ("median" = upper median, "mean" or "any" = max) of each
    model's percentile rank of its own flow. Ranks are taken over the
    timesteps at which every model has a value, so that every model's ranks
    refer to the same timesteps; other timesteps are NaN. All series must
    already be aligned and of equal length.
    """
    if not model_series:
        raise ValueError("model_series is empty")
    lengths = {len(v) for v in model_series.values()}
    if len(lengths) != 1:
        raise ValueError("all model series must be the same length (already aligned)")
    n = lengths.pop()
    values = [np.asarray(v, dtype=float) for v in model_series.values()]
    common = np.logical_and.reduce([np.isfinite(v) for v in values])
    m = int(common.sum())

    pct_ranks = np.full((len(values), n), np.nan)
    for i, v in enumerate(values):
        pct_ranks[i, common] = rankdata(v[common], method="average") / max(m, 1)

    if consensus == "median":
        # Upper median: >= q exactly when at least half of the models are at or above q.
        return np.quantile(pct_ranks, 0.5, axis=0, method="higher")
    if consensus == "mean":
        return np.mean(pct_ranks, axis=0)
    if consensus == "any":
        return np.max(pct_ranks, axis=0)
    raise ValueError("consensus must be 'median', 'mean', or 'any'")


def rank_consensus_regime(
        model_series: dict[str, np.ndarray],
        quantile_threshold: float = 0.90,
        consensus: str = "median",
) -> np.ndarray:
    """Boolean high-flow mask: consensus_rank compared with
    quantile_threshold; False where any model lacks a value."""
    rank = consensus_rank(model_series, consensus)
    return np.where(np.isfinite(rank), rank, -np.inf) >= quantile_threshold


def raw_events_from_mask(high_flow_mask: np.ndarray) -> DeclusterResult:
    """Group mask into events by pure contiguity, with no gap merging."""
    mask = np.asarray(high_flow_mask, dtype=bool)
    n = mask.size
    event_id = np.full(n, -1, dtype=np.int64)
    if not mask.any():
        return DeclusterResult.empty(n)

    idx = np.where(mask)[0]
    gaps = np.diff(idx)
    new_event_starts = np.concatenate([[True], gaps > 1])
    event_numbers = np.cumsum(new_event_starts) - 1
    event_id[idx] = event_numbers

    n_events = int(event_numbers.max() + 1)
    event_lengths = np.array([(event_id == e).sum() for e in range(n_events)])
    return DeclusterResult(event_id=event_id, n_events=n_events, event_lengths=event_lengths)


def group_runs_by_own_events(high_flow_mask: np.ndarray, own_windows: dict[str, DeclusterResult]) -> DeclusterResult:
    """Join adjacent contiguous high-flow runs into one event when at least
    half of the models have a single own event spanning both runs (the same
    own event id present in each). Runs are joined transitively, left to
    right. Returns a DeclusterResult whose event_id is set at high-flow
    timesteps only.
    """
    mask = np.asarray(high_flow_mask, dtype=bool)
    n = mask.size
    runs = raw_events_from_mask(mask)
    if runs.n_events == 0:
        return DeclusterResult.empty(n)
    run_positions = [np.flatnonzero(runs.event_id == r) for r in range(runs.n_events)]
    n_models = len(own_windows)

    def joined(left: np.ndarray, right: np.ndarray) -> bool:
        if n_models == 0:
            return False
        n_spanning = 0
        for dr in own_windows.values():
            ids_left = set(dr.event_id[left].tolist()) - {-1}
            ids_right = set(dr.event_id[right].tolist()) - {-1}
            n_spanning += bool(ids_left & ids_right)
        return n_spanning >= n_models / 2

    new_event_id = np.full(n, -1, dtype=np.int64)
    current = 0
    new_event_id[run_positions[0]] = current
    for r in range(1, runs.n_events):
        if not joined(run_positions[r - 1], run_positions[r]):
            current += 1
        new_event_id[run_positions[r]] = current
    n_events = current + 1
    event_lengths = np.array([(new_event_id == k).sum() for k in range(n_events)], dtype=np.int64)
    return DeclusterResult(event_id=new_event_id, n_events=n_events, event_lengths=event_lengths)


def independent_event_summary(decluster_result: DeclusterResult) -> dict:
    """Number of independent events and their median length."""
    return {
        "n_events": decluster_result.n_events,
        "median_event_length": float(np.median(decluster_result.event_lengths)) if decluster_result.n_events else 0.0,
    }


@dataclass
class HighFlowEvent:
    """One consensus event. start/end are inclusive positions of the event
    envelope; when runs were joined, timesteps inside it need not all be high
    flow, so duration_timesteps (span) can exceed n_high_flow_timesteps.
    peak_index is located on an obs-free proxy; times come from the series
    index.
    """
    event_id: int
    start: int                  # index of first high-flow timestep in this event
    end: int                    # index of last high-flow timestep in this event (inclusive)
    duration_timesteps: int     # end - start + 1
    n_high_flow_timesteps: int  # high-flow timesteps within [start, end]
    peak_index: int | None = None
    start_time: pd.Timestamp | None = None  # tz-aware (UTC), from the basin's obs DatetimeIndex
    end_time: pd.Timestamp | None = None
    peak_time: pd.Timestamp | None = None


def summarize_events(
        decluster_result: DeclusterResult,
        peak_series: np.ndarray | None = None,
        index: pd.DatetimeIndex | None = None,
) -> list[HighFlowEvent]:
    """One HighFlowEvent per event of decluster_result. peak_series (an obs-free
    proxy) locates each event's peak; index, when given, supplies timestamps.
    """
    events = []
    for e in range(decluster_result.n_events):
        idx = np.where(decluster_result.event_id == e)[0]
        start_i, end_i = int(idx.min()), int(idx.max())
        peak_i = int(idx[np.argmax(peak_series[idx])]) if peak_series is not None else None
        events.append(HighFlowEvent(
            event_id=e,
            start=start_i,
            end=end_i,
            duration_timesteps=end_i - start_i + 1,
            n_high_flow_timesteps=int(idx.size),
            peak_index=peak_i,
            start_time=index[start_i] if index is not None else None,
            end_time=index[end_i] if index is not None else None,
            peak_time=(index[peak_i] if (index is not None and peak_i is not None) else None),
        ))
    return events
