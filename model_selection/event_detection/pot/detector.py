"""
Peaks-over-threshold (POT) event detection on a single series, in four
independently testable stages:

  1. find_candidate_peaks       every local maximum (optionally at least
                                min_peak_separation apart, a noise floor
                                unrelated to basin timescales)
  2. qualify_peaks              keep peaks whose own flow percentile is at
                                least quantile_threshold
  3. group_peaks_by_independence  merge peaks that are not independent
                                (POT ratio criterion; Cunnane 1979; Lang,
                                Ouarda & Bobee 1999)
  4. resolve_event_boundaries   each event spans from where flow falls to
                                recession_ratio x its tallest peak on either
                                side; overlapping neighbours share the
                                minimum between their peaks

Pure minimum-seeking (Sloto & Crouse 1996) was not used for boundaries
because an exponential recession never reaches a minimum. Known
limitation: recession_ratio is relative to the event's own peak, so for a
very tall peak the boundary can pass a shallow local trough.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import rankdata

from model_selection.event_detection.base import DeclusterResult, DetectionResult, EventDetector

logger = logging.getLogger(__name__)

IndependenceTestFunc = Callable[[np.ndarray, int, int], bool]
"""independent(series, tallest_peak_idx, candidate_idx) -> bool: is the
candidate a separate event from the group whose tallest peak is given?"""


@dataclass(frozen=True)
class POTConfig:
    """POT settings.

    pot_ratio : two peaks are independent if flow between them recedes to at
        most this fraction of the smaller peak (2/3 is the usual value).
    recession_ratio : an event extends until flow falls to this fraction of
        its tallest peak; lower gives longer tails.
    min_peak_separation : minimum spacing of candidate peaks in timesteps
        (1 = every local maximum).
    """
    pot_ratio: float = 2.0 / 3.0
    recession_ratio: float = 0.25
    min_peak_separation: int = 1


def make_pot_ratio_independence_test(pot_ratio: float = 2.0 / 3.0) -> IndependenceTestFunc:
    """POT ratio independence test: the candidate is independent if the minimum
    flow between it and the tallest peak is at most pot_ratio times the
    smaller of the two peaks.
    """
    def _test(series: np.ndarray, tallest_peak: int, candidate: int) -> bool:
        trough = float(series[tallest_peak:candidate + 1].min())
        smaller_peak = float(min(series[tallest_peak], series[candidate]))
        return smaller_peak > 0 and (trough <= pot_ratio * smaller_peak)
    return _test


def find_candidate_peaks(series: np.ndarray, min_peak_separation: int = 1) -> np.ndarray:
    """Stage 1: indices of local maxima at least min_peak_separation apart."""
    peak_idx, _ = find_peaks(series, distance=max(1, int(min_peak_separation)))
    return peak_idx


def qualify_peaks(series: np.ndarray, peak_idx: np.ndarray, quantile_threshold: float,
                  reference: np.ndarray | None = None) -> np.ndarray:
    """Stage 2: the peaks whose percentile rank is at least
    quantile_threshold. Ranks are among the values of `reference` (default:
    the whole series), e.g. the series' values on the analysis basis when
    the series passed in has missing values filled for detection.
    """
    if reference is None:
        ranks = rankdata(series, method="average") / series.size
        return peak_idx[ranks[peak_idx] >= quantile_threshold]
    ref = np.sort(np.asarray(reference, dtype=float))
    v = series[peak_idx]
    ranks = (np.searchsorted(ref, v, "left") + np.searchsorted(ref, v, "right") + 1) / 2.0 / ref.size
    return peak_idx[ranks >= quantile_threshold]


def group_peaks_by_independence(
        series: np.ndarray,
        qualifying_idx: np.ndarray,
        independence_test: IndependenceTestFunc,
        log_progress_if_many: bool = False,
) -> tuple[list[list[int]], list[int]]:
    """Stage 3: group consecutive qualifying peaks that are not independent.

    Each candidate is tested against its group's current tallest member, not
    the most recently added one, so a short peak cannot make a later tall peak
    look dependent. log_progress_if_many logs progress for long lists, for use
    with slow custom tests. Returns (groups, tallest_idx_in_group).
    """
    n_transitivity_calls = len(qualifying_idx) - 1
    log_progress = log_progress_if_many and n_transitivity_calls > 20
    if log_progress:
        logger.info(f"[*] group_peaks_by_independence: transitivity-testing {n_transitivity_calls} "
                     f"candidate peak pairs with a custom independence_test...")
    t_start = time.time()
    groups: list[list[int]] = [[int(qualifying_idx[0])]]
    tallest_idx_in_group: list[int] = [int(qualifying_idx[0])]
    for call_i, p in enumerate(qualifying_idx[1:]):
        p = int(p)
        tallest_peak = tallest_idx_in_group[-1]
        independent = independence_test(series, tallest_peak, p)
        if independent:
            groups.append([p])
            tallest_idx_in_group.append(p)
        else:
            groups[-1].append(p)
            if series[p] > series[tallest_peak]:
                tallest_idx_in_group[-1] = p
        if log_progress and (call_i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (call_i + 1) / elapsed if elapsed > 0 else float("inf")
            eta = (n_transitivity_calls - call_i - 1) / rate if rate > 0 else float("nan")
            logger.info(f"[*] group_peaks_by_independence: {call_i + 1}/{n_transitivity_calls} pairs "
                         f"tested ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining).")
    if log_progress:
        logger.info(f"[+] group_peaks_by_independence: done in {time.time() - t_start:.1f}s, "
                     f"{len(groups)} groups from {len(qualifying_idx)} qualifying peaks.")
    return groups, tallest_idx_in_group


def resolve_event_boundaries(
        series: np.ndarray,
        groups: list[list[int]],
        tallest_idx_in_group: list[int],
        recession_ratio: float,
) -> DeclusterResult:
    """Stage 4: event spans from the grouped peaks.

    Pass 1: each group extends on both sides of its tallest peak until flow
    first falls to recession_ratio times that peak, widened to cover all its
    members. Pass 2: where adjacent spans overlap, the minimum flow between the
    two tallest peaks becomes the shared boundary (assigned to the earlier
    event). event_id covers each full span, including sub-threshold edges.
    """
    n = series.size

    def _natural_left(tallest_idx: int, peak_value: float) -> int:
        window = series[0:tallest_idx + 1]
        below = np.where(window <= recession_ratio * peak_value)[0]
        return int(below[-1]) if below.size > 0 else 0

    def _natural_right(tallest_idx: int, peak_value: float) -> int:
        window = series[tallest_idx:n]
        below = np.where(window <= recession_ratio * peak_value)[0]
        return tallest_idx + int(below[0]) if below.size > 0 else n - 1

    natural_left = [min(_natural_left(tallest_idx_in_group[k], series[tallest_idx_in_group[k]]), groups[k][0])
                     for k in range(len(groups))]
    natural_right = [max(_natural_right(tallest_idx_in_group[k], series[tallest_idx_in_group[k]]), groups[k][-1])
                       for k in range(len(groups))]

    final_left = list(natural_left)
    final_right = list(natural_right)
    for k in range(len(groups) - 1):
        if final_right[k] >= final_left[k + 1]:
            trough_lo, trough_hi = tallest_idx_in_group[k], tallest_idx_in_group[k + 1]
            trough = trough_lo + int(np.argmin(series[trough_lo:trough_hi + 1]))
            final_right[k] = trough
            final_left[k + 1] = trough + 1

    event_id = np.full(n, -1, dtype=np.int64)
    for k in range(len(groups)):
        event_id[final_left[k]:final_right[k] + 1] = k

    event_lengths = np.array([(event_id == k).sum() for k in range(len(groups))], dtype=np.int64)
    return DeclusterResult(event_id=event_id, n_events=len(groups), event_lengths=event_lengths)


def detect_model_events(
        series: np.ndarray,
        quantile_threshold: float,
        pot_ratio: float = 2.0 / 3.0,
        recession_ratio: float = 0.25,
        independence_test: IndependenceTestFunc | None = None,
        min_peak_separation: int = 1,
        return_peak_groups: bool = False,
        reference: np.ndarray | None = None,
) -> DeclusterResult | tuple[DeclusterResult, dict[int, list[int]]]:
    """POT event detection on one series (stages 1-4).

    independence_test replaces the POT ratio test if given. return_peak_groups
    also returns {event_id: qualifying peak indices merged into that event}.
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.size

    if independence_test is None:
        independence_test = make_pot_ratio_independence_test(pot_ratio)
        _using_custom_independence_test = False
    else:
        _using_custom_independence_test = True

    peak_idx = find_candidate_peaks(series, min_peak_separation)
    if peak_idx.size == 0:
        empty = DeclusterResult.empty(n)
        return (empty, {}) if return_peak_groups else empty

    qualifying_idx = qualify_peaks(series, peak_idx, quantile_threshold, reference)
    if qualifying_idx.size == 0:
        empty = DeclusterResult.empty(n)
        return (empty, {}) if return_peak_groups else empty

    groups, tallest_idx_in_group = group_peaks_by_independence(
        series, qualifying_idx, independence_test, log_progress_if_many=_using_custom_independence_test,
    )

    result = resolve_event_boundaries(series, groups, tallest_idx_in_group, recession_ratio)

    if return_peak_groups:
        peak_groups = {k: groups[k] for k in range(result.n_events)}
        return result, peak_groups
    return result


class POTDetector(EventDetector):
    """EventDetector for the peaks-over-threshold method."""
    name = "pot"

    def __init__(self, config: POTConfig | None = None, independence_test: IndependenceTestFunc | None = None):
        self.config = config or POTConfig()
        self.independence_test = independence_test

    def detect(self, values: np.ndarray, timestep: pd.Timedelta, quantile_threshold: float,
               reference: np.ndarray | None = None) -> DetectionResult:
        events = detect_model_events(
            values, quantile_threshold=quantile_threshold,
            pot_ratio=self.config.pot_ratio, recession_ratio=self.config.recession_ratio,
            independence_test=self.independence_test,
            min_peak_separation=self.config.min_peak_separation, reference=reference,
        )
        values = np.asarray(values, dtype=np.float64)
        peaks = np.array([int(np.flatnonzero(events.event_id == k)[np.argmax(values[events.event_id == k])])
                          for k in range(events.n_events)], dtype=int)
        return DetectionResult(events=events, metadata=None, peaks=peaks)
