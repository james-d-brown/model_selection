"""
The single entry point for detecting events in one series.

Every method sees a uniformly spaced series: the series is regularized
(gaps interpolated), detected on, and the event labels mapped back to the
original positions. Interpolated values are discarded afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_selection.event_detection.base import DeclusterResult, EventDetector
from model_selection.event_detection.config import EventConfig
from model_selection.event_detection.pot.detector import POTDetector
from model_selection.event_detection.wasp.detector import WASPDetector
from model_selection.preprocessing.regularize import regularize_series


@dataclass
class SeriesEventDetectionResult:
    """One series' events and peaks (by the method's own criterion; None if
    the method reports none) on its original positions, plus method metadata
    (None for POT)."""
    decluster_result: DeclusterResult
    metadata: dict | None
    peaks: np.ndarray | None = None


def make_detector(config: EventConfig) -> EventDetector:
    """The detector selected by config.method, configured from config."""
    if config.method == "pot":
        return POTDetector(config.pot)
    if config.method == "wasp":
        return WASPDetector(config.wasp)
    raise ValueError(f"unknown event detection method {config.method!r}")


def detect_events_for_series(
        series: pd.Series,
        config: EventConfig | None = None,
        detector: EventDetector | None = None,
) -> SeriesEventDetectionResult:
    """Detect events in a time-indexed series (regularize -> detect -> map back).

    config supplies the method and the high-flow quantile; detector, if
    given, overrides the method (e.g. a custom EventDetector). When the series
    has missing values (NaN), they are filled for detection only, and the
    high-flow quantile is taken among the values present, so that filled
    values never set the threshold.
    """
    config = config or EventConfig()
    detector = detector or make_detector(config)
    r = regularize_series(series)
    timestep = r.regularized_index[1] - r.regularized_index[0] if len(r.regularized_index) > 1 else None
    present = series.to_numpy(dtype=float)
    present = present[np.isfinite(present)]
    if present.size == r.values.size:
        detection = detector.detect(r.values, timestep, config.high_flow_quantile)
    else:
        detection = detector.detect(r.values, timestep, config.high_flow_quantile, reference=present)
    regularized = detection.events
    original_event_id = r.event_id_to_original(regularized.event_id)
    event_lengths = np.array(
        [(original_event_id == k).sum() for k in range(regularized.n_events)], dtype=np.int64)
    return SeriesEventDetectionResult(
        decluster_result=DeclusterResult(
            event_id=original_event_id, n_events=regularized.n_events, event_lengths=event_lengths),
        metadata=detection.metadata,
        peaks=(None if detection.peaks is None else
               np.unique([r.position_to_original(int(p)) for p in detection.peaks]).astype(int)),
    )


def detect_events_per_series(
        series_by_name: dict[str, pd.Series],
        config: EventConfig | None = None,
        detector: EventDetector | None = None,
) -> dict[str, SeriesEventDetectionResult]:
    """detect_events_for_series for every series of a dict."""
    return {name: detect_events_for_series(s, config, detector) for name, s in series_by_name.items()}
