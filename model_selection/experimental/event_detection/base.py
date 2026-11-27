"""
Common types and interface of the event detectors.

Every method detects events in one uniformly spaced series and returns a
DeclusterResult: an event_id per sample (-1 outside events) with each
event's span, and the positions of the series' peaks by the method's own
criterion. Method-specific diagnostics travel in DetectionResult.metadata.
Downstream code (consensus formation, windows, FPCA) depends only on these
types, never on how a method found its events.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DeclusterResult:
    """Event labelling of one series: event_id is -1 outside events and
    0..n_events-1 inside; event_lengths counts the samples of each event."""
    event_id: np.ndarray
    n_events: int
    event_lengths: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "DeclusterResult":
        return cls(event_id=np.full(n, -1, dtype=np.int64), n_events=0,
                   event_lengths=np.array([], dtype=np.int64))


@dataclass
class DetectionResult:
    """Output of an EventDetector: the events, the positions of the series'
    peaks by the method's own criterion (None if the method reports none),
    and optional method-specific metadata (None for methods that report
    none)."""
    events: DeclusterResult
    metadata: dict | None = None
    peaks: np.ndarray | None = None


class EventDetector(ABC):
    """Detects high-flow events in one series.

    detect() receives the series as a uniformly spaced array (gaps already
    filled, see preprocessing.regularize), its timestep, and the flow
    quantile above which a peak counts as high flow. When values were filled,
    `reference` holds the series' values on the analysis basis, among which
    the quantile is taken. It must use only the series itself (never
    observations of another series).
    """
    name: str = ""

    @abstractmethod
    def detect(self, values: np.ndarray, timestep: pd.Timedelta, quantile_threshold: float,
               reference: np.ndarray | None = None) -> DetectionResult:
        ...
