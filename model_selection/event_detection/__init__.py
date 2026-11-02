"""
Hydrologic event detection.

Per-series detection methods (pot, wasp) implement the common
EventDetector interface (base); detect.detect_events_for_series is the single
entry point. consensus and windows combine per-series events across model
formulations into the consensus events used by the high-flow pathway.

See docs/methods.md, Event detection methods.
"""
from model_selection.event_detection.base import DeclusterResult, DetectionResult, EventDetector
from model_selection.event_detection.config import EventConfig
from model_selection.event_detection.consensus import HighFlowEvent
from model_selection.event_detection.detect import (
    SeriesEventDetectionResult,
    detect_events_for_series,
    detect_events_per_series,
    make_detector,
)
from model_selection.event_detection.pot import POTConfig, POTDetector
from model_selection.event_detection.windows import ExpandedEvent
from model_selection.event_detection.wasp import WASPConfig, WASPDetector

__all__ = [
    "DeclusterResult", "DetectionResult", "EventDetector", "EventConfig", "HighFlowEvent",
    "SeriesEventDetectionResult", "detect_events_for_series", "detect_events_per_series", "make_detector",
    "POTConfig", "POTDetector", "ExpandedEvent", "WASPConfig", "WASPDetector",
]
