"""Peaks-over-threshold event detection."""
from model_selection.event_detection.pot.detector import (
    IndependenceTestFunc,
    POTConfig,
    POTDetector,
    detect_model_events,
    make_pot_ratio_independence_test,
)

__all__ = ["IndependenceTestFunc", "POTConfig", "POTDetector", "detect_model_events",
           "make_pot_ratio_independence_test"]
