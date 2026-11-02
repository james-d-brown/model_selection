"""WASP (wavelet-assessed scale-space persistence) event detection."""
from model_selection.event_detection.wasp.config import WASPConfig
from model_selection.event_detection.wasp.detector import WASPDetectionResult, WASPDetector, WASPEventDetector

__all__ = ["WASPConfig", "WASPDetectionResult", "WASPDetector", "WASPEventDetector"]
