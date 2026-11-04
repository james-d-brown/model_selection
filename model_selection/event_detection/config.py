"""Configuration of high-flow event detection and consensus formation."""
from __future__ import annotations

from dataclasses import dataclass, field

from model_selection.event_detection.pot.detector import POTConfig
from model_selection.event_detection.wasp.config import WASPConfig

DETECTION_METHODS = ("pot", "wasp")
DEPENDENCE_RULES = ("recession", "arrivals", "lag")
CONSENSUS_MODES = ("median", "mean", "any")


@dataclass(frozen=True)
class EventConfig:
    """How high-flow events are detected and combined across models.

    method : per-series detection method, "wasp" (default) or "pot"
        (docs/methods.md, Event detection methods).
    high_flow_quantile : flow percentile above which a peak (and a
        consensus timestep) counts as high flow.
    consensus : how per-model ranks combine into the regime mask: "median",
        "mean" or "any" (see event_detection.consensus).
    pot, wasp : method settings; only the one matching `method` is used.
    dependence : which events the tests treat as dependent
        (docs/methods.md, Dependence between samples):
        "recession" (default), events whose widened windows overlap, chained
        into clusters; "arrivals", events whose peaks are within the run
        length of Ferro and Segers' (2003) automatic declustering; "lag",
        events closer than the dependence lag.
    """
    method: str = "wasp"
    high_flow_quantile: float = 0.90
    consensus: str = "median"
    pot: POTConfig = field(default_factory=POTConfig)
    wasp: WASPConfig = field(default_factory=WASPConfig)
    dependence: str = "recession"

    def __post_init__(self):
        if self.method not in DETECTION_METHODS:
            raise ValueError(f"method must be one of {DETECTION_METHODS} -- got {self.method!r}")
        if self.dependence not in DEPENDENCE_RULES:
            raise ValueError(f"dependence must be one of {DEPENDENCE_RULES} -- got {self.dependence!r}")
        if self.consensus not in CONSENSUS_MODES:
            raise ValueError(f"consensus must be one of {CONSENSUS_MODES} -- got {self.consensus!r}")
