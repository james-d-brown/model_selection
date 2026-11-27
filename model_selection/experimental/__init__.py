"""
Experimental features: not part of the documented method, and not yet
supported by the checks the main pathway has passed (docs/methods.md,
Appendix E: Experimental high-flow event pathway).

  high_flow        the high-flow event pathway (one sample per event)
  event_detection  per-series event detection (POT, WASP) and consensus events
  fpca             functional principal component analysis of event hydrographs
  hydrology        e-folding timescale (WASP scale ceiling)
  visualization    hydrographs with events, and event galleries

Running the high-flow event pathway issues an ExperimentalWarning once per
process.
"""
import warnings


class ExperimentalWarning(UserWarning):
    """A feature of model_selection.experimental was used."""


_WARNED: set[str] = set()


def warn_experimental(feature: str) -> None:
    """Warn, once per process and feature, that `feature` is experimental."""
    if feature in _WARNED:
        return
    _WARNED.add(feature)
    warnings.warn(
        f"The {feature} is experimental: its outstanding problems are listed in docs/methods.md, "
        "Appendix E: Experimental high-flow event pathway.", ExperimentalWarning, stacklevel=3)


from model_selection.experimental.event_detection import EventConfig, POTConfig, WASPConfig  # noqa: E402
from model_selection.experimental.high_flow import FPCAConfig, run_high_flow_analysis  # noqa: E402

__all__ = [
    "ExperimentalWarning", "warn_experimental",
    "EventConfig", "POTConfig", "WASPConfig", "FPCAConfig", "run_high_flow_analysis",
]
