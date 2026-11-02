"""Output settings shared by both pathways."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputConfig:
    """What each pathway writes. Nothing is written when output_dir is None.

    assumption_plots : one diagnostic page per assumption of the tier in use,
        per tested round (docs/methods.md, Diagnostic plots).
    diagnostic_plot : high-flow hydrograph with the consensus events shaded.
    per_model_plots : high-flow hydrograph per model with its own events.
    event_gallery : high-flow FPCA event gallery (large for many events).
    dependence_plot : correlograms of the series that set the dependence lag
        ({gage}_dependence.html), written by either pathway.
    """
    output_dir: str | None = None
    assumption_plots: bool = True
    diagnostic_plot: bool = False
    per_model_plots: bool = False
    event_gallery: bool = False
    dependence_plot: bool = True
