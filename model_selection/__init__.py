"""
model_selection: hydrologic model selection by conditional association.

Model formulations are selected greedily by their association with
observations conditional on the models already chosen, with significance
testing, and optionally a cascade from the most powerful association
measure whose assumptions hold (partial correlation) to the most general
(conditional mutual information). Selection runs on the whole record
(pipeline.pooled) and on high-flow events (pipeline.high_flow).

Packages:
  io               reading model output and observations, writing results
  preprocessing    transforms and regularization of series
  hydrology        e-folding timescale (WASP scale ceiling)
  statistics       association measures, significance tests, adequacy tests, FPCA
  selection        greedy selection, the measure cascade, configuration
  event_detection  per-series event detection (POT, WASP) and consensus events
  pipeline         the pooled and high-flow pathways and their orchestration
  visualization    plots and HTML output
"""
from model_selection.event_detection import EventConfig, POTConfig, WASPConfig
from model_selection.io import BasinData, load_basin
from model_selection.pipeline import (
    FPCAConfig,
    OutputConfig,
    run_basin,
    run_high_flow_analysis,
    run_pooled_analysis,
)
from model_selection.selection import AdequacyConfig, SelectionConfig

__all__ = [
    "BasinData", "load_basin",
    "FPCAConfig", "OutputConfig", "run_basin", "run_high_flow_analysis", "run_pooled_analysis",
    "EventConfig", "POTConfig", "WASPConfig", "AdequacyConfig", "SelectionConfig",
]
