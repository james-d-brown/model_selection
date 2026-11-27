"""
model_selection: hydrologic model selection by conditional association.

Model formulations are selected greedily by their association with
observations conditional on the models already chosen, with significance
testing, and optionally a cascade from the most powerful association
measure whose assumptions hold (partial correlation) to the most general
(conditional mutual information). Selection runs on the whole record
(pipeline.pooled), optionally censored at high flow. An experimental
pathway on high-flow events is in experimental.

Packages:
  io               reading model output and observations, writing results
  preprocessing    transforms and regularization of series
  statistics       association measures, significance tests, adequacy tests
  selection        greedy selection, the measure cascade, configuration
  pipeline         the pooled pathway and its orchestration
  visualization    plots and HTML output
  experimental     the high-flow event pathway, event detection and FPCA
"""
from model_selection.io import BasinData, load_basin
from model_selection.pipeline import OutputConfig, run_basin, run_pooled_analysis
from model_selection.preprocessing.censoring import Censoring
from model_selection.selection import AdequacyConfig, SelectionConfig

__all__ = [
    "BasinData", "load_basin",
    "OutputConfig", "run_basin", "run_pooled_analysis", "Censoring",
    "AdequacyConfig", "SelectionConfig",
]
