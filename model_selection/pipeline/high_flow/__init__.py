"""The high-flow pathway (one sample per consensus event)."""
from model_selection.pipeline.high_flow.config import FPCAConfig
from model_selection.pipeline.high_flow.pathway import run_high_flow_analysis

__all__ = ["FPCAConfig", "run_high_flow_analysis"]
