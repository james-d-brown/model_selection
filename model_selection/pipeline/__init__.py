"""
The model-evaluation pipeline: a pooled pathway (one sample per timestep)
and a high-flow pathway (one sample per consensus event), orchestrated per
basin by run_basin.
"""
import logging

# Progress messages are logged at INFO; configure a default handler so they
# appear when the pipeline is run from a script (no-op if logging is
# already configured by the caller).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from model_selection.pipeline.config import OutputConfig  # noqa: E402
from model_selection.pipeline.high_flow import FPCAConfig, run_high_flow_analysis  # noqa: E402
from model_selection.pipeline.orchestration import run_basin  # noqa: E402
from model_selection.pipeline.pooled import run_pooled_analysis  # noqa: E402
from model_selection.preprocessing.seasonal import SeasonalConfig  # noqa: E402
from model_selection.pipeline.results import (  # noqa: E402
    BasinResult,
    HighFlowResult,
    PooledResult,
    basin_result_to_dict,
    high_flow_result_to_dict,
    outcome_line,
    pooled_result_to_dict,
)

__all__ = [
    "SeasonalConfig",
    "OutputConfig", "FPCAConfig", "run_high_flow_analysis", "run_basin", "run_pooled_analysis",
    "BasinResult", "HighFlowResult", "PooledResult",
    "basin_result_to_dict", "high_flow_result_to_dict", "pooled_result_to_dict", "outcome_line",
]
