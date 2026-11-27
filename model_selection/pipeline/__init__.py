"""
The model-evaluation pipeline: the pooled pathway (one sample per timestep),
optionally censored at high flow, orchestrated per basin by run_basin, which
can also run the experimental high-flow event pathway
(model_selection.experimental).
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
    "OutputConfig", "run_basin", "run_pooled_analysis",
    "BasinResult", "HighFlowResult", "PooledResult",
    "basin_result_to_dict", "high_flow_result_to_dict", "pooled_result_to_dict", "outcome_line",
]
