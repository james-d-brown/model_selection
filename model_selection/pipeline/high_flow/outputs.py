"""Writing the high-flow result JSON and hydrograph diagnostics."""
from __future__ import annotations

import copy
import logging
import os

import numpy as np

from model_selection.event_detection.base import DeclusterResult
from model_selection.event_detection.windows import attach_timestamps, own_window_to_expanded_events
from model_selection.io.basin import BasinData
from model_selection.io.results import write_json
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.results import (
    HIGH_FLOW_RESULT_JSON_FILENAME_TEMPLATE,
    HighFlowResult,
    high_flow_result_to_dict,
)

logger = logging.getLogger(__name__)

DIAGNOSTIC_PLOT_FILENAME_TEMPLATE = "{gage}_high_flow_diagnostic.html"
PER_MODEL_DIAGNOSTIC_PLOT_FILENAME_TEMPLATE = "{gage}_high_flow_diagnostic_{model}.html"


def write_high_flow_outputs(
        basin: BasinData,
        result: HighFlowResult,
        output: OutputConfig,
        own_windows: dict[str, DeclusterResult] | None = None,
        raw_formulations: dict[str, np.ndarray] | None = None,
        high_flow_quantile: float | None = None,
) -> tuple[str, str | None, dict[str, str]]:
    """Write {gage}_high_flow_result.json, plus (as configured) the basin
    hydrograph with the final events shaded and one hydrograph per model with
    that model's own events, with the median of the formulations' high-flow
    thresholds (high_flow_quantile of their values) as a dashed line. Returns (json path, diagnostic plot path or
    None, {model: per-model plot path})."""
    gage = result.gage
    output_dir = output.output_dir
    json_path = write_json(high_flow_result_to_dict(result), output_dir,
                           HIGH_FLOW_RESULT_JSON_FILENAME_TEMPLATE.format(gage=gage))

    plot_path = None
    per_model_paths: dict[str, str] = {}
    if output.diagnostic_plot or output.per_model_plots:
        from model_selection.visualization.html import write_standalone_html
        from model_selection.visualization.hydrograph import plot_basin_diagnostics

        if output.diagnostic_plot:
            plot_path = os.path.join(output_dir, DIAGNOSTIC_PLOT_FILENAME_TEMPLATE.format(gage=gage))
            write_standalone_html(plot_basin_diagnostics(basin, result.high_flow_events,
                                                         high_flow_quantile=high_flow_quantile), plot_path,
                                  f"{gage} high-flow diagnostic")

        if output.per_model_plots and own_windows and raw_formulations:
            for model_name, decluster_result in own_windows.items():
                model_events = own_window_to_expanded_events(raw_formulations[model_name], decluster_result)
                model_events = attach_timestamps(model_events, basin.obs.index)
                single_model_basin = copy.copy(basin)
                single_model_basin.formulations = {model_name: basin.formulations[model_name]}
                path = os.path.join(output_dir, PER_MODEL_DIAGNOSTIC_PLOT_FILENAME_TEMPLATE.format(
                    gage=gage, model=model_name))
                write_standalone_html(plot_basin_diagnostics(single_model_basin, model_events,
                                                             high_flow_quantile=high_flow_quantile), path,
                                      f"{gage} high-flow diagnostic ({model_name})")
                per_model_paths[model_name] = path
            logger.info(f"[+] Wrote {len(per_model_paths)} per-model high-flow diagnostics for {gage}.")

    logger.info(f"[+] Wrote high-flow outputs for {gage} to {output_dir}")
    return json_path, plot_path, per_model_paths
