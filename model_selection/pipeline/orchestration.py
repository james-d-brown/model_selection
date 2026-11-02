"""Running both pathways for one basin."""
from __future__ import annotations

import numpy as np

from model_selection.event_detection.config import EventConfig
from model_selection.io.basin import BasinData
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.high_flow.config import FPCAConfig
from model_selection.pipeline.high_flow.pathway import run_high_flow_analysis
from model_selection.pipeline.pooled.pathway import run_pooled_analysis
from model_selection.pipeline.results import BasinResult
from model_selection.preprocessing.seasonal import SeasonalConfig
from model_selection.selection.config import AdequacyConfig, SelectionConfig


def run_basin(
        basin: BasinData,
        selection: SelectionConfig | None = None,
        adequacy: AdequacyConfig | None = None,
        events: EventConfig | None = None,
        fpca: FPCAConfig | None = None,
        output: OutputConfig | None = None,
        run_high_flow: bool = True,
        high_flow_selection: SelectionConfig | None = None,
        rng: np.random.Generator | None = None,
        seasonal: SeasonalConfig | None = None,
) -> BasinResult:
    """Run the pooled pathway and (unless run_high_flow is False) the
    high-flow pathway on one basin. A generator passed as rng is shared by
    both; otherwise each pathway seeds its own from SelectionConfig.seed,
    or from a fresh seed it logs and records.
    high_flow_selection overrides `selection` for the high-flow pathway;
    seasonal configures conditioning on the time of year in both pathways."""
    pooled = run_pooled_analysis(basin, selection, adequacy, output, rng=rng, seasonal=seasonal)
    high_flow = None
    if run_high_flow:
        high_flow = run_high_flow_analysis(
            basin, high_flow_selection or selection, adequacy, events, fpca, output, rng=rng, seasonal=seasonal)
    return BasinResult(gage=basin.gage, pooled=pooled, high_flow=high_flow)
