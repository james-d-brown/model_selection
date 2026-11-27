"""Running the pathways for one basin."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from model_selection.io.basin import BasinData
from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.pooled.pathway import run_pooled_analysis
from model_selection.pipeline.results import BasinResult
from model_selection.preprocessing.censoring import Censoring
from model_selection.preprocessing.seasonal import SeasonalConfig
from model_selection.selection.config import AdequacyConfig, SelectionConfig

if TYPE_CHECKING:
    from model_selection.experimental.event_detection.config import EventConfig
    from model_selection.experimental.high_flow.config import FPCAConfig


def run_basin(
        basin: BasinData,
        selection: SelectionConfig | None = None,
        adequacy: AdequacyConfig | None = None,
        events: EventConfig | None = None,
        fpca: FPCAConfig | None = None,
        output: OutputConfig | None = None,
        run_high_flow: bool = False,
        high_flow_selection: SelectionConfig | None = None,
        rng: np.random.Generator | None = None,
        seasonal: SeasonalConfig | None = None,
        censoring: Censoring | None = None,
) -> BasinResult:
    """Run the pooled pathway on one basin, with censoring the pooled pathway
    censored at high flow too, and with run_high_flow the experimental
    high-flow event pathway (model_selection.experimental; docs/methods.md,
    Appendix E: Experimental high-flow event pathway). A generator passed as rng is shared by all of them;
    otherwise each pathway seeds its own from SelectionConfig.seed, or from a
    fresh seed it logs and records.
    events, fpca and high_flow_selection (which overrides `selection`)
    configure the experimental pathway and are used only with run_high_flow;
    seasonal configures conditioning on the time of year in the pathways;
    censoring (e.g. Censoring()) runs the censored pooled pathway
    (docs/methods.md, Censoring at high flow)."""
    pooled = run_pooled_analysis(basin, selection, adequacy, output, rng=rng, seasonal=seasonal)
    pooled_censored = None
    if censoring is not None:
        pooled_censored = run_pooled_analysis(basin, selection, adequacy, output, rng=rng, seasonal=seasonal,
                                              censoring=censoring)
    high_flow = None
    if run_high_flow:
        from model_selection.experimental.high_flow.pathway import run_high_flow_analysis
        high_flow = run_high_flow_analysis(
            basin, high_flow_selection or selection, adequacy, events, fpca, output, rng=rng, seasonal=seasonal)
    return BasinResult(gage=basin.gage, pooled=pooled, high_flow=high_flow, pooled_censored=pooled_censored)
