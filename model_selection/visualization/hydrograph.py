"""Interactive hydrographs of a basin with high-flow events shaded."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from model_selection.io.basin import BasinData

EVENT_SHADE_OPACITY = 0.18

OBS_LINE_COLOR = "black"

EVENT_SHADE_COLOR = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
                     "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#1f77b4"]


def plot_basin_diagnostics(
        basin: BasinData,
        events: list,
        title: str | None = None,
        high_flow_quantile: float | None = None,
) -> go.Figure:
    """Observed and simulated series as zoomable lines, with each event
    (anything with event_id, start_time and end_time) shaded. Events missing
    times are skipped and counted in an annotation. With high_flow_quantile,
    the median across formulations of their own high-flow thresholds (that
    quantile of each formulation's values present) is drawn as a dashed line,
    a guide to the level that defines high flow; each formulation's own
    threshold differs from it.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=basin.obs.index,
        y=basin.obs.values,
        name="Observed",
        mode="lines",
        line=dict(color=OBS_LINE_COLOR, width=2),
    ))

    palette = qualitative.Plotly
    for i, (name, series) in enumerate(basin.formulations.items()):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name=name,
            mode="lines",
            line=dict(width=1, color=color),
            opacity=0.75,
            legendgroup=name,
        ))

    if high_flow_quantile is not None:
        thresholds = [float(np.nanquantile(np.asarray(s.values, dtype=float), high_flow_quantile))
                      for s in basin.formulations.values() if np.isfinite(np.asarray(s.values, dtype=float)).any()]
        if thresholds:
            level = float(np.median(thresholds))
            index = basin.obs.index
            fig.add_trace(go.Scatter(
                x=[index[0], index[-1]], y=[level, level],
                name=f"high-flow threshold (median of formulations' {high_flow_quantile:g} quantiles)",
                mode="lines", line=dict(width=1.5, color="gray", dash="dash"), hoverinfo="name+y",
            ))

    # Shapes and labels are set in one layout update: adding them one at a
    # time (add_vrect) revalidates the whole layout each time, quadratic in
    # the number of events.
    n_skipped = 0
    shapes, labels = [], []
    for ev in events:
        if ev.start_time is None or ev.end_time is None:
            n_skipped += 1
            continue
        color = EVENT_SHADE_COLOR[ev.event_id % len(EVENT_SHADE_COLOR)]
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=ev.start_time, x1=ev.end_time, y0=0, y1=1,
                           fillcolor=color, opacity=EVENT_SHADE_OPACITY, line_width=0, layer="below"))
        labels.append(dict(x=ev.start_time + (ev.end_time - ev.start_time) / 2, xref="x", y=1, yref="paper",
                           text=f"E{ev.event_id}", showarrow=False, yanchor="bottom", font=dict(size=9)))
    fig.update_layout(shapes=shapes, annotations=labels)
    if n_skipped:

        fig.add_annotation(
            text=f"({n_skipped} event(s) omitted: missing start/end time)",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title=title or f"USGS basin {basin.gage}: simulated vs. observed, high-flow events shaded",
        xaxis_title="Time",
        yaxis_title="Streamflow (m<sup>3</sup>/s<sup>-1</sup>)",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_series_events(series, gage: str, output_path: str, config=None, series_label: str = "series") -> str:
    """Detect events in one series (e.g. a USGS record with no models) with
    event_detection.detect_events_for_series and write a hydrograph with its
    events shaded. Returns output_path."""
    from model_selection.event_detection.config import EventConfig
    from model_selection.event_detection.detect import detect_events_for_series
    from model_selection.event_detection.windows import attach_timestamps, own_window_to_expanded_events
    from model_selection.visualization.html import write_standalone_html

    config = config or EventConfig()
    result = detect_events_for_series(series, config)
    events = attach_timestamps(own_window_to_expanded_events(series.values, result.decluster_result), series.index)
    fig = plot_basin_diagnostics(BasinData(gage=gage, formulations={series_label: series}, obs=series), events)
    write_standalone_html(fig, output_path, f"{gage} event detection ({config.method})")
    return output_path
