"""
Correlograms of the series that set the dependence lag, and of the same
series before any transformation.

The dependence lag (docs/methods.md, Dependence between samples) is the largest
Politis decorrelation lag across the formulations' series prepared for it
(preprocessing.transforms.dependence_series). The upper panel draws each such series' sample
autocorrelation against lag in days, on a logarithmic axis so that hours to
years appear together, with the significance band that defines the
decorrelation lag at the record's length, each series' decorrelation lag,
and whole years. It shows how fast correlation decays, whether a slow or
long-memory component remains, and whether periodic structure remains after
the adjustment for the time of year. The lower panel draws the same for the
flows as supplied (no transform, no seasonal adjustment), so the
effect of the transformations on the decorrelation can be
seen; it plays no part in the analysis. The observations are drawn for
comparison only; they play no part in the dependence lag.
"""
from __future__ import annotations

import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from model_selection.statistics.significance.block_length import (
    C_THRESHOLD, DEFAULT_RUN_LENGTH, _autocovariances, estimate_dependence, searched_lags,
)

OBS_LABEL = "observations (for comparison)"


def _correlograms(fig: go.Figure, row: int | None, series: dict[str, np.ndarray], observations,
                  step_hours: float, run_length: int, show_legend: bool) -> float | None:
    """Add each series' correlogram and decorrelation lag to the figure (to
    subplot `row` when given); returns the significance band."""
    band = None
    palette = qualitative.Plotly
    where = {} if row is None else {"row": row, "col": 1}
    items = list(series.items()) + ([(OBS_LABEL, observations)] if observations is not None else [])
    for i, (name, values) in enumerate(items):
        x = np.asarray(values, dtype=float)
        n_obs = int(np.isfinite(x).sum())
        max_lag = searched_lags(x.size, run_length)
        acov = _autocovariances(x, max_lag)
        if acov[0] <= 0:
            continue
        acf = acov[1:] / acov[0]
        lag = estimate_dependence(x, run_length).decorrelation_lag
        days = np.arange(1, max_lag + 1) * step_hours / 24.0
        is_obs = name == OBS_LABEL
        color = "black" if is_obs else palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=days, y=acf, mode="lines", legendgroup=name, showlegend=show_legend,
                                 line=dict(width=1, color=color, dash="dot" if is_obs else "solid"),
                                 name=name if is_obs else f"{name}"), **where)
        fig.add_trace(go.Scatter(x=[lag * step_hours / 24.0] * 2, y=[-1, 1], mode="lines", legendgroup=name,
                                 showlegend=False, hovertemplate=f"{name}: decorrelation lag %{{x:.0f}} d<extra></extra>",
                                 line=dict(width=1.5, dash="dash", color=color),
                                 visible=None if not is_obs else False), **where)
        if not is_obs:
            band = C_THRESHOLD * np.sqrt(np.log10(n_obs) / n_obs)
    if band is not None:
        for sign in (1, -1):
            fig.add_hline(y=sign * band, line=dict(color="black", width=1, dash="dot"), **where)
    for year in range(1, 6):
        fig.add_vline(x=365.25 * year, line=dict(color="lightgray", width=1), **where)
    return band


def _lag_text(series: dict[str, np.ndarray], step_hours: float, run_length: int) -> str:
    return ", ".join(f"{name} {estimate_dependence(np.asarray(v, float), run_length).decorrelation_lag * step_hours / 24:.0f} d"
                     for name, v in series.items())


def plot_dependence(prepared: dict[str, np.ndarray], step_hours: float, gage: str,
                    observations: np.ndarray | None = None, run_length: int = DEFAULT_RUN_LENGTH,
                    seasonal: bool = False, raw: dict[str, np.ndarray] | None = None,
                    raw_observations: np.ndarray | None = None) -> go.Figure:
    """Correlograms of the formulation series from which the dependence lag
    is estimated (and, dotted, the observations prepared the same way), with
    the band and lags described in the module docstring; with `raw`, a second
    panel of the same for the flows as supplied. seasonal: whether the series
    were adjusted for the time of year (preprocessing.transforms.
    dependence_series), stated on the figure."""
    prepared_as = ("standardized signed square root of flow minus its fit on the seasonal phase"
                   if seasonal else "standardized signed square root of flow (seasonal cycle not adjusted)")
    if raw is None:
        fig = go.Figure()
        band = _correlograms(fig, None, prepared, observations, step_hours, run_length, True)
        fig.update_xaxes(type="log", title_text="lag (days)")
        fig.update_yaxes(title_text="autocorrelation")
        height, raw_band = 660, None
    else:
        wrap = lambda text: "<br>".join(textwrap.wrap(text, 150))   # long lists of lags would run off the page
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11, subplot_titles=(
            wrap(f"As used for the dependence lag ({prepared_as}). Decorrelation lags: " + _lag_text(prepared, step_hours, run_length)),
            wrap("Flow as supplied, before any transformation. Decorrelation lags: "
                 + _lag_text(raw, step_hours, run_length))))
        band = _correlograms(fig, 1, prepared, observations, step_hours, run_length, True)
        raw_band = _correlograms(fig, 2, raw, raw_observations, step_hours, run_length, False)
        fig.update_xaxes(type="log")
        fig.update_xaxes(title_text="lag (days)", row=2, col=1)
        fig.update_yaxes(title_text="autocorrelation")
        for a in fig.layout.annotations:
            a.update(font=dict(size=11), x=0, xanchor="left", align="left")
        height = 1100
    fig.update_layout(
        title=dict(text=f"Basin {gage}: autocorrelation of the series"
                   + (" as used for the dependence lag" if raw is None
                      else ", as used for the dependence lag and before transformation"), x=0.01,
                   **({} if raw is None else {"y": 0.99, "yanchor": "top"})),
        template="plotly_white", height=height, width=1200, margin=dict(t=190 if raw is not None else 130, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.06 if raw is not None else -0.18, xanchor="left", x=0))
    caption = ((f"Each curve is the autocorrelation of a series as used for the dependence lag: {prepared_as}, "
                "at every timestep. "
                if raw is None else
                "Upper panel: each series as used for the dependence lag, at every timestep; this is the panel that "
                "sets the dependence "
                "lag. Lower panel: the same flows as supplied, for comparison only; it plays no part in the analysis. ")
               + "Black dotted band: significance of an autocorrelation at the record's length"
               + (f" (±{band:.3g})" if band is not None else "")
               + ". Dashed lines, in each formulation's colour: its decorrelation lag, the first lag after which "
               "its autocorrelation stays inside the band (5 lags in a row). Grey: whole years.")
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.06 if raw is not None else 1.02, xanchor="left", yanchor="bottom",
        showarrow=False, align="left", font=dict(size=11, color="gray"), text="<br>".join(textwrap.wrap(caption, 150)))
    return fig
