"""
Correlograms of the series that set the dependence lag.

The dependence lag (docs/methods.md, Dependence between samples) is the largest
Politis decorrelation lag across the formulations' series as prepared for
the pooled pathway. This figure draws each prepared series' sample
autocorrelation against lag in days, on a logarithmic axis so that hours to
years appear together, with the significance band that defines the
decorrelation lag at the record's length, each series' decorrelation lag,
and whole years. It shows how fast correlation decays, whether a slow or
long-memory component remains, and whether periodic structure remains after
the calendar-day mean is removed. The observations are drawn for comparison
only; they play no part in the dependence lag.
"""
from __future__ import annotations

import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from model_selection.statistics.significance.block_length import (
    C_THRESHOLD, DEFAULT_RUN_LENGTH, _autocovariances, estimate_dependence, searched_lags,
)


def plot_dependence(prepared: dict[str, np.ndarray], step_hours: float, gage: str,
                    observations: np.ndarray | None = None, run_length: int = DEFAULT_RUN_LENGTH,
                    seasonal_window_days: int | None = None) -> go.Figure:
    """Correlograms of the prepared formulation series (and, dotted, the
    prepared observations), with the band and lags described in the module
    docstring. seasonal_window_days is the calendar-day mean's window used in
    the preparation (None: not removed), stated on the figure."""
    fig = go.Figure()
    band = None
    series = dict(prepared)
    if observations is not None:
        series["observations (for comparison)"] = observations
    palette = qualitative.Plotly
    for i, (name, values) in enumerate(series.items()):
        x = np.asarray(values, dtype=float)
        n_obs = int(np.isfinite(x).sum())
        max_lag = searched_lags(x.size, run_length)
        acov = _autocovariances(x, max_lag)
        if acov[0] <= 0:
            continue
        acf = acov[1:] / acov[0]
        lag = estimate_dependence(x, run_length).decorrelation_lag
        days = np.arange(1, max_lag + 1) * step_hours / 24.0
        is_obs = observations is not None and values is observations
        color = "black" if is_obs else palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=days, y=acf, mode="lines", line=dict(width=1, color=color,
                                                                       dash="dot" if is_obs else "solid"),
                                 name=name if is_obs else f"{name} (decorrelation lag {lag * step_hours / 24:.0f} d)"))
        if not is_obs:
            fig.add_vline(x=lag * step_hours / 24.0, line=dict(width=1.5, dash="dash", color=color))
            band = C_THRESHOLD * np.sqrt(np.log10(n_obs) / n_obs)
    if band is not None:
        for sign in (1, -1):
            fig.add_hline(y=sign * band, line=dict(color="black", width=1, dash="dot"))
    for year in range(1, 6):
        fig.add_vline(x=365.25 * year, line=dict(color="lightgray", width=1))
    fig.update_xaxes(type="log", title_text="lag (days)")
    fig.update_yaxes(title_text="autocorrelation")
    fig.update_layout(
        title=dict(text=f"Basin {gage}: autocorrelation of the series as analysed", x=0.01),
        template="plotly_white", height=660, width=1200, margin=dict(t=110, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
    prepared_as = ("log flow with its calendar-day mean removed "
                   f"({seasonal_window_days}-day window), rank-normal transformed"
                   if seasonal_window_days is not None else "log flow (seasonal cycle not removed), rank-normal transformed")
    caption = (f"Each curve is the autocorrelation of a series as analysed: {prepared_as}, at every timestep. "
               "Black dotted band: significance of an autocorrelation at the record's length"
               + (f" (±{band:.3g})" if band is not None else "")
               + ". Dashed lines, in each formulation's colour: its decorrelation lag, the first lag after which "
               "its autocorrelation stays inside the band (5 lags in a row). Grey: whole years.")
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.02, xanchor="left", yanchor="bottom", showarrow=False, align="left",
        font=dict(size=11, color="gray"), text="<br>".join(textwrap.wrap(caption, 150)))
    return fig
