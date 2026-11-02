"""
Diagnostic plots for the high-flow FPCA step: event-window agreement, the
event gallery (phase-normalized curves vs reconstructions), scree,
components, reconstructions, score scatter and the event-expansion summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model_selection.io.basin import OBS_KEY
from model_selection.statistics.fpca.fpca import FPCAResult, phase_normalize, reconstruct

_COLORWAY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_for(name: str, all_names: list[str]) -> str:
    if name == OBS_KEY:
        return "#000000"
    idx = [n for n in all_names if n != OBS_KEY].index(name)
    return _COLORWAY[idx % len(_COLORWAY)]


def _regularization_annotation(fpca: FPCAResult) -> str:
    """One-line title note: regularization lambda, per-component and cumulative
    explained variance, with a warning when the cumulative value exceeds 1
    (possible for lambda > 0, whose components are not exactly orthogonal; a
    sign lambda is distorting the basis).
    """
    per_component = ", ".join(f"PC{i + 1}={v:.3f}" for i, v in enumerate(fpca.explained_variance_ratio))
    cum_var = fpca.cumulative_explained_variance_ratio[-1] if len(fpca.cumulative_explained_variance_ratio) else float("nan")

    if fpca.regularization_lambda == 0.0:
        reg_note = "no regularization (lambda=0, ordinary FPCA)"
    else:
        reg_note = (f"regularized FPCA (lambda={fpca.regularization_lambda}), "
                    f"component orthogonality error={fpca.component_orthogonality_error:.4f}")

    flag = ""
    if cum_var > 1.02:
        flag = " -- WARNING: cumulative variance exceeds 1.0 (components are meaningfully non-orthogonal; consider reducing lambda)"

    return f"{reg_note}; per-component variance: [{per_component}]; cumulative={cum_var:.3f}{flag}"


def _grid_layout(n_rows: int, n_cols: int, row_h: int, row_gap: int, col_w: int, col_gap: int):
    """(total_width, total_height, vertical_spacing, horizontal_spacing) giving
    every row and gap a fixed pixel size whatever the number of rows (Plotly
    spacings are fractions of the whole canvas).
    """
    total_height = n_rows * row_h + max(n_rows - 1, 0) * row_gap
    total_width = n_cols * col_w + max(n_cols - 1, 0) * col_gap
    vertical_spacing = row_gap / total_height if n_rows > 1 else 0.0
    horizontal_spacing = col_gap / total_width if n_cols > 1 else 0.0
    return total_width, total_height, vertical_spacing, horizontal_spacing


def plot_full_event_gallery(
        event_curves: dict[int, dict[str, np.ndarray]],
        table: pd.DataFrame,
        fpca: FPCAResult,
        n_cols: int = 4,
) -> go.Figure:
    """One subplot per retained event: each series' phase-normalized curve
    (solid) against its FPCA reconstruction (dotted), titled with the window length
    and worst RMSE. Large for many events, hence opt-in.
    """
    event_ids = sorted(table["event_id"].unique())
    n = len(event_ids)
    n_rows = int(np.ceil(n / n_cols)) if n > 0 else 1
    pc_cols = [c for c in table.columns if c.startswith("pc")]

    # single pass: compute RAW-vs-reconstruction for every (event, series)
    # once, reused for both subplot titles (worst RMSE/R2) and the traces.
    per_event = {}  # event_id -> list of (name, color, norm_curve, recon, rmse, r2)
    all_names = sorted({name for curves in event_curves.values() for name in curves})
    for eid in event_ids:
        curves = event_curves.get(eid, {})
        sub = table[table["event_id"] == eid].set_index("series")
        rows = []
        for name, curve in curves.items():
            if name not in sub.index:
                continue  # filtered out by min_event_length -- no FPC score/reconstruction for it
            norm_curve = phase_normalize(curve, n_grid=fpca.n_grid)  # RAW curve
            scores = sub.loc[name, pc_cols].to_numpy(dtype=float)[None, :]
            recon = reconstruct(scores, fpca)[0]
            rmse = float(sub.loc[name, "reconstruction_rmse"])
            r2 = float(sub.loc[name, "reconstruction_r2"])
            rows.append((name, _color_for(name, all_names), norm_curve, recon, rmse, r2))
        per_event[eid] = rows

    subplot_titles = []
    for eid in event_ids:
        sub = table[table["event_id"] == eid]
        dur = int(sub["duration_timesteps"].iloc[0])
        # paired to the SAME series, not independently-worst RMSE and
        # independently-worst R2 (which could come from different series
        # and mislead -- e.g. "rmse=2.1, r2=-40" reads as one bad
        # reconstruction, but if those numbers came from two different
        # models neither one is actually that bad on its own).
        worst_row = max(per_event[eid], key=lambda r: r[4], default=None)
        if worst_row is not None:
            worst_rmse, worst_r2 = worst_row[4], worst_row[5]
            subplot_titles.append(f"E{eid}  d={dur}  rmse={worst_rmse:.2f}  R²={worst_r2:.2f}")
        else:
            subplot_titles.append(f"E{eid}  d={dur}")

    total_width, total_height, vspace, hspace = _grid_layout(
        n_rows, n_cols, row_h=220, row_gap=55, col_w=360, col_gap=40,
    )

    fig = make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=subplot_titles,
        vertical_spacing=vspace, horizontal_spacing=hspace,
    )

    legend_shown = set()

    for i, eid in enumerate(event_ids):
        row, col = divmod(i, n_cols)
        r, c = row + 1, col + 1
        for name, color, norm_curve, recon, _rmse, _r2 in per_event[eid]:
            show_legend = name not in legend_shown
            legend_shown.add(name)
            fig.add_trace(go.Scatter(
                x=fpca.grid, y=norm_curve, mode="lines",
                line=dict(color=color, width=2 if name == OBS_KEY else 1.3),
                name=name, legendgroup=name, showlegend=show_legend,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=fpca.grid, y=recon, mode="lines",
                line=dict(color=color, width=1, dash="dot"), opacity=0.85,
                name=f"{name} (reconstructed)", legendgroup=name, showlegend=False,
            ), row=r, col=c)

    fig.update_annotations(font_size=10)
    fig.update_layout(
        title=f"All high-flow events: original (solid) vs. FPCA reconstructed (dotted) streamflow m<sup>3</sup>/s<sup>-1</sup> against phase-normalized time"
              f"<br><sup>{_regularization_annotation(fpca)}</sup>",
        height=total_height, width=total_width, autosize=False,
        margin=dict(t=70),
    )
    return fig


def plot_fpca_scree(fpca: FPCAResult, max_components_shown: int = 10) -> go.Figure:
    """Explained-variance ratio per component of the full spectrum, with the
    cumulative curve.
    """
    k = min(max_components_shown, fpca.full_explained_variance_ratio.shape[0])
    ratios = fpca.full_explained_variance_ratio[:k]
    cumulative = np.cumsum(ratios)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(k)], y=ratios, name="explained variance ratio"))
    fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(k)], y=cumulative, name="cumulative",
                              mode="lines+markers", yaxis="y2"))
    n_kept = fpca.components.shape[0]
    fig.add_vline(x=n_kept - 0.5, line_dash="dash", line_color="gray",
                  annotation_text=f"kept n_components={n_kept}")
    fig.update_layout(
        title=f"FPCA scree (fit on: {fpca.fit_source}, n_curves={fpca.n_curves_fit})<br>"
              f"<sup>{fpca.selection_note}</sup>",
        yaxis=dict(title="explained variance ratio", range=[0, 1]),
        yaxis2=dict(title="cumulative", overlaying="y", side="right", range=[0, 1]),
        height=500, width=850, autosize=False,
    )
    return fig


def plot_fpca_components(fpca: FPCAResult, n_show: int = 3, scale: float = 1.0) -> go.Figure:
    """Mean curve and mean +/- scale * sqrt(variance) * component for the first
    n_show components: what each FPC does to the shape (Ramsay & Silverman).
    """
    n_show = min(n_show, fpca.components.shape[0])
    fig = make_subplots(rows=1, cols=n_show, subplot_titles=[
        f"PC{i+1} ({fpca.explained_variance_ratio[i]*100:.1f}% var)" for i in range(n_show)
    ], horizontal_spacing=0.06)
    for i in range(n_show):
        perturb = scale * np.sqrt(max(fpca.explained_variance[i], 0)) * fpca.components[i]
        fig.add_trace(go.Scatter(x=fpca.grid, y=fpca.mean_curve, mode="lines",
                                  line=dict(color="black", width=2), name="mean",
                                  showlegend=(i == 0), legendgroup="mean"), row=1, col=i + 1)
        fig.add_trace(go.Scatter(x=fpca.grid, y=fpca.mean_curve + perturb, mode="lines",
                                  line=dict(color="#d62728", width=1.5, dash="dash"), name="mean + PC",
                                  showlegend=(i == 0), legendgroup="plus"), row=1, col=i + 1)
        fig.add_trace(go.Scatter(x=fpca.grid, y=fpca.mean_curve - perturb, mode="lines",
                                  line=dict(color="#1f77b4", width=1.5, dash="dash"), name="mean - PC",
                                  showlegend=(i == 0), legendgroup="minus"), row=1, col=i + 1)
    fig.update_layout(title="FPCA components: effect on the mean shape", height=460, width=420 * n_show, autosize=False)
    return fig


def plot_reconstruction_gallery(
        table: pd.DataFrame,
        event_curves: dict[int, dict[str, np.ndarray]],
        fpca: FPCAResult,
        sample_rows: pd.DataFrame,
) -> go.Figure:
    """Phase-normalized curve vs reconstruction for selected (event, series)
    rows of the score table.
    """
    n = len(sample_rows)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    total_width, total_height, vspace, hspace = _grid_layout(
        n_rows, n_cols, row_h=280, row_gap=55, col_w=380, col_gap=45,
    )
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[
            f"E{r.event_id}/{r.series}  rmse={r.reconstruction_rmse:.2f}  R²={r.reconstruction_r2:.2f}"
            for r in sample_rows.itertuples()
        ],
        vertical_spacing=vspace, horizontal_spacing=hspace,
    )
    for i, r in enumerate(sample_rows.itertuples()):
        row, col = divmod(i, n_cols)
        curve = event_curves[r.event_id][r.series]
        norm_curve = phase_normalize(curve, n_grid=fpca.n_grid)  # RAW curve
        scores = np.array([[getattr(r, f"pc{c+1}") for c in range(fpca.components.shape[0])]])
        recon = reconstruct(scores, fpca)[0]
        fig.add_trace(go.Scatter(x=fpca.grid, y=norm_curve, mode="lines",
                                  line=dict(color="black", width=2), name="raw original",
                                  showlegend=(i == 0), legendgroup="orig"), row=row + 1, col=col + 1)
        fig.add_trace(go.Scatter(x=fpca.grid, y=recon, mode="lines",
                                  line=dict(color="#d62728", width=2, dash="dash"), name="reconstruction",
                                  showlegend=(i == 0), legendgroup="recon"), row=row + 1, col=col + 1)
    fig.update_annotations(font_size=11)
    fig.update_layout(title=f"Reconstruction fidelity (RAW original vs. top-k FPC reconstruction)"
                            f"<br><sup>{_regularization_annotation(fpca)}</sup>",
                       height=total_height, width=total_width, autosize=False)
    return fig


def plot_fpc_score_scatter(table: pd.DataFrame, pcs: tuple[str, str] = ("pc1", "pc2")) -> go.Figure:
    """Two FPC score dimensions coloured by series, as a face-validity check."""
    fig = go.Figure()
    all_names = sorted(table["series"].unique())
    for name in all_names:
        sub = table[table["series"] == name]
        fig.add_trace(go.Scatter(
            x=sub[pcs[0]], y=sub[pcs[1]], mode="markers",
            marker=dict(color=_color_for(name, all_names), size=8 if name == OBS_KEY else 6,
                        symbol="star" if name == OBS_KEY else "circle"),
            name=name,
            text=[f"event {e}" for e in sub["event_id"]], hoverinfo="text+x+y",
        ))
    fig.update_layout(
        title=f"FPC score scatter: {pcs[0]} vs {pcs[1]}, colored by series",
        xaxis_title=pcs[0], yaxis_title=pcs[1], height=650, width=850, autosize=False,
    )
    return fig


def plot_overview_diagnostics(fpca: FPCAResult) -> dict[str, go.Figure]:
    """The small overview figures (scree) as {title: figure}."""
    return {"scree": plot_fpca_scree(fpca)}


