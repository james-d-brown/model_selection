"""
Diagnostic plots for the escalation tests.

The plots are visual aids only: they are not dependence-corrected and do
not replace the tests' decisions. Each plot function returns a Plotly
figure; caption_html writes a plain-HTML caption for the page (Plotly
annotations wrap long paragraphs badly) from the round's PlotContext: the
candidate, the conditioning set, the test and its outcome. write_adequacy_page
combines the two into one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from model_selection.statistics.adequacy.linearity import ols_residuals as _ols_residuals
from model_selection.statistics.association.canonical import (
    first_canonical_projection as _first_canonical_projection,
)
from model_selection.statistics.association.canonical import residualize_on_conditioning
from model_selection.utils import as2d


@dataclass
class PlotContext:
    """What a diagnostic page shows, for its title and caption.

    selected : the formulations selected before this round, which with the
        sample's covariates form the conditioning set (the covariates come
        first in it, n_covariate_columns of them).
    result : the escalation test's outcome and what followed, in words;
        None for a descriptive plot made without a test.
    """
    tier: str
    round_no: int
    candidate: str
    selected: tuple[str, ...] = ()
    covariate_label: str | None = None
    n_covariate_columns: int = 0
    feature_names: tuple[str, ...] | None = None
    result: str | None = None

    def selected_text(self) -> str:
        names = [f"'{s}'" for s in self.selected]
        if not names:
            return ""
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        return f"the selected formulation{'s' if len(names) > 1 else ''} {joined}"

    def conditioning_text(self) -> str:
        parts = [t for t in (self.selected_text(), self.covariate_label) if t]
        return " and ".join(parts) if parts else "nothing"


TEST_NAMES = {"linear_specification": "linear adjustment test", "linearity": "candidate linearity test",
              "multivariate_linearity": "candidate linearity test",
              "state_independence": "candidate effect-modification test"}

N_STRATA = 3
_STRATUM_NAMES = ("lowest third", "middle third", "highest third")
_SEASONS = (("January to April", 0, 120), ("May to August", 120, 243), ("September to December", 243, 366))


def _feature_name(names: tuple[str, ...] | None, k: int, p: int) -> str:
    if names is not None and len(names) == p:
        return names[k]
    return "" if p == 1 else f"feature {k + 1}"


def state_strata(Z: np.ndarray, context: PlotContext | None) -> tuple[np.ndarray, list[str], str]:
    """(stratum of each sample, stratum names, what the strata divide) for
    the effect-modification plot: thirds of the selected formulations'
    state (their median percentile rank) when there are selected
    formulations; otherwise the time of year when the conditioning set is
    the seasonal phase; otherwise thirds of the conditioning set's state."""
    Z = as2d(Z)
    if context is not None and context.selected:
        score = composite_regime_score(Z[:, context.n_covariate_columns:])
        what = f"the state of {context.selected_text()}"
    elif context is not None and context.covariate_label and "seasonal phase" in context.covariate_label \
            and Z.shape[1] >= 2:
        day = (np.mod(np.arctan2(Z[:, 0], Z[:, 1]), 2 * np.pi) / (2 * np.pi)) * 365.0
        labels = np.digitize(day, [_SEASONS[1][1], _SEASONS[2][1]])
        return labels, [name for name, _, _ in _SEASONS], "the time of year"
    else:
        score = composite_regime_score(Z)
        what = "the state of the conditioning set"
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, N_STRATA + 1))[1:-1])
    labels = np.digitize(score, edges)
    names = list(_STRATUM_NAMES) if len(edges) == N_STRATA - 1 else [f"group {k + 1}" for k in range(len(edges) + 1)]
    return labels, names, what


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean y within each of n_bins equal-count bins of x, an assumption-free
    trend to compare against a fitted line.
    """
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    bins = np.array_split(np.arange(x.shape[0]), max(n_bins, 1))
    tx = np.array([x_sorted[idx].mean() for idx in bins if idx.size > 0])
    ty = np.array([y_sorted[idx].mean() for idx in bins if idx.size > 0])
    return tx, ty


def composite_regime_score(Z: np.ndarray) -> np.ndarray:
    """Scalar regime score of the conditioning set, used only to stratify plots:
    the median across conditioning models of each model's own percentile rank
    (the same obs-free rank-consensus idea as the high-flow regime).
    """
    Z = as2d(Z)
    n = Z.shape[0]
    ranks = np.column_stack([rankdata(Z[:, j], method="average") / n for j in range(Z.shape[1])])
    return np.median(ranks, axis=1)


def plot_linearity_diagnostics(
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray | None,
        n_bins: int = 30,
        max_scatter_points: int = 10000,
        rng: np.random.Generator | None = None,
        title: str | None = None,
):
    """Two-panel check of linearity for a scalar state.

    Panel A: residualized candidate vs residualized obs with the linear fit
    and a binned empirical trend. Panel B: linear-fit residuals vs fitted
    values. Scatter points are subsampled for rendering only; the trends use
    all points.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rng = rng or np.random.default_rng()
    candidate = np.asarray(candidate, dtype=np.float64).ravel()
    obs = np.asarray(obs, dtype=np.float64).ravel()
    n = candidate.shape[0]

    candidate_resid = _ols_residuals(candidate, conditioning)
    obs_resid = _ols_residuals(obs, conditioning)

    design = np.column_stack([np.ones(n), candidate_resid])
    coefs, _, _, _ = np.linalg.lstsq(design, obs_resid, rcond=None)
    fitted = design @ coefs
    linear_resid = obs_resid - fitted

    trend_x, trend_y = _binned_trend(candidate_resid, obs_resid, n_bins)
    trend_fx, trend_fy = _binned_trend(fitted, linear_resid, n_bins)

    sample_idx = (
        rng.choice(n, size=max_scatter_points, replace=False)
        if n > max_scatter_points else np.arange(n)
    )

    line_x = np.array([candidate_resid.min(), candidate_resid.max()])
    line_y = coefs[0] + coefs[1] * line_x

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Observations against the candidate, each given the conditioning set",
            "Residuals of the straight line against its fitted values",
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scattergl(
        x=candidate_resid[sample_idx], y=obs_resid[sample_idx],
        mode="markers", marker=dict(size=3, opacity=0.3),
        name="data" if n <= max_scatter_points else f"data ({max_scatter_points} of {n} points shown)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", name="straight-line fit", line=dict(color="red"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trend_x, y=trend_y, mode="lines+markers", name="binned mean",
        line=dict(color="orange", dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=fitted[sample_idx], y=linear_resid[sample_idx],
        mode="markers", marker=dict(size=3, opacity=0.3), name="residuals",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[fitted.min(), fitted.max()], y=[0, 0], mode="lines",
        line=dict(color="red", dash="dot"), name="zero", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=trend_fx, y=trend_fy, mode="lines+markers", name="binned residual trend",
        line=dict(color="orange", dash="dash"), showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=title or "Candidate linearity",
            font=dict(size=15),
            x=0.5, xanchor="center", y=0.97, yanchor="top",
        ),
        height=650,
        width=1500,
        margin=dict(l=70, r=50, t=130, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5),
    )
    fig.update_annotations(font_size=12)

    fig.update_xaxes(title_text="candidate, given the conditioning set", row=1, col=1, automargin=True)
    fig.update_yaxes(title_text="observations, given the conditioning set", row=1, col=1, automargin=True)
    fig.update_xaxes(title_text="fitted value of the straight line", row=1, col=2, automargin=True)
    fig.update_yaxes(title_text="residual", row=1, col=2, automargin=True)

    return fig


def plot_multivariate_linearity_diagnostics(
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray | None,
        n_bins: int = 15,
        max_scatter_points: int = 3000,
        rng: np.random.Generator | None = None,
        title: str | None = None,
        feature_names: tuple[str, ...] | None = None,
):
    """Linearity check for multivariate (FPC score) states.

    Top row (test-aligned): candidate and obs projected on their first
    canonical direction, the combination partial CCA is built from, with fit,
    binned trend and residual panels. Loading weights: how much each FPC
    dimension contributes to that projection. Per-FPC grid (physically
    interpretable, not test-aligned): candidate PCk vs obs PCk, meaningful
    because both share one FPCA basis.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rng = rng or np.random.default_rng()
    candidate_resid = residualize_on_conditioning(candidate, conditioning)
    obs_resid = residualize_on_conditioning(obs, conditioning)
    n = candidate_resid.shape[0]
    p, q = candidate_resid.shape[1], obs_resid.shape[1]
    n_fpc_panels = min(p, q)

    x_proj, y_proj, x_loadings, y_loadings, first_cc = _first_canonical_projection(
        candidate_resid, obs_resid)

    design = np.column_stack([np.ones(n), x_proj])
    coefs, _, _, _ = np.linalg.lstsq(design, y_proj, rcond=None)
    fitted = design @ coefs
    linear_resid = y_proj - fitted

    trend_x, trend_y = _binned_trend(x_proj, y_proj, n_bins)
    trend_fx, trend_fy = _binned_trend(fitted, linear_resid, n_bins)

    sample_idx = (
        rng.choice(n, size=max_scatter_points, replace=False)
        if n > max_scatter_points else np.arange(n)
    )
    line_x = np.array([x_proj.min(), x_proj.max()])
    line_y = coefs[0] + coefs[1] * line_x

    # Grid layout: row 1 = test-aligned panels A/B (2 cols); row 2 =
    # loading weights (spans both cols); rows 3+ = per-FPC grid, up to
    # 3 panels per row.
    fpc_cols = 3
    fpc_rows = int(np.ceil(n_fpc_panels / fpc_cols)) if n_fpc_panels > 0 else 0
    total_rows = 2 + fpc_rows

    specs = [[{}, {}, None], [{"colspan": 2}, None, None]]
    subplot_titles = [
        "Observations against the candidate, as their best-predicting combinations of features",
        "Residuals of the straight line against its fitted values",
        "Weight of each feature in those combinations",
    ]
    for r in range(fpc_rows):
        row_specs = []
        for c in range(fpc_cols):
            k = r * fpc_cols + c
            row_specs.append({} if k < n_fpc_panels else None)
        specs.append(row_specs)
        for c in range(fpc_cols):
            k = r * fpc_cols + c
            if k < n_fpc_panels:
                subplot_titles.append(f"{_feature_name(feature_names, k, n_fpc_panels) or 'feature'}: "
                                      "candidate against observations")

    fig = make_subplots(
        rows=total_rows, cols=fpc_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=min(0.25 / total_rows, 0.08),
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scattergl(
        x=x_proj[sample_idx], y=y_proj[sample_idx],
        mode="markers", marker=dict(size=3, opacity=0.3),
        name="data" if n <= max_scatter_points else f"data ({max_scatter_points} of {n} points shown)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", name="straight-line fit", line=dict(color="red"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=trend_x, y=trend_y, mode="lines+markers", name="binned mean",
        line=dict(color="orange", dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scattergl(
        x=fitted[sample_idx], y=linear_resid[sample_idx],
        mode="markers", marker=dict(size=3, opacity=0.3), name="residuals",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[fitted.min(), fitted.max()], y=[0, 0], mode="lines",
        line=dict(color="red", dash="dot"), name="zero", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=trend_fx, y=trend_fy, mode="lines+markers", name="binned residual trend",
        line=dict(color="orange", dash="dash"), showlegend=False,
    ), row=1, col=2)

    loading_labels = ([f"candidate {_feature_name(feature_names, i, p) or 'feature'}" for i in range(p)]
                      + [f"observations {_feature_name(feature_names, i, q) or 'feature'}" for i in range(q)])
    loading_values = list(x_loadings) + list(y_loadings)
    loading_colors = ["#1f77b4"] * p + ["#ff7f0e"] * q
    fig.add_trace(go.Bar(
        x=loading_labels, y=loading_values, marker=dict(color=loading_colors),
        name="loading weight", showlegend=False,
    ), row=2, col=1)

    for k in range(n_fpc_panels):
        r = 3 + (k // fpc_cols)
        c = 1 + (k % fpc_cols)
        cand_k = candidate_resid[:, k]
        obs_k = obs_resid[:, k]
        trend_kx, trend_ky = _binned_trend(cand_k, obs_k, min(n_bins, max(n // 10, 2)))
        fig.add_trace(go.Scattergl(
            x=cand_k[sample_idx], y=obs_k[sample_idx],
            mode="markers", marker=dict(size=3, opacity=0.3, color="#2ca02c"),
            showlegend=False,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=trend_kx, y=trend_ky, mode="lines+markers",
            line=dict(color="orange", dash="dash"), showlegend=False,
        ), row=r, col=c)

    # legend just below row 1 (panels A/B) so it never overlaps the title
    row1_bottom = fig.layout.yaxis.domain[0]
    fig.update_layout(
        title=dict(
            text=title or "Candidate linearity",
            font=dict(size=15),
            x=0.5, xanchor="center", y=0.99, yanchor="top",
        ),
        height=420 + 320 * total_rows,
        width=1500,
        margin=dict(l=70, r=50, t=130, b=60),
        legend=dict(orientation="h", yanchor="top", y=row1_bottom, xanchor="center", x=0.5),
    )
    fig.update_annotations(font_size=11)

    fig.update_xaxes(title_text="candidate combination, given the conditioning set", row=1, col=1, automargin=True)
    fig.update_yaxes(title_text="observations combination, given the conditioning set", row=1, col=1,
                     automargin=True)
    fig.update_xaxes(title_text="fitted value of the straight line", row=1, col=2, automargin=True)
    fig.update_yaxes(title_text="residual", row=1, col=2, automargin=True)
    fig.update_yaxes(title_text="weight", row=2, col=1, automargin=True)
    for k in range(n_fpc_panels):
        r = 3 + (k // fpc_cols)
        c = 1 + (k % fpc_cols)
        name = _feature_name(feature_names, k, n_fpc_panels) or "feature"
        fig.update_xaxes(title_text=f"candidate {name}, given the conditioning set", row=r, col=c, automargin=True)
        fig.update_yaxes(title_text=f"observations {name}", row=r, col=c, automargin=True)

    return fig


def plot_state_independence_diagnostics(
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray,
        n_bins_per_stratum: int = 10,
        max_scatter_points: int = 3000,
        rng: np.random.Generator | None = None,
        title: str | None = None,
        context: PlotContext | None = None,
):
    """Whether the candidate's relationship with the observations changes
    across states of the conditioning set.

    Candidate and observations are residualized on the conditioning set by
    the nonlinear tier's kernel regression, which removes the main effects
    of Z, and split into thirds by state_strata (the selected formulations'
    state, or the time of year in round 1 of the pooled pathway). Each third
    is drawn with its own binned trend. Under no effect modification the
    trends coincide. With several features (high-flow events), each third is
    reduced to its own best-predicting combination of features, shown with
    the features' weights and per-feature panels. Requires non-empty
    conditioning.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if conditioning is None or np.asarray(conditioning).size == 0:
        raise ValueError("the effect-modification plot requires a non-empty conditioning set")

    rng = rng or np.random.default_rng()
    candidate, obs, Z = _evenly_spaced(as2d(candidate), as2d(obs), as2d(conditioning))
    p, q = candidate.shape[1], obs.shape[1]
    multivariate = max(p, q) > 1
    feature_names = context.feature_names if context is not None else None
    n_feature_panels = min(p, q) if multivariate else 0

    stratum_labels, stratum_names, _ = state_strata(Z, context)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    per_stratum_points_budget = max(200, max_scatter_points // max(len(stratum_names), 1))

    cand_resid_all = _nonlinear_residuals(candidate, Z)
    obs_resid_all = _nonlinear_residuals(obs, Z)
    if multivariate:
        _, _, pooled_x_load, pooled_y_load, _ = _first_canonical_projection(cand_resid_all, obs_resid_all)
    stratum_data = []
    for k, name in enumerate(stratum_names):
        mask = stratum_labels == k
        n_k = int(mask.sum())
        if n_k < 5:
            continue  # too few points in this stratum to plot meaningfully
        cand_k = cand_resid_all[mask] - cand_resid_all[mask].mean(axis=0)
        obs_k = obs_resid_all[mask] - obs_resid_all[mask].mean(axis=0)
        if multivariate:
            x_proj_k, y_proj_k, x_load_k, y_load_k, _ = _first_canonical_projection(cand_k, obs_k)
            # A canonical pair's sign is arbitrary: flip it to agree with the pooled pair, so
            # the states' weights can be compared. Weights are scaled to unit length per side.
            if x_load_k @ pooled_x_load + y_load_k @ pooled_y_load < 0:
                x_proj_k, y_proj_k, x_load_k, y_load_k = -x_proj_k, -y_proj_k, -x_load_k, -y_load_k
            x_load_k = x_load_k / (np.linalg.norm(x_load_k) or 1.0)
            y_load_k = y_load_k / (np.linalg.norm(y_load_k) or 1.0)
        else:
            x_proj_k, y_proj_k, x_load_k, y_load_k = cand_k[:, 0], obs_k[:, 0], np.ones(1), np.ones(1)
        design_k = np.column_stack([np.ones(n_k), x_proj_k])
        coefs_k, _, _, _ = np.linalg.lstsq(design_k, y_proj_k, rcond=None)
        fitted_k = design_k @ coefs_k
        resid_k = y_proj_k - fitted_k
        bins = min(n_bins_per_stratum, max(n_k // 10, 2))
        trend_x_k, trend_y_k = _binned_trend(x_proj_k, y_proj_k, bins)
        trend_fx_k, trend_fy_k = _binned_trend(fitted_k, resid_k, bins)
        stratum_data.append(dict(
            k=k, n_k=n_k, cand_resid=cand_k, obs_resid=obs_k,
            x_proj=x_proj_k, y_proj=y_proj_k, x_loadings=x_load_k, y_loadings=y_load_k,
            fitted=fitted_k, resid=resid_k, trend_x=trend_x_k, trend_y=trend_y_k,
            trend_fx=trend_fx_k, trend_fy=trend_fy_k, label=f"{name} (n={n_k})",
        ))

    cols = 2
    feature_rows = int(np.ceil(n_feature_panels / cols)) if n_feature_panels else 0
    total_rows = 1 + (1 if multivariate else 0) + feature_rows
    specs = [[{}, {}]]
    what = "as their best-predicting combinations of features, " if multivariate else ""
    subplot_titles = [f"Observations against the candidate, {what}by state",
                      "Residuals of each state's straight line against its fitted values"]
    if multivariate:
        specs.append([{"colspan": 2}, None])
        subplot_titles.append("Weight of each feature in each state's combination")
    for r in range(feature_rows):
        specs.append([{} if r * cols + c < n_feature_panels else None for c in range(cols)])
        for c in range(cols):
            k = r * cols + c
            if k < n_feature_panels:
                subplot_titles.append(f"{_feature_name(feature_names, k, n_feature_panels) or 'feature'}: "
                                      "candidate against observations, by state")

    fig = make_subplots(rows=total_rows, cols=cols, specs=specs, subplot_titles=subplot_titles,
                        vertical_spacing=min(0.25 / total_rows, 0.08), horizontal_spacing=0.08)

    for i, sd in enumerate(stratum_data):
        color = palette[i % len(palette)]
        n_k = sd["n_k"]
        sample_idx = (rng.choice(n_k, size=per_stratum_points_budget, replace=False)
                      if n_k > per_stratum_points_budget else np.arange(n_k))
        fig.add_trace(go.Scattergl(x=sd["x_proj"][sample_idx], y=sd["y_proj"][sample_idx], mode="markers",
                                   marker=dict(size=3, opacity=0.25, color=color),
                                   legendgroup=sd["label"], showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=sd["trend_x"], y=sd["trend_y"], mode="lines+markers",
                                 line=dict(color=color, width=3), marker=dict(size=5, color=color),
                                 name=sd["label"], legendgroup=sd["label"]), row=1, col=1)
        fig.add_trace(go.Scattergl(x=sd["fitted"][sample_idx], y=sd["resid"][sample_idx], mode="markers",
                                   marker=dict(size=3, opacity=0.25, color=color),
                                   legendgroup=sd["label"], showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=sd["trend_fx"], y=sd["trend_fy"], mode="lines+markers",
                                 line=dict(color=color, width=3), marker=dict(size=5, color=color),
                                 legendgroup=sd["label"], showlegend=False), row=1, col=2)
        if multivariate:
            labels = ([f"candidate {_feature_name(feature_names, j, p) or 'feature'}" for j in range(p)]
                      + [f"observations {_feature_name(feature_names, j, q) or 'feature'}" for j in range(q)])
            fig.add_trace(go.Bar(x=labels, y=list(sd["x_loadings"]) + list(sd["y_loadings"]),
                                 marker=dict(color=color), legendgroup=sd["label"], showlegend=False), row=2, col=1)
            for fk in range(n_feature_panels):
                r, c = 3 + fk // cols, 1 + fk % cols
                cand_fk, obs_fk = sd["cand_resid"][:, fk], sd["obs_resid"][:, fk]
                tfx, tfy = _binned_trend(cand_fk, obs_fk, min(n_bins_per_stratum, max(n_k // 10, 2)))
                fig.add_trace(go.Scattergl(x=cand_fk[sample_idx], y=obs_fk[sample_idx], mode="markers",
                                           marker=dict(size=3, opacity=0.25, color=color),
                                           legendgroup=sd["label"], showlegend=False), row=r, col=c)
                fig.add_trace(go.Scatter(x=tfx, y=tfy, mode="lines+markers", line=dict(color=color, width=3),
                                         marker=dict(size=5, color=color), legendgroup=sd["label"],
                                         showlegend=False), row=r, col=c)

    height = 600 if total_rows == 1 else 200 + 380 * total_rows
    fig.update_layout(
        title=dict(text=title or "Candidate effect modification", font=dict(size=15), x=0.5, xanchor="center",
                   y=1 - 16 / height, yanchor="top"),
        height=height, width=1500, barmode="group",
        margin=dict(l=70, r=230, t=120, b=60),
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02, font=dict(size=12),
                    title=dict(text="State")),
    )
    fig.update_annotations(font_size=12)
    xname = "candidate combination" if multivariate else "candidate"
    yname = "observations combination" if multivariate else "observations"
    fig.update_xaxes(title_text=f"{xname}, given the conditioning set", row=1, col=1, automargin=True)
    fig.update_yaxes(title_text=f"{yname}, given the conditioning set", row=1, col=1, automargin=True)
    fig.update_xaxes(title_text="fitted value of the straight line", row=1, col=2, automargin=True)
    fig.update_yaxes(title_text="residual", row=1, col=2, automargin=True)
    if multivariate:
        fig.update_yaxes(title_text="relative weight", row=2, col=1, automargin=True)
    for fk in range(n_feature_panels):
        r, c = 3 + fk // cols, 1 + fk % cols
        name = _feature_name(feature_names, fk, n_feature_panels) or "feature"
        fig.update_xaxes(title_text=f"candidate {name}, given the conditioning set", row=r, col=c, automargin=True)
        fig.update_yaxes(title_text=f"observations {name}", row=r, col=c, automargin=True)
    return fig


KERNEL_PLOT_MAX_N = 2000
"""Largest sample a diagnostic plot fits a kernel regression to. Longer
samples (pooled records in the linear tier) are plotted on this many evenly
spaced rows, as the kernel-based tests themselves are run on at most this
many evenly spaced samples (AdequacyConfig.kernel_check_max_n)."""


def _evenly_spaced(*arrays, max_n: int = KERNEL_PLOT_MAX_N):
    """The arrays restricted to at most max_n evenly spaced rows (unchanged
    when they are no longer than that)."""
    n = as2d(arrays[0]).shape[0]
    if n <= max_n:
        return arrays
    rows = np.unique(np.rint(np.linspace(0, n - 1, max_n)).astype(int))
    return tuple(as2d(a)[rows] for a in arrays)


_REGRESSION_CACHE: dict = {}
"""The most recent plotting regression, keyed on its conditioning set. A
regression's kernel eigendecompositions depend only on Z, so every plot of a
round (each column of the specification plot, the candidate and the
observations of the effect-modification plot) shares one; residuals() does
not change the regression, so the plotted values are those of a fresh one."""


def _plot_regression(Z: np.ndarray):
    """The kernel regression on Z used by the plots, built once per
    conditioning set (cache of one, to bound memory)."""
    import hashlib

    from model_selection.statistics.regression.kernel_residuals import ConditioningRegression

    Z = np.ascontiguousarray(as2d(np.asarray(Z, dtype=float)))
    key = (Z.shape, hashlib.sha1(Z.tobytes()).hexdigest())
    if key not in _REGRESSION_CACHE:
        _REGRESSION_CACHE.clear()
        _REGRESSION_CACHE[key] = ConditioningRegression(Z, np.arange(Z.shape[0]))
    return _REGRESSION_CACHE[key]


def _nonlinear_residuals(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Cross-fitted kernel ridge residuals of Y on Z, as used by the
    nonlinear tier (no buffer)."""
    resid, _ = _plot_regression(Z).residuals(as2d(Y))
    return resid


def plot_regression_specification_diagnostics(
        candidate: np.ndarray,
        obs: np.ndarray,
        conditioning: np.ndarray,
        max_scatter_points: int = 3000,
        rng: np.random.Generator | None = None,
        title: str | None = None,
        feature_names: tuple[str, ...] | None = None,
):
    """Residuals against fitted values of the least-squares regressions that
    remove the conditioning set in the linear tier, one panel per column of
    the candidate and of the observations, labelled by feature_names when
    given (for example the high-flow pathway's principal components). A
    binned trend near zero everywhere means the regression has removed the
    conditioning set's influence; systematic curvature means it has not.
    Requires non-empty conditioning.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from model_selection.statistics.regression.linear import ols_fit

    if conditioning is None or np.asarray(conditioning).size == 0:
        raise ValueError("the specification plot requires a non-empty conditioning set")
    rng = rng or np.random.default_rng()
    C, O, Z = as2d(candidate), as2d(obs), as2d(conditioning)
    columns = [("candidate", C, j) for j in range(C.shape[1])] + [("observations", O, j) for j in range(O.shape[1])]

    def panel_title(label, M, j):
        if feature_names is not None and len(feature_names) == M.shape[1]:
            return f"{label}, {feature_names[j]}"
        return label if M.shape[1] == 1 else f"{label}, feature {j + 1}"

    n_cols = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[panel_title(label, M, j) for label, M, j in columns],
                        vertical_spacing=min(0.3 / n_rows, 0.12), horizontal_spacing=0.08)
    for i, (label, M, j) in enumerate(columns):
        fitted, resid = ols_fit(M[:, [j]], Z)
        f, r = fitted.ravel(), resid.ravel()
        n = f.size
        idx = rng.choice(n, size=max_scatter_points, replace=False) if n > max_scatter_points else np.arange(n)
        tx, ty = _binned_trend(f, r, min(15, max(n // 10, 2)))
        row, col = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(go.Scattergl(x=f[idx], y=r[idx], mode="markers", showlegend=False,
                                   marker=dict(size=3, opacity=0.3, color="#1f77b4")), row=row, col=col)
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines+markers", showlegend=False,
                                 line=dict(color="#d62728", width=2)), row=row, col=col)
        fig.add_hline(y=0.0, line=dict(color="gray", dash="dash"), row=row, col=col)
        fig.update_xaxes(title_text="fitted value", row=row, col=col)
        fig.update_yaxes(title_text="residual", row=row, col=col)
    fig.update_layout(height=320 * n_rows + 120, width=1400,
                      title=dict(text=title or "Linear adjustment"))
    return fig


def write_adequacy_page(fig, captions_html: str, path: str, page_title: str) -> str:
    """Write a figure plus its HTML caption block to a standalone HTML page."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    page_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 1520px;
          margin: 20px auto; padding: 0 20px; color: #222; }}
  .captions {{ display: flex; gap: 48px; margin-top: 12px; font-size: 13px; line-height: 1.55; }}
  .captions > div {{ flex: 1; }}
  b {{ color: #111; }}
</style>
</head>
<body>
{chart_html}
<div class="captions">
{captions_html}
</div>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page_html)
    return path


def caption_html(kind: str, context: PlotContext | None, multivariate: bool) -> str:
    """Plain-HTML caption for a diagnostic page: what is shown, how to read
    it, and the formal test's outcome, specific to the round in `context`."""
    ctx = context
    cand = f"candidate '{ctx.candidate}'" if ctx else "the candidate"
    cond = ctx.conditioning_text() if ctx else "the conditioning set"
    test = TEST_NAMES[kind]
    tier = f"{ctx.tier} tier" if ctx else "tier"
    feats = ""
    if multivariate and ctx and ctx.feature_names:
        feats = " (" + ", ".join(ctx.feature_names) + ")"
    if kind == "linear_specification":
        shown = (f"For the {cand} and for the observations{', feature by feature' + feats if multivariate else ''}, "
                 f"the residual of its least-squares regression on {cond}, against the fitted value. The red line is "
                 "the mean residual in bins of the fitted value.")
        read = (f"If {cond} can be removed by a straight-line regression, the red line stays near zero everywhere. "
                f"A curve means part of the dependence on {cond} is left in the residuals, which the {tier}'s test "
                f"could mistake for information in the candidate. The {test} looks for any smooth nonlinear "
                f"dependence on {cond}, including in directions this plot does not show, so it can reject when the "
                "lines look flat.")
    elif kind in ("linearity", "multivariate_linearity"):
        reduce = (f" Each is first reduced to the combination of its features{feats} that best predicts the other; "
                  "the middle panel shows the weights, and the lower panels compare each feature separately, for "
                  "interpretation only." if multivariate else "")
        shown = (f"Left: the observations against the {cand}, each with {cond} removed by least squares, with "
                 "the straight-line fit (red) and the mean in bins of the candidate (orange). Right: the residuals of "
                 f"that line against its fitted values.{reduce}")
        read = ("If the relationship is a straight line, the binned mean follows the red line across the whole "
                "range, including the tails, and the trend on the right stays flat at zero. A bend is curvature a "
                f"straight line misses, which the {test} looks for; the {tier}'s statistic would then understate the "
                "candidate's contribution. A spread that widens or narrows is unequal variance, which the test "
                "allows for.")
    else:
        what = "the conditioning set's state"
        if ctx is not None:
            if ctx.selected:
                rank = "their median percentile rank" if len(ctx.selected) > 1 else "its percentile rank"
                what = f"the state of {ctx.selected_text()} ({rank})"
            elif ctx.covariate_label and "seasonal phase" in ctx.covariate_label:
                what = "the time of year"
        reduce = (f" With several features{feats}, each third is reduced to its own best-predicting combination of "
                  "features; the middle panel shows the weights, scaled to unit length per side with signs aligned "
                  "across thirds, and the lower panels compare each feature separately." if multivariate else "")
        shown = (f"The {cand} and the observations, each with {cond} removed by kernel regression, split into "
                 f"thirds by {what}, one colour per third. Left: observations against candidate, with each third's "
                 f"binned mean; right: the residuals of each third's own straight line.{reduce}")
        read = ("With no effect modification the thirds' trends coincide. Trends that differ in slope or direction "
                f"mean the candidate's relationship with the observations depends on {what}, which the {tier}'s "
                f"statistic averages over and can miss. The {test} detects any interaction of the candidate with "
                f"{cond}, not only with this summary of it, so the picture and the test can differ.")
    result = (ctx.result if ctx and ctx.result else
              "No formal test was made for this page (single-tier selection): the picture is descriptive only.")
    return (f'<div><b>What is shown.</b> {shown}</div>'
            f'<div><b>How to read it.</b> {read}</div>'
            f'<div><b>Result.</b> {result}</div>')


_PLOTTERS = {
    "linear_specification": lambda c, o, z, title, ctx: plot_regression_specification_diagnostics(
        c, o, z, title=title, feature_names=ctx.feature_names if ctx else None),
    "linearity": lambda c, o, z, title, ctx: plot_linearity_diagnostics(c, o, z, title=title),
    "multivariate_linearity": lambda c, o, z, title, ctx: plot_multivariate_linearity_diagnostics(
        c, o, z, title=title, feature_names=ctx.feature_names if ctx else None),
    "state_independence": lambda c, o, z, title, ctx: plot_state_independence_diagnostics(
        c, o, z, title=title, context=ctx),
}
PLOT_KINDS = tuple(_PLOTTERS)
_PAGE_HEADINGS = {"linear_specification": "Linear adjustment", "linearity": "Candidate linearity",
                  "multivariate_linearity": "Candidate linearity", "state_independence": "Candidate effect modification"}


def _page_path(output_dir, gage, pathway, tier_label, kind, round_label):
    safe = round_label.replace("/", "_").replace(" ", "_")
    return os.path.join(output_dir, f"{gage}_{pathway}_{tier_label}_{kind}_{safe}.html")


def make_adequacy_plot_func(test_kind: str, tier_label: str, output_dir: str, gage: str, pathway: str,
                            feature_names: tuple[str, ...] | None = None):
    """Plot function writing one page per tested round to
    {output_dir}/{gage}_{pathway}_{tier}_{kind}_{round_label}.html.

    The function takes (candidate, obs, conditioning, round_label, context);
    context (PlotContext) gives the page its specific title and caption.
    Returns the written path, or None for a round with nothing to plot
    (plots that need a non-empty conditioning set)."""
    plot = _PLOTTERS[test_kind]

    def _plot(candidate, obs, conditioning, round_label, context: PlotContext | None = None):
        if context is not None and context.feature_names is None and feature_names is not None:
            context.feature_names = feature_names
        heading = _PAGE_HEADINGS[test_kind]
        where = f"{tier_label} tier, {pathway.replace('_', '-')} pathway, gage {gage}"
        if context is not None:
            title = (f"{heading}: round {context.round_no}, {where}<br><sup>candidate '{context.candidate}' given "
                     f"{context.conditioning_text()}</sup>")
        else:
            title = f"{heading}: {where}<br><sup>{round_label}</sup>"
        try:
            fig = plot(candidate, obs, conditioning, title, context)
        except ValueError:
            return None
        multivariate = max(as2d(candidate).shape[1], as2d(obs).shape[1]) > 1
        path = _page_path(output_dir, gage, pathway, tier_label, test_kind, round_label)
        page_title = f"{heading}: {where}, {round_label}"
        return write_adequacy_page(fig, caption_html(test_kind, context, multivariate), path, page_title)

    return _plot
