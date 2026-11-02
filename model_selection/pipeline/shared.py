"""Steps used by both the pooled and the high-flow pathways."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model_selection.io.basin import OBS_KEY, BasinData, complete_mask, on_regular_grid
from model_selection.pipeline.config import OutputConfig
from model_selection.selection.tiers import AdequacyPlotFactory

ZERO_FLOW_FLAG_THRESHOLD = 0.10
"""Zero-flow fraction above which a series is flagged (KSG ties)."""


logger = logging.getLogger(__name__)


def resolve_rng(rng: np.random.Generator | None, seed: int | None,
                gage: str, pathway: str) -> tuple[np.random.Generator, int | None]:
    """The run's random generator and its seed. A generator passed by the
    caller is used as given (seed None); otherwise the configured seed, or a
    freshly drawn one, which is logged so the run can be repeated."""
    if rng is not None:
        return rng, None
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    logger.info(f"[i] {gage} {pathway}: random seed {seed} (SelectionConfig(seed={seed}) repeats this run)")
    return np.random.default_rng(seed), seed


def common_basis(basin: BasinData, notes: list[str]) -> tuple[BasinData, dict]:
    """The basin on its regular time grid with every series set to missing
    (NaN) wherever any series (observations or a formulation) lacks a value,
    so that every step that compares formulations uses the same timesteps for
    every series, and a summary of what was excluded. Event detection is the
    exception: it uses each series' own values (docs/methods.md, Missing
    values).

    Returns (basin, summary). summary: n_grid_timesteps, n_common_timesteps,
    missing_by_series (each series' own missing timesteps), and
    excluded_by_formulation: for each formulation, the timesteps it removes
    from the common basis where the observations have values (expected to be
    zero; formulation_missing is True when any is not).
    """
    basin = on_regular_grid(basin)
    common = complete_mask([basin.obs.values] + [s.values for s in basin.formulations.values()])
    obs_present = np.isfinite(basin.obs.values.astype(float))
    missing = {OBS_KEY: int((~obs_present).sum())}
    excluded = {}
    for name, s in basin.formulations.items():
        absent = ~np.isfinite(s.values.astype(float))
        missing[name] = int(absent.sum())
        excluded[name] = int((absent & obs_present).sum())
    summary = {
        "n_grid_timesteps": int(common.size), "n_common_timesteps": int(common.sum()),
        "missing_by_series": missing, "excluded_by_formulation": excluded,
        "formulation_missing": any(v > 0 for v in excluded.values()),
    }
    if common.sum() < common.size:
        notes.append(f"Common basis: {int(common.sum())} of {common.size} timesteps have values for every series; "
                     "every series is set to missing at the others for the whole analysis (missing timesteps per "
                     "series: " + ", ".join(f"{k}={v}" for k, v in missing.items()) + ").")
    masked = BasinData(gage=basin.gage,
                       formulations={k: s.where(common) for k, s in basin.formulations.items()},
                       obs=basin.obs.where(common))
    return masked, summary


def zero_flow_check(basin: BasinData, notes: list[str]) -> tuple[dict[str, float], bool]:
    """Fraction of zero-or-negative flow among each series' available values
    (obs under OBS_KEY) and whether any exceeds ZERO_FLOW_FLAG_THRESHOLD
    (noted if so)."""
    def fraction(series: pd.Series) -> float:
        values = series.dropna()
        return float((values <= 0).mean()) if len(values) else 0.0

    fractions = {OBS_KEY: fraction(basin.obs)}
    for name, series in basin.formulations.items():
        fractions[name] = fraction(series)
    flag = any(f > ZERO_FLOW_FLAG_THRESHOLD for f in fractions.values())
    if flag:
        notes.append(
            "high zero-flow fraction detected in at least one series "
            f"(threshold={ZERO_FLOW_FLAG_THRESHOLD}); tied zero flows are ordered at random by the rank "
            "transform, which imposes an arbitrary order among them (docs/methods.md, Samples)."
        )
    return fractions, flag


def max_decorrelation_lag(series_by_name: dict[str, np.ndarray], run_length: int) -> tuple[int, dict[str, int]]:
    """The dependence lag (timesteps): the largest Politis decorrelation lag
    across the formulations' series, observation-free, with each
    formulation's value. The largest is used because a lag that separates
    samples should be chosen generously (Hegger et al., 1999). The series are
    on the regular grid; NaN marks missing values, which the estimate skips
    (docs/methods.md, Dependence between samples)."""
    from model_selection.statistics.significance.block_length import estimate_dependence
    lags = {}
    for name, values in series_by_name.items():
        lags[name] = int(estimate_dependence(np.asarray(values, dtype=float), run_length).decorrelation_lag)
    return max(lags.values()), lags


def log_review_flags(result, pathway: str) -> None:
    """Log every review flag of a pathway result as a WARNING, prominently."""
    import logging

    from model_selection.pipeline.results import review_flags

    flags = review_flags(result)
    if not flags:
        return
    log = logging.getLogger("model_selection.review")
    log.warning(f"*** REVIEW REQUIRED: {result.gage} {pathway} pathway, {len(flags)} flag(s) ***")
    for f in flags:
        log.warning(f"*** [{f['code']}] {f['message']}")


def write_dependence_plot(output: OutputConfig, gage: str, prepared: dict, observations, index,
                          run_length: int, seasonal_window_days: int | None) -> str | None:
    """Write {gage}_dependence.html (visualization.dependence) when an output
    directory is set, the plot is enabled and the series are time-indexed;
    returns its path."""
    import pandas as pd

    if (not (output.dependence_plot and output.output_dir is not None) or len(index) < 2
            or not isinstance(index, pd.DatetimeIndex)):  # lags are drawn in days
        return None
    import os

    from model_selection.visualization.dependence import plot_dependence
    from model_selection.visualization.html import write_standalone_html

    step_hours = (index[1] - index[0]) / np.timedelta64(1, "h")
    path = os.path.join(output.output_dir, f"{gage}_dependence.html")
    write_standalone_html(plot_dependence(prepared, step_hours, gage, observations, run_length,
                                          seasonal_window_days), path,
                          f"{gage} dependence")
    return path


def adequacy_plot_factory(output: OutputConfig, gage: str, pathway: str,
                          feature_names: tuple[str, ...] | None = None) -> AdequacyPlotFactory | None:
    """Plot factory for the cascade's adequacy checks, or None when plots are
    disabled or there is no output directory. feature_names label the
    columns of multivariate samples."""
    if not (output.assumption_plots and output.output_dir is not None):
        return None
    from model_selection.visualization.adequacy import make_adequacy_plot_func

    def _factory(test_kind: str, tier_label: str):
        return make_adequacy_plot_func(test_kind, tier_label, output.output_dir, gage, pathway, feature_names)

    return _factory


def pearson_with_observations(obs: np.ndarray, formulations: dict[str, np.ndarray],
                              mask: np.ndarray | None = None) -> dict[str, float]:
    """Pearson correlation of each formulation's flow with the observed flow,
    at the time steps in mask where every series has a value. Reported for
    comparison with an evaluation that scores each formulation on its own;
    selection does not use it."""
    obs = np.asarray(obs, dtype=float)
    keep = np.isfinite(obs) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(obs))
    for values in formulations.values():
        keep &= np.isfinite(np.asarray(values, dtype=float))
    out = {}
    for name, values in formulations.items():
        x, y = np.asarray(values, dtype=float)[keep], obs[keep]
        out[name] = float(np.corrcoef(x, y)[0, 1]) if keep.sum() > 2 and x.std() > 0 and y.std() > 0 else float("nan")
    return out


def log_pearson(gage: str, pathway: str, what: str, r: dict[str, float],
                r_transformed: dict[str, float] | None = None) -> None:
    """Log each formulation's Pearson correlation with the observations,
    ranked largest first: on the raw flow and, when given, on the series as
    transformed for selection (pearson_with_observations)."""
    def ranked(d):
        order = sorted(d.items(), key=lambda kv: (-np.nan_to_num(kv[1], nan=-np.inf), kv[0]))
        return ", ".join(f"{k} {v:.3f}" for k, v in order)
    if not r:
        return
    text = (f"[i] {gage} {pathway}: each formulation on its own, as a traditional evaluation would rank them. "
            f"Pearson correlation with the observations on the raw flow ({what}): {ranked(r)}.")
    if r_transformed:
        text += (" Pearson correlation on the series as transformed for selection (log flow less its calendar-day "
                 f"mean, as normal scores, at the same time steps): {ranked(r_transformed)}.")
    text += " Selection instead asks what each formulation adds beyond those already selected."
    logger.info(text)
