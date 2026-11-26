"""
Known-answer basins for the forward selection, the backward pass and the
comparisons (docs/methods.md, Backward pass; Interchangeable formulations).
A diagnostic only: the package is not changed.

Each replicate builds a synthetic daily basin whose minimal set of
formulations is known, runs the pooled pathway (default: the full cascade,
with the backward pass and the comparisons), and reports the forward
selection, the final set, what the backward pass removed, the classes, the
number of tests the post-selection steps made, and the time.

Cases (--case):

  simple      observations = seasonal cycle + baseflow + quickflow + noise;
              1 baseflow, 2 quickflow, 3 quickflow + noise (redundant given 2),
              4 constant + noise (uninformative), 5 baseflow + quickflow + noise
              (informative, redundant given 1 and 2). Minimal set {1, 2}.
  many        observations = seasonal cycle + baseflow + quickflow + interflow
              + noise (the baseflow's spread doubled, so that each component is
              detectable in 20 years), eight formulations, the composites noisy:
              1 baseflow, 2 quickflow, 3 interflow, 4 baseflow + quickflow +
              noise, 5 quickflow + interflow + noise, 6 a near-copy of
              quickflow (exchangeable with 2), 7 noise (uninformative),
              8 baseflow + quickflow + interflow + noise. Minimal set {1, 2, 3},
              with 6 in place of 2 equally correct.

  many_censored  the eight formulations of "many", built so that the minimal
              set is the same with and without censoring at a high quantile: a
              wet season raises storms, interflow and baseflow together, so
              each component is high when the observations are high (each
              contributes about a fifth or more of the mean high flow); the
              composites and the noisy copy carry multiplicative errors (a
              relative error of 0.3), so they are as degraded at high flows as
              at low flows; the observations carry a multiplicative error
              (0.15); 7 is an independent flow series. Minimal set {1, 2, 3},
              with 6 in place of 2 equally correct, in both pathways.

  many_linear  the roles of "many", additive on the square-root scale on
              which the pathway analyses every series (every flow is the square
              of its square-root-scale value), so the linear tier holds by
              construction; components as in many_censored; noise sd 0.5 on
              the square-root scale for the noisy formulations and the
              observations. Minimal set {1, 2, 3}, with 6 in place of 2.

With --censored, the censored pooled pathway runs instead: the candidates
and the observations are censored at their own --quantile (default 0.9),
the selected formulations enter the conditioning set uncensored
(docs/methods.md, Censoring at high flow). High observed flow depends on
every component, so the minimal sets are the same; power is lower, so a
final set that lacks a weak component (most likely the slow baseflow) is a
power shortfall, not an error. So is a noisy composite that stays in place
of a component the data cannot single out: given the formulations that were
selected, it still adds information. What must hold is that no
uninformative formulation is selected, and that every formulation in the
final set adds information given the others.

A replicate is correct when its final set is a minimal set. The run also
counts the final sets that contain an uninformative formulation (4 in
"simple", 7 in "many") and those that keep a formulation that is
redundant given the minimal set (a composite, or a noisy copy).

Each finished stage of a replicate (preparation, each round's tests, the
backward pass, the comparisons) is printed as it finishes, with its time,
and each replicate ends with its time by stage. With --profile, each
replicate also runs under Python's profiler: its statistics are written to
known_answer_<case>_seed<seed>.prof and the functions with the most
cumulative time are printed.

Plots show a case's construction, not a run: one realization (seed
PLOT_SEED), and nothing else is done. --plot writes an interactive page of
the observations, the components and every formulation, coloured by role
(known_answer_<case>.html); --plot-static writes one calendar year
(PLOT_YEAR) of the observations and the components as a figure
(known_answer_<case>.svg; needs matplotlib), as in docs/methods.md,
Appendix G. With --censored, both show each series' censoring level at
--quantile.

Usage, from the project root:
    python -m scripts.diagnostics.known_answer_check --case simple --reps 6 --workers 6
    python -m scripts.diagnostics.known_answer_check --case many --reps 6 --workers 6
    python -m scripts.diagnostics.known_answer_check --case many --reps 3 --workers 3 --censored [--quantile 0.9]
    python -m scripts.diagnostics.known_answer_check --case many_censored --plot-static --censored   (no run)
    [--years 20] [--measure cascade] [--n-perm 499] [--profile] [--quiet]
Each worker is single-threaded; one Ctrl+C stops it and reports the
replicates already finished.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, logging, multiprocessing as mp, signal, time, traceback, warnings
import numpy as np
import pandas as pd
from scipy.signal import lfilter

CASES = {
    "simple": {"minimal": [{"1_baseflow", "2_quickflow"}],
                  "uninformative": {"4_constant_noise"},
                  "redundant": {"5_base_quick_noise", "3_quickflow_noise"}},
    "many": {"minimal": [{"1_baseflow", "2_quickflow", "3_interflow"},
                         {"1_baseflow", "6_quickflow_copy", "3_interflow"}],
             "uninformative": {"7_noise"},
             "redundant": {"4_base_quick_noise", "5_quick_inter_noise", "8_all_noise"}},
    "many_censored": {"minimal": [{"1_baseflow", "2_quickflow", "3_interflow"},
                                  {"1_baseflow", "6_quickflow_copy", "3_interflow"}],
                      "uninformative": {"7_unrelated_flow"},
                      "redundant": {"4_base_quick_mult", "5_quick_inter_mult", "8_all_mult"}},
    "many_linear": {"minimal": [{"1_baseflow", "2_quickflow", "3_interflow"},
                                {"1_baseflow", "6_quickflow_copy", "3_interflow"}],
                    "uninformative": {"7_unrelated_flow"},
                    "redundant": {"4_base_quick_noise", "5_quick_inter_noise", "8_all_noise"}},
}


def _components(rng, n, idx, base_spread=0.3):
    seasonal = 10 + 5 * np.sin(2 * np.pi * idx.dayofyear.values / 365.25)
    base = 5 * np.exp(base_spread * lfilter([1], [1, -0.995], rng.normal(scale=0.1, size=n)))
    pulses = np.where(rng.random(n) < 0.05, rng.exponential(30, size=n), 0.0)
    quick = lfilter([1], [1, -np.exp(-1 / 3)], pulses) + 0.01
    # Interflow: a slower response to its own, partly shared, recharge pulses.
    recharge = np.where(rng.random(n) < 0.03, rng.exponential(20, size=n), 0.0) + 0.3 * pulses
    inter = 0.15 * lfilter([1], [1, -np.exp(-1 / 20)], recharge) + 0.01
    return seasonal, base, quick, inter


def _wet_season_parts(seed: int, years: int, relative_error: float = 0.3, obs_error: float = 0.15) -> dict:
    """The "many_censored" case: built so that its minimal set is the same with
    and without censoring at a high quantile.

    - A wet season (a smooth annual weight, highest with the seasonal cycle)
      raises the storm frequency, the interflow recharge and the baseflow
      together, so every component is high when the observations are high:
      each component's own top decile overlaps the observations' top decile,
      and each contributes materially to high flows.
    - The noisy formulations (composites, a copy) carry multiplicative
      errors, a lognormal factor with the given relative error, so they are
      as degraded relative to the exact components at high flows as at low
      flows: wherever the observations are high, the exact components carry
      strictly more information than any composite.
    - The observations carry a multiplicative error too (heteroscedastic, as
      gauge and model errors are).
    - The uninformative formulation is an independent flow series (its own
      storms, routed), flow-like but unrelated to the observations."""
    rng = np.random.default_rng(seed)
    n = int(years * 365.25)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    phase = 2 * np.pi * idx.dayofyear.values / 365.25
    seasonal = 10 + 5 * np.sin(phase)
    wet = 0.5 * (1 + np.sin(phase))
    base = 10 * np.exp(0.4 * lfilter([1], [1, -0.995], rng.normal(scale=0.1, size=n))) * (0.3 + 1.4 * wet)
    pulses = np.where(rng.random(n) < 0.02 + 0.08 * wet, rng.exponential(30, size=n), 0.0)
    quick = lfilter([1], [1, -np.exp(-1 / 3)], pulses) + 0.01
    recharge = np.where(rng.random(n) < 0.01 + 0.05 * wet, rng.exponential(20, size=n), 0.0) + 0.3 * pulses
    inter = 0.35 * lfilter([1], [1, -np.exp(-1 / 20)], recharge) + 0.01
    mult = lambda v, sd=relative_error: v * np.exp(sd * rng.normal(size=n) - sd ** 2 / 2)
    obs = mult(seasonal + base + quick + inter, obs_error)
    other = lfilter([1], [1, -np.exp(-1 / 3)], np.where(rng.random(n) < 0.05, rng.exponential(30, size=n), 0.0))
    s = lambda v: pd.Series(np.clip(v, 0.1, None), index=idx)
    forms = {"1_baseflow": s(base), "2_quickflow": s(quick), "3_interflow": s(inter),
             "4_base_quick_mult": s(mult(base + quick)), "5_quick_inter_mult": s(mult(quick + inter)),
             "6_quickflow_copy": s(mult(quick, 0.01)), "7_unrelated_flow": s(10 + other),
             "8_all_mult": s(mult(base + quick + inter))}
    return {"index": idx, "components": {"seasonal cycle": seasonal, "baseflow": base, "quickflow": quick,
                                         "interflow": inter},
            "obs": pd.Series(np.clip(obs, 0.1, None), index=idx), "formulations": forms}


def _square_root_scale_parts(seed: int, years: int, noise_sd: float = 0.5, obs_sd: float = 0.5) -> dict:
    """The "many_linear" case: the roles of "many", additive on the square-root
    scale on which the pathway analyses every series, so that the
    observations' conditional mean is linear in the formulations there and
    the linear tier holds by construction.

    The components are the square roots of the "many_censored" components
    (a wet season raising storms, interflow and baseflow together). On the
    square-root scale the observations are their sum plus the seasonal cycle
    plus noise (sd obs_sd), and each noisy formulation is its components'
    sum plus noise (sd noise_sd); every flow is the square of its
    square-root-scale value. The near-copy carries noise of 0.01 noise_sd,
    and the uninformative formulation is an independent flow series. Flows
    are kept at least 0.0025 (a square-root-scale floor of 0.05), which the
    noisy composites reach on about 1.5% of days."""
    parts = _wet_season_parts(seed, years)
    rng = np.random.default_rng(seed + 10_000)
    idx = parts["index"]
    n = len(idx)
    c = {k: np.sqrt(v) for k, v in parts["components"].items()}
    noise = lambda sd: rng.normal(scale=sd, size=n)
    flow = lambda r: pd.Series(np.square(np.clip(r, 0.05, None)), index=idx)
    obs_root = c["seasonal cycle"] + c["baseflow"] + c["quickflow"] + c["interflow"] + noise(obs_sd)
    other = parts["formulations"]["7_unrelated_flow"]
    forms = {"1_baseflow": flow(c["baseflow"]), "2_quickflow": flow(c["quickflow"]),
             "3_interflow": flow(c["interflow"]),
             "4_base_quick_noise": flow(c["baseflow"] + c["quickflow"] + noise(noise_sd)),
             "5_quick_inter_noise": flow(c["quickflow"] + c["interflow"] + noise(noise_sd)),
             "6_quickflow_copy": flow(c["quickflow"] + noise(0.01 * noise_sd)), "7_unrelated_flow": other,
             "8_all_noise": flow(c["baseflow"] + c["quickflow"] + c["interflow"] + noise(noise_sd))}
    return {"index": idx, "components": {k: np.square(v) for k, v in c.items()}, "obs": flow(obs_root),
            "formulations": forms}


def basin_parts(case: str, seed: int, years: int, noise_sd: float = 2.0) -> dict:
    """The case's components, observations and formulations (make_basin's
    construction), for plotting: {"index", "components", "obs", "formulations"}."""
    if case == "many_censored":
        return _wet_season_parts(seed, years)
    if case == "many_linear":
        return _square_root_scale_parts(seed, years)
    rng = np.random.default_rng(seed)
    n = int(years * 365.25)
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    # "many" doubles the baseflow's spread, so that each component's information is detectable in 20 years.
    seasonal, base, quick, inter = _components(rng, n, idx, base_spread=0.6 if case == "many" else 0.3)
    noise = lambda sd=noise_sd: rng.normal(scale=sd, size=n)
    s = lambda v: pd.Series(np.clip(v, 0.1, None), index=idx)
    if case == "simple":
        obs = seasonal + base + quick + noise()
        forms = {"1_baseflow": s(base), "2_quickflow": s(quick), "3_quickflow_noise": s(quick + noise()),
                 "4_constant_noise": s(10 + noise()), "5_base_quick_noise": s(base + quick + noise())}
    elif case == "many":
        obs = seasonal + base + quick + inter + noise()
        forms = {"1_baseflow": s(base), "2_quickflow": s(quick), "3_interflow": s(inter),
                 "4_base_quick_noise": s(base + quick + noise()), "5_quick_inter_noise": s(quick + inter + noise()),
                 "6_quickflow_copy": s(quick + noise(0.02 * noise_sd)), "7_noise": s(10 + noise()),
                 "8_all_noise": s(base + quick + inter + noise())}
    else:
        raise ValueError(f"unknown case {case!r}")
    components = {"seasonal cycle": seasonal, "baseflow": base, "quickflow": quick}
    if case == "many":
        components["interflow"] = inter
    return {"index": idx, "components": components, "obs": pd.Series(np.clip(obs, 0.1, None), index=idx),
            "formulations": forms}


def make_basin(case: str, seed: int, years: int, noise_sd: float = 2.0):
    from model_selection.io.basin import BasinData
    parts = basin_parts(case, seed, years, noise_sd)
    return BasinData(gage=f"{case.upper()}{seed}", formulations=parts["formulations"], obs=parts["obs"])


ROLES = {
    "simple": {"1_baseflow": "minimal set", "2_quickflow": "minimal set",
                  "3_quickflow_noise": "noisy copy (redundant)", "4_constant_noise": "uninformative",
                  "5_base_quick_noise": "noisy composite (redundant)"},
    "many": {"1_baseflow": "minimal set", "2_quickflow": "minimal set", "3_interflow": "minimal set",
             "4_base_quick_noise": "noisy composite (redundant)", "5_quick_inter_noise": "noisy composite (redundant)",
             "6_quickflow_copy": "near-copy of 2 (exchangeable)", "7_noise": "uninformative",
             "8_all_noise": "noisy composite (redundant)"},
    "many_censored": {"1_baseflow": "minimal set", "2_quickflow": "minimal set", "3_interflow": "minimal set",
                      "4_base_quick_mult": "noisy composite (redundant)",
                      "5_quick_inter_mult": "noisy composite (redundant)",
                      "6_quickflow_copy": "near-copy of 2 (exchangeable)", "7_unrelated_flow": "uninformative",
                      "8_all_mult": "noisy composite (redundant)"},
    "many_linear": {"1_baseflow": "minimal set", "2_quickflow": "minimal set", "3_interflow": "minimal set",
                    "4_base_quick_noise": "noisy composite (redundant)",
                    "5_quick_inter_noise": "noisy composite (redundant)",
                    "6_quickflow_copy": "near-copy of 2 (exchangeable)", "7_unrelated_flow": "uninformative",
                    "8_all_noise": "noisy composite (redundant)"},
}
PLOT_SEED = 1          # the realization the plots show
PLOT_YEAR = 2008       # the calendar year of --plot-static (the record starts in 2000)

ROLE_COLOURS = {"minimal set": "#1f77b4", "near-copy of 2 (exchangeable)": "#17becf",
                "noisy composite (redundant)": "#ff7f0e", "noisy copy (redundant)": "#ff7f0e",
                "uninformative": "#7f7f7f"}
COMPONENT_COLOURS = {"seasonal cycle": "#9467bd", "baseflow": "#2ca02c", "quickflow": "#d62728",
                     "interflow": "#8c564b"}


def plot_case(case: str, seed: int, years: int, path: str, quantile: float | None = None,
              window_years: float = 2.0) -> None:
    """One page of linked time-series panels for a case: the observations;
    the components that generate them; and each formulation, coloured by its
    role in the known answer. The view opens on the last window_years of the
    record (zoom out to see all of it). With quantile, each censored series
    (the observations and the formulations) shows its censoring level as a
    dashed line."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from model_selection.visualization.html import write_standalone_html
    parts = basin_parts(case, seed, years)
    idx, forms = parts["index"], parts["formulations"]
    names = list(forms)
    titles = ["Observations", "Components (the observations are their sum plus noise)"] + [
        f"{nm}: {ROLES[case][nm]}" for nm in names]
    rows = len(titles)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.25 / rows,
                        subplot_titles=titles)

    def threshold(row, values):
        if quantile is not None:
            level = float(np.quantile(values, quantile))
            fig.add_trace(go.Scatter(x=[idx[0], idx[-1]], y=[level, level], mode="lines", showlegend=False,
                                     line=dict(color="black", width=1, dash="dash"),
                                     hovertemplate=f"censoring level (q = {quantile:g}): {level:.2f}<extra></extra>"),
                          row=row, col=1)

    fig.add_trace(go.Scatter(x=idx, y=parts["obs"].values, name="observations", mode="lines",
                             line=dict(color="black", width=1)), row=1, col=1)
    threshold(1, parts["obs"].values)
    for nm, v in parts["components"].items():
        fig.add_trace(go.Scatter(x=idx, y=v, name=nm, mode="lines", line=dict(color=COMPONENT_COLOURS[nm], width=1)),
                      row=2, col=1)
    for r, nm in enumerate(names, start=3):
        role = ROLES[case][nm]
        fig.add_trace(go.Scatter(x=idx, y=forms[nm].values, name=nm, mode="lines", showlegend=False,
                                 line=dict(color=ROLE_COLOURS[role], width=1)), row=r, col=1)
        threshold(r, forms[nm].values)
    start = idx[-1] - pd.Timedelta(days=int(365.25 * window_years))
    fig.update_xaxes(range=[start, idx[-1]])
    fig.update_yaxes(title_text="flow", title_font_size=10)
    mode = f", censored at the {quantile:g} quantile" if quantile is not None else ""
    fig.update_layout(height=170 * rows + 120, width=1300, template="plotly_white",
                      title=f"Known-answer basin '{case}', seed {seed}{mode}: " + "; ".join(
                          f"{colour}, {meaning}" for colour, meaning, present in (
                              ("blue", "the minimal set", True), ("orange", "redundant", True),
                              ("grey", "uninformative", True),
                              ("teal", "exchangeable", any("exchangeable" in r for r in ROLES[case].values())))
                          if present),
                      margin=dict(t=150), legend=dict(orientation="h", yanchor="bottom", y=1.0 + 45 / (170 * rows),
                                                      x=0))
    write_standalone_html(fig, path, f"Known-answer basin {case}, seed {seed}")


def plot_static(case: str, seed: int, years: int, path: str, quantile: float | None = None,
                year: int = PLOT_YEAR) -> None:
    """One calendar year of a case as a figure (SVG): the observations, with
    their censoring level when quantile is given, above the components that
    generate them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    parts = basin_parts(case, seed, years)
    idx, obs = parts["index"], parts["obs"].values
    m = idx.year == year
    if not m.any():
        raise ValueError(f"the record ({idx[0].year}-{idx[-1].year}) does not include {year}")
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "svg.fonttype": "none"})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    top.plot(idx[m], obs[m], color="#222222", lw=0.9, label="Observations")
    if quantile is not None:
        top.axhline(float(np.quantile(obs, quantile)), color="#b2182b", lw=0.9, ls="--",
                    label=f"{100 * quantile:g}th percentile of the record")
    colours = {"baseflow": "#2166ac", "interflow": "#1b7837", "quickflow": "#e08214"}
    for name, colour in colours.items():
        if name in parts["components"]:
            bottom.plot(idx[m], parts["components"][name][m], color=colour, lw=0.9, label=name.capitalize())
    bottom.plot(idx[m], parts["components"]["seasonal cycle"][m], color="#888888", lw=0.9, ls=":",
                label="Seasonal cycle")
    for ax, ncol in ((top, 2), (bottom, 4)):
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Flow")
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _post_selection_tests(sel) -> dict:
    """The tests the backward pass and the comparisons ran (implied and reused results excluded)."""
    backward = sum(1 for r in sel.backward if not r["reused_forward"] and r["p_value"] is not None)
    comparisons = sum(len(rec["comparisons"]) for rec in sel.equivalence)
    sets = 0
    for s in sel.equivalent_sets:
        sets += sum(1 for t in s["members"].values() if not t.get("implied"))
        beyond = s.get("selected_adds_beyond_set")
        sets += int(beyond is not None and "by" not in beyond)
    return {"backward": backward, "comparisons": comparisons, "sets": sets}


def _stage_kind(label: str) -> str:
    """A stage's kind, for the time by stage: the tier and the kind of test."""
    for key, kind in (("preparation", "preparation"), ("backward pass", "backward pass"),
                      ("comparisons", "comparisons"), ("equally supported", "equally supported sets")):
        if key in label:
            return kind
    tier = label.split(" round ")[0] if " round " in label else ""
    for key, kind in (("significance test", "significance tests"), ("effect sizes", "effect sizes"),
                      ("validity test", "validity tests"), ("power tests", "power tests"),
                      ("effective sample size", "effective sample size")):
        if key in label:
            return f"{tier}: {kind}" if tier else kind
    return "other"


def _stage_logging(seed: int, quiet: bool) -> None:
    """Print the package's stage times (model_selection.timing) as they come, and nothing else."""
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("model_selection").setLevel(logging.ERROR)   # the package's own messages and flags
    timing_logger = logging.getLogger("model_selection.timing")
    timing_logger.handlers.clear()
    timing_logger.propagate = False
    if quiet:
        timing_logger.setLevel(logging.WARNING)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"    seed {seed}: %(message)s"))
    timing_logger.addHandler(handler)
    timing_logger.setLevel(logging.INFO)


def run_replicate(task):
    case, seed, years, measure, n_perm, profile, quiet, quantile = task
    warnings.simplefilter("ignore")
    _stage_logging(seed, quiet)
    try:
        from model_selection.pipeline import run_pooled_analysis
        from model_selection.selection import SelectionConfig
        t0 = time.time()
        cfg = SelectionConfig(measure=measure, n_permutations=n_perm, seed=seed, n_jobs=1)
        basin = make_basin(case, seed, years)
        censoring = None
        if quantile is not None:
            from model_selection.preprocessing import Censoring
            censoring = Censoring(quantile)
        run = lambda b, c: run_pooled_analysis(b, c, censoring=censoring)
        top = None
        if profile:
            import cProfile, io, pstats
            prof = cProfile.Profile()
            res = prof.runcall(run, basin, cfg)
            prof.dump_stats(f"known_answer_{case}_seed{seed}.prof")
            out = io.StringIO()
            pstats.Stats(prof, stream=out).sort_stats("cumulative").print_stats(30)
            top = out.getvalue()
        else:
            res = run(basin, cfg)
        sel = res.selection
        by_kind = {}
        for t in sel.timings:
            if t["stage"] in ("backward pass", "comparisons with the formulations not selected"):
                continue  # totals of stages whose parts are recorded too
            k = _stage_kind(t["stage"])
            by_kind[k] = by_kind.get(k, 0.0) + t["seconds"]
        return {"seed": seed, "committed": res.committed_tier, "forward": list(sel.forward_selected),
                "final": list(sel.selected),
                "removed": [(r["formulation"], round(r["p_value"], 4)) for r in sel.backward if r["removed"]],
                "steps": [(st.tier, st.formulation, st.accepted, round(st.p_value, 4)) for st in sel.trajectory],
                "classes": sel.classes, "tests": _post_selection_tests(sel),
                "minutes": (time.time() - t0) / 60, "by_kind": by_kind, "profile": top}
    except Exception:
        return {"seed": seed, "error": traceback.format_exc()}


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=tuple(CASES), default="simple")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--measure", default="cascade")
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--profile", action="store_true", help="run each replicate under Python's profiler")
    ap.add_argument("--quiet", action="store_true", help="do not print each stage as it finishes")
    ap.add_argument("--plot", action="store_true",
                    help="write the case's series (observations, components, formulations) to "
                         "known_answer_<case>.html, and do not run")
    ap.add_argument("--plot-static", action="store_true",
                    help="write one calendar year of the observations and components to "
                         "known_answer_<case>.svg (needs matplotlib), and do not run")
    ap.add_argument("--censored", action="store_true", help="the censored pooled pathway")
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="with --censored, the quantile at which each series is censored (default 0.9)")
    a = ap.parse_args()
    spec = CASES[a.case]
    n_cpu = os.cpu_count() or 2
    workers = min(a.workers or max(1, n_cpu // 2), max(1, n_cpu - 1), a.reps)
    if a.censored and not 0.0 < a.quantile < 1.0:
        ap.error("--quantile must lie in (0, 1)")
    quantile = a.quantile if a.censored else None
    if a.plot or a.plot_static:
        tag = f"_censored{a.quantile:g}" if a.censored else ""
        for wanted, draw, ext in ((a.plot, plot_case, "html"), (a.plot_static, plot_static, "svg")):
            if wanted:
                path = f"known_answer_{a.case}{tag}.{ext}"
                draw(a.case, PLOT_SEED, a.years, path, quantile)
                print(f"  wrote {path}", flush=True)
        return
    tasks = [(a.case, s, a.years, a.measure, a.n_perm, a.profile, a.quiet, quantile)
             for s in range(a.first_seed, a.first_seed + a.reps)]
    rows, t0 = [], time.time()
    pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)
    try:
        it = pool.imap_unordered(run_replicate, tasks)
        while len(rows) < len(tasks):
            try:
                r = it.next(timeout=1.0)
                rows.append(r)
                print(f"  seed {r['seed']} done ({len(rows)}/{len(tasks)})  [{(time.time() - t0) / 60:.1f} min]",
                      flush=True)
            except mp.TimeoutError:
                continue
        pool.close()
    except KeyboardInterrupt:
        print("\nInterrupted: stopping workers.", flush=True)
        pool.terminate()
    pool.join()
    minimal = spec["minimal"]
    mode = f"censored at the {a.quantile:g} quantile" if a.censored else "uncensored"
    print(f"\n{a.case}, {mode}: {a.measure}, {a.years} years daily, {a.n_perm} resamples; minimal set(s) "
          + " or ".join(str(sorted(m)) for m in minimal))
    for r in sorted(rows, key=lambda r: r["seed"]):
        if "error" in r:
            print(f"\nseed {r['seed']}: ERROR\n{r['error']}")
            continue
        print(f"\nseed {r['seed']}: committed tier {r['committed']}  [{r['minutes']:.1f} min]")
        for st in r["steps"]:
            print(f"   {st[0]:9} {st[1]:24} accepted={st[2]!s:5} p={st[3]}")
        print(f"   forward {r['forward']} -> final {r['final']}  correct: {set(r['final']) in minimal}")
        if r["removed"]:
            print("   removed by the backward pass: " + ", ".join(f"{m} (p={p})" for m, p in r["removed"]))
        c = r["classes"] or {}
        print(f"   classes: distinct {c.get('distinct')}, exchangeable {c.get('exchangeable')}, "
              f"redundant {c.get('redundant')}, not tested {c.get('not_tested')}")
        print(f"   post-selection tests: {r['tests']}")
        total = sum(r["by_kind"].values())
        print("   time by stage: " + "; ".join(f"{k} {v / 60:.1f} min ({100 * v / total:.0f}%)" for k, v in
                                              sorted(r["by_kind"].items(), key=lambda kv: -kv[1])))
        if r["profile"]:
            print(r["profile"])
    ok = [r for r in rows if "error" not in r]
    if ok:
        bad = spec["uninformative"]
        print(f"\nForward selection minimal: {sum(set(r['forward']) in minimal for r in ok)}/{len(ok)}; "
              f"final set minimal: {sum(set(r['final']) in minimal for r in ok)}/{len(ok)}; "
              f"uninformative selected: {sum(bool(bad & set(r['final'])) for r in ok)}/{len(ok)}; "
              f"redundant left: {sum(bool(spec['redundant'] & set(r['final'])) for r in ok)}/{len(ok)}; "
              f"median minutes per replicate: {np.median([r['minutes'] for r in ok]):.1f}")


if __name__ == "__main__":
    main()
