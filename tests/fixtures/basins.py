"""Synthetic basins for pipeline tests: an hourly series with storm events
on a low-flow baseline, one informative formulation and one noise
formulation. The default record spans three years, so that every calendar
day recurs and the seasonal conditioning is estimable, as on a real gauge
record; storms occur once every 200 hours on average."""
import numpy as np
import pandas as pd

from model_selection.io.basin import BasinData


HOURS_PER_EVENT = 200


def make_synthetic_basin(n_hours=3 * 8760, n_events=None, seed=0):
    if n_events is None:
        n_events = max(1, round(n_hours / HOURS_PER_EVENT))
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_hours, freq="h")
    baseline = 5.0 + 0.5 * rng.normal(size=n_hours)
    baseline = np.clip(baseline, 0.1, None)
    storm_signal = np.zeros(n_hours)
    event_starts = rng.choice(np.arange(50, n_hours - 50), size=n_events, replace=False)
    for start in event_starts:
        width = rng.integers(10, 30)
        peak = rng.uniform(50, 300)
        t = np.arange(width)
        storm_signal[start:start + width] += peak * np.exp(-((t - width / 3) ** 2) / (2 * (width / 6) ** 2))
    true_flow = baseline + storm_signal
    obs = true_flow + rng.normal(scale=0.05 * (true_flow + 1))
    obs = np.clip(obs, 0, None)
    informative = true_flow * (1 + rng.normal(scale=0.1, size=n_hours))
    informative = np.clip(informative, 0, None)
    noise_formulation = np.clip(baseline + rng.normal(scale=20, size=n_hours), 0, None)
    return BasinData(
        gage="TESTGAGE01",
        formulations={
            "formulation_informative": pd.Series(informative, index=idx),
            "formulation_noise": pd.Series(noise_formulation, index=idx),
        },
        obs=pd.Series(obs, index=idx),
    )


def make_synthetic_basin_with_gap(n_hours=3 * 8760, n_events=None, seed=0, gap_start=1500, gap_hours=48):
    """Same construction as make_synthetic_basin, but with a genuine,
    multi-hour data gap spliced into every series' own index -- for
    testing the regularize-before-detection step of the high-flow pathway."""
    basin = make_synthetic_basin(n_hours=n_hours, n_events=n_events, seed=seed)
    idx = basin.obs.index
    keep_mask = ~((idx >= idx[gap_start]) & (idx < idx[gap_start] + pd.Timedelta(hours=gap_hours)))
    gapped_idx = idx[keep_mask]

    def _apply_gap(series: pd.Series) -> pd.Series:
        return series.loc[gapped_idx]

    return BasinData(
        gage=basin.gage,
        formulations={name: _apply_gap(s) for name, s in basin.formulations.items()},
        obs=_apply_gap(basin.obs),
    )
