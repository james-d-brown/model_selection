"""
Transforms applied before association estimation (pooled pathway, and the
series from which the dependence lag is estimated in both pathways).

prepare_series applies, in order:

1. log(x + offset): streamflow is skewed and its seasonal variation largely
   multiplicative, so its seasonal mean is described in log space.
   Because the last step depends only on the ordering of values, the log
   matters through step 2.
2. Optionally, the series' calendar-day mean is removed
   (preprocessing.seasonal), as part of conditioning every test on the time
   of year.
3. A rank / normal-score (Gaussian copula) transform, putting every series on
   a common, unitless scale with its dependence structure unchanged. Tied
   values (zero flow, quantized readings) are ordered at random, so that the
   transformed series has no ties, as the nearest-neighbour CMI estimator
   requires; the order among tied values is arbitrary (Genest and
   Nešlehová, 2007), which is why series with much zero flow are flagged.

See docs/methods.md, Samples.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def log_transform(x: np.ndarray, offset: float | None = None) -> np.ndarray:
    """log(x + offset), with negative values clipped to zero. offset=None uses
    1% of the smallest positive value."""
    x = np.asarray(x, dtype=np.float64)
    if offset is None:
        pos = x[x > 0]
        offset = 0.01 * np.min(pos) if pos.size else 1e-6
    return np.log(np.clip(x, 0, None) + offset)


def rank_normal_transform(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Empirical rank -> standard normal quantile, (rank - 0.5) / n. Tied
    values receive distinct ranks in random order; NaN stays NaN."""
    x = np.asarray(x, dtype=np.float64)
    rng = rng or np.random.default_rng()
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    values = x[ok]
    n = values.size
    if n == 0:
        return out
    order = np.lexsort((rng.random(n), values))  # by value, ties broken at random
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    out[ok] = norm.ppf((ranks - 0.5) / n)
    return out


def prepare_series(x: np.ndarray, rng: np.random.Generator | None = None,
                   index=None, seasonal_window_days: int | None = None) -> np.ndarray:
    """log -> optional removal of the calendar-day mean (when
    seasonal_window_days is given; index supplies the timestamps) ->
    rank/normal score with random tie-breaking."""
    x = log_transform(x)
    if seasonal_window_days is not None:
        from model_selection.preprocessing.seasonal import calendar_day_anomalies
        if index is None:
            raise ValueError("removing the calendar-day mean needs the series' timestamps (index)")
        x = calendar_day_anomalies(x, index, seasonal_window_days)
    return rank_normal_transform(x, rng)
