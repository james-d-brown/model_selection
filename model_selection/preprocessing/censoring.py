"""
Censoring of series at high flow (docs/methods.md, Censoring at high flow).

A series is censored at a high quantile of its own record: every value below
the quantile is set to the quantile value itself. Censoring is applied to
the series as the pooled pathway transforms it
(preprocessing.transforms.prepare_series), which orders the values as flow
does, so the censored values lie at the quantile, adjacent to the values
above it, as in the threshold-weighted CRPS (Gneiting and
Ranjan, 2011; Allen et al., 2023). A function of the candidate and a
function of the observations are conditionally independent whenever the
candidate and the observations are (Dawid, 1979), so censoring them keeps
the null hypothesis.

Censoring : the configuration, one quantile shared with the high-flow
    events (EventConfig.high_flow_quantile).
CensoredSeries : what a censored sample carries besides the censored
    values: each series' uncensored values and its threshold, so that
    surrogate candidates can be built uncensored and then censored at the
    candidate's own threshold.
censor_and_record, censoring_record : a censored array is recorded with its
    uncensored values and threshold, so that the nonlinear tier can estimate
    its conditional means by fitting the uncensored values and then applying
    the censoring (statistics.regression.censored). The record is kept while
    the array exists and is looked up by the array itself.
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass

import numpy as np

from model_selection.utils import as2d

# High flow relative to each series: shared by the high-flow events
# (EventConfig.high_flow_quantile) and censoring at high flow.
DEFAULT_HIGH_FLOW_QUANTILE = 0.90


@dataclass(frozen=True)
class Censoring:
    """Censoring of the candidate and the observations at a quantile of
    each series' own record; the selected formulations are not censored.

    quantile : in (0, 1); the high-flow quantile of the high-flow events
        by default, so one setting defines high flow in both
        (Censoring.at_high_flow(events) uses an EventConfig's).
    """
    quantile: float = DEFAULT_HIGH_FLOW_QUANTILE

    def __post_init__(self):
        q = float(self.quantile)
        if not (np.isfinite(q) and 0.0 < q < 1.0):
            raise ValueError(f"Censoring.quantile must lie in (0, 1) -- got {self.quantile!r}")

    @classmethod
    def at_high_flow(cls, events) -> "Censoring":
        """Censoring at the high-flow quantile of an EventConfig."""
        return cls(quantile=events.high_flow_quantile)

    def threshold(self, values: np.ndarray) -> float:
        """The series' quantile over its finite values (linear interpolation
        between order statistics, numpy's default)."""
        v = np.asarray(values, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            raise ValueError("cannot censor a series with no finite values")
        return float(np.quantile(v, self.quantile))

    @staticmethod
    def apply(values: np.ndarray, threshold: float) -> np.ndarray:
        """Every value below the threshold set to the threshold; NaN stays NaN."""
        v = np.asarray(values, dtype=float)
        return np.where(np.isnan(v), np.nan, np.maximum(v, threshold))


_RECORDS: dict[int, tuple] = {}
_RECORDS_LOCK = threading.Lock()


def censor_and_record(uncensored: np.ndarray, threshold: float) -> np.ndarray:
    """The values censored at the threshold (Censoring.apply), recorded with
    their uncensored values and the threshold."""
    uncensored = np.asarray(uncensored, dtype=float)
    censored = Censoring.apply(uncensored, threshold)
    key = id(censored)

    def _forget(_ref, key=key):
        with _RECORDS_LOCK:
            entry = _RECORDS.get(key)
            if entry is not None and entry[0] is _ref:
                del _RECORDS[key]

    with _RECORDS_LOCK:
        _RECORDS[key] = (weakref.ref(censored, _forget), uncensored, float(threshold))
    return censored


def censoring_record(values) -> tuple[np.ndarray, float] | None:
    """(uncensored values, threshold) of an array made by censor_and_record,
    or None for any other array."""
    with _RECORDS_LOCK:
        entry = _RECORDS.get(id(values))
    if entry is None or entry[0]() is not values:
        return None
    return entry[1], entry[2]


@dataclass(frozen=True)
class CensoredSeries:
    """The uncensored values and the censoring threshold of each censored
    candidate (by name) and of the observations, aligned with the sample's
    rows."""
    uncensored: dict[str, np.ndarray]
    thresholds: dict[str, float]
    obs_threshold: float
    obs_uncensored: np.ndarray | None = None

    def censor(self, name: str, values: np.ndarray) -> np.ndarray:
        """`values` (e.g. a surrogate of candidate `name`, uncensored) censored
        at that candidate's threshold, and recorded (censor_and_record)."""
        return censor_and_record(values, self.thresholds[name])

    def censored_candidates(self) -> dict[str, np.ndarray]:
        """Every candidate censored at its threshold, and recorded."""
        return {k: self.censor(k, v) for k, v in self.uncensored.items()}

    def censored_obs(self) -> np.ndarray:
        """The observations censored at their threshold, and recorded."""
        if self.obs_uncensored is None:
            raise ValueError("the observations' uncensored values are not available")
        return censor_and_record(self.obs_uncensored, self.obs_threshold)

    def subset(self, idx: np.ndarray) -> "CensoredSeries":
        """The same series at rows idx; the thresholds stay those of the
        whole record."""
        return CensoredSeries({k: v[idx] for k, v in self.uncensored.items()}, dict(self.thresholds),
                              self.obs_threshold, None if self.obs_uncensored is None else self.obs_uncensored[idx])

    def shares_at_threshold(self, censored: dict[str, np.ndarray], obs: np.ndarray) -> dict[str, float]:
        """Share of each censored series (and of the observations, key
        "observations") at its threshold."""
        out = {k: float(np.mean(as2d(v)[:, 0] <= self.thresholds[k])) for k, v in censored.items()}
        out["observations"] = float(np.mean(as2d(obs)[:, 0] <= self.obs_threshold))
        return out
