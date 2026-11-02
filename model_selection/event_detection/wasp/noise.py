"""
The noise level of a flow series, as a function of flow.

WASP keeps a feature only where it stands well above the noise: a peak by
its prominence, an onset by its curvature. Hydrologic errors commonly grow
with flow (Schoups & Vrugt 2010), so the noise standard deviation is
modelled as linear in flow, sd(Q) = a + b Q. It is estimated from the
second-difference pseudo-residuals (x[t-1] - 2 x[t] + x[t+1]) / sqrt(6)
(Gasser, Sroka & Jennen-Steinmetz 1986), which cancel a locally linear
signal, so a smooth rise or recession does not count as noise. Within each
of NOISE_BINS equal-count flow bins the standard deviation is estimated
robustly, as the median absolute pseudo-residual / 0.6745 (Donoho &
Johnstone 1994), and a + b Q is fitted to the bin estimates. The noise is
never below the rounding error of the recorded values, step / sqrt(12),
with step the smallest nonzero change between consecutive values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_selection.event_detection.wasp.config import NOISE_BINS

_MAD_TO_SD = 0.6745
"""Median absolute value of a standard normal variable."""


@dataclass(frozen=True)
class NoiseModel:
    """Noise standard deviation sd(Q) = max(intercept + slope Q, floor)."""
    intercept: float
    slope: float
    floor: float

    def sd(self, flow) -> np.ndarray:
        """Noise standard deviation at each flow value."""
        return np.maximum(self.intercept + self.slope * np.asarray(flow, dtype=float), self.floor)


def fit_noise_model(series: np.ndarray, n_bins: int = NOISE_BINS) -> NoiseModel:
    """Fits sd(Q) = a + b Q to robust noise estimates in equal-count flow bins.

    The slope is constrained to be non-negative (noise does not shrink with
    flow); the intercept is the median of the bin estimates less the slope
    term, constrained to be non-negative.
    """
    x = np.asarray(series, dtype=float)
    if x.size < 3:
        return NoiseModel(0.0, 0.0, 0.0)
    pseudo = (x[:-2] - 2 * x[1:-1] + x[2:]) / np.sqrt(6.0)
    level = x[1:-1]

    steps = np.abs(np.diff(x))
    steps = steps[steps > 0]
    floor = float(steps.min() / np.sqrt(12.0)) if steps.size else 0.0

    bins = np.array_split(np.argsort(level, kind="stable"), min(n_bins, pseudo.size))
    bin_level = np.array([np.median(level[b]) for b in bins])
    bin_sd = np.array([np.median(np.abs(pseudo[b])) / _MAD_TO_SD for b in bins])

    slope = 0.0
    if np.ptp(bin_level) > 0:
        slope = max(float(np.polyfit(bin_level, bin_sd, 1)[0]), 0.0)
    intercept = max(float(np.median(bin_sd - slope * bin_level)), 0.0)
    return NoiseModel(intercept=intercept, slope=slope, floor=floor)
