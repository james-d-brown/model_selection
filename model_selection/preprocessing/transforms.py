"""
Transforms applied before association estimation (pooled pathway), and to
the series from which the dependence lag is estimated (both pathways).

prepare_series gives the series the pooled tests analyse: the signed square
root of flow, sign(x) |x|^(1/2), standardized to zero mean and unit variance
over its values. It is the Box-Cox power transform with exponent 1/2 (Box and
Cox, 1964) in the signed form of Bickel and Doksum (1981), defined for zero
and negative flow, which keep their value and sign; zero flows are a point
mass at one value. Standardization makes the result independent of the flow
units. Every test conditions on the seasonal phase (preprocessing.seasonal).

dependence_series gives the series from which the dependence lag is
estimated: the series the tests analyse (prepare_series), adjusted for the
time of year when the tests condition on it, by subtracting its fit on the
seasonal phase (preprocessing.seasonal.phase_residuals).

See docs/methods.md, Samples.
"""
from __future__ import annotations

import numpy as np

EXPONENT = 0.5
"""Exponent of the signed power transform (the square root)."""


def signed_power(x: np.ndarray, exponent: float = EXPONENT) -> np.ndarray:
    """sign(x) |x|^exponent; NaN stays NaN."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.abs(x) ** exponent


def standardize_series(x: np.ndarray) -> np.ndarray:
    """x centred and scaled to unit standard deviation over its finite
    values (a constant series is centred only); NaN stays NaN."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if not ok.any():
        return out
    centred = x[ok] - x[ok].mean()
    sd = centred.std()
    out[ok] = centred / sd if sd > 0 else centred
    return out


def prepare_series(x: np.ndarray) -> np.ndarray:
    """The standardized signed square root of flow. Deterministic."""
    return standardize_series(signed_power(x))


def dependence_series(x: np.ndarray, index=None, seasonal: bool = False) -> np.ndarray:
    """The series the tests analyse (prepare_series), minus its fit on the
    seasonal phase when `seasonal` (index supplies the timestamps): the
    series from which the dependence lag is estimated."""
    z = prepare_series(x)
    if not seasonal:
        return z
    from model_selection.preprocessing.seasonal import phase_residuals
    if index is None:
        raise ValueError("adjusting for the time of year needs the series' timestamps (index)")
    return phase_residuals(z, index)
