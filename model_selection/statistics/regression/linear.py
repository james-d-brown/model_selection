"""Ordinary least squares with intercept, for multi-column targets."""
from __future__ import annotations

import numpy as np

from model_selection.utils import as2d


def _design(Z: np.ndarray | None, n: int) -> np.ndarray:
    if Z is None or np.asarray(Z).size == 0:
        return np.ones((n, 1))
    return np.column_stack([np.ones(n), as2d(Z)])


def ols_fit(Y: np.ndarray, Z: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Fitted values and residuals of every column of Y regressed on Z with an
    intercept. Z=None regresses on the intercept alone. Returns (fitted, residuals)."""
    Y = as2d(Y)
    X = _design(Z, Y.shape[0])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    fitted = X @ beta
    return fitted, Y - fitted


