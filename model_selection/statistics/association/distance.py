"""
Distance-covariance building blocks (Székely and Rizzo, 2014): pairwise
distance matrices, U-centring and the U-centred inner product, from which
the bias-corrected squared distance correlation is formed. The formulas
match the `dcor` package to machine precision (see the tests).
"""
from __future__ import annotations

import numpy as np

from model_selection.utils import as2d


def pairwise_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """(n, n) Euclidean distance matrix of the rows of X (n, d)."""
    X = as2d(X)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=-1))


def u_center(a: np.ndarray) -> np.ndarray:
    """U-centring of a distance matrix (Szekely & Rizzo 2013/2014), which makes
    u_product an unbiased estimator of squared distance covariance. Requires n > 2.
    """
    n = a.shape[0]
    row_sum = a.sum(axis=1)
    col_sum = a.sum(axis=0)
    total = a.sum()
    A = a - row_sum[:, None] / (n - 2) - col_sum[None, :] / (n - 2) + total / ((n - 1) * (n - 2))
    np.fill_diagonal(A, 0.0)
    return A


def u_product(A: np.ndarray, B: np.ndarray) -> float:
    """Inner product of two U-centred matrices. Requires n > 3; can be slightly
    negative at finite n (the unbiased estimator is not positive semi-definite).
    """
    n = A.shape[0]
    return float(np.sum(A * B)) / (n * (n - 3))
