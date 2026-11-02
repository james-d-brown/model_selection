"""Small array helpers shared across the package."""
from __future__ import annotations

import numpy as np


def as2d(a: np.ndarray) -> np.ndarray:
    """Return `a` as a float64 2-D array: (n,) -> (n, 1); (n, d) unchanged."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return a
