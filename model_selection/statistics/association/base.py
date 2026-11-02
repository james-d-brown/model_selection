"""
Common interface for association estimators.

An estimator is constructed once per greedy-selection round with the
observations and the conditioning set (the already-selected models) fixed,
then called repeatedly with only the candidate varying. This lets every
estimator cache whatever depends on obs/conditioning, and lets the
selection and significance-testing code work with any measure without
knowing which one it holds.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


class CachedAssociationEstimator(ABC):
    """Association between a candidate and a fixed (obs, conditioning) pair.

    Subclasses capture obs/conditioning at construction. The obs and
    conditioning arguments of __call__ exist only for signature compatibility
    and are ignored.
    """

    @abstractmethod
    def __call__(
            self,
            candidate: np.ndarray,
            obs: np.ndarray | None = None,
            conditioning: np.ndarray | None = None,
    ) -> float:
        ...

@dataclass
class StructuralCheckOutcome:
    """Shape-only well-posedness check for a measure (no data required)."""
    well_posed: bool
    reason: str


StructuralCheckFunc = Callable[[int, int, int, int], StructuralCheckOutcome]
"""structural_check(n, p, q, k) -> StructuralCheckOutcome, where n is the
sample count, p and q the candidate and obs dimensionality, and k the
conditioning dimensionality (0 when unconditional)."""
