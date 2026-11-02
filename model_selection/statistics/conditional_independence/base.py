"""Interface and result type shared by the three conditional independence tests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointTestResult:
    """Effect sizes of every candidate and their null draws under one shared
    resampling.

    observed : (K,) effect size per candidate, on the test's own scale.
    null : (B, K) effect sizes under H0; row b uses the same resampling for
        every candidate, so null.max(axis=1) is the null of the maximum.
    diagnostics : per-candidate dict of assumption diagnostics.
    block_length : block length (or multiplier bandwidth) of the resampling,
        in time units; 0 or 1 means samples were resampled independently.
    dependence : how the block length was obtained (ResolvedBlock.to_record).
    """
    names: list[str]
    observed: np.ndarray
    null: np.ndarray
    diagnostics: dict[str, dict] = field(default_factory=dict)
    block_length: int | None = None
    dependence: dict | None = None

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def winner(self) -> str:
        return self.names[int(np.argmax(self.observed))]


class ConditionalIndependenceTest(ABC):
    """One tier of the cascade.

    run() receives every remaining candidate, the observations and the
    selected set (None when empty), each as an (n,) or (n, d) array on the
    same samples, the samples' time positions and the dependence lag in time
    units. grid_step is the step of the regular time grid the samples lie
    on, with missing points allowed (timestep samples); None for samples not
    on a grid (event samples).
    """
    name: str = ""

    @abstractmethod
    def run(
            self,
            candidates: dict[str, np.ndarray],
            obs: np.ndarray,
            conditioning: np.ndarray | None,
            n_resamples: int,
            rng: np.random.Generator,
            time_positions: np.ndarray,
            dependence_lag: float,
            grid_step: float | None = None,
    ) -> JointTestResult:
        ...

    def magnitude(
            self,
            candidate: np.ndarray,
            obs: np.ndarray,
            conditioning: np.ndarray | None,
            time_positions: np.ndarray,
            dependence_lag: float,
            observed: float,
            null_draws: np.ndarray,
            grid_step: float | None = None,
    ) -> tuple[float, float | None]:
        """The candidate's effect size on the scale of the effect-size floor,
        and the null mean subtracted from it (None when none is).

        Default: the test's own effect size minus the mean of the candidate's
        null draws (bias correction). This is the common scale for tests whose
        effect size is the set-correlation R^2 or converts to it exactly in the
        Gaussian case; a test whose statistic does not overrides it
        (docs/methods.md, p-values and effect sizes)."""
        null_mean = float(np.mean(null_draws))
        return float(observed) - null_mean, null_mean
