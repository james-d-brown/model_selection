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
    fixed_b : (B,) the draws of the fixed-b variance functional Q that each
        row of null was divided by (multipliers.fixed_b_draws), or None.
    """
    names: list[str]
    observed: np.ndarray
    null: np.ndarray
    diagnostics: dict[str, dict] = field(default_factory=dict)
    block_length: int | None = None
    dependence: dict | None = None
    fixed_b: np.ndarray | None = None

    def index(self, name: str) -> int:
        return self.names.index(name)

    def width_draws(self, k: int) -> np.ndarray:
        """Candidate k's null draws for quantities that use their mean (the
        chance level of an effect size, the effective sample size): the
        undivided draws over the mean of Q, whose mean is the multipliers'
        variance corrected for the bandwidth's downward bias, E[Q] = 1 - b +
        b^2 / 3 (Kiefer and Vogelsang, 2005). The divided draws, which give
        the p-value, have the larger mean E[1/Q] times the undivided one.
        The null draws themselves without fixed-b."""
        column = self.null[:, k]
        if self.fixed_b is None:
            return column
        return column * self.fixed_b / float(np.mean(self.fixed_b))

    def null_degrees_of_freedom(self, k: int = 0, q: int = 1) -> float | None:
        """Degrees of freedom nu of the variance behind candidate k's null, for
        a test of q restrictions (q = the candidate's columns times the
        observations'), used in the power calculation (statistics.power). On
        a regular grid, 2 E[Q]^2 / var(Q) from the fixed-b draws, the
        chi-square whose first two moments match Q's (Satterthwaite, 1946).
        For independent units (clusters of events), G - 1 for G clusters: the
        degrees of freedom of a scale estimated from G cluster sums (Bester,
        Conley and Hansen, 2011, for one restriction). Their F(q, G - q) for q
        restrictions belongs to a Wald statistic that inverts a q x q
        variance matrix estimated from the clusters, which the tiers'
        statistics do not; for them G - q overstates the null's imprecision
        (in simulated event samples with 3 features, 15 to 30 clusters, the
        test detected an effect of the size so reported 89% to 100% of the
        time, against the intended 80%). G - 1 is also conservative there (85%
        to 93%) and close at many clusters (82% at 128). It can be zero or
        negative: no effect is then detectable. The effective number of
        clusters (diagnostics, significance.clusters.effective_clusters) is
        reported, not used here. None otherwise (the null's precision is not
        assessed). k and q are unused for clusters; q is kept for callers."""
        dep = self.dependence or {}
        if self.fixed_b is not None and self.fixed_b.size > 1 and np.var(self.fixed_b) > 0:
            return float(2.0 * np.mean(self.fixed_b) ** 2 / np.var(self.fixed_b))
        if dep.get("independent_units") and dep.get("n_blocks"):
            return float(dep["n_blocks"]) - 1.0
        return None

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
