"""
Free-form tier: conditional mutual information (CMI).

The k-nearest-neighbour estimator of Frenzel and Pompe (2007) gives CMI in
nats, with k set relative to the sample size as Runge (2018) recommends. Its null distribution uses local permutation (Runge, 2018): candidate
values are exchanged only between samples that are near neighbours in Z,
excluding neighbours closer than the dependence lag in time (Theiler, 1986).
The same Theiler window applies in the estimator: samples closer than it in
time are not each other's neighbours, so serial correlation does not inflate
the estimate (Theiler, 1986).
Before the estimate and the neighbour search, every column of the candidate,
the observations and Z is replaced by its ranks (Runge, 2018): CMI is
unchanged by such a transform, but the estimator's max-norm distances are
not, and would otherwise be dominated by the column with the widest spread.
With Z empty, a time-block permutation is used, with the Politis–White
block length for the products of the candidate and the observations on
evenly spaced samples. The local permutation treats samples as exchangeable;
on evenly spaced samples, whether they are indistinguishable from independent
sampling is estimated and recorded. The neighbour structure
depends only on Z, so all candidates share each permutation.

See docs/methods.md, Free-form tier.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from model_selection.statistics.association.information import CachedCMI
from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult
from model_selection.statistics.significance.neighborhood import compute_neighbor_index, local_permutation
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH, resolve_block, samples_independent
from model_selection.statistics.significance.resampling import time_block_permutation
from model_selection.utils import as2d


def ranks(X: np.ndarray) -> np.ndarray:
    """Each column of X replaced by the ranks of its values, 1 to n (ties,
    already broken, are ordered by position)."""
    X = as2d(X)
    return np.column_stack([rankdata(X[:, j], method="ordinal") for j in range(X.shape[1])]).astype(float)


def break_ties(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """X with tied values in any column separated by noise of 1e-10 times
    the column's standard deviation, as Kraskov et al. (2004) recommend for
    the nearest-neighbour estimator, which assumes no two samples coincide
    (e.g. event durations of zero). Columns without ties are unchanged."""
    X = np.array(as2d(X), dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.unique(col).size < col.size:
            sd = col.std()
            X[:, j] = col + 1e-10 * (sd if sd > 0 else 1.0) * rng.standard_normal(col.size)
    return X


class CMITest(ConditionalIndependenceTest):
    """KSG conditional mutual information with a local-permutation null.

    k_fraction : estimator neighbour count as a fraction of n (Runge, 2018:
        0.1 to 0.2); k, if given, fixes the count instead.
    n_neighbors : local-permutation neighbourhood size (Runge, 2018: 5 to 10).
    run_length : decorrelation-lag run length for the block permutation used
        when Z is empty (docs/methods.md, Dependence between samples).
    theiler_window : neighbours closer than this in time are not
        local-permutation partners; None uses the dependence lag. Only the
        local permutation uses it: the block permutation with Z empty keeps
        the dependence lag.
    """
    name = "cmi"

    def __init__(self, k: int | None = None, k_fraction: float = 0.1, n_neighbors: int = 10, workers: int = -1,
                 run_length: int = DEFAULT_RUN_LENGTH, theiler_window: int | None = None):
        self.run_length = run_length
        self.theiler_window = theiler_window
        self.k = k
        self.k_fraction = k_fraction
        self.n_neighbors = n_neighbors
        self.workers = workers

    def estimator_k(self, n: int) -> int:
        """Neighbour count used for a sample of size n."""
        if self.k is not None:
            return self.k
        return max(1, min(n - 1, int(round(self.k_fraction * n))))

    def _permuter(self, conditioning, n, time_positions, dependence_lag):
        if conditioning is None:
            return lambda rng: time_block_permutation(n, time_positions, dependence_lag, rng)
        window = dependence_lag if self.theiler_window is None else self.theiler_window
        nn_idx, _ = compute_neighbor_index(
            as2d(conditioning), self.n_neighbors, time_positions=time_positions,
            min_temporal_lag=int(round(window)), workers=self.workers,
        )
        return lambda rng: local_permutation(nn_idx, rng)[0]

    def _setup(self, candidates, obs, conditioning, rng, time_positions, dependence_lag, grid_step):
        """Tie-broken inputs, the estimator, the observed values, the
        permutation and the dependence record shared by run and
        sequential_p_value."""
        candidates = {nm: break_ties(c, rng) for nm, c in candidates.items()}
        obs = break_ties(obs, rng)
        conditioning = None if conditioning is None else break_ties(conditioning, rng)
        names = list(candidates)
        n = as2d(obs).shape[0]
        k = self.estimator_k(n)
        ranked = {nm: ranks(c) for nm, c in candidates.items()}
        ranked_conditioning = None if conditioning is None else ranks(conditioning)
        window = dependence_lag if self.theiler_window is None else self.theiler_window
        estimator = CachedCMI(ranks(obs), ranked_conditioning, k=k, workers=self.workers,
                              time_positions=time_positions, theiler_window=window)
        observed = np.array([float(estimator(ranked[nm])) for nm in names])
        block, dependence = None, None
        if conditioning is None:
            resolved = resolve_block(time_positions, dependence_lag,
                                     [(candidates[nm], obs) for nm in names], self.run_length, grid_step)
            block, dependence = resolved.length, resolved.to_record()
        else:
            dependence = samples_independent(time_positions, [obs] + [candidates[nm] for nm in names],
                                             self.run_length, grid_step)
        permute = self._permuter(ranked_conditioning, n, time_positions, dependence_lag if block is None else block)
        return ranked, ranked_conditioning, names, n, k, estimator, observed, block, dependence, permute

    def sequential_p_value(self, candidate, obs, conditioning, max_resamples, rng, time_positions, dependence_lag,
                           exceedances: int, grid_step=None) -> tuple[float, np.ndarray, float]:
        """Sequential Monte Carlo p-value of one candidate (Besag and
        Clifford, 1991): null draws stop once `exceedances` of them reach the
        observed value, after L draws, giving p = exceedances / L; otherwise
        all max_resamples are drawn and p = (M + 1) / (max_resamples + 1).
        The p-value is valid either way. With exceedances from
        significance.base.stopping_exceedances, the decision at alpha is the
        same as with all max_resamples draws: draws stop only once the
        candidate can no longer be significant. Returns (observed, null draws
        made, p-value)."""
        candidates, _, _, _, _, estimator, observed, _, _, permute = self._setup(
            {"c": candidate}, obs, conditioning, rng, time_positions, dependence_lag, grid_step)
        x = as2d(candidates["c"])
        obs_value = float(observed[0])
        null = []
        hits = 0
        for _ in range(max_resamples):
            value = float(estimator(x[permute(rng)]))
            null.append(value)
            hits += value >= obs_value
            if hits >= exceedances:
                return obs_value, np.array(null), exceedances / len(null)
        return obs_value, np.array(null), (hits + 1) / (max_resamples + 1)

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        candidates, conditioning, names, n, k, estimator, observed, block, dependence, permute = self._setup(
            candidates, obs, conditioning, rng, time_positions, dependence_lag, grid_step)
        null = np.empty((n_resamples, len(names)))
        for b in range(n_resamples):
            perm = permute(rng)
            for j, nm in enumerate(names):
                null[b, j] = float(estimator(as2d(candidates[nm])[perm]))
        k_dim = 0 if conditioning is None else as2d(conditioning).shape[1]
        diagnostics = {nm: {"samples_per_conditioning_dimension": n / max(k_dim, 1), "estimator_k": k,
                            "permutation_neighbours": self.n_neighbors} for nm in names}
        return JointTestResult(names, observed, null, diagnostics, block_length=block, dependence=dependence)
