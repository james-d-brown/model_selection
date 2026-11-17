"""
Free-form tier: conditional mutual information (CMI).

The k-nearest-neighbour estimator of Mesner and Shalizi (2021) gives CMI in
nats, with k set relative to the sample size as Runge (2018) recommends. It
extends the Frenzel and Pompe (2007) estimator that Runge uses to data with
ties, such as a censoring threshold,
and remains consistent there (statistics.association.information). Its null distribution uses local permutation (Runge, 2018): candidate
values are exchanged only between samples that are near neighbours in Z,
excluding neighbours closer than the dependence lag in time (Theiler, 1986);
a sample may keep its own value, as in Runge's implementation.
The same Theiler window applies in the estimator: samples closer than it in
time are not each other's neighbours, so serial correlation does not inflate
the estimate (Theiler, 1986).
Before the estimate, every column of the candidate, the observations and Z
is standardized to mean zero and unit variance. CMI
is unchanged by such a rescaling, but the estimator's max-norm distances are
not, and would otherwise be dominated by the column with the widest spread.
A fixed linear rescaling keeps the estimator in the setting of Mesner and
Shalizi (2021), who estimate on the values themselves: tied values stay
tied, so a point mass stays one value, and no other ties are created. Ranks
(Runge, 2018) are not used: integer ranks put distinct samples at exactly
equal distances, which the Mesner-Shalizi estimator counts as it counts a
point mass. Ties are not broken with noise, which would make the estimator
inconsistent (Gao et al., 2017). The local permutation's neighbour search in
Z uses the ranks of Z, as in Runge (2018), tied values sharing their average
rank: it counts nothing at a boundary distance, so rank ties there only
leave the order among equidistant neighbours arbitrary.
With Z empty, a time-block permutation is used, with blocks as long as the
dependence of the products of the candidate and the observations on evenly
spaced samples: the shorter of the two series' decorrelation lags
(significance.block_length), beyond which the products' autocovariance
vanishes under the null. The local permutation treats samples as exchangeable;
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
    """Each column of X replaced by the ranks of its values, 1 to n; tied
    values share their average rank."""
    X = as2d(X)
    return np.column_stack([rankdata(X[:, j], method="average") for j in range(X.shape[1])]).astype(float)


def standardize(X: np.ndarray) -> np.ndarray:
    """Each column of X centred and scaled to unit variance; a constant
    column is only centred. Tied values stay tied."""
    X = np.asarray(as2d(X), dtype=float)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


class CMITest(ConditionalIndependenceTest):
    """Nearest-neighbour conditional mutual information (Mesner and Shalizi,
    2021) with a local-permutation null.

    k_fraction : estimator neighbour count as a fraction of n (Runge, 2018:
        0.1 to 0.2); k, if given, fixes the count instead.
    n_neighbors : local-permutation neighbourhood size, counting the sample
        itself, which may keep its own value (Runge, 2018: 5 to 10; 5 is the
        default of his implementation, tigramite's CMIknn). At least 2.
    run_length : decorrelation-lag run length for the block permutation used
        when Z is empty (docs/methods.md, Dependence between samples).
    theiler_window : neighbours closer than this in time are not
        local-permutation partners; None uses the dependence lag. Only the
        local permutation uses it: the block permutation with Z empty keeps
        the dependence lag.
    """
    name = "cmi"

    def __init__(self, k: int | None = None, k_fraction: float = 0.1, n_neighbors: int = 5, workers: int = -1,
                 run_length: int = DEFAULT_RUN_LENGTH, theiler_window: int | None = None):
        if n_neighbors < 2:
            raise ValueError("n_neighbors counts the sample itself and must be at least 2")
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
            as2d(conditioning), self.n_neighbors - 1, time_positions=time_positions,
            min_temporal_lag=int(round(window)), workers=self.workers,
        )
        return lambda rng: local_permutation(nn_idx, rng)[0]

    def _setup(self, candidates, obs, conditioning, rng, time_positions, dependence_lag, grid_step):
        """Standardized inputs, the estimator, the observed values, the
        permutation and the dependence record shared by run and
        sequential_p_value."""
        names = list(candidates)
        n = as2d(obs).shape[0]
        k = self.estimator_k(n)
        scaled = {nm: standardize(c) for nm, c in candidates.items()}
        scaled_conditioning = None if conditioning is None else standardize(conditioning)
        window = dependence_lag if self.theiler_window is None else self.theiler_window
        estimator = CachedCMI(standardize(obs), scaled_conditioning, k=k, workers=self.workers,
                              time_positions=time_positions, theiler_window=window)
        observed = np.array([float(estimator(scaled[nm])) for nm in names])
        block, dependence = None, None
        if conditioning is None:
            resolved = resolve_block(time_positions, dependence_lag,
                                     [(candidates[nm], obs) for nm in names], self.run_length, grid_step)
            # Blocks as long as the products' dependence, the lag beyond which it
            # vanishes under the null (the shorter factor's decorrelation lag),
            # so that whole blocks exchanged are independent. The multipliers'
            # testing bandwidth (resolved.length) sets a variance estimate's
            # weights, not the extent of the dependence, and is not used here.
            block = resolved.decorrelation_lag if resolved.estimated else resolved.length
            # A permutation of G units reaches p-values down to 1 / G!, not the
            # sign-flip floor 2^(1 - G) of the multiplier tests.
            dependence = {**resolved.to_record(), "block_length": block, "smallest_p_value": None}
        else:
            dependence = samples_independent(time_positions, [obs] + [candidates[nm] for nm in names],
                                             self.run_length, grid_step)
        permute = self._permuter(None if conditioning is None else ranks(conditioning), n, time_positions,
                                 dependence_lag if block is None else block)
        return scaled, scaled_conditioning, names, n, k, estimator, observed, block, dependence, permute

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
