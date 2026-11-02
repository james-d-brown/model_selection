"""
Nonlinear tier: nonlinear-regression residual test.

Each variable's kernel feature map (its standardized columns plus a radial
basis function kernel, regression.kernel_residuals.feature_kernel) is
regressed on Z by kernel ridge regression with cross-fitting, and the
residual cross-covariance operator is measured by its squared
Hilbert-Schmidt norm, normalised as an RV coefficient (Robert and Escoufier,
1976). Every quantity is computed from the residual feature maps' inner
products with the exact kernels. The construction follows the generalised
Hilbertian covariance measure (Lundborg et al., 2022) and its kernel
extension (Bergen et al., 2026). Under H0 every residual product has zero
conditional mean, so a multiplier bootstrap of the products gives the null
distribution without resampling the candidate; the multipliers follow the
dependent wild bootstrap (Shao, 2010). All candidates share each set of
multipliers. On samples on a regular time grid the multipliers' bandwidth is
the Politis–White block length of the statistic's own per-sample terms, the
residual feature-map products, from the trace of their autocovariance
matrices (significance.block_length.trace_autocovariance).

See docs/methods.md, Nonlinear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.conditional_independence.base import ConditionalIndependenceTest, JointTestResult
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights
from model_selection.statistics.regression.kernel_residuals import (
    ConditioningRegression, feature_kernel, kernel_factor, standardize,
)
from model_selection.statistics.significance.block_length import (
    DEFAULT_RUN_LENGTH, resolve_block_from_inner_products,
)
from model_selection.utils import as2d


class NonlinearResidualTest(ConditionalIndependenceTest):
    """Residual cross-covariance of kernel feature maps after kernel ridge
    removal of Z, calibrated by a multiplier bootstrap.

    run_length : decorrelation-lag run length for the multiplier bandwidth
        (docs/methods.md, Dependence between samples).
    """
    name = "nonlinear"

    def __init__(self, run_length: int = DEFAULT_RUN_LENGTH):
        self.run_length = run_length
        self._factors: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _factor(self, X: np.ndarray) -> np.ndarray:
        """Kernel factor of a variable, cached by array identity: the same
        arrays recur in every round of a selection trial."""
        cached = self._factors.get(id(X))
        if cached is None or cached[0] is not X:
            cached = (X, kernel_factor(feature_kernel(X)))
            self._factors[id(X)] = cached
        return cached[1]

    def _residuals(self, X: np.ndarray, regression: ConditioningRegression | None):
        """(centred residual coordinates of the feature map, residuals of the
        variable's own standardized columns, cross-fitted R^2 of those columns)."""
        L = self._factor(X)
        own = standardize(X)
        if regression is None:
            return L - L.mean(axis=0), own, 0.0
        E, E_own = regression.residuals(L, own)
        tss = float(np.sum(own ** 2))
        r2_own = 1.0 - float(np.sum(E_own ** 2)) / tss if tss > 0 else 0.0
        return E - E.mean(axis=0), E_own - E_own.mean(axis=0), r2_own

    def _regression(self, conditioning, time_positions, dependence_lag) -> ConditioningRegression | None:
        """Cross-fitted regression on the conditioning set, reused for the
        same arrays (the significance test and the magnitude of its winner)."""
        if conditioning is None or np.asarray(conditioning).size == 0:
            return None
        key = (conditioning, time_positions, dependence_lag)
        cached = getattr(self, "_last_regression", None)
        if cached is None or cached[0][0] is not conditioning or cached[0][1] is not time_positions \
                or cached[0][2] != dependence_lag:
            cached = (key, ConditioningRegression(conditioning, time_positions, buffer=dependence_lag))
            self._last_regression = cached
        return cached[1]

    def magnitude(self, candidate, obs, conditioning, time_positions, dependence_lag, observed, null_draws,
                  grid_step=None):
        """The candidate's share of the variation in the observations left
        unexplained by Z: one minus the ratio of the generalized variances
        (determinants of the residual covariance matrices) of the
        observations' cross-fitted residuals on (Z, C) and on Z, both by the
        tier's kernel ridge regression. For linear conditional means this is
        the set-correlation R^2 of the linear tier (Cohen, 1982). Cross-fitted
        residuals carry no in-sample optimism, so no null mean is
        subtracted."""
        own_o = standardize(obs)
        reduced = self._regression(conditioning, time_positions, dependence_lag)
        E_r = own_o - own_o.mean(axis=0) if reduced is None else reduced.residuals(own_o)[0]
        regressors = as2d(candidate) if reduced is None else np.column_stack([as2d(conditioning), as2d(candidate)])
        E_f = ConditioningRegression(regressors, time_positions, buffer=dependence_lag).residuals(own_o)[0]
        sign_r, logdet_r = np.linalg.slogdet(E_r.T @ E_r)
        sign_f, logdet_f = np.linalg.slogdet(E_f.T @ E_f)
        if sign_r <= 0 or sign_f <= 0:
            return 0.0, None
        return float(1.0 - np.exp(logdet_f - logdet_r)), None

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        names = list(candidates)
        n = as2d(obs).shape[0]
        regression = self._regression(conditioning, time_positions, dependence_lag)
        E_o, own_o, r2_o = self._residuals(obs, regression)
        G_o = E_o @ E_o.T
        norm_o = float(np.sqrt(np.sum(G_o ** 2)))
        fitted = {nm: self._residuals(candidates[nm], regression) for nm in names}

        def centred_products(E_c):
            """Inner products of the per-sample terms of the statistic (the
            residual products, centred at their mean) and the normaliser."""
            G_c = E_c @ E_c.T
            A = G_c * G_o
            row = A.mean(axis=1)
            return A, A - row[:, None] - row[None, :] + A.mean(), float(np.sqrt(np.sum(G_c ** 2))) * norm_o

        # Multiplier bandwidth from the persistence of the statistic's own
        # per-sample terms (all candidates share one set of multipliers).
        resolved = resolve_block_from_inner_products(
            time_positions, dependence_lag, [centred_products(fitted[nm][0])[1] for nm in names],
            self.run_length, grid_step)
        block = resolved.length
        weights = dependent_wild_weights(time_positions, block, n_resamples, rng)

        observed = np.empty(len(names))
        null = np.empty((n_resamples, len(names)))
        diagnostics = {}
        k = 0 if conditioning is None else as2d(conditioning).shape[1]
        for j, nm in enumerate(names):
            E_c, _, r2_c = fitted[nm]
            A, P, denom = centred_products(E_c)
            if denom > 0:
                observed[j] = float(A.sum()) / denom
                null[:, j] = np.einsum("bi,bi->b", weights @ P, weights) / denom
            else:
                observed[j] = 0.0
                null[:, j] = 0.0
            diagnostics[nm] = {
                "r2_candidate_on_selected": r2_c,
                "r2_obs_on_selected": r2_o,
                "samples_per_conditioning_dimension": n / max(k, 1),
            }
        dependence = resolved.to_record()
        dependence["crossfit_buffer_dropped_samples"] = 0 if regression is None else regression.buffer_dropped_samples
        return JointTestResult(names, observed, null, diagnostics, block_length=block, dependence=dependence)
