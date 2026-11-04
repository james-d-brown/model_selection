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
    BANDWIDTH_MULTIPLIERS, ConditioningRegression, cached_regression, feature_factor_with_cap, low_rank,
    shared_regression, spectrum_tasks, standardize,
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
        self._factors: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        self.magnitude_diagnostics: dict[int, dict] = {}
        """Regression diagnostics of each effect size, by id of the candidate
        array: whether a cross-fitting record was short, and the largest share
        of a kernel's trace left out by a factor at the rank cap."""

    def _factor(self, X: np.ndarray) -> np.ndarray:
        """Kernel factor of a variable, cached by array identity: the same
        arrays recur in every round of a selection trial."""
        cached = self._factors.get(id(X))
        if cached is None or cached[0] is not X:
            cached = (X, *feature_factor_with_cap(X))
            self._factors[id(X)] = cached
        return cached[1]

    def _factor_cap(self, X: np.ndarray) -> float:
        cached = self._factors.get(id(X))
        return cached[2] if cached is not None and cached[0] is X else 0.0

    def _prepare(self, variables: list, conditioning, time_positions, dependence_lag) -> None:
        """Compute, in one parallel pool, every eigendecomposition the round
        needs that is not yet kept: the conditioning regression's, one per
        trial bandwidth, and each variable's feature factor. They are
        independent and use no random numbers, so results do not depend on
        the number of workers."""
        from model_selection.parallel import run_tasks
        tasks, slots = [], []
        if conditioning is not None and np.asarray(conditioning).size \
                and cached_regression(conditioning, time_positions, dependence_lag) is None:
            spectra = spectrum_tasks(conditioning)
            tasks += spectra
            slots += [("regression", None)] * len(spectra)
        for X in variables:
            cached = self._factors.get(id(X))
            if cached is None or cached[0] is not X:
                tasks.append(lambda X=X: feature_factor_with_cap(X))
                slots.append(("factor", X))
        if not tasks:
            return
        results = run_tasks(tasks)
        spectra = [r for (kind, _), r in zip(slots, results) if kind == "regression"]
        if spectra:
            shared_regression(conditioning, time_positions, buffer=dependence_lag, spectra=spectra)
        for (kind, X), r in zip(slots, results):
            if kind == "factor":
                self._factors[id(X)] = (X, *r)

    def magnitudes(self, candidates: dict, obs, conditioning, time_positions, dependence_lag) -> dict:
        """magnitude() of several candidates, with the eigendecompositions of
        their regressions on (Z, C) computed in shared parallel pools, in
        batches sized to the workers so that memory in flight is bounded."""
        from model_selection.parallel import batches, parallel_map, run_tasks
        out = {}
        reduced = self._regression(conditioning, time_positions, dependence_lag)
        for group in batches(list(candidates), len(BANDWIDTH_MULTIPLIERS)):
            regressors = {nm: (as2d(candidates[nm]) if reduced is None
                               else np.column_stack([as2d(conditioning), as2d(candidates[nm])])) for nm in group}
            todo = [nm for nm in group if cached_regression(regressors[nm], time_positions, dependence_lag) is None]
            tasks = [task for nm in todo for task in spectrum_tasks(regressors[nm])]
            results = run_tasks(tasks) if tasks else []
            k = len(BANDWIDTH_MULTIPLIERS)
            full = {nm: ConditioningRegression(regressors[nm], time_positions, buffer=dependence_lag,
                                               spectra=results[i * k:(i + 1) * k]) for i, nm in enumerate(todo)}
            for nm in group:
                if nm not in full:
                    full[nm] = shared_regression(regressors[nm], time_positions, buffer=dependence_lag)
            values = parallel_map(lambda nm: self.magnitude(candidates[nm], obs, conditioning, time_positions,
                                                            dependence_lag, 0.0, np.zeros(1), full=full[nm]), group)
            out.update(zip(group, values))
        return out

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
        return shared_regression(conditioning, time_positions, buffer=dependence_lag)

    def magnitude(self, candidate, obs, conditioning, time_positions, dependence_lag, observed, null_draws,
                  grid_step=None, full: ConditioningRegression | None = None):
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
        if full is None:
            full = shared_regression(regressors, time_positions, buffer=dependence_lag)
        E_f = full.residuals(own_o)[0]
        regs = [full] + ([] if reduced is None else [reduced])
        self.magnitude_diagnostics[id(candidate)] = {
            "crossfit_blocks_short": any(r.blocks_short for r in regs),
            "low_rank_cap_share": max(r.cap_share for r in regs),
            "low_rank_rank_change": max(r.rank_change for r in regs),
        }
        sign_r, logdet_r = np.linalg.slogdet(E_r.T @ E_r)
        sign_f, logdet_f = np.linalg.slogdet(E_f.T @ E_f)
        if sign_r <= 0 or sign_f <= 0:
            return 0.0, None
        return float(1.0 - np.exp(logdet_f - logdet_r)), None

    def run(self, candidates, obs, conditioning, n_resamples, rng, time_positions, dependence_lag, grid_step=None):
        names = list(candidates)
        n = as2d(obs).shape[0]
        self._prepare([obs] + [candidates[nm] for nm in names], conditioning, time_positions, dependence_lag)
        regression = self._regression(conditioning, time_positions, dependence_lag)
        E_o, own_o, r2_o = self._residuals(obs, regression)
        # In the low-rank path the statistic is computed from the per-sample
        # vectors u_i = e_C,i (x) e_O,i (Kronecker product of the residual
        # coordinates), whose inner products are the entries of A below, so
        # that no n x n matrix is formed; the two forms are equal.
        features = low_rank() is not None and n > low_rank()
        G_o = None if features else E_o @ E_o.T
        norm_o = float(np.linalg.norm(E_o.T @ E_o)) if features else float(np.sqrt(np.sum(G_o ** 2)))
        from model_selection.parallel import parallel_map
        fitted = dict(zip(names, parallel_map(lambda nm: self._residuals(candidates[nm], regression), names)))

        def feature_terms(E_c):
            """(sum of the u_i, centred u_i as rows, normaliser)."""
            F = (E_c[:, :, None] * E_o[:, None, :]).reshape(n, -1)
            total = F.sum(axis=0)
            return total, F - total / n, float(np.linalg.norm(E_c.T @ E_c)) * norm_o

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
            time_positions, dependence_lag,
            [(lambda nm=nm: feature_terms(fitted[nm][0])[1]) if features else centred_products(fitted[nm][0])[1]
             for nm in names],
            self.run_length, grid_step)
        block = resolved.length
        weights = dependent_wild_weights(time_positions, block, n_resamples, rng)

        observed = np.empty(len(names))
        null = np.empty((n_resamples, len(names)))
        diagnostics = {}
        k = 0 if conditioning is None else as2d(conditioning).shape[1]
        for j, nm in enumerate(names):
            E_c, _, r2_c = fitted[nm]
            if features:
                total, Fc, denom = feature_terms(E_c)
            else:
                A, P, denom = centred_products(E_c)
            if denom > 0 and features:
                observed[j] = float(total @ total) / denom
                null[:, j] = np.sum((weights @ Fc) ** 2, axis=1) / denom
            elif denom > 0:
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
        dependence["crossfit_blocks_short"] = bool(regression is not None and regression.blocks_short)
        dependence["low_rank_cap_share"] = 0.0 if regression is None else regression.cap_share
        dependence["low_rank_rank_change"] = 0.0 if regression is None else regression.rank_change
        dependence["feature_factor_cap_share"] = max(self._factor_cap(X) for X in [obs] + [candidates[nm] for nm in names])
        return JointTestResult(names, observed, null, diagnostics, block_length=block, dependence=dependence)
