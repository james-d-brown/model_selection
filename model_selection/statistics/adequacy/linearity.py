"""
Linearity adequacy tests for the linear-regression tier (partial canonical
correlation, PC).

LinearityTest (scalar candidate/obs): Ramsey's RESET (1969) applied to the
residuals of candidate and obs, each regressed on the conditioning set.
Testing residuals isolates exactly what partial correlation assumes: once
the conditioning set's linear effect is removed from both sides, the
remaining relationship is linear. RESET needs only fitted values, so its
power does not degrade with conditioning dimension.

MultivariateLinearityTest (multivariate candidate/obs, e.g. FPC scores):
a kernel-machine score test. Candidate and obs are residualized on the
conditioning set; the null model is a kernel ridge fit with the linear
kernel K_lin = C C'; the score is trace(r' K_rbf r) on the null residuals r,
so any structure the RBF kernel finds is structure a linear model cannot
explain.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.adequacy.base import AdequacyTestFunc, FunctionalFormTest, as_adequacy_test
from model_selection.statistics.adequacy.kernels import (
    kernel_ridge_fit,
    median_heuristic_bandwidth,
    rbf_kernel,
    reml_kernel_ridge_lambda,
    variance_shares,
)
from model_selection.statistics.association.canonical import residualize_on_conditioning
from model_selection.statistics.regression.linear import ols_fit
from model_selection.statistics.significance.block_length import DEFAULT_RUN_LENGTH
from model_selection.utils import as2d


def _ols_leverage(regressors: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix of least squares on an intercept and the
    regressors."""
    X = np.column_stack([np.ones(as2d(regressors).shape[0]), as2d(regressors)])
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, sv > sv[0] * max(X.shape) * np.finfo(float).eps]
    return np.sum(U ** 2, axis=1)


def ols_residuals(y: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """OLS residuals of scalar y regressed on Z with an intercept; Z=None returns
    y minus its mean.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    if Z is None:
        design = np.ones((n, 1))
    else:
        Z = as2d(Z)
        design = np.hstack([np.ones((n, 1)), Z])
    coefs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefs
    return y - fitted


def reset_statistic(x_resid: np.ndarray, y_resid: np.ndarray, powers: tuple[int, ...] = (2, 3)) -> tuple[float, int, int, float, float]:
    """RESET F-statistic for whether powers of the fitted values of
    y_resid ~ x_resid add explanatory power beyond the linear fit.

    Returns (F, df_num, df_den, partial_r2, semipartial_r2). partial_r2 is the
    nonlinear terms' share of the variance the linear fit left unexplained;
    semipartial_r2 is their share of all variance in y_resid (Cohen & Cohen
    1983), the decision-relevant scale, since partial_r2 can look large merely
    because the linear fit left little residual variance.
    """
    x_resid = np.asarray(x_resid, dtype=np.float64).ravel()
    y_resid = np.asarray(y_resid, dtype=np.float64).ravel()
    n = x_resid.shape[0]

    # Restricted model: y_resid ~ intercept + x_resid
    design_restricted = np.column_stack([np.ones(n), x_resid])
    coefs_r, _, _, _ = np.linalg.lstsq(design_restricted, y_resid, rcond=None)
    fitted_r = design_restricted @ coefs_r
    rss_restricted = float(np.sum((y_resid - fitted_r) ** 2))

    # Augmented model: y_resid ~ intercept + x_resid + fitted_r^2 + fitted_r^3 + ...
    power_terms = np.column_stack([fitted_r ** p for p in powers])
    design_augmented = np.column_stack([design_restricted, power_terms])
    coefs_a, _, _, _ = np.linalg.lstsq(design_augmented, y_resid, rcond=None)
    fitted_a = design_augmented @ coefs_a
    rss_augmented = float(np.sum((y_resid - fitted_a) ** 2))

    df_num = len(powers)
    df_den = n - design_augmented.shape[1]

    delta_rss = rss_restricted - rss_augmented
    partial_r2 = delta_rss / rss_restricted if rss_restricted > 0 else 0.0
    partial_r2 = float(min(max(partial_r2, 0.0), 1.0))

    tss = float(np.sum((y_resid - y_resid.mean()) ** 2))
    semipartial_r2 = delta_rss / tss if tss > 0 else 0.0
    semipartial_r2 = float(min(max(semipartial_r2, 0.0), 1.0))

    if df_den <= 0 or rss_augmented <= 0:
        return 0.0, df_num, max(df_den, 1), partial_r2, semipartial_r2

    f_stat = (delta_rss / df_num) / (rss_augmented / df_den)
    return max(f_stat, 0.0), df_num, df_den, partial_r2, semipartial_r2


class LinearityTest(FunctionalFormTest):
    """RESET-on-residuals test of whether partial correlation adequately
    summarizes the candidate-obs relationship given the conditioning set.
    Scalar candidate and obs only. An F statistic of a regression, so its
    independent bootstrap weights are Rademacher (Davidson and Flachaire, 2008).
    """
    independent_multipliers = "rademacher"

    @property
    def name(self) -> str:
        return "LinearityTest(RESET-on-residuals)"

    def _compute_statistic(
            self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
    ) -> tuple[float, int, int, float, float]:
        candidate_resid = ols_residuals(candidate, conditioning)
        obs_resid = ols_residuals(obs, conditioning)
        return reset_statistic(candidate_resid, obs_resid)

    def _null_fit(self, candidate, obs, conditioning):
        """Null model: obs linear in the conditioning set and the candidate."""
        regressors = as2d(candidate)
        if conditioning is not None and np.asarray(conditioning).size:
            regressors = np.column_stack([regressors, as2d(conditioning)])
        return ols_fit(obs, regressors)[0].reshape(obs.shape)

    def _null_leverage(self, candidate, obs, conditioning):
        regressors = as2d(candidate)
        if conditioning is not None and np.asarray(conditioning).size:
            regressors = np.column_stack([regressors, as2d(conditioning)])
        return _ols_leverage(regressors)


class MultivariateLinearityTest(FunctionalFormTest):
    """Kernel score test of whether a linear (CCA-type) relationship between
    multivariate candidate and obs is adequate given the conditioning set.

    The ridge penalty defaults to a REML estimate on the linear kernel at every
    permutation draw. Effect sizes come from an augmented fit with
    K_lin + K_rbf at the same penalty. classical_p_value is NaN (Davies'
    method is not implemented); decisions use the permutation p-value.
    """

    _supports_multivariate = True

    def __init__(self, alpha: float = 0.05, effect_size_floor: float = 0.02,
                 ridge_lambda: float | None = None, run_length: int = DEFAULT_RUN_LENGTH):
        super().__init__(alpha=alpha, effect_size_floor=effect_size_floor, run_length=run_length)
        self.ridge_lambda = ridge_lambda

    @property
    def name(self) -> str:
        return "MultivariateLinearityTest(kernel-linear-vs-nonlinear-score)"

    def _classical_p_value(self, statistic: float, df_num: int, df_den: int) -> float:
        return float("nan")  # Davies' (1980) exact reference not implemented

    def _compute_statistic(
            self, candidate: np.ndarray, obs: np.ndarray, conditioning: np.ndarray | None,
    ) -> tuple[float, int, int, float, float]:
        candidate_resid = residualize_on_conditioning(candidate, conditioning)
        obs_resid = residualize_on_conditioning(obs, conditioning)

        K_linear = candidate_resid @ candidate_resid.T
        bw = median_heuristic_bandwidth(candidate_resid)
        K_nonlinear = rbf_kernel(candidate_resid, bw)

        ridge_lambda = self.ridge_lambda
        if ridge_lambda is None:
            ridge_lambda = reml_kernel_ridge_lambda(K_linear, obs_resid)

        _, resid_null = kernel_ridge_fit(K_linear, obs_resid, ridge_lambda)

        score = float(np.sum(resid_null * (K_nonlinear @ resid_null)))
        score = max(score, 0.0)

        K_full = K_linear + K_nonlinear
        _, resid_aug = kernel_ridge_fit(K_full, obs_resid, ridge_lambda)
        partial_es, semipartial_es = variance_shares(obs_resid, resid_null, resid_aug)

        return score, 1, 1, partial_es, semipartial_es  # degrees of freedom unused

    def _null_fit(self, candidate, obs, conditioning):
        """Null model: obs linear in the conditioning set, plus a ridge fit
        with the linear kernel on the candidate's residuals."""
        candidate_resid = residualize_on_conditioning(candidate, conditioning)
        obs_2d = as2d(obs)
        obs_resid = residualize_on_conditioning(obs_2d, conditioning)
        K_linear = candidate_resid @ candidate_resid.T
        ridge_lambda = self.ridge_lambda
        if ridge_lambda is None:
            ridge_lambda = reml_kernel_ridge_lambda(K_linear, obs_resid)
        _, resid_null = kernel_ridge_fit(K_linear, obs_resid, ridge_lambda)
        return (obs_2d - resid_null).reshape(obs.shape)



def make_linearity_adequacy_test(alpha: float = 0.05, effect_size_floor: float = 0.02,
                                  run_length: int = DEFAULT_RUN_LENGTH) -> AdequacyTestFunc:
    """PC adequacy check for scalar samples (LinearityTest)."""
    return as_adequacy_test(LinearityTest(alpha=alpha, effect_size_floor=effect_size_floor, run_length=run_length))


def make_multivariate_linearity_adequacy_test(
        alpha: float = 0.05, effect_size_floor: float = 0.02, ridge_lambda: float | None = None,
        run_length: int = DEFAULT_RUN_LENGTH,
) -> AdequacyTestFunc:
    """PC adequacy check for multivariate samples (MultivariateLinearityTest)."""
    return as_adequacy_test(MultivariateLinearityTest(
        alpha=alpha, effect_size_floor=effect_size_floor, ridge_lambda=ridge_lambda, run_length=run_length))
