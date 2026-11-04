"""Kernel construction and kernel ridge regression for the kernel-machine
adequacy tests."""
from __future__ import annotations

import numpy as np

from model_selection.utils import as2d


def median_heuristic_bandwidth(X: np.ndarray) -> float:
    """Median-heuristic RBF bandwidth (Gretton et al. 2012): the median of
    all pairwise Euclidean distances between rows of X; 1.0 if degenerate."""
    from scipy.spatial.distance import pdist

    X = as2d(np.asarray(X, dtype=float))
    if X.shape[0] < 2:
        return 1.0
    med = float(np.median(pdist(X)))
    return med if med > 0 else 1.0


def rbf_kernel(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """(n, n) RBF kernel K[i, j] = exp(-||x_i - x_j||^2 / (2 bandwidth^2))."""
    X = as2d(X)
    diffs = X[:, None, :] - X[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    return np.exp(-sq_dists / (2.0 * bandwidth ** 2))


def kernel_ridge_fit(K: np.ndarray, y_centered: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Kernel ridge regression of centred y on K. Returns (fitted, residuals)."""
    n = K.shape[0]
    alpha = np.linalg.solve(K + ridge_lambda * np.eye(n), y_centered)
    fitted = K @ alpha
    return fitted, y_centered - fitted


def reml_lambda_from_spectrum(
        eigvals: np.ndarray, proj: np.ndarray, bounds: tuple[float, float] = (1e-6, 1e6),
        n_total: int | None = None, null_energy: np.ndarray | None = None,
) -> float:
    """REML ridge penalty (see reml_kernel_ridge_lambda) from K's eigenvalues
    and the data projected on K's eigenvectors, proj = V' y (m, q).

    With a thin basis (m < n eigenvectors, the rest of K's spectrum zero, as
    for a low-rank factor), n_total is n and null_energy the squared norm of
    each column of y outside the basis, ||y||^2 - ||V' y||^2; the n - m zero
    eigenvalues then enter the likelihood as they would with the full
    eigendecomposition. Without them the basis is taken to be complete."""
    from scipy.optimize import minimize_scalar

    eigvals = np.clip(eigvals, 0.0, None)
    m, q = proj.shape
    n = m if n_total is None else int(n_total)
    d0 = n - m
    null = np.zeros(q) if null_energy is None else np.clip(np.asarray(null_energy, dtype=float).ravel(), 0.0, None)

    def _profile_negative_log_likelihood(log_lam: float) -> float:
        lam = np.exp(log_lam)
        denom = eigvals + lam
        total = q * (np.sum(np.log(denom)) + d0 * log_lam)
        for j in range(q):
            quad = np.sum(proj[:, j] ** 2 / denom) + (null[j] / lam if d0 else 0.0)
            if quad > 0:
                total += n * np.log(quad)
        return total

    result = minimize_scalar(_profile_negative_log_likelihood, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                             method="bounded")
    return float(np.exp(result.x))


def kernel_spectrum(n: int, diag: np.ndarray, column, exact) -> tuple[np.ndarray, np.ndarray, float]:
    """(eigenvalues e, eigenvectors V, cap share) of a kernel for the kernel
    score tests: the full eigendecomposition of exact() when n is within the
    rank cap, else the thin spectrum of its pivoted-Cholesky factor G (K ~ G
    G' = V diag(e) V'), with its cap share (regression.kernel_residuals)."""
    from model_selection.statistics.regression.kernel_residuals import _use_low_rank, low_rank, pivoted_cholesky
    if not _use_low_rank(n):
        e, V = np.linalg.eigh(exact())
        return np.clip(e, 0.0, None), V, 0.0
    G, share = pivoted_cholesky(diag, column, low_rank(), kind="escalation test")
    U, sv, _ = np.linalg.svd(G, full_matrices=False)
    e = sv ** 2
    keep = e > max(float(e.max()), 0.0) * n * np.finfo(float).eps
    return e[keep], U[:, keep], share


def project(V: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(V' Y, squared norm of each column of Y outside V's span)."""
    P = V.T @ Y
    return P, np.clip(np.sum(Y ** 2, axis=0) - np.sum(P ** 2, axis=0), 0.0, None)


def reml_kernel_ridge_lambda(
        K: np.ndarray,
        y: np.ndarray,
        bounds: tuple[float, float] = (1e-6, 1e6),
) -> float:
    """Ridge penalty for a kernel ridge fit of y on K, estimated by restricted
    maximum likelihood, as SKAT does for its null-model variance component
    (Wu et al. 2011).

    Model: y = h + eps, h ~ N(0, tau K), eps ~ N(0, sigma^2 I), with
    lambda = sigma^2 / tau. tau is profiled out analytically, leaving a bounded
    one-dimensional search over log(lambda), O(n) per evaluation after one
    eigendecomposition of K. Multivariate y shares K and lambda across columns,
    with each column's tau profiled separately. A fixed penalty that ignores K's
    scale inflates the null distribution of the kernel score tests; this
    estimate removes that inflation without sacrificing power.
    """
    from scipy.optimize import minimize_scalar

    n = K.shape[0]
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.clip(eigvals, 0.0, None)  # K is PSD by construction; guards tiny negative numerical noise only

    y2d = as2d(y)
    q = y2d.shape[1]
    proj = eigvecs.T @ y2d  # (n, q): each column's data, expressed in K's own eigenbasis

    def _profile_negative_log_likelihood(log_lam: float) -> float:
        lam = np.exp(log_lam)
        denom = eigvals + lam
        logdet = np.sum(np.log(denom))  # log|K + lambda*I|, cheap given eigenvalues
        total = 0.0
        for j in range(q):
            quad = np.sum(proj[:, j] ** 2 / denom)  # y_j' (K+lambda*I)^-1 y_j, cheap given eigenvalues
            # Guard against a genuinely all-zero column (quad == 0) -- would make log(quad) = -inf,
            # which minimize_scalar cannot handle; contributes nothing informative to the objective anyway.
            if quad > 0:
                total += n * np.log(quad)
        total += q * logdet
        return total

    result = minimize_scalar(
        _profile_negative_log_likelihood,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
    )
    return float(np.exp(result.x))


def variance_shares(resid_z: np.ndarray, resid_null: np.ndarray, resid_aug: np.ndarray) -> tuple[float, float]:
    """Effect sizes of a departure on the set-correlation scale used by every
    effect-size floor: generalized variances (determinants of the residual
    cross-product matrices) of the observations after removing Z alone (g_z),
    under the null model (g_null) and under the augmented model (g_aug).

    partial = 1 - g_aug / g_null, the departure's share of what the null
    model leaves unexplained; semipartial = (g_null - g_aug) / g_z, its share
    of what Z leaves unexplained (the adequacy effect size). For a single
    column these are the usual ratios of residual sums of squares.
    """
    def logdet(r):
        r = as2d(r)
        sign, value = np.linalg.slogdet(r.T @ r)
        return value if sign > 0 else -np.inf

    l_z, l_null, l_aug = logdet(resid_z), logdet(resid_null), logdet(resid_aug)
    if not np.isfinite(l_null) or not np.isfinite(l_z):
        return 0.0, 0.0
    partial = 1.0 - np.exp(l_aug - l_null)
    semipartial = np.exp(l_null - l_z) - np.exp(l_aug - l_z)
    return float(min(max(partial, 0.0), 1.0)), float(min(max(semipartial, 0.0), 1.0))
