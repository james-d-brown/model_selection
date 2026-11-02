"""
Functional principal component analysis on curves sampled on a common grid.

This is the discretized form of FPCA (Ramsay & Silverman, ch. 8): ordinary
PCA via SVD on curves already resampled onto one evenly spaced grid, which
is exact FPCA in the limit of a fine grid. Because every curve shares the
grid, the unweighted SVD is consistent with the L2 inner product up to a
common constant, which affects neither explained-variance ratios,
component order, nor score comparisons. A roughness-penalized variant
(Silverman 1996) is provided for smoother components.

Curves with missing points (observed curves with gauge gaps) are scored by
conditional expectation given their observed points (PACE; Yao, Mueller and
Wang, 2005) under the fitted basis, with the noise variance of probabilistic
PCA (Tipping and Bishop, 1999): the variance the kept components leave
unexplained, averaged over the remaining dimensions.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FPCAResult:
    """A fitted FPCA basis.

    components has orthonormal rows (approximately so when regularized; see
    component_orthogonality_error, the largest off-diagonal of the
    components' Gram matrix). explained_variance_ratio is relative to the
    full spectrum, which full_explained_variance_ratio keeps for scree plots.
    selection_note records how the component count was chosen.
    """
    mean_curve: np.ndarray                 # (n_grid,)
    components: np.ndarray                 # (n_components, n_grid)
    explained_variance: np.ndarray         # (n_components,)
    explained_variance_ratio: np.ndarray   # (n_components,), of the full spectrum
    cumulative_explained_variance_ratio: np.ndarray  # (n_components,)
    n_grid: int
    grid: np.ndarray                       # (n_grid,), phase in [0, 1]
    fit_source: str
    n_curves_fit: int
    full_explained_variance_ratio: np.ndarray = field(repr=False, default=None)
    selection_note: str = ""
    regularization_lambda: float = 0.0
    component_orthogonality_error: float = 0.0
    noise_variance: float = 0.0            # per grid point, unexplained by the kept components


def n_components_for_variance(matrix: np.ndarray, variance_threshold: float,
                              max_components: int | None = None) -> tuple[int, bool]:
    """Smallest number of components whose cumulative explained variance reaches
    variance_threshold, limited by the data's rank and, if given, by
    max_components. Returns (k, capped).
    """
    n_curves, n_grid = matrix.shape
    cap = min(n_curves - 1, n_grid) if max_components is None else min(max_components, n_curves - 1, n_grid)
    mean_curve = matrix.mean(axis=0)
    centered = matrix - mean_curve
    S = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    explained_variance = (S ** 2) / max(n_curves - 1, 1)
    total_variance = explained_variance.sum()
    ratio = explained_variance / total_variance if total_variance > 0 else explained_variance
    cumulative = np.cumsum(ratio)

    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    capped = k > cap
    k = min(max(k, 1), cap)
    return k, capped


def _noise_variance(centered: np.ndarray, kept_variance: np.ndarray) -> float:
    """Probabilistic-PCA noise variance (Tipping and Bishop, 1999): the total
    variance the kept components leave unexplained, per remaining dimension
    of the grid."""
    n_curves, n_grid = centered.shape
    total = float(np.sum(centered ** 2)) / max(n_curves - 1, 1)
    remaining = n_grid - kept_variance.size
    return max(total - float(np.sum(kept_variance)), 0.0) / remaining if remaining > 0 else 0.0


def fit_functional_pca(matrix: np.ndarray, n_components: int = 3, fit_source: str = "",
                        selection_note: str = "") -> FPCAResult:
    """SVD-based FPCA of an (n_curves, n_grid) matrix of curves on a common grid."""
    n_curves, n_grid = matrix.shape
    if n_components >= n_curves:
        raise ValueError(
            f"n_components={n_components} must be < n_curves={n_curves} "
            "(need more events than components to fit a non-degenerate basis)"
        )
    mean_curve = matrix.mean(axis=0)
    centered = matrix - mean_curve
    # full_matrices=False: economy SVD, U is (n_curves, min(n_curves,n_grid))
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance_full = (S ** 2) / max(n_curves - 1, 1)
    total_variance = explained_variance_full.sum()
    full_ratio = explained_variance_full / total_variance if total_variance > 0 else explained_variance_full

    components = Vt[:n_components]
    explained_variance = explained_variance_full[:n_components]
    explained_variance_ratio = full_ratio[:n_components]
    cumulative = np.cumsum(explained_variance_ratio)

    return FPCAResult(
        mean_curve=mean_curve,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance_ratio=cumulative,
        n_grid=n_grid,
        grid=np.linspace(0.0, 1.0, n_grid),
        fit_source=fit_source,
        selection_note=selection_note,
        n_curves_fit=n_curves,
        full_explained_variance_ratio=full_ratio,
        noise_variance=_noise_variance(centered, explained_variance),
    )


def _second_difference_matrix(n_grid: int) -> np.ndarray:
    """(n_grid - 2, n_grid) central second-difference operator on the grid."""
    D = np.zeros((n_grid - 2, n_grid))
    for i in range(n_grid - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def roughness_penalty_matrix(n_grid: int) -> np.ndarray:
    """Roughness penalty K = (n_grid - 1)^4 D'D, the discretization of the
    integrated squared second derivative relative to the integrated square on
    phase [0, 1] (grid spacing h = 1 / (n_grid - 1): sum (D v)^2 / h^4 against
    sum v^2), so that a given lambda smooths equally on any grid. Its null
    space is constant and linear functions."""
    D = _second_difference_matrix(n_grid)
    return (n_grid - 1) ** 4 * (D.T @ D)


def fit_regularized_functional_pca(matrix: np.ndarray, n_components: int = 3, lam: float = 0.0,
                                    fit_source: str = "", selection_note: str = "") -> FPCAResult:
    """Roughness-penalized FPCA (Silverman 1996).

    Maximizes v' Sigma v subject to v' (I + lam K) v = 1, solved by transforming
    the data with (I + lam K)^(-1/2), taking an ordinary SVD and transforming
    back. lam = 0 is exactly fit_functional_pca. Components are renormalized to
    unit L2 norm and reordered by captured variance, so for lam > 0 they are
    not exactly orthogonal; component_orthogonality_error reports by how much.
    project() and reconstruct() work unchanged on the result.
    """
    n_curves, n_grid = matrix.shape
    if n_components >= n_curves:
        raise ValueError(
            f"n_components={n_components} must be < n_curves={n_curves} "
            "(need more events than components to fit a non-degenerate basis)"
        )
    if lam == 0.0:
        return fit_functional_pca(matrix, n_components=n_components, fit_source=fit_source,
                                   selection_note=selection_note)

    mean_curve = matrix.mean(axis=0)
    centered = matrix - mean_curve

    K = roughness_penalty_matrix(n_grid)
    W = np.eye(n_grid) + lam * K

    # M = W^(1/2), M_inv = W^(-1/2), via eigendecomposition of the
    # symmetric positive-definite W -- exact and numerically stable at
    # grid sizes used here (up to a few thousand points).
    w_eigvals, w_eigvecs = np.linalg.eigh(W)
    w_eigvals = np.clip(w_eigvals, a_min=1e-12, a_max=None)  # guard tiny negative numerical noise
    m_inv = w_eigvecs @ np.diag(1.0 / np.sqrt(w_eigvals)) @ w_eigvecs.T

    transformed = centered @ m_inv  # X_c @ M^-1
    U, S, Vt = np.linalg.svd(transformed, full_matrices=False)
    # Vt's rows are ordinary-orthonormal directions of the TRANSFORMED
    # problem; recover the actual regularized components v_i = M^-1 u_i
    # (m_inv symmetric, so u_i @ m_inv == m_inv @ u_i as a row op here).
    components_raw = Vt[:n_components] @ m_inv

    explained_variance_full = (S ** 2) / max(n_curves - 1, 1)
    total_variance = explained_variance_full.sum()
    full_ratio = explained_variance_full / total_variance if total_variance > 0 else explained_variance_full

    # Renormalize to ordinary unit L2 norm (see docstring), then
    # re-derive explained variance from the ACTUAL ordinary-inner-product
    # projection of the data onto each renormalized component -- the raw
    # eigenproblem's variance accounting was defined w.r.t. the
    # transformed problem and isn't directly interpretable in original
    # units once components are rescaled.
    norms = np.linalg.norm(components_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    components = components_raw / norms

    scores_kept = centered @ components.T
    explained_variance_kept = scores_kept.var(axis=0, ddof=1)
    # Re-sort descending by ACTUAL captured variance -- renormalization
    # can occasionally reorder components relative to the raw
    # eigenproblem's rho ordering, since rho was defined pre-renormalization.
    order = np.argsort(-explained_variance_kept)
    components = components[order]
    explained_variance_kept = explained_variance_kept[order]
    # Kept variance is measured in the curves' own space, so the ratio uses
    # the total variance in that space, not in the penalized (transformed) one.
    total_original = float(np.sum(centered ** 2)) / max(n_curves - 1, 1)
    explained_variance_ratio = (explained_variance_kept / total_original if total_original > 0
                                else explained_variance_kept)
    cumulative = np.cumsum(explained_variance_ratio)

    gram = components @ components.T
    orthogonality_error = float(np.abs(gram - np.eye(n_components)).max())

    return FPCAResult(
        mean_curve=mean_curve,
        components=components,
        explained_variance=explained_variance_kept,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance_ratio=cumulative,
        n_grid=n_grid,
        grid=np.linspace(0.0, 1.0, n_grid),
        fit_source=fit_source,
        selection_note=selection_note + f"; regularized FPCA (lambda={lam})",
        n_curves_fit=n_curves,
        full_explained_variance_ratio=full_ratio,
        regularization_lambda=lam,
        component_orthogonality_error=orthogonality_error,
        noise_variance=_noise_variance(centered, explained_variance_kept),
    )


def project(matrix: np.ndarray, fpca: FPCAResult) -> np.ndarray:
    """(n_curves, n_grid) -> (n_curves, n_components) scores on a fitted basis."""
    return (matrix - fpca.mean_curve) @ fpca.components.T


def conditional_scores(matrix: np.ndarray, fpca: FPCAResult) -> np.ndarray:
    """(n_curves, n_grid) curves, possibly with missing (NaN) points ->
    (n_curves, n_components) scores: the conditional expectation of each
    curve's scores given its observed points (PACE; Yao, Mueller and Wang,
    2005), under scores with variances fpca.explained_variance and
    independent noise of variance fpca.noise_variance at every grid point:

        E[xi | y_o] = (Phi_o Phi_o' + s2 Lambda^-1)^-1 Phi_o (y_o - mu_o)

    (Phi_o: the components at the observed points). For a complete curve this
    is the projection shrunk by lambda_k / (lambda_k + s2) per component; with
    s2 = 0, the least-squares fit of the components to the observed points.
    A curve with no observed point has undefined (NaN) scores."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    Phi = fpca.components
    k = Phi.shape[0]
    with np.errstate(divide="ignore"):
        prior_precision = np.where(fpca.explained_variance > 0, 1.0 / fpca.explained_variance, np.inf)
    scores = np.full((matrix.shape[0], k), np.nan)
    for i, y in enumerate(matrix):
        observed = np.isfinite(y)
        if not observed.any():
            continue
        Phi_o = Phi[:, observed]
        A = Phi_o @ Phi_o.T + fpca.noise_variance * np.diag(np.where(np.isfinite(prior_precision),
                                                                      prior_precision, 1e12))
        b = Phi_o @ (y[observed] - fpca.mean_curve[observed])
        scores[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    return scores


def reconstruct(scores: np.ndarray, fpca: FPCAResult) -> np.ndarray:
    """(n_curves, n_components) scores -> (n_curves, n_grid) reconstruction."""
    return scores @ fpca.components + fpca.mean_curve


def reconstruction_error(matrix: np.ndarray, matrix_hat: np.ndarray) -> pd.DataFrame:
    """Per-curve RMSE and R^2 (fraction of the curve's own variance captured by
    the kept components) of a reconstruction, over each curve's observed
    (finite) points.
    """
    resid = matrix - matrix_hat
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing rows give NaN
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        curve_var = np.nanmean((matrix - np.nanmean(matrix, axis=1, keepdims=True)) ** 2, axis=1)
        resid_var = np.nanmean(resid ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(curve_var > 0, 1.0 - resid_var / curve_var, np.nan)
    return pd.DataFrame({"rmse": rmse, "r2": r2})


def phase_normalize(curve: np.ndarray, n_grid: int = 50) -> np.ndarray:
    """Resample a curve of any length onto n_grid evenly spaced points of
    relative phase [0, 1] by linear interpolation (a single-sample curve gives
    a constant). Absolute duration is not encoded. A grid point between two
    samples of which either is missing (NaN) is missing: gaps are never
    filled.
    """
    curve = np.asarray(curve, dtype=np.float64)
    m = curve.shape[0]
    if m == 1:
        return np.full(n_grid, curve[0])
    src_phase = np.linspace(0.0, 1.0, m)
    dst_phase = np.linspace(0.0, 1.0, n_grid)
    return np.interp(dst_phase, src_phase, curve)
