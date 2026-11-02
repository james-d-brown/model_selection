"""
Canonical correlation between residual matrices (Hotelling, 1936), used by
the linear tier, and the structural check of that tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.association.base import StructuralCheckOutcome
from model_selection.utils import as2d


def residualize_on_conditioning(Y: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """OLS residuals of every column of Y regressed on Z (with intercept).
    Z=None returns each column's deviation from its mean.
    """
    Y = as2d(Y)
    n = Y.shape[0]
    if Z is None or np.asarray(Z).size == 0:
        design = np.ones((n, 1))
    else:
        Z = as2d(Z)
        design = np.hstack([np.ones((n, 1)), Z])
    coefs, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
    fitted = design @ coefs
    return Y - fitted


def _whitener(M: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Symmetric whitening matrix W such that M @ W has ~identity covariance.

    Stabilized by a relative eigenvalue floor (rtol * largest eigenvalue, as in
    numpy.linalg.pinv) so it stays defined for single or collinear columns
    without perturbing well-conditioned cases.
    """
    n = M.shape[0]
    cov = (M.T @ M) / max(n - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    max_eigval = float(eigval.max()) if eigval.size > 0 and eigval.max() > 0 else 1.0
    floor = rtol * max_eigval
    eigval = np.clip(eigval, a_min=floor, a_max=None)
    return eigvec @ np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T


def canonical_correlations(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Canonical correlations of already-centred X (n, p) and Y (n, q): singular
    values of the whitened cross-covariance. Returns min(p, q) values in [0, 1],
    descending.
    """
    X = as2d(X)
    Y = as2d(Y)
    n = X.shape[0]

    Wx = _whitener(X)
    Wy = _whitener(Y)
    Xw = X @ Wx
    Yw = Y @ Wy
    cross = (Xw.T @ Yw) / max(n - 1, 1)
    svals = np.linalg.svd(cross, compute_uv=False)
    return np.clip(svals, 0.0, 1.0)


def partial_cca_structural_adequacy(n: int, p: int, q: int, k: int) -> StructuralCheckOutcome:
    """Structural pre-check for PC (a StructuralCheckFunc).

    Residualizing on k conditioning columns plus an intercept leaves
    n_eff = n - k - 1 samples. Whitening a d-dimensional covariance needs
    n_eff > d; below that the estimate is dominated by regularization rather
    than data (Le Cao et al. 2009). This checks only that hard threshold, not
    the wider 10-20x rule of thumb for reliable CCA.
    """
    n_eff = n - k - 1
    max_dim = max(p, q)
    well_posed = n_eff > max_dim
    if well_posed:
        reason = (
            f"structurally well-posed: n_eff={n_eff} (n={n}, k={k} conditioning dims, "
            f"-1 for the intercept) exceeds max(p,q)={max_dim} (p={p}, q={q})"
        )
    else:
        reason = (
            f"structurally ill-posed: n_eff={n_eff} (n={n}, k={k} conditioning dims, -1 for "
            f"the intercept) does not exceed max(p,q)={max_dim} (p={p}, q={q}) -- PartialCCA's "
            f"whitening step cannot estimate genuine covariance structure here (Le Cao et al. "
            f"2009): with only {n_eff} effective samples available to estimate a "
            f"{max_dim}-dimensional covariance matrix, the whitened result is dominated by "
            f"ridge regularization, not by data."
        )
    return StructuralCheckOutcome(well_posed=well_posed, reason=reason)


def first_canonical_projection(
        X: np.ndarray, Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """First canonical variate pair of already-centred X (n, p) and Y (n, q),
    using the same whitening as canonical_correlations but keeping the
    canonical vectors. corr(x_proj, y_proj) equals canonical_correlations(X, Y)[0].
    The sign is fixed so the relationship is positively sloped (display only).

    Returns (x_proj, y_proj, x_loadings, y_loadings, first_canonical_corr).
    """
    X = as2d(X)
    Y = as2d(Y)
    n = X.shape[0]
    Wx = _whitener(X)
    Wy = _whitener(Y)
    Xw = X @ Wx
    Yw = Y @ Wy
    cross = (Xw.T @ Yw) / max(n - 1, 1)
    U, svals, Vt = np.linalg.svd(cross, full_matrices=False)
    x_loadings = Wx @ U[:, 0]
    y_loadings = Wy @ Vt[0, :]
    x_proj = X @ x_loadings
    y_proj = Y @ y_loadings
    if x_proj.std() > 0 and y_proj.std() > 0 and np.corrcoef(x_proj, y_proj)[0, 1] < 0:
        y_loadings = -y_loadings
        y_proj = -y_proj
    first_cc = float(np.clip(svals[0], 0.0, 1.0)) if svals.size > 0 else 0.0
    return x_proj, y_proj, x_loadings, y_loadings, first_cc
