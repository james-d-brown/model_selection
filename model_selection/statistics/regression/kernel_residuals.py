"""
Cross-fitted kernel ridge residuals of kernel feature maps.

The nonlinear tier regresses each variable's feature map phi(X), the element
of a reproducing kernel Hilbert space with kernel k, on the conditioning set
Z, and works with the residuals. A kernel matrix K = k(X_i, X_j) factorises as
K = L L' with L = V_+ diag(sqrt(e_+)), keeping the eigenvalues above the
numerical-rank tolerance e_max n eps (the threshold numerical linear algebra
uses to decide the rank of a matrix); rows of L are then coordinates of the
phi(X_i) in which every inner product, and therefore every quantity computed
from residual inner products, is the kernel's own to working precision.
Kernel ridge regression is linear in its targets, so regressing L on Z gives
the residual feature maps exactly.

The conditioning-set regression is kernel ridge regression with an
unpenalized intercept and a radial basis function kernel on the standardized
Z. The bandwidth (a multiple of the median pairwise distance) and the penalty
are chosen once, on all samples, by closed-form leave-one-out error (Rifkin
and Lippert, 2007) over a range wide enough to run from near-interpolation to
a near-constant fit (penalties 10^-6 to 10^6 against the kernel's unit
diagonal), summed over the coordinates of L, which is the leave-one-out error
measured in the Hilbert-space norm.

Residuals are cross-fitted by h-block cross-validation (Burman, Chow and
Nolan, 1994): each sample's residual is that of the fit made without the
samples within one dependence lag of it. Kernel ridge regression with a fixed
bandwidth and penalty is penalized least squares, so the residuals of the
fit without a set S follow exactly from the fit on all samples:
e_S(-S) = (I - H_SS)^-1 e_S, with H the hat matrix (the leave-set-out
form of the leave-one-out identity). One kernel eigendecomposition per
bandwidth serves every sample and every variable regressed on that Z. Where
the samples outside a sample's buffer would be too few to fit (fewer than 10,
or fewer than the dimensions of Z plus 2), only the sample itself is left out
and the number of such samples is recorded (buffer_dropped_samples).

See docs/methods.md, Nonlinear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth
from model_selection.utils import as2d

BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_PENALTIES = tuple(10.0 ** np.arange(-6.0, 6.5, 0.5))


def standardize(X: np.ndarray) -> np.ndarray:
    """Columns centred and scaled to unit standard deviation (constant
    columns centred only)."""
    X = as2d(np.asarray(X, dtype=float))
    sd = X.std(axis=0)
    sd[sd <= 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def feature_kernel(X: np.ndarray) -> np.ndarray:
    """Kernel of a variable's features: the linear kernel of its standardized
    columns plus a radial basis function kernel with the median-heuristic
    bandwidth (Gretton et al., 2012), so that the features contain the
    variable itself and its nonlinear transformations."""
    Xs = standardize(X)
    bw = median_heuristic_bandwidth(Xs)
    return Xs @ Xs.T + np.exp(-_sq_dists(Xs, Xs) / (2.0 * bw ** 2))


def kernel_factor(K: np.ndarray) -> np.ndarray:
    """L with L L' = K to working precision: eigenvectors scaled by the square
    roots of the eigenvalues above e_max * n * eps."""
    e, V = np.linalg.eigh((K + K.T) / 2.0)
    tol = max(float(e.max()), 0.0) * K.shape[0] * np.finfo(float).eps
    keep = e > tol
    if not np.any(keep):
        return np.zeros((K.shape[0], 1))
    return V[:, keep] * np.sqrt(e[keep])


def _centred_spectrum(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues above the numerical-rank tolerance, and their eigenvectors,
    of the doubly centred kernel C K C (C = I - 11'/n). With an unpenalized
    intercept, the hat matrix at penalty lam is 11'/n + U diag(s/(s+lam)) U'."""
    Kc = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
    s, U = np.linalg.eigh((Kc + Kc.T) / 2.0)
    keep = s > max(float(s.max()), 0.0) * K.shape[0] * np.finfo(float).eps
    return s[keep], U[:, keep]


class ConditioningRegression:
    """Kernel ridge regression on one conditioning set, h-block cross-fitted
    (module docstring), with the kernel eigendecompositions shared by all the
    variables regressed on it."""

    def __init__(self, Z: np.ndarray, time_positions: np.ndarray, buffer: float = 0.0):
        Z = standardize(as2d(np.asarray(Z, dtype=float)))
        self.n = n = Z.shape[0]
        sq = _sq_dists(Z, Z)
        base = median_heuristic_bandwidth(Z)
        self.spectra = [_centred_spectrum(np.exp(-sq / (2.0 * (base * m) ** 2))) for m in BANDWIDTH_MULTIPLIERS]
        t = np.asarray(time_positions, dtype=float)
        order = np.argsort(t, kind="stable")
        lo = np.searchsorted(t[order], t - buffer, side="left")
        hi = np.searchsorted(t[order], t + buffer, side="right")
        self.left_out: list[tuple[np.ndarray, int]] = []
        self.buffer_dropped_samples = 0
        for i in range(n):
            S = order[lo[i]:hi[i]]
            if n - S.size < max(10, Z.shape[1] + 2):
                S = np.array([i])  # too few samples outside the buffer to fit
                self.buffer_dropped_samples += 1
            self.left_out.append((S, int(np.flatnonzero(S == i)[0])))

    def _left_out_residuals(self, U: np.ndarray, shrink: np.ndarray, E: np.ndarray) -> np.ndarray:
        """Each sample's residual from the fit without its left-out set S:
        row i of (I - H_SS)^-1 E_S."""
        out = np.empty_like(E)
        for i, (S, pos) in enumerate(self.left_out):
            U_S = U[S]
            H_SS = 1.0 / self.n + (U_S * shrink) @ U_S.T
            out[i] = np.linalg.solve(np.eye(S.size) - H_SS, E[S])[pos]
        return out

    def residuals(self, target: np.ndarray, extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Cross-fitted residuals of `target` (n, r) and, with the same fitted
        smoother, of `extra` (n, s). The bandwidth and penalty minimize the
        leave-one-out squared error summed over the columns of `target`."""
        T = as2d(np.asarray(target, dtype=float))
        X = None if extra is None else as2d(np.asarray(extra, dtype=float))
        Y = T if X is None else np.column_stack([T, X])
        r = T.shape[1]
        Yc = Y - Y.mean(axis=0)
        best = (np.inf, None)
        for s, U in self.spectra:
            UtY = U.T @ Yc
            leverage_base = U ** 2
            for lam in RIDGE_PENALTIES:
                shrink = s / (s + lam)
                E = Yc - U @ (shrink[:, None] * UtY)
                h = 1.0 / self.n + leverage_base @ shrink
                err = float(np.sum((E[:, :r] / np.maximum(1.0 - h, 1e-8)[:, None]) ** 2))
                if err < best[0]:
                    best = (err, (U, shrink, E))
        U, shrink, E = best[1]
        resid = self._left_out_residuals(U, shrink, E)
        return resid[:, :r], (None if X is None else resid[:, r:])


def gram(E: np.ndarray) -> np.ndarray:
    """Inner products of the rows of E."""
    return E @ E.T
