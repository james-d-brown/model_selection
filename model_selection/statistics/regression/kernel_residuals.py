"""
Cross-fitted kernel ridge residuals of kernel feature maps.

The nonlinear tier regresses each variable's feature map phi(X), the element
of a reproducing kernel Hilbert space with kernel k, on the conditioning set
Z, and works with the residuals. Every kernel matrix K is represented by a
factor L with L L' = K: rows of L are coordinates of the phi(X_i) in which
every inner product, and therefore every quantity computed from residual
inner products, is the kernel's own. Kernel ridge regression is linear in its
targets, so regressing L on Z gives the residual feature maps.

Factors. Up to the rank cap (set_low_rank; SelectionConfig.nonlinear_rank)
the factor of a kernel on n <= cap samples is exact: eigenvectors scaled by
the square roots of the eigenvalues above the numerical-rank tolerance
e_max n eps. On longer samples it is a pivoted-Cholesky factor (Fine and
Scheinberg, 2001; Harbrecht, Peters and Schneider, 2012): columns are added
at the sample with the largest remaining diagonal, until that diagonal falls
to the same tolerance or the rank reaches the cap. Below the cap the factor
is the kernel's to working precision; at the cap its trace-norm error is
recorded (APPROXIMATIONS, and cap_share on each regression). Kernel ridge
regression on a factor is ridge regression on its columns (the Nystrom
method, Williams and Seeger, 2001), at a cost of O(n m^2) for rank m.

Regression. Kernel ridge regression with an unpenalized intercept and a
radial basis function kernel on the standardized Z, over a range of
bandwidths (multiples of the median pairwise distance) and penalties wide
enough to run from near-interpolation to a near-constant fit (10^-6 to 10^6
against the kernel's unit diagonal).

Cross-fitting (Semenova et al., 2023). The samples' time span is divided
into CROSSFIT_BLOCKS contiguous blocks of equal span, each at least one
dependence lag long (fewer blocks on short records, at least
MIN_CROSSFIT_BLOCKS; records shorter than that many lags are recorded as
blocks_short). Each block's residuals come from the fit to the samples
outside the block and its two neighbouring blocks. The bandwidth and penalty
are chosen once per target by the same blocked cross-validation: the
squared error of every block's predictions, summed over the coordinates of
the target, which is that error in the Hilbert-space norm. For each
bandwidth and block, one eigendecomposition of the factor's columns over the
fitting samples serves every penalty and every variable regressed on Z.

See docs/methods.md, Nonlinear tier.
"""
from __future__ import annotations

import numpy as np

from model_selection.statistics.adequacy.kernels import median_heuristic_bandwidth
from model_selection.utils import as2d

BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
RIDGE_PENALTIES = tuple(10.0 ** np.arange(-6.0, 6.5, 0.5))
CROSSFIT_BLOCKS = 20
MIN_CROSSFIT_BLOCKS = 4
"""Fewest blocks: with three, the middle block's neighbours would leave its
fit no samples."""
MIN_FIT_SAMPLES = 10
MEDIAN_MAX_N = 8000
"""Samples beyond which the median-heuristic bandwidth is taken from this
many evenly spaced samples (all pairwise distances would need n^2 / 2
values)."""
DEFAULT_RANK_CAP = 2000

_LOW_RANK: int | None = DEFAULT_RANK_CAP
APPROXIMATIONS: list = []
"""(kind, n, rank, share of the kernel's trace left out) of every
pivoted-Cholesky factor built; read and cleared by callers that report them."""


def set_low_rank(max_rank: int | None) -> None:
    """Cap on the rank of kernel factors; None for exact factors at every
    sample size. Factors are pivoted-Cholesky only for samples longer than
    the cap."""
    global _LOW_RANK
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"the rank cap must be at least 1 -- got {max_rank}")
    _LOW_RANK = max_rank


def low_rank() -> int | None:
    return _LOW_RANK


def _use_low_rank(n: int) -> bool:
    return _LOW_RANK is not None and n > _LOW_RANK


def pivoted_cholesky(diag: np.ndarray, column, max_rank: int, kind: str = "kernel") -> tuple[np.ndarray, float]:
    """(G, cap share): G (n, m) with G G' approximating the kernel whose
    diagonal is `diag` and whose column p is column(p), by greedy pivots on
    the largest remaining diagonal, stopping at the numerical-rank tolerance
    (max diag * n * eps) or at max_rank columns. The cap share is the share
    of the kernel's trace left out when the cap was reached, else 0."""
    d = np.asarray(diag, dtype=float).copy()
    n = d.size
    total = float(d.sum())
    tol = max(float(d.max()), 0.0) * n * np.finfo(float).eps
    G = np.zeros((n, min(max_rank, n)))
    m = 0
    while m < G.shape[1]:
        p = int(np.argmax(d))
        if d[p] <= tol:
            break
        g = column(p) - G[:, :m] @ G[p, :m]
        g /= np.sqrt(d[p])
        G[:, m] = g
        d -= g ** 2
        np.maximum(d, 0.0, out=d)
        d[p] = 0.0
        m += 1
    share = float(d.sum()) / total if total > 0 else 0.0
    APPROXIMATIONS.append((kind, n, m, share))
    capped = m >= max_rank and m < n and float(d.max()) > tol
    return (G[:, :m] if m else np.zeros((n, 1))), (share if capped else 0.0)


def _median_bandwidth(Xs: np.ndarray) -> float:
    if Xs.shape[0] > MEDIAN_MAX_N:
        Xs = Xs[np.linspace(0, Xs.shape[0] - 1, MEDIAN_MAX_N).round().astype(int)]
    return median_heuristic_bandwidth(Xs)


def _rbf_column(Zs: np.ndarray, sq_norms: np.ndarray, bw: float):
    def column(p):
        d = np.maximum(sq_norms + sq_norms[p] - 2.0 * Zs @ Zs[p], 0.0)
        return np.exp(-d / (2.0 * bw ** 2))
    return column


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


def feature_factor_with_cap(X: np.ndarray) -> tuple[np.ndarray, float]:
    """(factor L of the feature kernel of X, cap share): exact up to the rank
    cap's sample size, pivoted Cholesky beyond it."""
    Xs = standardize(X)
    if not _use_low_rank(Xs.shape[0]):
        return kernel_factor(feature_kernel(X)), 0.0
    bw = _median_bandwidth(Xs)
    sq = (Xs ** 2).sum(1)
    rbf = _rbf_column(Xs, sq, bw)
    return pivoted_cholesky(sq + 1.0, lambda p: Xs @ Xs[p] + rbf(p), _LOW_RANK, kind="features")


def feature_factor(X: np.ndarray) -> np.ndarray:
    return feature_factor_with_cap(X)[0]


def crossfit_blocks(time_positions: np.ndarray, dependence_lag: float) -> tuple[np.ndarray, int, bool]:
    """(block of each sample, number of blocks, too short): CROSSFIT_BLOCKS
    blocks of equal time span, fewer when a block would be shorter than the
    dependence lag, at least MIN_CROSSFIT_BLOCKS; too short when even those
    are shorter than the lag."""
    t = np.asarray(time_positions, dtype=float)
    span = float(t.max() - t.min()) if t.size else 0.0
    k = CROSSFIT_BLOCKS
    if dependence_lag > 0 and span > 0:
        k = min(CROSSFIT_BLOCKS, int(span // dependence_lag))
    short = k < MIN_CROSSFIT_BLOCKS
    k = max(k, MIN_CROSSFIT_BLOCKS)
    edges = t.min() + span * np.arange(1, k) / k if span > 0 else np.array([])
    return np.searchsorted(edges, t, side="right"), k, short


def _folds(block: np.ndarray, k: int, n_regressors: int) -> tuple[list, bool]:
    """(predicted samples, fitting samples) per block, and whether any fit
    had to keep the neighbouring blocks for lack of samples."""
    folds, kept = [], False
    minimum = max(MIN_FIT_SAMPLES, n_regressors + 2)
    for b in range(k):
        pred = np.flatnonzero(block == b)
        if pred.size == 0:
            continue
        fit = np.flatnonzero(np.abs(block - b) > 1)
        if fit.size < minimum:
            fit = np.flatnonzero(block != b)
            kept = True
        folds.append((pred, fit))
    return folds, kept


def _fold_decomposition(s: np.ndarray, U: np.ndarray, pred: np.ndarray, fit: np.ndarray, n: int):
    """(predicted samples, fitting samples, eigenvalues e and vectors V of the
    centred Gram matrix of the factor's columns Phi = U diag(sqrt(s)) over
    the fitting samples, and Q = (Phi_pred - their mean) V). Phi has centred
    columns and Phi' Phi = diag(s), so the Gram matrix over the fitting
    samples is diag(s) minus that over the samples left out."""
    sq = np.sqrt(s)
    drop = np.setdiff1d(np.arange(n), fit, assume_unique=True)
    Phi_drop = U[drop] * sq
    mu = -Phi_drop.sum(axis=0) / fit.size
    gram = np.diag(s) - Phi_drop.T @ Phi_drop - fit.size * np.outer(mu, mu)
    e, V = np.linalg.eigh((gram + gram.T) / 2.0)
    return pred, fit, np.maximum(e, 0.0), V, (U[pred] * sq - mu) @ V


def _centred_spectrum(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues above the numerical-rank tolerance, and their eigenvectors,
    of the doubly centred kernel C K C (C = I - 11'/n)."""
    Kc = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
    s, U = np.linalg.eigh((Kc + Kc.T) / 2.0)
    keep = s > max(float(s.max()), 0.0) * K.shape[0] * np.finfo(float).eps
    return s[keep], U[:, keep]


def _spectrum_of_factor(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centred spectrum of G G', from the thin singular value decomposition
    of G with centred columns."""
    U, sv, _ = np.linalg.svd(G - G.mean(axis=0), full_matrices=False)
    s = sv ** 2
    keep = s > max(float(s.max()), 0.0) * G.shape[0] * np.finfo(float).eps
    return s[keep], U[:, keep]


def _low_rank_spectrum(Zs: np.ndarray, sq_norms: np.ndarray, bw: float):
    """(centred spectrum s, U, cap share, half) of the pivoted-Cholesky
    approximation of the radial basis function kernel. When the factor
    reached the cap, half is the spectrum of its first half of columns, which
    is the pivoted-Cholesky factor of half the rank (the pivots are greedy),
    for the rank check (ConditioningRegression.rank_change); else None."""
    G, share = pivoted_cholesky(np.ones(Zs.shape[0]), _rbf_column(Zs, sq_norms, bw), _LOW_RANK,
                                kind="conditioning")
    s, U = _spectrum_of_factor(G)
    half = _spectrum_of_factor(G[:, :max(1, G.shape[1] // 2)]) if share > 0 else None
    return s, U, share, half


def spectrum_tasks(Z: np.ndarray) -> list:
    """One task per trial bandwidth, each returning (centred spectrum s, U,
    cap share, half-rank spectrum or None; see _low_rank_spectrum) of the
    radial basis function kernel on the standardized Z: the
    decompositions a ConditioningRegression needs first, as independent tasks
    that can share a parallel pool with others."""
    Zs = standardize(as2d(np.asarray(Z, dtype=float)))
    base = _median_bandwidth(Zs)
    if _use_low_rank(Zs.shape[0]):
        sq_norms = (Zs ** 2).sum(1)
        return [lambda m=m: _low_rank_spectrum(Zs, sq_norms, base * m) for m in BANDWIDTH_MULTIPLIERS]
    sq = _sq_dists(Zs, Zs)
    return [lambda m=m: (*_centred_spectrum(np.exp(-sq / (2.0 * (base * m) ** 2))), 0.0, None)
            for m in BANDWIDTH_MULTIPLIERS]


class ConditioningRegression:
    """Kernel ridge regression on one conditioning set, cross-fitted on
    blocks (module docstring), with the decompositions shared by all the
    variables regressed on it.

    buffer : the dependence lag, the shortest block length.
    blocks_short : the record spans fewer than MIN_CROSSFIT_BLOCKS lags, or a
        fit kept its neighbouring blocks for lack of samples.
    cap_share : largest share of a kernel's trace left out by a factor that
        reached the rank cap (0 when none did).
    rank_change : largest relative change in a fit's cross-fitted residual
        sum of squares between the factor of half the rank and the factor at
        the cap, over the fits whose chosen bandwidth reached the cap (0 when
        none did). With error decaying at least geometrically in the rank
        (Harbrecht, Peters and Schneider, 2012), it bounds the change still
        to come from raising the rank.
    """

    def __init__(self, Z: np.ndarray, time_positions: np.ndarray, buffer: float = 0.0,
                 spectra: list | None = None):
        """spectra: the results of spectrum_tasks(Z), when they were computed
        elsewhere; computed here otherwise. The blocks' decompositions, one per
        bandwidth and block, are computed here as parallel tasks."""
        Z = standardize(as2d(np.asarray(Z, dtype=float)))
        self.n = Z.shape[0]
        block, k, short = crossfit_blocks(time_positions, buffer)
        self.n_blocks = k
        folds, kept = _folds(block, k, Z.shape[1])
        self.blocks_short = bool(short or kept)
        from model_selection.parallel import run_tasks
        if spectra is None:
            spectra = run_tasks(spectrum_tasks(Z))
        n = self.n
        jobs = [(i, False, sp[0], sp[1]) for i, sp in enumerate(spectra)]
        jobs += [(i, True, sp[3][0], sp[3][1]) for i, sp in enumerate(spectra) if sp[3] is not None]
        decomps = run_tasks([lambda s=s_, U=U_, pr=pr, ft=ft: _fold_decomposition(s, U, pr, ft, n)
                             for _, _, s_, U_ in jobs for pr, ft in folds])
        k = len(folds)
        by_job = {(i, half): decomps[j * k:(j + 1) * k] for j, (i, half, _, _) in enumerate(jobs)}
        self.spectra = [(sp[0], sp[1], sp[2], by_job[(i, False)],
                         None if sp[3] is None else (sp[3][0], sp[3][1], by_job[(i, True)]))
                        for i, sp in enumerate(spectra)]
        self.cap_share = max(float(sp[2]) for sp in spectra)
        self.rank_change = 0.0
        import threading
        self._lock = threading.Lock()

    @staticmethod
    def _fold_terms(Yc: np.ndarray, s: np.ndarray, U: np.ndarray, folds: list, n: int) -> list:
        """Per fold: (predicted samples, fitting mean, g = V' Phi_fit'(Y_fit -
        mean), Q, centred predicted targets) for one bandwidth."""
        sq = np.sqrt(s)
        PhiY = (U * sq).T @ Yc                         # Phi' Y over all samples
        out = []
        for pred, fit, e, V, Q in folds:
            drop = np.setdiff1d(np.arange(n), fit, assume_unique=True)
            y_mean = Yc[fit].mean(axis=0)
            PhiY_fit = PhiY - (U[drop] * sq).T @ Yc[drop]   # Phi_fit' Y_fit
            Phi_sum = -(U[drop] * sq).sum(axis=0)           # sum of Phi over the fitting samples
            g = V.T @ (PhiY_fit - np.outer(Phi_sum, y_mean))
            out.append((pred, y_mean, g, Q, Yc[pred] - y_mean))
        return out

    @staticmethod
    def _residuals_at(terms: list, folds: list, lam: float, shape) -> np.ndarray:
        resid = np.empty(shape)
        for (pred, _, g, Q, a), (_, _, e, _, _) in zip(terms, folds):
            resid[pred] = a - Q @ (g / (e + lam)[:, None])
        return resid

    def _fit(self, Y: np.ndarray, r: int):
        """(spectrum index, penalty, cross-fitted residuals of every column of
        Y): the bandwidth and penalty minimize the blocked cross-validation
        error of the first r columns. When the chosen bandwidth's factor
        reached the rank cap, the same fit at half the rank updates
        rank_change."""
        Yc = Y - Y.mean(axis=0)
        best = (np.inf, None)
        for b, (s, U, _, folds, _) in enumerate(self.spectra):
            terms = self._fold_terms(Yc, s, U, folds, self.n)
            err = np.zeros(len(RIDGE_PENALTIES))
            for (pred, _, g, Q, a), (_, _, e, _, _) in zip(terms, folds):
                for j, lam in enumerate(RIDGE_PENALTIES):
                    err[j] += float(np.sum((a[:, :r] - Q @ (g[:, :r] / (e + lam)[:, None])) ** 2))
            j = int(np.argmin(err))
            if err[j] < best[0]:
                best = (err[j], (b, RIDGE_PENALTIES[j], terms, folds))
        rss, (b, lam, terms, folds) = best
        resid = self._residuals_at(terms, folds, lam, Yc.shape)
        half = self.spectra[b][4]
        if half is not None and rss > 0:
            s_h, U_h, folds_h = half
            resid_h = self._residuals_at(self._fold_terms(Yc[:, :r], s_h, U_h, folds_h, self.n), folds_h, lam,
                                         (Yc.shape[0], r))
            change = abs(float(np.sum(resid_h ** 2)) - rss) / rss
            with self._lock:
                self.rank_change = max(self.rank_change, change)
        return b, lam, resid

    def effective_df(self, target: np.ndarray) -> float:
        """Effective degrees of freedom of the fit to `target`, excluding the
        intercept: the trace of its centred smoother matrix on all samples
        (Buja, Hastie and Tibshirani, 1989), on the scale of the number of
        regressors."""
        T = as2d(np.asarray(target, dtype=float))
        b, lam, _ = self._fit(T, T.shape[1])
        s = self.spectra[b][0]
        return float(np.sum(s / (s + lam)))

    def residuals(self, target: np.ndarray, extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Cross-fitted residuals of `target` (n, r) and, with the same
        bandwidth and penalty, of `extra` (n, s). The bandwidth and penalty
        minimize the blocked cross-validation error summed over the columns of
        `target`."""
        T = as2d(np.asarray(target, dtype=float))
        X = None if extra is None else as2d(np.asarray(extra, dtype=float))
        Y = T if X is None else np.column_stack([T, X])
        r = T.shape[1]
        _, _, resid = self._fit(Y, r)
        return resid[:, :r], (None if X is None else resid[:, r:])


def gram(E: np.ndarray) -> np.ndarray:
    """Inner products of the rows of E."""
    return E @ E.T


import threading

_SHARED: list = []
_SHARED_LOCK = threading.Lock()
SHARED_REGRESSIONS = 2
"""How many conditioning regressions shared_regression keeps (each holds one
n x n eigendecomposition per trial bandwidth)."""


def _shared_key(Z2: np.ndarray, t: np.ndarray, buffer: float):
    import hashlib
    columns = sorted(hashlib.sha1(np.ascontiguousarray(Z2[:, j]).tobytes()).hexdigest() for j in range(Z2.shape[1]))
    return (tuple(columns), Z2.shape[0], hashlib.sha1(t.tobytes()).hexdigest(), float(buffer))


def cached_regression(Z: np.ndarray, time_positions: np.ndarray, buffer: float = 0.0) -> ConditioningRegression | None:
    """The shared regression on Z if one is kept, else None (nothing built)."""
    key = _shared_key(as2d(np.asarray(Z, dtype=float)), np.asarray(time_positions, dtype=float), buffer)
    with _SHARED_LOCK:
        return next((reg for k, reg in _SHARED if k == key), None)


def shared_regression(Z: np.ndarray, time_positions: np.ndarray, buffer: float = 0.0,
                      spectra: list | None = None) -> ConditioningRegression:
    """A ConditioningRegression on Z, reused when the same conditioning set
    recurs, as when a comparison tests C beyond (Z, W) and then W beyond
    (Z, C): both effect sizes regress on {Z, W, C}. The fit does not depend
    on the order of Z's columns (standardized columns, distances summed over
    them), so the key ignores it. Exact: a reused regression is the one that
    would be built again. spectra: precomputed results of spectrum_tasks(Z).
    The cache is guarded by a lock; a regression is built outside it, so
    regressions on different sets are built in parallel."""
    Z2 = as2d(np.asarray(Z, dtype=float))
    t = np.asarray(time_positions, dtype=float)
    key = _shared_key(Z2, t, buffer)
    with _SHARED_LOCK:
        for i, (k, reg) in enumerate(_SHARED):
            if k == key:
                _SHARED.insert(0, _SHARED.pop(i))
                return reg
    reg = ConditioningRegression(Z2, t, buffer=buffer, spectra=spectra)
    with _SHARED_LOCK:
        _SHARED.insert(0, (key, reg))
        del _SHARED[SHARED_REGRESSIONS:]
    return reg
