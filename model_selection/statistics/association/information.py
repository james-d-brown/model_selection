"""
k-nearest-neighbour estimators of (conditional) mutual information for
continuous, discrete or mixed data.

knn_cmi: Mesner and Shalizi (2021), I(X;Y|Z), eqs. (14), (18), (19).
knn_mi:  the same with Z empty, the mixture estimator of Gao et al. (2017).

The k-th neighbour distance rho_i is taken in the max-norm joint space. All
counts exclude the sample itself and include samples at exactly rho_i
(Mesner and Shalizi, 2021, eq. 14). k~_i counts the joint-space samples
within rho_i (eq. 18). It exceeds k only where samples lie exactly at
rho_i: where samples coincide, or where a point mass puts many samples at
the same distance, as samples just above a censoring threshold are from
every censored sample near them in Z. Each sample's term is
    psi(k) - psi(n_xz) - psi(n_yz) + psi(n_z)    when k~_i = k,
    log(k~_i) - log(n_xz) - log(n_yz) + log(n_z) otherwise (eq. 19),
and the estimate is the mean of the terms. With Z empty, n_z is the number
of other samples. The estimate is not clipped at zero: a permutation test
compares it with null draws on the same scale.

On continuous values k~_i = k almost surely, and the estimator differs from
Frenzel and Pompe (2007) only in finite samples: it counts boundary samples
and uses psi(n) where Frenzel and Pompe use psi(n + 1). On data with ties it
remains consistent (Mesner and Shalizi, 2021, Theorem 3.1); the Frenzel-Pompe
and KSG estimators do not, and adding noise to break the ties degrades them
(Gao et al., 2017). Values on a regular grid, such as integer ranks, break
the almost-sure equality: distinct samples then often lie exactly at rho_i,
and eq. 18 counts them as it counts a point mass.
Tree queries use `workers=-1` (all cores; scipy releases the GIL).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from model_selection.statistics.association.base import CachedAssociationEstimator
from model_selection.utils import as2d


def _count_within(tree: cKDTree, points: np.ndarray, radii: np.ndarray, workers: int = -1) -> np.ndarray:
    """Count neighbours (excluding the point itself) within each point's
    Chebyshev radius, boundary included."""
    counts = tree.query_ball_point(points, r=np.asarray(radii, dtype=np.float64), p=np.inf, workers=workers,
                                   return_length=True)
    return np.asarray(counts, dtype=np.int64) - 1  # exclude self


def _terms(k: int, k_tilde: np.ndarray, n_xz: np.ndarray, n_yz: np.ndarray, n_z: np.ndarray) -> np.ndarray:
    """Mesner and Shalizi (2021, eq. 19): digamma where no samples coincide
    at the k-th neighbour distance (k~ = k), log otherwise. Counts are at
    least 1, which matters only for a sample whose Theiler window leaves
    fewer than k others; Mesner and Shalizi (2021, Section 4) apply the same
    floor to the Frenzel-Pompe counts."""
    cont = k_tilde == k
    k_tilde, n_xz, n_yz, n_z = (np.maximum(a, 1) for a in (k_tilde, n_xz, n_yz, n_z))
    return np.where(cont, digamma(k) - digamma(n_xz) - digamma(n_yz) + digamma(n_z),
                    np.log(k_tilde) - np.log(n_xz) - np.log(n_yz) + np.log(n_z))


def knn_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray | None, k: int = 5, workers: int = -1) -> float:
    """Mesner and Shalizi (2021) estimate of I(X;Y|Z) in nats; with z None or
    empty, the Gao et al. (2017) estimate of I(X;Y)."""
    return float(CachedCMI(y, z, k=k, workers=workers)(x))


def knn_mi(x: np.ndarray, y: np.ndarray, k: int = 5, workers: int = -1) -> float:
    """Gao et al. (2017) estimate of I(X;Y) in nats (knn_cmi with Z empty)."""
    return knn_cmi(x, y, None, k=k, workers=workers)


class CachedCMI(CachedAssociationEstimator):
    """knn_cmi/knn_mi with obs and conditioning fixed.

    The KD-trees that do not depend on the candidate (y, yz, z) are built once
    at construction; only the joint and xz trees are rebuilt per call. This is
    what makes the permutation-test loop affordable.

    theiler_window : with time_positions, samples closer than this in time
        are excluded from each other's neighbour searches and counts (the
        Theiler window), as Theiler (1986) proposed for nearest-neighbour
        estimates from time series: temporal neighbours lie close in every
        variable and would otherwise inflate the estimate. 0 excludes nothing.
    """

    def __init__(self, obs: np.ndarray, conditioning: np.ndarray | None, k: int = 5, workers: int = -1,
                 time_positions: np.ndarray | None = None, theiler_window: float = 0.0):
        self.k = k
        self.workers = workers
        self.obs = as2d(obs)
        self.n = self.obs.shape[0]
        if conditioning is None or np.asarray(conditioning).size == 0:
            self.z = None
            self.y_tree = cKDTree(self.obs)
        else:
            self.z = as2d(conditioning)
            if self.z.shape[0] != self.n:
                raise ValueError("obs and conditioning must have the same number of samples")
            self.yz = np.hstack([self.obs, self.z])
            self.yz_tree = cKDTree(self.yz)
            self.z_tree = cKDTree(self.z)
        self.t = None
        self.window = float(theiler_window) if time_positions is not None else 0.0
        self.excluded = None  # (n, m) indices of each sample's temporal neighbours, -1 padded
        if self.window > 0:
            t = np.asarray(time_positions, dtype=float)
            if t.shape[0] != self.n:
                raise ValueError("time_positions must have the same number of samples as obs")
            order = np.argsort(t, kind="stable")
            lo = np.searchsorted(t[order], t - self.window, side="right")
            hi = np.searchsorted(t[order], t + self.window, side="left")
            lists = [np.setdiff1d(order[lo[i]:hi[i]], [i]) for i in range(self.n)]
            m = max((len(lst) for lst in lists), default=0)
            if m > 0:
                self.t = t
                self.excluded = np.full((self.n, m), -1, dtype=np.int64)
                for i, lst in enumerate(lists):
                    self.excluded[i, :len(lst)] = lst

    def _radius_and_k_tilde(self, joint: np.ndarray, tree: cKDTree) -> tuple[np.ndarray, np.ndarray]:
        """rho, the distance to each sample's k-th nearest neighbour in the
        joint space outside its Theiler window, and k~, the samples within rho
        (Mesner and Shalizi, 2021, eq. 18). Without a Theiler window, k~ = k
        exactly where the (k+1)-th neighbour lies beyond rho, so the joint
        count is made only where it lies at rho."""
        k, n = self.k, self.n
        if self.excluded is None and k + 2 <= n:
            dists, _ = tree.query(joint, k=k + 2, p=np.inf, workers=self.workers)
            eps = dists[:, k]
            k_tilde = np.full(n, k, dtype=np.int64)
            at = dists[:, k + 1] <= eps
            if np.any(at):
                k_tilde[at] = _count_within(tree, joint[at], eps[at], workers=self.workers)
            return eps, k_tilde
        eps = self._kth_distance(joint, tree)
        return eps, self._count(tree, joint, eps)

    def _kth_distance(self, joint: np.ndarray, tree: cKDTree) -> np.ndarray:
        """Distance to each sample's k-th nearest neighbour in the joint space,
        among the samples outside its Theiler window."""
        k, n, workers = self.k, self.n, self.workers
        if self.excluded is None:
            dists, _ = tree.query(joint, k=k + 1, p=np.inf, workers=workers)
            return dists[:, -1]
        k_query = min(n, k + 1 + self.excluded.shape[1])
        dists, idx = tree.query(joint, k=k_query, p=np.inf, workers=workers)
        dists, idx = np.atleast_2d(dists), np.atleast_2d(idx)
        allowed = np.abs(self.t[idx] - self.t[:, None]) >= self.window
        rank = np.cumsum(allowed, axis=1)
        hit = allowed & (rank == k)
        eps = np.where(hit.any(axis=1), dists[np.arange(n), np.argmax(hit, axis=1)], dists[:, -1])
        return eps

    def _count(self, tree: cKDTree, points: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Neighbours within each sample's radius, boundary included, outside
        its Theiler window."""
        counts = _count_within(tree, points, eps, workers=self.workers)
        if self.excluded is None:
            return counts
        valid = self.excluded >= 0
        diff = np.abs(points[np.where(valid, self.excluded, 0)] - points[:, None, :]).max(axis=2)
        return counts - np.sum(valid & (diff <= eps[:, None]), axis=1)


    def __call__(self, candidate: np.ndarray, obs: np.ndarray | None = None,
                 conditioning: np.ndarray | None = None) -> float:
        x = as2d(candidate)
        k, n = self.k, self.n
        if x.shape[0] != n:
            raise ValueError("candidate must have the same number of samples as obs")
        if k >= n:
            raise ValueError(f"k={k} must be < n_samples={n}")

        if self.z is None:
            joint = np.hstack([x, self.obs])
            eps, k_tilde = self._radius_and_k_tilde(joint, cKDTree(joint))
            n_x = self._count(cKDTree(x), x, eps)
            n_y = self._count(self.y_tree, self.obs, eps)
            # With Z empty every other sample outside the window is within any radius.
            n_z = (n - 1) if self.excluded is None else (n - 1 - np.sum(self.excluded >= 0, axis=1))
            return float(np.mean(_terms(k, k_tilde, n_x, n_y, np.broadcast_to(n_z, n_x.shape))))

        joint = np.hstack([x, self.obs, self.z])
        eps, k_tilde = self._radius_and_k_tilde(joint, cKDTree(joint))
        xz = np.hstack([x, self.z])
        n_xz = self._count(cKDTree(xz), xz, eps)
        n_yz = self._count(self.yz_tree, self.yz, eps)
        n_z = self._count(self.z_tree, self.z, eps)
        return float(np.mean(_terms(k, k_tilde, n_xz, n_yz, n_z)))
