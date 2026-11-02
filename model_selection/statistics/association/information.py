"""
KSG k-nearest-neighbour information estimators.

ksg_mi:  Kraskov, Stogbauer & Grassberger (2004), Algorithm 1: I(X;Y).
ksg_cmi: Frenzel & Pompe (2007) conditional extension: I(X;Y|Z).

Both use the max-norm joint-space kNN distance, make no parametric or
linearity assumption, and accept multivariate inputs. An empty
conditioning set reduces ksg_cmi to ksg_mi, so the same estimator serves the
unconditional first pick and every later conditional step of selection.
Tree queries use `workers=-1` (all cores; scipy releases the GIL).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from model_selection.statistics.association.base import CachedAssociationEstimator
from model_selection.utils import as2d


def _count_within(tree: cKDTree, points: np.ndarray, radii: np.ndarray, workers: int = -1) -> np.ndarray:
    """Count neighbours (excluding the point itself) strictly within each
    point's Chebyshev radius.
    """
    radii = np.asarray(radii, dtype=np.float64)
    # query_ball_point returns points with distance <= r; KSG needs strict "<".
    # Standard fix: shrink r by a small relative epsilon before counting.
    shrunk = np.where(radii > 0, radii * (1 - 1e-12), 0.0)
    counts = tree.query_ball_point(points, r=shrunk, p=np.inf, workers=workers, return_length=True)
    return np.asarray(counts, dtype=np.int64) - 1  # exclude self


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5, workers: int = -1) -> float:
    """KSG Algorithm 1 estimate of I(X;Y) in nats.

    x, y : (n,) or (n, d), already in the space MI should be computed in.
    k    : joint-space neighbour count (5 is the usual bias/variance balance).
    workers : passed to scipy tree queries (-1 = all cores).
    """
    x = as2d(x)
    y = as2d(y)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of samples")
    if k >= n:
        raise ValueError(f"k={k} must be < n_samples={n}")

    joint = np.hstack([x, y])
    joint_tree = cKDTree(joint)
    # distance to kth neighbor (excluding self) in max-norm joint space
    dists, _ = joint_tree.query(joint, k=k + 1, p=np.inf, workers=workers)
    eps = dists[:, -1]

    x_tree = cKDTree(x)
    y_tree = cKDTree(y)
    n_x = _count_within(x_tree, x, eps, workers=workers)
    n_y = _count_within(y_tree, y, eps, workers=workers)

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(mi)


def ksg_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray | None, k: int = 5, workers: int = -1) -> float:
    """Frenzel & Pompe (2007) estimate of I(X;Y|Z) in nats; reduces to ksg_mi when
    z is None or empty.

    CMI = psi(k) + < psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1) >, with the kth-NN
    distance taken in the joint (X,Y,Z) space.
    """
    x = as2d(x)
    y = as2d(y)
    if z is None or (np.asarray(z).size == 0):
        return ksg_mi(x, y, k=k, workers=workers)
    z = as2d(z)
    n = x.shape[0]
    if not (y.shape[0] == n and z.shape[0] == n):
        raise ValueError("x, y, z must have the same number of samples")
    if k >= n:
        raise ValueError(f"k={k} must be < n_samples={n}")

    joint = np.hstack([x, y, z])
    joint_tree = cKDTree(joint)
    dists, _ = joint_tree.query(joint, k=k + 1, p=np.inf, workers=workers)
    eps = dists[:, -1]

    xz = np.hstack([x, z])
    yz = np.hstack([y, z])

    xz_tree = cKDTree(xz)
    yz_tree = cKDTree(yz)
    z_tree = cKDTree(z)

    n_xz = _count_within(xz_tree, xz, eps, workers=workers)
    n_yz = _count_within(yz_tree, yz, eps, workers=workers)
    n_z = _count_within(z_tree, z, eps, workers=workers)

    cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
    return float(cmi)


class CachedCMI(CachedAssociationEstimator):
    """ksg_cmi/ksg_mi with obs and conditioning fixed.

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

    def _kth_distance(self, joint: np.ndarray) -> np.ndarray:
        """Distance to each sample's k-th nearest neighbour in the joint space,
        among the samples outside its Theiler window."""
        k, n, workers = self.k, self.n, self.workers
        tree = cKDTree(joint)
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
        """Neighbours strictly within each sample's radius, outside its Theiler window."""
        counts = _count_within(tree, points, eps, workers=self.workers)
        if self.excluded is None:
            return counts
        valid = self.excluded >= 0
        diff = np.abs(points[np.where(valid, self.excluded, 0)] - points[:, None, :]).max(axis=2)
        shrunk = np.where(eps > 0, eps * (1 - 1e-12), 0.0)
        return counts - np.sum(valid & (diff <= shrunk[:, None]), axis=1)

    def __call__(self, candidate: np.ndarray, obs: np.ndarray | None = None,
                 conditioning: np.ndarray | None = None) -> float:
        x = as2d(candidate)
        k, n, workers = self.k, self.n, self.workers
        if x.shape[0] != n:
            raise ValueError("candidate must have the same number of samples as obs")
        if k >= n:
            raise ValueError(f"k={k} must be < n_samples={n}")

        if self.z is None:
            joint = np.hstack([x, self.obs])
            eps = self._kth_distance(joint)
            n_x = self._count(cKDTree(x), x, eps)
            n_y = self._count(self.y_tree, self.obs, eps)
            # Each sample's neighbours are drawn from the samples outside its window.
            n_ref = n if self.excluded is None else n - np.sum(self.excluded >= 0, axis=1)
            mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + np.mean(digamma(n_ref))
            return float(mi)

        joint = np.hstack([x, self.obs, self.z])
        eps = self._kth_distance(joint)
        xz = np.hstack([x, self.z])
        n_xz = self._count(cKDTree(xz), xz, eps)
        n_yz = self._count(self.yz_tree, self.yz, eps)
        n_z = self._count(self.z_tree, self.z, eps)

        cmi = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))
        return float(cmi)
