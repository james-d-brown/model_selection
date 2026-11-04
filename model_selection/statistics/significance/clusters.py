"""
How the tests use clusters of dependent samples (docs/methods.md,
Dependence between samples; Appendix C, High-flow event samples).

Every resampling component takes samples' time positions and a dependence
lag, and treats samples less than one lag apart as dependent and samples at
least one lag apart as independent: the dependent wild bootstrap's
multipliers (their Bartlett correlation is one at no separation and zero at
one lag), the cross-fitting and permutation blocks, and the free-form tier's
Theiler window. Clusters are therefore given to them as coordinates: every
sample of a cluster at the same position, and consecutive clusters exactly
CLUSTER_SPACING apart, with a dependence lag of CLUSTER_SPACING. Then
- the samples of a cluster share one multiplier (the wild cluster
  bootstrap; Cameron, Gelbach and Miller, 2008), and different clusters
  receive independent multipliers;
- a cluster is never split between cross-fitting or permutation blocks;
- samples of the same cluster are never permutation partners.
The rule that forms the clusters (event_detection.dependence) is separate
from this.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CLUSTER_SPACING = 2
"""Spacing of consecutive clusters, and the dependence lag that goes with it
(an integer above one, so that the multipliers' Bartlett construction, which
is evaluated on whole time steps, applies)."""


def cluster_coordinates(labels: np.ndarray, time_positions: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """(position of each sample, dependence lag): each cluster at its own
    position, in time order of its first sample (or label order without
    time positions), CLUSTER_SPACING apart. The last cluster is placed one
    unit further out, so that the positions are never evenly spaced: evenly
    spaced positions would be read as a regular time grid, on which the
    block length is estimated from the samples' order, whereas cluster
    positions carry no time (statistics.significance.block_length)."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if time_positions is None:
        rank = {lab: k for k, lab in enumerate(uniq)}
    else:
        t = np.asarray(time_positions, dtype=float)
        first = np.array([t[labels == lab].min() for lab in uniq])
        rank = {lab: k for k, lab in enumerate(uniq[np.argsort(first, kind="stable")])}
    last = len(uniq) - 1
    positions = np.array([rank[lab] * CLUSTER_SPACING + (rank[lab] == last and last > 0) for lab in labels],
                         dtype=np.int64)
    return positions, CLUSTER_SPACING


def cluster_summary(labels: np.ndarray) -> dict:
    """Number of samples and clusters, and the number of clusters of each size."""
    sizes = np.bincount(np.unique(np.asarray(labels), return_inverse=True)[1])
    by_size = np.bincount(sizes)
    return {"n_samples": int(sizes.sum()), "n_clusters": int(sizes.size),
            "n_in_multi_sample_clusters": int(sizes[sizes > 1].sum()), "largest_cluster": int(sizes.max()),
            "clusters_by_size": {int(s): int(c) for s, c in enumerate(by_size) if c and s}}


def consecutive_cluster_correlation(terms: np.ndarray, positions: np.ndarray, alpha: float = 0.05) -> dict:
    """Correlation of a statistic's terms between consecutive clusters.

    terms : (n, p) per-sample terms of the statistic (for example the
        products of two residual series), centred here. positions : the
        cluster coordinates of the samples (cluster_coordinates).
    The terms are summed within each cluster, and the lag-one correlation
    of the cluster sums, in the inner product of the terms, is
    sum_c <S_c, S_c+1> / sum_c <S_c, S_c>. Under independence of the
    clusters its standard error is about 1/sqrt(m) for m clusters (Bartlett,
    1946), and it is significant when it exceeds the two-sided normal
    quantile at alpha times that. Returns {correlation, n_clusters,
    standard_error, significant}; correlation None with fewer than three
    clusters."""
    X = np.asarray(terms, dtype=float)
    X = X.reshape(X.shape[0], -1)
    X = X - X.mean(axis=0)
    pos = np.asarray(positions)
    uniq, inverse = np.unique(pos, return_inverse=True)
    m = uniq.size
    if m < 3:
        return {"correlation": None, "n_clusters": int(m), "standard_error": None, "significant": False}
    S = np.zeros((m, X.shape[1]))
    np.add.at(S, inverse, X)
    denom = float(np.sum(S * S))
    r = float(np.sum(S[:-1] * S[1:]) / denom) if denom > 0 else 0.0
    se = 1.0 / np.sqrt(m)
    return {"correlation": r, "n_clusters": int(m), "standard_error": float(se),
            "significant": bool(abs(r) > stats.norm.ppf(1 - alpha / 2) * se)}
