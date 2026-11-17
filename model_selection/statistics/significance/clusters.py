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

With one Rademacher sign per independent unit, the tests' nulls draw from
2^G patterns of signs (the wild cluster restricted bootstrap; multipliers
module). For least squares on unscaled residuals, a quadratic statistic
takes the same value under a pattern and its negation, and the observed
value is that of the pattern of all +1, so no p-value can be smaller than
2 / 2^G = 2^(1 - G). The tests depart from that ideal: the CR2 scaling of
the residuals (cluster_scaled_residuals) changes their direction within a
cluster, so no draw need reproduce the observed statistic, and the
nonlinear tier's rebuilt draws are not even in the signs. The floor is
therefore a necessary condition, not a guarantee: a test at level alpha is
treated as untestable, rather than not significant, with fewer than
minimum_units(alpha) units, where even the ideal test could not reject. Above it, the null's level rests on the number of clusters
growing (Djogbenou, MacKinnon and Nielsen, 2019); no result fixes it at a
few clusters of a few events.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CLUSTER_SPACING = 2
"""Spacing of consecutive clusters, and the dependence lag that goes with it
(an integer above one, so that the multipliers' Bartlett construction, which
is evaluated on whole time steps, applies)."""


def smallest_p_value(n_units: int) -> float:
    """The smallest p-value a sign-flip randomization of n_units independent
    units can give for a quadratic statistic, 2^(1 - n_units) (module
    docstring)."""
    return float(2.0 ** (1 - int(n_units)))


def minimum_units(level: float) -> int:
    """The fewest independent units at which a sign-flip randomization can
    reject at `level`: the smallest G with 2^(1 - G) < level."""
    g = 1
    while smallest_p_value(g) >= level:
        g += 1
    return g


def orthonormal_residual_maker(design: np.ndarray) -> np.ndarray:
    """I - P for least squares on `design` (n, k; include the intercept), P
    the projection onto its column space at numerical rank."""
    U, sv, _ = np.linalg.svd(np.asarray(design, dtype=float), full_matrices=False)
    U = U[:, sv > sv[0] * max(U.shape) * np.finfo(float).eps] if sv.size and sv[0] > 0 else U[:, :0]
    return np.eye(U.shape[0]) - U @ U.T


def cluster_scaled_residuals(residuals: np.ndarray, residual_maker: np.ndarray,
                             positions: np.ndarray) -> np.ndarray:
    """Residuals of a null model's fit with each unit's (cluster's) residual
    vector multiplied by B_g^(-1/2), B_g = (R R')_gg the unit's block of R R'
    for the fit's residual maker R (r = R y): the CR2 correction (Bell and
    McCaffrey, 2002; Pustejovsky and Tipton, 2018), which makes the residuals'
    covariance that of the errors, block by block, when the errors are
    independent with equal variance. For least squares R R' = I - P and
    B_g = I - P_gg; for a unit of one sample it is the HC2 correction, 1 /
    sqrt(1 - h_t) (Davidson and Flachaire, 2008). Restricted residuals
    understate the errors by the fit's projection, and with regressors and
    errors shared within clusters that loss is a whole block, not a diagonal,
    so the wild cluster bootstrap's residuals are scaled by cluster
    (MacKinnon, Nielsen and Webb, 2023, use cluster-level corrections
    throughout). Scaling the bootstrap's residuals by CR2 follows their
    logic; it is not their published method, which corrects variance
    estimates and scores. B_g^(-1/2) is taken on B_g's positive eigenvalues
    (its pseudo-inverse square root, as Bell and McCaffrey do when a block is
    singular). positions: each sample's unit; samples sharing a position form
    one unit."""
    r = np.asarray(residuals, dtype=float)
    r2 = r.reshape(r.shape[0], -1)
    R = np.asarray(residual_maker, dtype=float)
    _, unit = np.unique(np.asarray(positions), return_inverse=True)
    out = np.empty_like(r2)
    for g in range(int(unit.max()) + 1):
        idx = np.flatnonzero(unit == g)
        Rg = R[idx]
        vals, vecs = np.linalg.eigh(Rg @ Rg.T)
        keep = vals > max(float(vals.max()), 0.0) * 1e-10 if vals.size else vals > 0
        inv_sqrt = (vecs[:, keep] / np.sqrt(vals[keep])) @ vecs[:, keep].T
        out[idx] = inv_sqrt @ r2[idx]
    return out.reshape(r.shape)


def effective_clusters(residuals: np.ndarray, positions: np.ndarray) -> float:
    """The effective number of clusters G* = (sum_g gamma_g)^2 / sum_g
    gamma_g^2, gamma_g the squared norm of unit g's summed residuals (Carter,
    Schnepel and Steigerwald, 2017, in their conservative case of perfect
    correlation within a cluster): G for equal clusters, fewer when a few
    clusters dominate. Computed from the candidate's residuals only, so that
    it does not depend on the tested association. With clusters of single
    events gamma_g is each event's squared residual, so G* then measures how
    unevenly the candidate's residuals are spread, not how the events
    cluster; it is reported as a diagnostic, and the power calculation uses
    the number of clusters (statistics.power)."""
    r = np.asarray(residuals, dtype=float)
    r2 = r.reshape(r.shape[0], -1)
    _, unit = np.unique(np.asarray(positions), return_inverse=True)
    sums = np.zeros((int(unit.max()) + 1, r2.shape[1]))
    np.add.at(sums, unit, r2)
    gamma = np.sum(sums ** 2, axis=1)
    total = float(np.sum(gamma ** 2))
    return float(np.sum(gamma)) ** 2 / total if total > 0 else 0.0


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
