"""Tests of how the resampling components treat clusters of dependent
samples given as cluster coordinates (statistics.significance.clusters):
one multiplier per cluster, clusters never split, no permutation partners
within a cluster, clusters counted as the independent units, and the check
of correlation between consecutive clusters."""
import numpy as np
import pytest

from model_selection.statistics.conditional_independence import LinearResidualTest
from model_selection.statistics.conditional_independence.multipliers import dependent_wild_weights
from model_selection.statistics.significance.block_length import resolve_block, sampling_stride
from model_selection.statistics.significance.clusters import (
    CLUSTER_SPACING, cluster_coordinates, cluster_summary, consecutive_cluster_correlation,
)

LABELS = np.array([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])


def test_cluster_coordinates_put_a_cluster_at_one_position_and_clusters_a_lag_apart():
    pos, lag = cluster_coordinates(LABELS, np.arange(LABELS.size) * 10.0)
    assert lag == CLUSTER_SPACING
    for lab in np.unique(LABELS):
        assert np.unique(pos[LABELS == lab]).size == 1
    distinct = np.unique(pos)
    assert distinct.size == 6 and np.all(np.diff(distinct) >= lag)
    assert sampling_stride(pos) is None     # never read as a regular time grid
    # Clusters are placed in time order of their first sample.
    pos2, _ = cluster_coordinates(np.array([1, 1, 0]), np.array([0.0, 5.0, 9.0]))
    assert pos2[0] == pos2[1] < pos2[2]
    assert cluster_summary(LABELS) == {"n_samples": 10, "n_clusters": 6, "n_in_multi_sample_clusters": 7,
                                       "largest_cluster": 3, "clusters_by_size": {1: 3, 2: 2, 3: 1}}


@pytest.mark.parametrize("independent", ["gaussian", "rademacher"])
def test_samples_of_a_cluster_share_one_multiplier_and_clusters_are_independent(independent):
    pos, lag = cluster_coordinates(LABELS)
    W = dependent_wild_weights(pos, lag, 20000, np.random.default_rng(1), independent=independent)
    for lab in np.unique(LABELS):
        cols = W[:, LABELS == lab]
        assert np.all(cols == cols[:, :1])
    first = [np.flatnonzero(LABELS == lab)[0] for lab in np.unique(LABELS)]
    C = np.corrcoef(W[:, first].T)
    assert np.max(np.abs(C - np.eye(len(first)))) < 0.03
    if independent == "rademacher":
        assert set(np.unique(W)) == {-1.0, 1.0}


def test_cross_fitting_and_permutation_blocks_never_split_a_cluster():
    from model_selection.statistics.regression.kernel_residuals import crossfit_blocks
    from model_selection.statistics.significance.resampling import time_block_permutation
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(60), rng.integers(1, 4, size=60))
    pos, lag = cluster_coordinates(labels)
    block, k, short = crossfit_blocks(pos, lag)
    assert k == 20 and not short
    for lab in np.unique(labels):
        assert np.unique(block[labels == lab]).size == 1
    for _ in range(20):
        perm = time_block_permutation(labels.size, pos, lag, rng)
        moved = labels[perm]
        # each cluster's samples stay together and in order
        runs = np.split(moved, np.flatnonzero(np.diff(moved)) + 1)
        assert len(runs) == np.unique(labels).size


def test_samples_of_a_cluster_are_never_permutation_partners():
    from model_selection.statistics.significance.neighborhood import compute_neighbor_index
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(40), 3)
    z = rng.normal(size=(labels.size, 1))
    z[1::3] = z[0::3] + 1e-6          # cluster-mates are each other's nearest neighbours in z
    pos, lag = cluster_coordinates(labels)
    nn, pool = compute_neighbor_index(z, 5, pos, lag, workers=1)
    for i in range(labels.size):
        partners = nn[i, :pool[i]]
        assert pool[i] == 5 and not np.any(labels[partners] == labels[i])


def test_clusters_are_counted_as_the_independent_units():
    pos, lag = cluster_coordinates(LABELS)
    res = resolve_block(pos, lag, [(np.zeros((10, 1)) + np.arange(10)[:, None], np.ones((10, 1)))])
    assert res.independent_units and res.n_blocks == 6 and not res.estimated


def test_the_wild_cluster_bootstrap_holds_its_level_with_dependence_within_clusters():
    """Candidate and observations share a strong error component within each
    cluster but are independent across clusters. Treated as independent
    events the null is too narrow; given as clusters it holds the level."""
    rng = np.random.default_rng(4)
    reps, m, size = 300, 40, 4
    labels = np.repeat(np.arange(m), size)
    n = labels.size
    pos, lag = cluster_coordinates(labels)
    rej_cluster = rej_naive = 0
    for _ in range(reps):
        c = np.repeat(rng.normal(size=m), size) + 0.3 * rng.normal(size=n)
        o = np.repeat(rng.normal(size=m), size) + 0.3 * rng.normal(size=n)
        z = rng.normal(size=(n, 1))
        # Treated as independent events: irregular positions further apart than a lag of one.
        naive = np.cumsum(rng.integers(2, 50, size=n))
        for positions, l, name in ((pos, lag, "cluster"), (naive, 1, "naive")):
            r = LinearResidualTest().run({"c": c}, o, z, 99, rng, positions, l)
            p = (1 + np.sum(r.null[:, 0] >= r.observed[0])) / 100
            if p < 0.05:
                if name == "cluster":
                    rej_cluster += 1
                else:
                    rej_naive += 1
    assert rej_cluster / reps <= 0.09
    assert rej_naive / reps > 2 * max(rej_cluster / reps, 0.05)


def test_consecutive_cluster_correlation_detects_dependence_across_clusters():
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(150), 2)
    pos, _ = cluster_coordinates(labels)
    independent = consecutive_cluster_correlation(rng.normal(size=(300, 4)), pos)
    assert abs(independent["correlation"]) < 3 * independent["standard_error"] and independent["n_clusters"] == 150
    # Terms that follow a slowly varying level across clusters.
    level = np.zeros(150)
    for k in range(1, 150):
        level[k] = 0.8 * level[k - 1] + rng.normal()
    persistent = consecutive_cluster_correlation(np.repeat(level, 2)[:, None] + 0.1 * rng.normal(size=(300, 1)), pos)
    assert persistent["significant"] and persistent["correlation"] > 0.5
    assert consecutive_cluster_correlation(np.ones((4, 1)), np.array([0, 0, 2, 2]))["correlation"] is None
