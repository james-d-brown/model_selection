"""Tests of how the resampling components treat clusters of dependent
samples given as cluster coordinates (statistics.significance.clusters):
one multiplier per cluster, clusters never split, no permutation partners
within a cluster, clusters counted as the independent units, and the check
of correlation between consecutive clusters."""
import numpy as np
import pytest

from model_selection.statistics.adequacy.linearity import LinearityTest
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


def test_minimum_units_follows_the_smallest_sign_flip_p_value():
    from model_selection.statistics.significance.clusters import minimum_units, smallest_p_value
    assert smallest_p_value(6) == 2.0 ** -5
    assert minimum_units(0.05) == 6         # 2^-4 = 0.0625 >= 0.05 > 2^-5
    assert minimum_units(0.005) == 9        # 2^-7 >= 0.005 > 2^-8


def test_nonlinear_cluster_draw_with_every_sign_positive_is_the_observed_statistic():
    """The nonlinear tier's cluster null rebuilds the observations' feature
    map and residualizes it with the observed fit's tuning, so the draw with
    every sign +1 is the observed statistic, with or without a conditioning
    set, and the run records that it was reproduced."""
    from model_selection.statistics.conditional_independence import NonlinearResidualTest
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(12), 3)
    pos, lag = cluster_coordinates(labels)
    z = np.repeat(rng.normal(size=(12, 1)), 3, axis=0) + 0.3 * rng.normal(size=(labels.size, 1))
    o = np.column_stack([np.cos(z[:, 0]), z[:, 0]]) + 0.5 * rng.normal(size=(labels.size, 2))
    for conditioning in (None, z):
        test = NonlinearResidualTest()
        test.uncensored_feature_means = "regression"
        r = test.run({"c": rng.normal(size=(labels.size, 2))}, o, conditioning, 99, rng, pos, lag)
        assert r.dependence["independent_units"] and r.dependence["cluster_draws_reproduce_statistic"]


def test_cluster_scaling_is_hc2_for_single_events_and_cr2_for_clusters():
    from model_selection.statistics.significance.clusters import cluster_scaled_residuals, orthonormal_residual_maker
    rng = np.random.default_rng(1)
    n = 30
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    M = orthonormal_residual_maker(X)
    r = M @ rng.normal(size=n)
    h = np.sum(np.linalg.qr(X)[0] ** 2, axis=1)
    assert np.allclose(cluster_scaled_residuals(r, M, np.arange(n)), r / np.sqrt(1 - h))
    labels = np.repeat(np.arange(10), 3)
    scaled = cluster_scaled_residuals(r, M, labels)
    for g in range(10):                       # (I - P_gg)^(-1/2) r_g (Bell and McCaffrey, 2002)
        idx = labels == g
        vals, vecs = np.linalg.eigh(M[np.ix_(idx, idx)])
        assert np.allclose(scaled[idx], vecs @ ((vecs.T @ r[idx]) / np.sqrt(vals)))


def test_sign_flip_null_holds_its_level_with_ten_clusters():
    """Ten clusters of four events with a shared component within each
    cluster: with one Gaussian multiplier per cluster the null is the normal
    law with the clusters' sample variance and rejects about 10% at 5%
    (Cameron, Gelbach and Miller, 2008); one sign per cluster holds the
    level. 400 replicates, standard error about 1.1 points."""
    rng = np.random.default_rng(11)
    reps, m, size = 400, 10, 4
    labels = np.repeat(np.arange(m), size)
    n = labels.size
    pos, lag = cluster_coordinates(labels)
    rejections = 0
    for _ in range(reps):
        c = np.repeat(rng.normal(size=m), size) + 0.5 * rng.normal(size=n)
        o = np.repeat(rng.normal(size=m), size) + 0.5 * rng.normal(size=n)
        r = LinearResidualTest().run({"c": c}, o, None, 199, rng, pos, lag)
        rejections += (1 + np.sum(r.null[:, 0] >= r.observed[0])) / 200 < 0.05
    assert rejections / reps <= 0.075


def test_a_round_with_too_few_clusters_is_untestable_not_unselected():
    from model_selection.selection import SelectionSample, build_tier, greedy_forward_selection
    from model_selection.selection.config import SelectionConfig
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(5), 4)
    pos, lag = cluster_coordinates(labels)
    o = rng.normal(size=labels.size)
    sample = SelectionSample({"a": o + 0.1 * rng.normal(size=o.size), "b": rng.normal(size=o.size)}, o, pos, lag,
                             clusters=labels)
    tier = build_tier("linear", SelectionConfig())
    result = greedy_forward_selection(sample, tier, n_permutations=199, rng=rng)
    assert result.untestable and result.selected == [] and "untestable" in result.stopped_reason


def test_event_tiers_take_the_regression_path_whatever_the_number_of_components():
    """The smear path is the pooled pathway's method for a scalar series; an
    event sample with one component takes the regression path, as event
    samples with several do."""
    from model_selection.selection import build_tier
    from model_selection.selection.config import SelectionConfig
    assert build_tier("nonlinear", SelectionConfig(), events=True).test.uncensored_feature_means == "regression"
    assert build_tier("nonlinear", SelectionConfig()).test.uncensored_feature_means == "smear"


def test_adequacy_draws_on_clusters_recompute_the_whole_statistic():
    """On independent units the RESET draws are recomputed whole, with their
    own denominators (adequacy.base): _statistic_parts serves only the
    regular-grid path, so it is never called."""
    calls = []

    class Recording(LinearityTest):
        def _statistic_parts(self, candidate, obs, conditioning):
            calls.append(1)
            return super()._statistic_parts(candidate, obs, conditioning)

    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(15), 3)
    pos, lag = cluster_coordinates(labels)
    c, z = rng.normal(size=labels.size), rng.normal(size=(labels.size, 1))
    Recording().run(c, c + rng.normal(size=labels.size), z, time_positions=pos, min_temporal_lag=lag,
                    n_permutations=49, rng=rng)
    assert calls == []


def test_power_reports_nothing_detectable_without_degrees_of_freedom():
    from model_selection.statistics.power import minimum_detectable_r2
    assert minimum_detectable_r2(200, 3, 3, 1, 0.05, nu=0.0) == 1.0
    assert minimum_detectable_r2(200, 3, 3, 1, 0.05, nu=-2.0) == 1.0
    assert minimum_detectable_r2(200, 3, 3, 1, 0.05, nu=5.0) < 1.0


def test_unrestricted_width_is_not_widened_by_a_strong_effect():
    """On clusters of events the restricted residuals carry the candidate's
    effect into every draw and widen the null; the unrestricted residuals do
    not, so the effective sample size from them is about the same with or
    without the effect (LinearResidualTest.unrestricted_width_draws)."""
    from model_selection.statistics.power import effective_sample_size
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(60), rng.integers(1, 4, size=60))
    pos, lag = cluster_coordinates(labels)
    n = labels.size
    z, c = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    test = LinearResidualTest()
    n_eff = {}
    for label, o in (("none", rng.normal(size=(n, 2))), ("strong", 2.5 * c + rng.normal(size=(n, 2)))):
        r = test.run({"c": c}, o, z, 199, rng, pos, lag)
        n_eff[label, "restricted"] = effective_sample_size(n, r.width_draws(0), 2, 2, 2)
        n_eff[label, "unrestricted"] = effective_sample_size(
            n, test.unrestricted_width_draws(c, o, z, 199, rng, pos, lag), 2, 2, 2)
    assert abs(n_eff["none", "unrestricted"] / n_eff["none", "restricted"] - 1) < 0.25
    assert n_eff["strong", "restricted"] < 0.8 * n_eff["strong", "unrestricted"]
    assert abs(n_eff["strong", "unrestricted"] / n_eff["none", "unrestricted"] - 1) < 0.25


def test_water_year_blocks_of_clusters_never_split_a_cluster():
    from model_selection.statistics.regression.kernel_residuals import crossfit_blocks
    from model_selection.statistics.significance.clusters import cluster_coordinates, cluster_period_starts
    year = 8760.0
    # Twelve water years with two clusters each; the second cluster of each year runs across the next start.
    peaks, labels = [], []
    for y in range(12):
        peaks += [y * year + 100, y * year + year - 50, (y + 1) * year + 20]
        labels += [2 * y, 2 * y + 1, 2 * y + 1]
    peaks, labels = np.array(peaks), np.array(labels)
    pos, lag = cluster_coordinates(labels, peaks)
    starts = cluster_period_starts(pos, peaks, np.arange(1, 12) * year)
    block, k, short = crossfit_blocks(pos.astype(float), lag, starts)
    assert not short and k == 12
    assert all(len(set(block[labels == c])) == 1 for c in np.unique(labels))
