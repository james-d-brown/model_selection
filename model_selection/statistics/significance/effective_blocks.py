"""
The effective number of blocks a resampling test rests on.

A block or cluster resampling counts its blocks, but when a few blocks carry
most of the variation that matters, the test behaves as if it had far fewer:
the cluster-robust t-test and the wild cluster bootstrap then reject too
often (Carter et al., 2017; MacKinnon and Webb, 2017, 2018). Censored
series, whose terms are nonzero only at exceedances, are an example.

Carter, Schnepel and Steigerwald (2017) measure this with the effective
number of clusters G* = G / (1 + Gamma), Gamma being the squared coefficient
of variation over clusters of gamma_g, each cluster's contribution to the
variance. Its feasible version sets each cluster's error covariance to a
matrix of ones (perfect correlation within a cluster), so that gamma_g
depends on the design alone, never on the outcome: "using the same data to
approximate G would lead to dependence between Z and the critical value".
Here the design is the candidate's residuals on the conditioning set, the
side of the test that does not involve the observations; with errors fully
correlated within a block, gamma_g is the square of the sum of those
residuals over the block, and

    G_A = (sum_g gamma_g)^2 / sum_g gamma_g^2 = G / (1 + CV^2(gamma)).

Blocks are the resampling's units: clusters of events when the samples are
independent units, otherwise consecutive stretches of the block length on
the sample's time positions. The dependent wild bootstrap draws smoothly
correlated multipliers rather than disjoint blocks, so for it the stretches
are an approximation of its units. For a candidate of several columns the
smallest G_A over its columns is used, a conservative extension: Carter et
al. define G* for one contrast at a time.

Carter et al. find the size of the test rising sharply once G* falls below
about 10, and advise caution at about 20 (their application, G_A = 19).
Because the feasible version assumes perfect correlation within a block, it
is a lower bound on G*: for a Gaussian design the block sums vary as
chi-square(1), whose squared coefficient of variation is 2, so G_A is about
G / 3 even when no block dominates. With one sample per block (events
resampled as independent units) G_A is the number of samples divided by the
kurtosis of the design.
"""
from __future__ import annotations

import numpy as np

from model_selection.utils import as2d

# An acceptance resting on fewer effective blocks than MIN_EFFECTIVE_BLOCKS_REVIEW is flagged for review;
# below MIN_EFFECTIVE_BLOCKS_SEVERE the size of such tests rises sharply (Carter et al., 2017).
MIN_EFFECTIVE_BLOCKS_SEVERE = 10.0
MIN_EFFECTIVE_BLOCKS_REVIEW = 20.0


def block_labels(time_positions: np.ndarray, block_length: float, independent_units: bool) -> np.ndarray:
    """The resampling unit of each sample: its distinct position when the
    samples are independent units (clusters), otherwise the consecutive
    stretch of `block_length` time units it falls in."""
    t = np.asarray(time_positions, dtype=float)
    if independent_units:
        return np.unique(t, return_inverse=True)[1]
    width = max(float(block_length), 1.0)
    return np.unique(np.floor((t - t.min()) / width), return_inverse=True)[1]


def effective_number_of_blocks(design: np.ndarray, time_positions: np.ndarray, block_length: float,
                               independent_units: bool = False) -> float | None:
    """Feasible effective number of blocks (Carter et al., 2017) of the
    design's columns (see the module docstring); the smallest over columns.
    None when no column varies."""
    X = as2d(np.asarray(design, dtype=float))
    labels = block_labels(time_positions, block_length, independent_units)
    sums = np.zeros((labels.max() + 1, X.shape[1]))
    np.add.at(sums, labels, X)
    gamma = sums ** 2
    denom = (gamma ** 2).sum(axis=0)
    ok = denom > 0
    if not np.any(ok):
        return None
    g = gamma[:, ok].sum(axis=0) ** 2 / denom[ok]
    return float(np.min(g))
