"""
Conditioning-space neighbourhoods for local permutation (Runge, 2018).

Candidate values are exchanged only among samples that are near neighbours
in Z-space, which keeps each sample's candidate value plausible for its own
Z. Neighbours closer than the dependence lag in time are excluded (the
Theiler window; Theiler, 1986), because temporally adjacent samples are
close in Z-space for reasons unrelated to conditional exchangeability.
time_positions are in original time units: np.arange(n) for timestep
samples, each event's peak index for event samples.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_neighbor_index(
    z: np.ndarray,
    n_neighbors: int,
    time_positions: np.ndarray | None = None,
    min_temporal_lag: int = 0,
    workers: int = -1,
    max_expansion_factor: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-space kNN index excluding neighbours within min_temporal_lag time units.

    Queries more than n_neighbors candidates and expands up to
    max_expansion_factor * n_neighbors when exclusion leaves a point short.
    Points that still fall short keep however many valid neighbours they have.

    Returns (nn_idx, pool_size): nn_idx is (n, n_neighbors), ordered by
    distance and right-padded with -1; pool_size counts valid entries per row.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    n = z.shape[0]
    tree = cKDTree(z)

    if time_positions is None:
        time_positions = np.arange(n)
    else:
        time_positions = np.asarray(time_positions)
        if time_positions.shape[0] != n:
            raise ValueError("time_positions must match z's length")

    nn_idx = np.full((n, n_neighbors), -1, dtype=np.int64)
    pool_size = np.zeros(n, dtype=np.int64)

    if min_temporal_lag <= 0:
        # No exclusion requested: behaves exactly as the original
        # (unfiltered) neighbor search.
        k_query = min(n_neighbors + 1, n)
        _, idx = tree.query(z, k=k_query, p=np.inf, workers=workers)
        if k_query == 1:
            idx = idx[:, None]
        for i in range(n):
            row = idx[i][idx[i] != i][:n_neighbors]
            nn_idx[i, : row.shape[0]] = row
            pool_size[i] = row.shape[0]
        return nn_idx, pool_size

    # With exclusion: query a generous pool, filter by temporal separation,
    # expand only as needed.
    k_query = min(n, (n_neighbors + 1) * 2)
    max_k = min(n, n_neighbors * max_expansion_factor + 1)
    still_short = np.arange(n)

    while still_short.size > 0:
        _, idx = tree.query(z[still_short], k=k_query, p=np.inf, workers=workers)
        if k_query == 1:
            idx = idx[:, None]
        next_short = []
        for row_pos, i in enumerate(still_short):
            candidates = idx[row_pos]
            candidates = candidates[candidates != i]

            time_sep = np.abs(time_positions[candidates] - time_positions[i])
            lag_ok = time_sep >= min_temporal_lag

            valid = candidates[lag_ok][:n_neighbors]
            nn_idx[i, : valid.shape[0]] = valid
            pool_size[i] = valid.shape[0]
            if valid.shape[0] < n_neighbors and k_query < max_k:
                next_short.append(i)
        if not next_short or k_query >= max_k:
            break
        still_short = np.asarray(next_short)
        k_query = min(max_k, k_query * 2)

    return nn_idx, pool_size


def local_permutation(
    nn_idx: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Restricted permutation of Runge (2018, Algorithm 1).

    Points are visited in random order. Each point i receives the value of
    one of its own Z-neighbours, drawn at random among those whose value has
    not yet been used; if every neighbour's value has been used, the last
    neighbour tried is used again. Every sample therefore receives a value
    from its own neighbourhood, at the cost of occasional repeats. A point
    with no valid neighbour keeps its own value.

    Returns (index, n_repeats): index[i] is the sample whose value i
    receives; n_repeats counts the samples that received an already used
    value.
    """
    n = nn_idx.shape[0]
    index = np.arange(n)
    used = np.zeros(n, dtype=bool)
    n_repeats = 0
    for i in rng.permutation(n):
        neighbours = nn_idx[i][nn_idx[i] >= 0]
        if neighbours.size == 0:
            continue
        neighbours = rng.permutation(neighbours)
        free = neighbours[~used[neighbours]]
        if free.size:
            j = free[0]
        else:
            j = neighbours[-1]
            n_repeats += 1
        index[i] = j
        used[j] = True
    return index, n_repeats
