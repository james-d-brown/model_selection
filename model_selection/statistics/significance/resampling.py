"""
Dependence-aware resampling primitives shared by the significance and
adequacy tests: time-distance block permutation, and subsampling of long
records on a regular time grid for estimators too expensive to run on a full
record.
"""
from __future__ import annotations

import numpy as np


def time_block_permutation(
    n: int,
    time_positions: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block permutation of positions 0..n-1 where blocks are defined by time
    proximity, not array-position count.

    time_positions are sorted, a random phase in [0, block_length) is drawn,
    and points in the same block_length-wide time window form one block; block
    order is then shuffled. Within-block adjacency is preserved while
    longer-range alignment is randomized (a permutation analogue of the moving
    block bootstrap, Kunsch 1989). Time-based blocks matter for irregularly
    spaced samples (high-flow events), where a fixed index count could group
    points far apart in time. Deliberately not circular, so the record's ends
    are never merged into one block.

    A block_length <= 0 means independent samples (free permutation). A
    block_length longer than half the time span is reduced to half the span,
    so that at least two blocks are exchanged; the dependence is then longer
    than the record can represent, which the caller reports. Returns an
    index array into 0..n-1.
    """
    time_positions = np.asarray(time_positions, dtype=np.float64)
    if n < 2:
        return rng.permutation(n)
    order_by_time = np.argsort(time_positions)
    sorted_times = time_positions[order_by_time]
    span = float(sorted_times[-1] - sorted_times[0])
    if block_length <= 0 or span <= 0:
        return rng.permutation(n)
    block_length = min(float(block_length), span / 2.0)

    phase = rng.uniform(0.0, block_length)
    block_id = np.floor((sorted_times - sorted_times[0] + phase) / block_length).astype(int)
    # block_id is non-decreasing along sorted time, so each block is a
    # contiguous run of order_by_time (empty blocks do not occur in a split).
    starts = np.flatnonzero(np.diff(block_id)) + 1
    blocks = np.split(order_by_time, starts)
    block_order = rng.permutation(len(blocks))
    perm = np.concatenate([blocks[b] for b in block_order])
    return perm


def _slots(present: np.ndarray, interval: float, start: float) -> tuple[np.ndarray, np.ndarray]:
    """Systematic sample with a fractional interval: slot k sits at the
    nominal grid position start + k * interval and selects the grid point
    nearest to it. Returns (slot numbers, grid points) of the slots whose
    grid point is present."""
    nominal = start + interval * np.arange(int(np.floor((present.size - 1 - start) / interval)) + 1)
    points = np.rint(nominal).astype(np.int64)
    keep = (points < present.size) & present[np.minimum(points, present.size - 1)]
    return np.flatnonzero(keep), points[keep]


def grid_subsample(
        time_positions: np.ndarray, grid_step: float, min_temporal_lag: float, max_n: int, rng: np.random.Generator,
        *, at_lag_spacing: bool,
) -> tuple[np.ndarray, np.ndarray, float, int | None]:
    """Evenly spaced subsample of samples on a regular time grid (possibly
    with missing points), by systematic sampling from a random start
    (Cochran, 1977, chapter 8).

    Returns (indices, positions, step, natural_n): the selected samples, the
    time positions to use for them, the step of the regular grid those
    positions lie on, and the number of samples at the lag's spacing (None
    when nothing is subsampled).

    at_lag_spacing False (tests whose nulls account for dependence): the
    record is used whole when it has at most max_n samples; otherwise exactly
    max_n samples are kept, spread evenly over the record, as far apart as
    the cap allows. Thinning loses precision (MacEachern and Berliner, 1994) and is
    used only as computation requires.

    at_lag_spacing True (tests whose nulls treat samples as exchangeable):
    samples are at least one dependence lag apart, so they are approximately
    independent; where that leaves more than max_n, exactly max_n are kept,
    spread evenly over the record.

    The sampling interval is generally fractional: slot k is
    the grid point nearest to the nominal position start + k * interval, so
    the spacing of the selected points differs from the interval by at most
    one grid step. Each selected sample is given its nominal position, within
    half a grid step of its time, so the subsample lies on a regular grid of
    step interval * grid_step for the dependence estimates. Slots whose grid
    point is missing are missing from the subsample; where the interval that
    fills the cap leaves a few slots too many, those are dropped evenly.
    """
    t = np.asarray(time_positions, dtype=float)
    if t.size == 0:
        return np.arange(0), t, float(grid_step), None
    origin = t.min()
    g = np.rint((t - origin) / grid_step).astype(np.int64)
    length = int(g.max()) + 1
    present = np.zeros(length, dtype=bool)
    present[g] = True
    row_of = np.full(length, -1, dtype=np.int64)
    row_of[g] = np.arange(t.size)
    lag_steps = max(1, int(np.ceil(max(min_temporal_lag, 0) / grid_step)))
    fraction = rng.random()

    def start_of(interval):
        start = fraction * interval
        return np.floor(start) if interval == np.floor(interval) else start  # integer intervals keep exact times

    def count(interval):
        return _slots(present, interval, start_of(interval))[0].size

    shortest = lag_steps if at_lag_spacing else 1
    if shortest == 1 and t.size <= max_n:
        return np.arange(t.size), t, float(grid_step), None
    if count(shortest) <= max_n:
        interval = float(shortest)
    else:
        lo, hi = float(shortest), float(length)  # count(lo) > max_n >= count(hi)
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if count(mid) > max_n else (lo, mid)
        # an interval that keeps exactly max_n samples (preferably a whole number of steps, which keeps every
        # sample at its own time), or else the longest that keeps more
        interval = next((c for c in (float(np.ceil(lo)), hi) if count(c) == max_n), lo)
    start = start_of(interval)
    slot, points = _slots(present, interval, start)
    if slot.size > max_n:
        keep = np.unique(np.rint(np.linspace(0, slot.size - 1, max_n)).astype(np.int64))
        slot, points = slot[keep], points[keep]
    idx = row_of[points]
    positions = origin + (start + interval * slot) * grid_step
    natural_n = int(np.sum((g - int(np.rint(start))) % lag_steps == 0))
    return idx, positions, float(interval * grid_step), natural_n
