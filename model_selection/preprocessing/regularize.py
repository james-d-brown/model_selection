"""
Temporary regularization of series for event detection.

Event detectors treat adjacent array samples as equally spaced and need a
value at every sample, so a data gap would look like an instantaneous jump.
Each series is reconstructed on a uniform index (its own modal step), with
missing timestamps and missing values (NaN) linearly interpolated; events are
detected on that, and every result is mapped back to the original positions.
Interpolated values never reach association estimation: events whose windows
contain missing values are excluded from the event sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegularizedSeries:
    """A uniformly spaced reconstruction of a series plus the position mappings
    back to the original index.
    """
    original_index: pd.DatetimeIndex
    regularized_index: pd.DatetimeIndex
    values: np.ndarray  # regularized, gap-interpolated values -- for event detection ONLY
    orig_to_regularized: np.ndarray  # length == len(original_index)
    regularized_to_orig: np.ndarray  # length == len(regularized_index); nearest original position

    def position_to_original(self, regularized_position: int) -> int:
        """Map a position on the regularized series to the nearest preceding
        original position.
        """
        return int(self.regularized_to_orig[regularized_position])

    def event_id_to_original(self, event_id_regularized: np.ndarray) -> np.ndarray:
        """Map a regularized-length event_id array back to original positions."""
        return event_id_regularized[self.orig_to_regularized]


def regularize_series(series: pd.Series) -> RegularizedSeries:
    """Reconstruct `series` on a uniform index at its modal step, linearly
    interpolating missing samples. Already-uniform series map by identity.
    """
    idx = series.index
    diffs = idx.to_series().diff().dropna()

    if len(diffs) == 0 or (diffs == diffs.mode().iloc[0]).all():
        n = len(series)
        identity = np.arange(n)
        values = series.astype(np.float64)
        if values.isna().any():  # missing values on a regular index: filled for detection only
            values = values.interpolate(method="linear", limit_direction="both")
        return RegularizedSeries(
            original_index=idx, regularized_index=idx,
            values=values.to_numpy(),
            orig_to_regularized=identity, regularized_to_orig=identity,
        )

    modal_step = diffs.mode().iloc[0]
    full_index = pd.date_range(idx[0], idx[-1], freq=modal_step)

    reindexed = series.reindex(full_index)
    interpolated = reindexed.astype(np.float64).interpolate(method="linear", limit_direction="both")

    orig_to_regularized = full_index.get_indexer(idx)
    if (orig_to_regularized < 0).any():
        raise ValueError(
            "regularize_series: every original timestamp must exist in the reconstructed, "
            "uniform index -- this indicates the original index is not aligned to its own "
            "modal step (e.g. sub-step jitter in timestamps), which this function does not "
            "currently handle."
        )

    # nearest preceding original position for every regularized position
    regularized_to_orig = np.clip(
        np.searchsorted(orig_to_regularized, np.arange(len(full_index)), side="right") - 1,
        0, len(series) - 1,
    )

    return RegularizedSeries(
        original_index=idx, regularized_index=full_index,
        values=interpolated.values.astype(np.float64),
        orig_to_regularized=orig_to_regularized, regularized_to_orig=regularized_to_orig,
    )
