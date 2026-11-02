"""
WASP stage 1: genuine peaks by global scale-space connectivity.

The series is smoothed at every scale up to sigma_max (its own ACF e-folding
time, a characteristic timescale of the series' response, not a
decorrelation length) and every local maximum marked in a (scale, position) grid, the
scalogram. Maxima are linked across scales into "worms" by one global
connected-component labelling (scipy.ndimage.label) rather than by tracking
ridges scale by scale, which can drift through chains of unrelated
noise maxima. A worm is a genuine peak if it spans at least MIN_SCALE_LEVELS
scales and its prominence, at some scale, is at least NOISE_MULTIPLE noise
standard deviations at the peak's flow (the signal-to-noise rule of Du,
Kibbe & Lin 2006; noise model in wasp.noise). Its position is followed
from its finest scale down to the unsmoothed series.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d
from scipy.ndimage import label as _ndimage_label
from scipy.signal import find_peaks

from model_selection.event_detection.wasp.config import NOISE_MULTIPLE
from model_selection.event_detection.wasp.noise import NoiseModel, fit_noise_model


def _build_scalogram(series: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Builds the full (n_sigma, n_pos) scalogram: binary[i, j] = True
    iff position j is a local maximum of the series smoothed at
    sigmas[i]; prom[i, j] = that local maximum's own prominence there
    (0 elsewhere)."""
    n_sigma, n_pos = len(sigmas), len(series)
    binary = np.zeros((n_sigma, n_pos), dtype=bool)
    prom = np.zeros((n_sigma, n_pos), dtype=float)
    for i, sigma in enumerate(sigmas):
        smoothed = gaussian_filter1d(series, sigma=float(sigma), mode="nearest")
        peaks, props = find_peaks(smoothed, prominence=0)
        binary[i, peaks] = True
        prom[i, peaks] = props["prominences"]
    return binary, prom


def _label_worms(binary: np.ndarray, sigmas: np.ndarray, alpha: float) -> np.ndarray:
    """Label worms in the scalogram by connected components.

    Each scale row is first dilated horizontally by tol(sigma) =
    round(alpha * sigma), so legitimate position drift grows with scale; rows
    are then labelled with ordinary 3x3 connectivity and the labels mapped back
    onto the original (undilated) local-maximum positions only.
    """
    dilated = np.zeros_like(binary)
    for i, sigma in enumerate(sigmas):
        tol = max(1, int(round(alpha * float(sigma))))
        horiz_structure = np.ones((1, 2 * tol + 1), dtype=bool)
        dilated[i:i + 1, :] = binary_dilation(binary[i:i + 1, :], structure=horiz_structure)
    labels_dilated, _ = _ndimage_label(dilated, structure=np.ones((3, 3), dtype=bool))
    return np.where(binary, labels_dilated, 0)


def _basin(smoothed: np.ndarray, pos: int) -> tuple[int, int]:
    """The extent of the local maximum of `smoothed` at pos: from the
    neighbouring local minimum on the left to the one on the right."""
    lo = pos
    while lo > 0 and smoothed[lo - 1] <= smoothed[lo]:
        lo -= 1
    hi = pos
    while hi < len(smoothed) - 1 and smoothed[hi + 1] <= smoothed[hi]:
        hi += 1
    return lo, hi


def _worm_stats(labels: np.ndarray, series: np.ndarray, prom: np.ndarray,
                sigmas: np.ndarray) -> list[dict]:
    """Per worm: n_sigma_levels (its persistence; the bounding-box row extent,
    since 3x3 connectivity cannot skip a scale), max_prominence (its largest
    prominence at any scale) and peak_pos. The peak is followed from the
    worm's finest scale down to the unsmoothed series: the highest flow
    within the extent of the worm's maxima at its finest scale, between the
    neighbouring minima of the series smoothed at that scale.
    """
    from scipy.ndimage import find_objects
    smoothed_at: dict[int, np.ndarray] = {}
    stats = []
    for label_id, sl in enumerate(find_objects(labels), start=1):
        if sl is None:
            continue
        row_slice, col_slice = sl
        member = labels[sl] == label_id
        finest = row_slice.start
        if finest not in smoothed_at:
            smoothed_at[finest] = gaussian_filter1d(series, sigma=float(sigmas[finest]), mode="nearest")
        cols_at_finest = col_slice.start + np.where(member[0])[0]
        lo = _basin(smoothed_at[finest], int(cols_at_finest.min()))[0]
        hi = _basin(smoothed_at[finest], int(cols_at_finest.max()))[1]
        stats.append({
            "label": label_id,
            "n_sigma_levels": row_slice.stop - row_slice.start,
            "max_prominence": float(prom[sl][member].max()),
            "peak_pos": lo + int(np.argmax(series[lo:hi + 1])),
        })
    return stats


MIN_SCALE_LEVELS = 2
"""Fewest scale levels a worm must span: a maximum at one scale only is not
linked across scales."""


def _empty(scale_range_too_short: bool, n_raw: int = 0) -> dict:
    return {
        "peaks": np.array([], dtype=int), "peaks_before_quantile_filter": np.array([], dtype=int),
        "n_raw_candidates": n_raw, "n_peaks_before_quantile_filter": 0,
        "scale_range_too_short": scale_range_too_short,
    }


def detect_peaks(
        series: np.ndarray,
        sigma_max: float,
        sigma_step: float = 1.0,
        alpha: float = 0.1,
        quantile_threshold: float | None = None,
        reference: np.ndarray | None = None,
        noise: NoiseModel | None = None,
) -> dict:
    """Stage 1. Returns a dict with peaks (genuine peak positions after the
    quantile filter, ascending), peaks_before_quantile_filter (every genuine
    peak), n_raw_candidates (worms tested), n_peaks_before_quantile_filter
    and scale_range_too_short (True when the scale ceiling leaves fewer than
    MIN_SCALE_LEVELS scales, so no worm can qualify; no peaks are returned).

    alpha : worm connectivity tolerance (see _label_worms).
    quantile_threshold : keeps only the genuine peaks among the highest flows
        of `reference` (default: the series); None applies no filter.
    noise : the series' noise model; None fits it to the series.
    """
    sigmas = np.arange(sigma_step, sigma_max, sigma_step)
    if len(sigmas) < MIN_SCALE_LEVELS:
        return _empty(scale_range_too_short=True)
    noise = fit_noise_model(series) if noise is None else noise

    binary, prom = _build_scalogram(series, sigmas)
    labels = _label_worms(binary, sigmas, alpha)
    worms = _worm_stats(labels, series, prom, sigmas)
    if not worms:
        return _empty(scale_range_too_short=False)

    genuine = sorted({
        w["peak_pos"] for w in worms
        if w["n_sigma_levels"] >= MIN_SCALE_LEVELS
        and w["max_prominence"] >= NOISE_MULTIPLE * float(noise.sd(series[w["peak_pos"]]))
    })
    genuine_peaks = np.array(genuine, dtype=int)
    peaks = genuine_peaks
    if quantile_threshold is not None and genuine_peaks.size > 0:
        threshold_value = np.quantile(series if reference is None else reference, quantile_threshold)
        peaks = genuine_peaks[series[genuine_peaks] >= threshold_value]

    return {
        "peaks": peaks, "peaks_before_quantile_filter": genuine_peaks,
        "n_raw_candidates": len(worms), "n_peaks_before_quantile_filter": len(genuine_peaks),
        "scale_range_too_short": False,
    }
