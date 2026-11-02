"""
WASP stages 2 and 3: event onsets and ends, resolved sequentially.

Events are resolved in time order over the full set of genuine peaks
(including peaks later dropped by the quantile filter), so each onset search
is bounded below by the previous event's resolved end and an unreported
event's recession can never be mistaken for the next onset. The onset is
the sharpest upturn of the rise: the largest curvature (second derivative of
the smoothed flow) at the finest scale where it is reliable, that is, at
least NOISE_MULTIPLE standard deviations of the curvature that noise alone
would produce there (the minimum reliable scale of Elder & Zucker 1998).
Positions whose flow exceeds the event's own peak are excluded. The end is
where flow returns to the onset value, or the minimum between two peaks if
that comes first. Known limitation: the end rule is threshold based, not
scale-space based (an exponential recession has no curvature to detect).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter1d

from model_selection.event_detection.base import DeclusterResult
from model_selection.event_detection.wasp.config import NOISE_MULTIPLE
from model_selection.event_detection.wasp.noise import NoiseModel, fit_noise_model

_TRUNCATE = 4.0
"""Gaussian kernel radius in standard deviations (scipy's default)."""


@lru_cache(maxsize=None)
def _curvature_noise_gain(sigma: float) -> float:
    """Standard deviation of the smoothed second derivative of unit white
    noise at scale sigma: the L2 norm of the discrete kernel."""
    radius = int(_TRUNCATE * sigma + 0.5)
    impulse = np.zeros(2 * radius + 1)
    impulse[radius] = 1.0
    kernel = gaussian_filter1d(impulse, sigma=sigma, order=2, mode="constant", truncate=_TRUNCATE)
    return float(np.sqrt(np.sum(kernel ** 2)))


def _detect_onset(series: np.ndarray, window_lo: int, peak_idx: int,
                  sigmas: np.ndarray, noise: NoiseModel) -> int:
    """Stage 2 for one event: the onset position in [window_lo, peak_idx).

    Candidates are the positions before the peak whose flow does not exceed
    the peak's. Scales are tried from finest to coarsest; at the first scale
    where some candidate's curvature reaches NOISE_MULTIPLE noise standard
    deviations, the onset is the candidate of largest curvature relative to
    that threshold. If no scale reaches it, the coarsest scale's is used.
    Returns window_lo when there is no candidate.
    """
    if window_lo >= peak_idx:
        return window_lo
    candidates = series[window_lo:peak_idx] <= series[peak_idx]
    if not candidates.any():
        return window_lo

    n = len(series)
    pad = int(_TRUNCATE * float(sigmas[-1]) + 0.5) + 1
    lo, hi = max(0, window_lo - pad), min(n, peak_idx + pad)
    segment = series[lo:hi]
    noise_sd = noise.sd(series[window_lo:peak_idx])
    tiny = np.finfo(float).tiny

    score = None
    for sigma in sigmas:
        curvature = gaussian_filter1d(segment, sigma=float(sigma), order=2, mode="nearest",
                                      truncate=_TRUNCATE)[window_lo - lo:peak_idx - lo]
        threshold = np.maximum(NOISE_MULTIPLE * noise_sd * _curvature_noise_gain(float(sigma)), tiny)
        score = np.where(candidates, curvature / threshold, -np.inf)
        best = int(np.argmax(score))
        if score[best] >= 1.0:
            return window_lo + best
    return window_lo + int(np.argmax(score))


def _resolve_end(series: np.ndarray, peak_idx: int, onset_val: float, next_peak_idx: int | None) -> int:
    """Stage 3 for one event: the first position after the peak where flow returns
    to the onset value; if that reaches the next peak, the minimum between the
    two peaks instead.
    """
    n = len(series)
    search_hi = next_peak_idx if next_peak_idx is not None else n - 1
    below = np.where(series[peak_idx:search_hi + 1] <= onset_val)[0]
    natural_end = peak_idx + int(below[0]) if below.size > 0 else search_hi
    if next_peak_idx is not None and natural_end >= next_peak_idx:
        natural_end = peak_idx + int(np.argmin(series[peak_idx:next_peak_idx + 1]))
    return natural_end


def resolve_events_sequential(
        series: np.ndarray,
        peaks: np.ndarray,
        sigmas: np.ndarray,
        noise: NoiseModel | None = None,
        all_peaks: np.ndarray | None = None,
) -> tuple[DeclusterResult, np.ndarray]:
    """Stages 2 and 3 for all events in time order.

    sigmas : the scales searched for onsets, finest first (stage 1's scales).
    noise : the series' noise model; None fits it to the series.

    For each peak of all_peaks (default: peaks): find its onset with the
    previous event's end as the lower bound; if the previous end reaches this
    onset, cap it at onset - 1 (this only revises an already computed value, so
    it terminates); then resolve this event's end, bounded by the next peak.
    Only the events of `peaks` are returned. Returns (DeclusterResult, onsets).
    """
    n = len(series)
    sigmas = np.asarray(sigmas, dtype=float)
    noise = fit_noise_model(series) if noise is None else noise
    full_sequence = [int(p) for p in (all_peaks if all_peaks is not None else peaks)]
    reported = [int(p) for p in peaks]
    missing = set(reported) - set(full_sequence)
    if missing:
        raise ValueError(
            f"peaks contains position(s) {sorted(missing)} not present in all_peaks -- "
            "every reported peak must also appear in the full sequence used for windowing."
        )
    all_spans: dict[int, list] = {}  # peak position -> [onset, end]

    window_lo = 0
    for k, peak_idx in enumerate(full_sequence):
        onset_x = _detect_onset(series, window_lo, peak_idx, sigmas, noise)

        if k > 0:
            prev_peak = full_sequence[k - 1]
            all_spans[prev_peak][1] = min(all_spans[prev_peak][1], onset_x - 1)

        next_peak = full_sequence[k + 1] if k + 1 < len(full_sequence) else None
        onset_val = series[onset_x]
        end_x = _resolve_end(series, peak_idx, onset_val, next_peak)

        all_spans[peak_idx] = [onset_x, end_x]
        window_lo = end_x

    spans = [[all_spans[int(p)][0], all_spans[int(p)][1], int(p)] for p in peaks]

    event_id = np.full(n, -1, dtype=np.int64)
    for k, (s, e, _) in enumerate(spans):
        event_id[max(s, 0):e + 1] = k
    event_lengths = np.array([(event_id == k).sum() for k in range(len(spans))], dtype=np.int64)
    onsets = np.array([s[0] for s in spans], dtype=int)

    return DeclusterResult(event_id=event_id, n_events=len(spans), event_lengths=event_lengths), onsets
