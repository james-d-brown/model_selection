"""WASP (wavelet-assessed scale-space persistence) event detection: stage 1 finds
genuine peaks in scale space (wasp.peaks), stages 2-3 resolve onsets and
ends (wasp.onsets)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_selection.event_detection.base import DeclusterResult, DetectionResult, EventDetector
from model_selection.event_detection.wasp.config import WASPConfig
from model_selection.event_detection.wasp.noise import fit_noise_model
from model_selection.event_detection.wasp.onsets import resolve_events_sequential
from model_selection.event_detection.wasp.peaks import detect_peaks
from model_selection.hydrology.timescale import estimate_basin_timescale


@dataclass
class WASPDetectionResult:
    """WASP events plus how they were reached: validated peaks and onsets, the
    fitted noise model (sd = noise_intercept + noise_slope x flow), the series
    timescale used as sigma_max, and the candidate counts before and after the
    quantile filter.
    """
    event_id: np.ndarray
    n_events: int
    event_lengths: np.ndarray
    peaks: np.ndarray               # validated peak positions, ascending, one per event
    genuine_peaks: np.ndarray       # every genuine peak (before the quantile filter), ascending
    onsets: np.ndarray               # per-peak onset positions, same order as peaks
    noise_intercept: float          # noise model: sd = intercept + slope x flow
    noise_slope: float
    basin_timescale: float          # series timescale used as sigma_max (timesteps)
    n_raw_peak_candidates: int      # number of worms tested
    quantile_threshold: float | None
    n_peaks_before_quantile_filter: int
    scale_range_too_short: bool = False   # too few scales below the ceiling: no peaks resolvable

    def to_decluster_result(self) -> DeclusterResult:
        """The events as a DeclusterResult."""
        return DeclusterResult(event_id=self.event_id, n_events=self.n_events, event_lengths=self.event_lengths)

    def metadata(self) -> dict:
        """Diagnostics as a JSON-serializable dict."""
        return {
            "noise_intercept": self.noise_intercept,
            "noise_slope": self.noise_slope,
            "basin_timescale": self.basin_timescale,
            "n_raw_peak_candidates": self.n_raw_peak_candidates,
            "quantile_threshold": self.quantile_threshold,
            "n_peaks_before_quantile_filter": self.n_peaks_before_quantile_filter,
            "n_peaks_after_quantile_filter": len(self.peaks),
            "scale_range_too_short": self.scale_range_too_short,
        }


class WASPEventDetector:
    """Runs stage 1 (detect_peaks) and stages 2-3 (resolve_events_sequential)."""

    def __init__(self, config: WASPConfig | None = None):
        self.config = config or WASPConfig()

    def detect(self, series: np.ndarray, sigma_max: float | None = None,
               quantile_threshold: float | None = None,
               reference: np.ndarray | None = None,
               native_timestep: "pd.Timedelta | None" = None) -> WASPDetectionResult:
        """Detect events in a uniformly spaced series.

        sigma_max : scale ceiling; None (normal use) uses the series' own ACF
            e-folding timescale, which requires native_timestep.
        quantile_threshold : keep only genuine peaks among the highest flows
            (None = no filter).
        native_timestep : spacing of `series` (required when sigma_max is None).
        """
        series = np.asarray(series, dtype=np.float64)
        n = series.size
        if sigma_max is not None:
            basin_timescale = float(sigma_max)
        else:
            if native_timestep is None:
                raise ValueError(
                    "WASPEventDetector.detect(): sigma_max=None requires native_timestep "
                    "(a pandas.Timedelta) to estimate the series timescale; pass "
                    "native_timestep, or an explicit sigma_max."
                )
            synthetic_index = pd.date_range("2000-01-01", periods=n, freq=native_timestep)
            indexed_series = pd.Series(series, index=synthetic_index)
            basin_timescale = estimate_basin_timescale(indexed_series).timescale

        noise = fit_noise_model(series)
        peak_result = detect_peaks(
            series, sigma_max=basin_timescale,
            sigma_step=self.config.sigma_step, alpha=self.config.worm_tolerance_alpha,
            quantile_threshold=quantile_threshold, reference=reference, noise=noise,
        )
        peaks = peak_result["peaks"]
        all_peaks = peak_result["peaks_before_quantile_filter"]

        if len(peaks) == 0:
            empty = DeclusterResult.empty(n)
            return WASPDetectionResult(
                event_id=empty.event_id, n_events=0, event_lengths=empty.event_lengths,
                peaks=peaks, genuine_peaks=np.asarray(all_peaks, dtype=int), onsets=np.array([], dtype=int),
                noise_intercept=noise.intercept, noise_slope=noise.slope, basin_timescale=basin_timescale,
                n_raw_peak_candidates=peak_result["n_raw_candidates"],
                quantile_threshold=quantile_threshold,
                n_peaks_before_quantile_filter=peak_result["n_peaks_before_quantile_filter"],
                scale_range_too_short=peak_result["scale_range_too_short"],
            )

        sigmas = np.arange(self.config.sigma_step, basin_timescale, self.config.sigma_step)
        result, onsets = resolve_events_sequential(series, peaks, sigmas, noise=noise, all_peaks=all_peaks)

        return WASPDetectionResult(
            event_id=result.event_id, n_events=result.n_events, event_lengths=result.event_lengths,
            peaks=peaks, genuine_peaks=np.asarray(all_peaks, dtype=int), onsets=onsets,
            noise_intercept=noise.intercept, noise_slope=noise.slope, basin_timescale=basin_timescale,
            n_raw_peak_candidates=peak_result["n_raw_candidates"],
            quantile_threshold=quantile_threshold,
            n_peaks_before_quantile_filter=peak_result["n_peaks_before_quantile_filter"],
            scale_range_too_short=peak_result["scale_range_too_short"],
        )


class WASPDetector(EventDetector):
    """EventDetector for the WASP method."""
    name = "wasp"

    def __init__(self, config: WASPConfig | None = None):
        self.config = config or WASPConfig()

    def detect(self, values: np.ndarray, timestep: pd.Timedelta, quantile_threshold: float,
               reference: np.ndarray | None = None) -> DetectionResult:
        result = WASPEventDetector(config=self.config).detect(
            values, quantile_threshold=quantile_threshold, native_timestep=timestep, reference=reference)
        return DetectionResult(events=result.to_decluster_result(), metadata=result.metadata(),
                               peaks=result.genuine_peaks)
