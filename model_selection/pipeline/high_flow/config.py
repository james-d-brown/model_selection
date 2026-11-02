"""Configuration of the high-flow event sample (curves -> FPC scores)."""
from __future__ import annotations

from dataclasses import dataclass

BASIS_FIT_SOURCES = ("pooled", "pooled_with_obs", "obs")


@dataclass(frozen=True)
class FPCAConfig:
    """How event curves become the per-event feature vectors.

    n_components : fixed number of FPCs; None chooses the smallest number
        reaching variance_threshold.
    max_components : optional cap on that number; None (default) applies no
        cap, so the representation always reaches the variance target. Whether
        the resulting dimension can be tested is assessed separately (see
        SelectionConfig.meaningful_r2).
    n_grid : points of the common phase grid each curve is resampled onto;
        None (default) uses the length in timesteps of the longest event, so
        that no event curve is resampled onto fewer points than it has and
        its peak is not lost between grid points.
    min_event_length : events shorter than this (timesteps) are dropped.
    basis_fit_source : curves the basis is fitted on: "pooled_with_obs"
        (default: every formulation's and the observations' complete
        curves, so the common space represents the observations as well as
        the formulations), "pooled" (the formulations only), "obs", or one
        model's name. The basis does not use which observed curve belongs to
        which event's formulations, so it cannot create an association
        between them (docs/methods.md, Event representation).
    regularization_lambda : roughness penalty (0 = ordinary FPCA).
    ensemble_r2_threshold, ensemble_fraction_threshold : flag low confidence
        when more than ensemble_fraction_threshold of events have an obs
        reconstruction R^2, under a basis fitted to the formulations only,
        below ensemble_r2_threshold (the models' collective shapes may not
        represent what was observed).
    """
    n_components: int | None = None
    variance_threshold: float | None = 0.90
    max_components: int | None = None
    n_grid: int | None = None
    min_event_length: int = 2
    basis_fit_source: str = "pooled_with_obs"
    regularization_lambda: float = 0.0
    ensemble_r2_threshold: float = 0.5
    ensemble_fraction_threshold: float = 0.2
