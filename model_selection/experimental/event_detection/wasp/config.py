"""Free choices of the WASP event detector."""
from __future__ import annotations

from dataclasses import dataclass

NOISE_MULTIPLE = 3.0
"""A feature counts only where it is at least this many noise standard
deviations: a peak's prominence, an onset's curvature. The threshold of Du,
Kibbe & Lin (2006) for wavelet peak detection."""

NOISE_BINS = 10
"""Equal-count flow bins in which the noise level is estimated."""


@dataclass
class WASPConfig:
    """WASP settings.

    sigma_step : spacing of the scale sweep in timesteps; affects precision
        and cost, not what counts as a genuine peak.
    worm_tolerance_alpha : scale-relative connectivity tolerance,
        tol(sigma) = round(alpha * sigma), used when linking local maxima
        across scales. 0.1 is a reasonable, not a tuned, value. (Lindeberg's
        scale-space theory could derive this bound from the signal itself;
        not implemented.)
    recession_target : "onset_value", the only option implemented: an event
        ends when flow returns to its onset value.
    """
    sigma_step: float = 1.0
    worm_tolerance_alpha: float = 0.1
    recession_target: str = "onset_value"
