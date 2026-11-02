"""The series' autocorrelation e-folding time, used as the WASP detector's
scale ceiling."""
from model_selection.hydrology.timescale import BasinTimescaleResult, estimate_basin_timescale

__all__ = ["BasinTimescaleResult", "estimate_basin_timescale"]
