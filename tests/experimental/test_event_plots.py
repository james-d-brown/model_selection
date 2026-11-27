"""Plots of the experimental high-flow event pathway."""
import numpy as np
import pytest


def test_hydrograph_draws_the_median_high_flow_threshold():
    from model_selection.experimental.visualization.hydrograph import plot_basin_diagnostics
    from tests.fixtures.basins import make_synthetic_basin
    basin = make_synthetic_basin(seed=1)
    fig = plot_basin_diagnostics(basin, [], high_flow_quantile=0.9)
    (line,) = [t for t in fig.data if t.name.startswith("high-flow threshold")]
    expected = np.median([np.nanquantile(s.values, 0.9) for s in basin.formulations.values()])
    assert line.y[0] == pytest.approx(float(expected))
