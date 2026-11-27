"""Tests of the assumption diagnostic plots and of their attachment to every
tier, in cascade and single-tier modes."""
import os

import numpy as np
import pytest

from model_selection.pipeline.config import OutputConfig
from model_selection.pipeline.shared import adequacy_plot_factory
from model_selection.selection import AdequacyConfig, SelectionConfig, build_cascade, build_tier
from model_selection.visualization.adequacy import (
    PLOT_KINDS, PlotContext, make_adequacy_plot_func,
    plot_regression_specification_diagnostics, plot_state_independence_diagnostics,
)


def _state(n=150, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 1))
    c = np.tanh(z) + 0.3 * rng.normal(size=(n, 1))
    o = z ** 2 + 0.5 * c + 0.3 * rng.normal(size=(n, 1))
    return c, o, z


def test_specification_plot_has_a_panel_per_column():
    c, o, z = _state()
    fig = plot_regression_specification_diagnostics(np.column_stack([c, c]), o, z)
    assert len([a for a in fig.layout.annotations if a.text.startswith(("candidate", "observations"))]) == 3


def test_specification_plot_labels_panels_by_feature_name():
    c, o, z = _state()
    fig = plot_regression_specification_diagnostics(np.column_stack([c, c]), np.column_stack([o, o]), z,
                                                    feature_names=("PC1", "PC2"))
    titles = {a.text for a in fig.layout.annotations}
    assert {"candidate, PC1", "candidate, PC2", "observations, PC1", "observations, PC2"} <= titles


def test_specification_plot_requires_conditioning():
    c, o, _ = _state()
    with pytest.raises(ValueError):
        plot_regression_specification_diagnostics(c, o, None)


def test_effect_modification_plot_uses_nonlinear_residuals():
    c, o, z = _state()
    fig = plot_state_independence_diagnostics(c, o, z)
    assert len(fig.data) > 0
    # a scalar candidate and observations: the relationship and residual panels only, no feature weights
    assert not any(t.type == "bar" for t in fig.data)


def test_every_plot_kind_writes_a_page(tmp_path):
    c, o, z = _state()
    for kind in PLOT_KINDS:
        if kind == "multivariate_linearity":
            continue
        plot = make_adequacy_plot_func(kind, "linear", str(tmp_path), "G", "pooled")
        ctx = PlotContext(tier="linear", round_no=2, candidate="x", selected=("y",), result="Not rejected: p=0.4.")
        path = plot(c, o, z, "round2_x_given_y", ctx)
        assert path and os.path.getsize(path) > 1000 and f"_{kind}_round2" in path
        page = open(path).read()
        assert "candidate &#x27;x&#x27;" in page or "candidate 'x'" in page
        assert "Not rejected: p=0.4." in page


def _factory(tmp_path):
    return adequacy_plot_factory(OutputConfig(output_dir=str(tmp_path)), "G", "pooled")


def test_cascade_attaches_plots_to_every_assumption(tmp_path):
    tiers = {t.name: t for t in build_cascade(SelectionConfig(), AdequacyConfig(), False,
                                              plot_factory=_factory(tmp_path))}
    assert all(c.plot is not None for c in tiers["linear"].adequacy_checks)
    assert tiers["nonlinear"].diagnostic_plots == []               # no formal check, no plot
    assert tiers["nonlinear"].adequacy_checks[0].plot is not None  # effect modification
    assert tiers["cmi"].diagnostic_plots == [] and tiers["cmi"].adequacy_checks == []


@pytest.mark.parametrize("name,expected", [("linear", 3), ("nonlinear", 1), ("cmi", 1)])
def test_single_tier_mode_still_plots(tmp_path, name, expected):
    tier = build_tier(name, SelectionConfig(measure=name), plot_factory=_factory(tmp_path))
    assert len(tier.diagnostic_plots) == expected


@pytest.mark.integration
def test_dependence_plot_is_written_by_the_pooled_pathway(tmp_path):
    from model_selection.pipeline import OutputConfig, run_pooled_analysis
    from model_selection.selection import SelectionConfig
    from model_selection.visualization.dependence import plot_dependence
    from tests.fixtures.basins import make_synthetic_basin
    basin = make_synthetic_basin(seed=1)
    run_pooled_analysis(basin, SelectionConfig(measure="linear", n_permutations=20),
                        output=OutputConfig(output_dir=str(tmp_path), assumption_plots=False),
                        rng=np.random.default_rng(0))
    assert (tmp_path / f"{basin.gage}_dependence.html").exists()
    rng = np.random.default_rng(2)
    fig = plot_dependence({"a": rng.standard_normal(500)}, 1.0, "G", observations=rng.standard_normal(500))
    assert len([t for t in fig.data if t.showlegend is not False]) == 2   # one curve per series
    # With the flows as supplied, a second panel draws their correlograms for comparison.
    raw = np.exp(np.convolve(rng.standard_normal(520), np.ones(21) / 21, "valid"))
    fig = plot_dependence({"a": rng.standard_normal(500)}, 1.0, "G", observations=rng.standard_normal(500),
                          raw={"a": raw}, raw_observations=raw)
    curves = [t for t in fig.data if t.mode == "lines" and t.hovertemplate is None]
    assert {t.yaxis for t in curves} == {"y", "y2"} and len(curves) == 4
    assert "before any transformation" in fig.layout.annotations[1].text
