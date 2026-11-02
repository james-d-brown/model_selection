"""The event-merging diagnostic agrees with the high-flow pathway's own event
construction and its stage counts are consistent."""
import numpy as np

from model_selection.event_detection import EventConfig
from model_selection.pipeline import run_high_flow_analysis
from model_selection.selection import SelectionConfig
from scripts.diagnostics.event_merging import event_merging_report
from tests.fixtures.basins import make_synthetic_basin


def test_stage_counts_are_consistent_and_match_the_pathway(tmp_path):
    basin = make_synthetic_basin(n_hours=6000, n_events=40, seed=3)
    events = EventConfig(method="wasp", high_flow_quantile=0.9)
    summary, table = event_merging_report(basin, events, output_dir=str(tmp_path))
    assert summary["raw_runs"] - summary["joins"] == summary["events_after_joining"] == len(table)
    assert table["raw_runs"].sum() == summary["raw_runs"]
    win = table[table["window"].notna()]
    assert (win["start"].iloc[1:].values > win["end"].iloc[:-1].values).all()  # windows never overlap
    assert len(win) == len(table) - summary["consensus_events_without_peak"]
    assert (win["formulations_with_peak"] >= 1).all()
    assert table["formulations_with_own_event"].between(0, summary["formulations"]).all()
    assert (tmp_path / f"{basin.gage}_event_merging_windows.csv").exists()

    result = run_high_flow_analysis(basin, SelectionConfig(measure="linear", n_permutations=20), events=events,
                                    rng=np.random.default_rng(0))
    assert result.n_raw_consensus_events == summary["events_after_joining"]
    assert len(result.high_flow_events) == summary["event_samples"]
