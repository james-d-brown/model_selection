"""Censoring at high flow: the configuration object and the record a censored
sample carries (docs/methods.md, Censoring at high flow)."""
import numpy as np
import pytest

from model_selection.event_detection import EventConfig
from model_selection.preprocessing import CensoredSeries, Censoring
from model_selection.preprocessing.censoring import DEFAULT_HIGH_FLOW_QUANTILE
from model_selection.preprocessing.transforms import prepare_series


def test_one_default_quantile_defines_high_flow_for_events_and_censoring():
    assert Censoring().quantile == EventConfig().high_flow_quantile == DEFAULT_HIGH_FLOW_QUANTILE
    assert Censoring.at_high_flow(EventConfig(high_flow_quantile=0.95)).quantile == 0.95


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan"), float("inf")])
def test_quantile_must_lie_strictly_between_zero_and_one(bad):
    with pytest.raises(ValueError, match="quantile"):
        Censoring(bad)


def test_values_below_the_quantile_are_set_to_the_quantile_value_itself():
    c = Censoring(0.8)
    x = np.array([5.0, 1.0, np.nan, 3.0, 9.0, 2.0, 7.0])
    t = c.threshold(x)
    assert t == pytest.approx(np.quantile(x[np.isfinite(x)], 0.8))      # finite values only
    out = c.apply(x, t)
    assert np.isnan(out[2])
    ok = np.isfinite(x)
    assert np.array_equal(out[ok], np.maximum(x[ok], t))
    assert np.all(out[ok & (x > t)] == x[ok & (x > t)])                  # values above it unchanged
    with pytest.raises(ValueError, match="no finite values"):
        c.threshold(np.array([np.nan, np.nan]))


def test_censoring_normal_scores_leaves_the_point_mass_adjacent_to_the_exceedances():
    """Censoring is applied after the normal-score transform of log flow, so
    the censored values lie at the quantile's normal score, next to the
    smallest exceedance, not at the mid-rank score of a censored group."""
    rng = np.random.default_rng(0)
    flow = rng.gamma(2.0, size=5000)
    scores = prepare_series(flow)
    c = Censoring(0.9)
    t = c.threshold(scores)
    out = c.apply(scores, t)
    assert np.mean(out == t) == pytest.approx(0.9, abs=0.001)
    above = out[out > t]
    assert t == pytest.approx(1.2816, abs=0.01)                          # the 0.9 normal quantile
    assert above.min() - t < 0.01                                        # no gap at the threshold
    # The order of the flows above the threshold is kept.
    hi = flow > np.quantile(flow, 0.9)
    assert np.array_equal(np.argsort(out[hi]), np.argsort(flow[hi]))


def test_censored_series_censor_each_candidate_at_its_own_threshold_and_subset_rows():
    unc = {"a": np.arange(10.0), "b": np.arange(10.0) * 2}
    cs = CensoredSeries(unc, {"a": 7.0, "b": 3.0}, obs_threshold=5.0)
    assert np.array_equal(cs.censor("a", np.array([1.0, 8.0])), [7.0, 8.0])
    assert np.array_equal(cs.censor("b", np.array([1.0, 8.0])), [3.0, 8.0])
    sub = cs.subset(np.array([0, 9]))
    assert np.array_equal(sub.uncensored["a"], [0.0, 9.0]) and sub.thresholds == cs.thresholds
    shares = cs.shares_at_threshold({"a": np.maximum(unc["a"], 7.0), "b": np.maximum(unc["b"], 3.0)},
                                    np.maximum(np.arange(10.0), 5.0))
    assert shares == {"a": 0.8, "b": 0.2, "observations": 0.6}
