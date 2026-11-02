"""
Unit tests for the POT detector and for event_detection.consensus: the
consensus high-flow regime (rank_consensus_regime), contiguous runs
(raw_events_from_mask), the grouping of runs by the models' own events
(group_runs_by_own_events), and event summaries.
"""
import numpy as np
import pytest

from model_selection.event_detection.base import DeclusterResult
from model_selection.event_detection.consensus import (
    independent_event_summary,
    rank_consensus_regime,
    group_runs_by_own_events,
    raw_events_from_mask,
)
from model_selection.event_detection.pot.detector import (
    detect_model_events,
    find_candidate_peaks,
    group_peaks_by_independence,
    make_pot_ratio_independence_test,
    qualify_peaks,
    resolve_event_boundaries,
)

# ---------------------------------------------------------------------------
# rank_consensus_regime
# ---------------------------------------------------------------------------

def test_rank_consensus_never_uses_obs_by_construction():
    """Signature-level check: rank_consensus_regime's only argument is the
    model-series dict; obs cannot leak in because it is never passed."""
    import inspect
    sig = inspect.signature(rank_consensus_regime)
    assert "obs" not in sig.parameters


def test_rank_consensus_flags_top_decile_correctly():
    rng = np.random.default_rng(0)
    n = 2000
    # two models with very different magnitude scales but correlated shape,
    # to check the regime isn't just picking the larger-magnitude model's peaks
    shape = np.sin(np.linspace(0, 40, n)) + rng.normal(scale=0.2, size=n)
    model_small = shape * 1.0
    model_large = shape * 1000.0 + rng.normal(scale=50, size=n)
    mask = rank_consensus_regime({"m1": model_small, "m2": model_large}, quantile_threshold=0.90)
    assert 0.05 < mask.mean() < 0.15  # roughly top decile


def test_rank_consensus_consensus_modes_differ_as_documented():
    """"any" (max of per-model ranks) should flag at least as many
    timesteps as "median" -- a single model reaching the threshold is
    sufficient for "any" but not necessarily for "median"."""
    rng = np.random.default_rng(1)
    n = 1000
    # one model spikes independently of the other -- "any" should catch
    # this even where "median" (needing both models' agreement) does not.
    m1 = rng.normal(size=n)
    m2 = rng.normal(size=n)
    m1[500:510] += 20  # m1-only spike

    mask_median = rank_consensus_regime({"m1": m1, "m2": m2}, quantile_threshold=0.90, consensus="median")
    mask_any = rank_consensus_regime({"m1": m1, "m2": m2}, quantile_threshold=0.90, consensus="any")
    assert mask_any.sum() >= mask_median.sum()
    assert mask_any[500:510].all()  # "any" definitely catches the m1-only spike


def test_rank_consensus_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        rank_consensus_regime({"m1": np.zeros(10), "m2": np.zeros(20)})


def test_rank_consensus_rejects_empty_dict():
    with pytest.raises(ValueError, match="empty"):
        rank_consensus_regime({})


# ---------------------------------------------------------------------------
# raw_events_from_mask
# ---------------------------------------------------------------------------

def test_raw_events_from_mask_is_zero_tolerance_contiguity():
    """Contiguous runs only: a one-timestep gap separates two runs."""
    mask = np.zeros(60, dtype=bool)
    mask[10:15] = True
    mask[16:20] = True
    raw = raw_events_from_mask(mask)
    assert raw.n_events == 2
    assert independent_event_summary(raw)["n_events"] == 2
    assert raw_events_from_mask(np.zeros(50, dtype=bool)).n_events == 0


# ---------------------------------------------------------------------------
# detect_model_events -- the core per-model event-detection algorithm.
# ---------------------------------------------------------------------------

def _ramp_series(control_x, control_y, n):
    """Piecewise-linear series through named control points -- builds a
    realistic, continuous hydrograph shape rather than isolated spikes, so
    find_peaks and the trough/recession searches see the actual intended
    shape everywhere, not just at a few hand-set indices."""
    x = np.arange(n)
    return np.interp(x, control_x, control_y)


def test_detect_model_events_single_clean_peak():
    series = _ramp_series([0, 20, 25, 30, 60], [1, 1, 10, 1, 1], 61)
    result = detect_model_events(series, quantile_threshold=0.5,
                                   pot_ratio=2 / 3, recession_ratio=0.25)
    assert result.n_events == 1
    assert result.event_id[25] == 0  # the peak itself is inside its own event


def test_detect_model_events_no_qualifying_peaks_returns_empty():
    series = _ramp_series([0, 20, 25, 30, 60], [1, 1, 10, 1, 1], 61)
    result = detect_model_events(series, quantile_threshold=1.01,
                                   pot_ratio=2 / 3, recession_ratio=0.25)
    assert result.n_events == 0
    assert (result.event_id == -1).all()


def test_detect_model_events_flat_series_has_no_peaks():
    series = np.full(50, 5.0)
    result = detect_model_events(series, quantile_threshold=0.5)
    assert result.n_events == 0


def test_detect_model_events_two_well_separated_peaks_stay_independent():
    series = _ramp_series([0, 10, 15, 20, 40, 45, 50, 70], [1, 1, 10, 1, 1, 10, 1, 1], 71)
    result = detect_model_events(series, quantile_threshold=0.5,
                                   pot_ratio=2 / 3, recession_ratio=0.9)
    assert result.n_events == 2
    assert result.event_id[15] != result.event_id[45]


def test_detect_model_events_transitivity_uses_tallest_not_most_recent_group_member():
    """Direct regression test for the transitivity fix: peak A (height 10)
    absorbs peak B (height 5); a third, genuinely separate peak C (height
    8) must then be tested against A (the group's tallest member, per
    this function's own documented design), not against B (the most
    recently absorbed, shorter member). Comparing against B would
    incorrectly also absorb C into the same event -- comparing against A
    correctly keeps C independent. Constructed and verified precisely: the
    trough between B and C alone is NOT low enough relative to B's height
    to separate them, but the trough between A and C IS low enough
    relative to A's height."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)

    # Confirm the test is actually isolating the intended mechanism before
    # trusting the result: compared to B (height 5), trough B-C=4.5 is
    # NOT independent (4.5 > 2/3*5=3.33); compared to A (height 10),
    # trough A-C=4.0 IS independent (4.0 <= 2/3*8=5.33).
    assert series[20:31].min() > (2 / 3) * series[20]  # not independent vs B
    assert series[10:31].min() <= (2 / 3) * min(series[10], series[30])  # independent vs A

    result = detect_model_events(series, quantile_threshold=0.0,
                                   pot_ratio=2 / 3, recession_ratio=0.9)
    assert result.n_events == 2
    assert result.event_id[10] == result.event_id[20]  # A, B merged
    assert result.event_id[30] != result.event_id[10]  # C independent of the A/B group


def test_detect_model_events_pot_ratio_controls_merge_vs_split():
    """A trough between two peaks that merges them at a low pot_ratio but
    separates them at a high one -- direct control over the independence
    test's own sensitivity."""
    series = _ramp_series([0, 10, 15, 20, 40], [1, 10, 5, 10, 1], 41)
    # trough=5, peaks both 10 -> ratio = 5/10 = 0.5
    result_low = detect_model_events(series, quantile_threshold=0.0,
                                       pot_ratio=0.3, recession_ratio=0.9)  # 0.5 > 0.3 -> merge
    result_high = detect_model_events(series, quantile_threshold=0.0,
                                        pot_ratio=0.7, recession_ratio=0.9)  # 0.5 <= 0.7 -> independent
    assert result_low.n_events == 1
    assert result_high.n_events == 2


def test_detect_model_events_recession_ratio_controls_boundary_length():
    """A lower recession_ratio requires flow to decline further before an
    event is considered over -- strictly longer or equal event spans."""
    series = _ramp_series([0, 20, 25, 30, 60], [1, 1, 10, 1, 1], 61)
    result_tight = detect_model_events(series, quantile_threshold=0.5,
                                         pot_ratio=2 / 3, recession_ratio=0.75)  # short tail required
    result_loose = detect_model_events(series, quantile_threshold=0.5,
                                         pot_ratio=2 / 3, recession_ratio=0.1)  # long tail required
    assert result_loose.event_lengths[0] >= result_tight.event_lengths[0]


def test_detect_model_events_never_produces_overlapping_spans():
    """Direct structural check across a busier, noisier series: no two
    events' spans should ever overlap, regardless of how many peaks merge
    or how boundaries extend."""
    rng = np.random.default_rng(3)
    n = 400
    series = np.abs(np.sin(np.linspace(0, 30, n))) * 10 + rng.normal(scale=0.5, size=n).clip(min=-2)
    series = np.clip(series, 0.01, None)
    result = detect_model_events(series, quantile_threshold=0.6,
                                   pot_ratio=2 / 3, recession_ratio=0.25)
    if result.n_events >= 2:
        for k in range(result.n_events - 1):
            this_idx = np.where(result.event_id == k)[0]
            next_idx = np.where(result.event_id == k + 1)[0]
            assert this_idx.max() < next_idx.min()  # strictly non-overlapping, in order


def _multi_storm_series(seed=0):
    """Four genuine, well-separated storms with real intervening dry
    periods -- the exact shape of scenario that exposed the boundary bug: with min_peak_separation small, many
    qualifying peaks exist, and an earlier boundary design made each
    event's END land exactly at the START of the next event's peak, with
    zero gap between them."""
    rng = np.random.default_rng(seed)
    n = 3000
    series = np.full(n, 3.0)

    def add_storm(vals, peak_t, peak_q, k):
        for i in range(peak_t - 15, len(vals)):
            if i < peak_t:
                vals[i] = max(vals[i], 3.0 + (peak_q - 3.0) * (i - (peak_t - 15)) / 15)
            else:
                vals[i] = max(vals[i], 3.0 + (peak_q - 3.0) * np.exp(-0.04 * (i - peak_t)))

    for c in [300, 900, 1600, 2400]:
        add_storm(series, c, rng.uniform(20, 50), 0.04)
    return np.clip(series * (1 + rng.normal(scale=0.015, size=n)), 0.1, None)


def test_detect_model_events_boundaries_do_not_touch_neighboring_peaks():
    """Direct regression test for a real, severe bug found on real basin
    data: with min_peak_separation small
    (many qualifying peaks), a group's boundary search was anchored to
    a group member whose own value could be too low for
    recession_ratio*value to ever be reached before the next storm's
    peak -- causing one event to end (or the next to start) exactly AT
    a neighboring peak, with zero gap, rather than at a genuine trough.
    Every event's start/end value here must be substantially receded
    from its own peak -- not sitting at a neighboring event's peak
    height."""
    series = _multi_storm_series()
    result, peak_groups = detect_model_events(
        series, quantile_threshold=0.90, recession_ratio=0.25,
        min_peak_separation=6, return_peak_groups=True,
    )
    assert result.n_events == 4
    for eid in range(result.n_events):
        positions = np.where(result.event_id == eid)[0]
        span_lo, span_hi = positions.min(), positions.max()
        peak_height = series[peak_groups[eid]].max()
        # start/end values must be well below this event's own peak --
        # a genuine recession, not sitting at a neighboring peak's height
        assert series[span_lo] < 0.3 * peak_height
        assert series[span_hi] < 0.3 * peak_height


def test_detect_model_events_leaves_genuine_gaps_between_isolated_events():
    """Direct regression test: isolated events (separated by a real dry
    period) must leave that period unclaimed by either event, not have
    one event's span extend all the way to the next event's peak. This
    holds for ANY gap in the record, not just the record's own start/
    end -- verified here on three INTERIOR gaps between four storms, none
    of which are at the record boundary."""
    series = _multi_storm_series()
    result = detect_model_events(series, quantile_threshold=0.90,
                                   recession_ratio=0.25, min_peak_separation=6)
    assert result.n_events == 4
    for k in range(result.n_events - 1):
        this_max = np.where(result.event_id == k)[0].max()
        next_min = np.where(result.event_id == k + 1)[0].min()
        assert next_min - this_max > 100  # a real, substantial unclaimed gap, not near-zero


def test_detect_model_events_isolation_more_common_at_higher_quantile_threshold():
    """Raising quantile_threshold removes peaks, widening gaps
    between remaining groups -- so the SAME resolution logic (compare
    two independently-computed natural boundaries) should, if anything,
    produce equal or larger gaps at a higher threshold, not smaller
    ones, since fewer qualifying peaks means fewer opportunities for
    natural boundaries to overlap and force a shared-trough resolution."""
    series = _multi_storm_series()
    result_low, _ = detect_model_events(series, quantile_threshold=0.90,
                                          recession_ratio=0.25, min_peak_separation=6,
                                          return_peak_groups=True)
    result_high, _ = detect_model_events(series, quantile_threshold=0.97,
                                           recession_ratio=0.25, min_peak_separation=6,
                                           return_peak_groups=True)
    assert result_low.n_events == result_high.n_events == 4
    gaps_low = [np.where(result_low.event_id == k + 1)[0].min() - np.where(result_low.event_id == k)[0].max()
                 for k in range(3)]
    gaps_high = [np.where(result_high.event_id == k + 1)[0].min() - np.where(result_high.event_id == k)[0].max()
                  for k in range(3)]
    assert all(gh >= gl for gh, gl in zip(gaps_high, gaps_low))


def test_detect_model_events_span_always_covers_every_group_member():
    """Direct regression test for a second, related bug found while
    fixing the first: anchoring a group's
    natural boundary purely to its TALLEST member's own recession range
    can leave a shorter, already-confirmed group member entirely outside
    the resulting span -- contradicting step 3's own decision that it
    belongs to this event. Verified against the precisely-known A/B/C
    scenario, where B's own recession is much faster than A's (the
    group's tallest, and hence the boundary anchor)."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    result, peak_groups = detect_model_events(
        series, quantile_threshold=0.0, recession_ratio=0.9,
        return_peak_groups=True,
    )
    for event_id, members in peak_groups.items():
        for m in members:
            assert result.event_id[m] == event_id


def test_detect_model_events_does_not_silently_drop_a_genuine_nearby_peak():
    """Direct regression test for a real, severe bug found on actual
    basin data: peak detection previously
    passed distance=min_inter_event_time to find_peaks, which silently
    discards any peak within that many timesteps of a taller one --
    scipy's own find_peaks(distance=...) semantics, verified directly.
    This is exactly the multi-peak separation decision the independence
    test (step 3) exists to make, and it was being made wrongly, by
    proximity alone, before step 3 ever ran. A dominant peak (height 30)
    with a genuine, distinct secondary peak (height 14) only 7 timesteps
    away must now surface BOTH as candidates -- min_inter_event_time=8
    (exceeding that 7-timestep gap) is deliberately chosen so the old,
    buggy behavior would have dropped the secondary peak entirely."""
    n = 100
    series = np.full(n, 1.0)
    series[40:60] = [1, 2, 5, 10, 20, 30, 25, 18, 12, 8, 6, 9, 14, 10, 6, 4, 3, 2, 1.5, 1]

    result = detect_model_events(series, quantile_threshold=0.0,
                                   recession_ratio=0.9)
    assert result.n_events == 2
    assert result.event_id[45] != result.event_id[52]  # dominant (45) and secondary (52) peaks


def test_detect_model_events_correctly_merges_noisy_single_storm_despite_many_candidates():
    """The other side of the same fix, verified together: removing the
    distance pre-filter means a single, genuine storm's own noisy
    recession tail can now surface dozens of trivial local maxima as
    candidates (confirmed directly during development: 22 "qualifying"
    peaks from ONE storm, at quantile_threshold=0.90, on a realistic 1%-
    noise recession). The independence test (step 3) must still
    correctly reject all of them and produce ONE event -- confirming
    step 3 alone, not a proximity pre-filter, is what's actually
    responsible for merge/split decisions now."""
    rng = np.random.default_rng(0)
    n = 2000
    series = np.full(n, 3.0)
    peak_t, peak_q, k = 500, 100.0, 0.03
    for i in range(peak_t - 20, n):
        if i < peak_t:
            series[i] = 3.0 + (peak_q - 3.0) * (i - (peak_t - 20)) / 20
        else:
            series[i] = 3.0 + (peak_q - 3.0) * np.exp(-k * (i - peak_t))
    series = np.clip(series * (1 + rng.normal(scale=0.01, size=n)), 0.1, None)

    result = detect_model_events(series, quantile_threshold=0.90,
                                   recession_ratio=0.25)
    assert result.n_events == 1


def test_detect_model_events_min_peak_separation_default_unaffected():
    """Backward-compatibility check: min_peak_separation's default (1)
    must produce identical behavior to before this parameter existed --
    the exact scenario that exposed the original distance-filter bug."""
    n = 100
    series = np.full(n, 1.0)
    series[40:60] = [1, 2, 5, 10, 20, 30, 25, 18, 12, 8, 6, 9, 14, 10, 6, 4, 3, 2, 1.5, 1]
    result = detect_model_events(series, quantile_threshold=0.0,
                                   recession_ratio=0.9)
    assert result.n_events == 2
    assert result.event_id[45] != result.event_id[52]


def test_detect_model_events_min_peak_separation_rejects_nearby_micro_peaks():
    """min_peak_separation, set to a small, FIXED value (deliberately
    NOT derived from min_inter_event_time -- see this function's own
    docstring for why conflating the two was the original bug), should
    reject a genuine but very closely-spaced secondary peak, unlike the
    default (1, no-op)."""
    n = 100
    series = np.full(n, 1.0)
    series[40:60] = [1, 2, 5, 10, 20, 30, 25, 18, 12, 8, 6, 9, 14, 10, 6, 4, 3, 2, 1.5, 1]
    result_filtered = detect_model_events(series, quantile_threshold=0.0,
                                            recession_ratio=0.9, min_peak_separation=8)
    result_default = detect_model_events(series, quantile_threshold=0.0,
                                           recession_ratio=0.9, min_peak_separation=1)
    assert result_filtered.n_events == 1  # secondary peak (7 steps away) now rejected at detection
    assert result_default.n_events == 2   # unaffected -- both peaks still detected


def test_detect_model_events_return_peak_groups_default_false_unchanged_type():
    """Backward-compatibility check: without return_peak_groups=True,
    the return type must be exactly DeclusterResult, not a tuple."""
    from model_selection.event_detection.base import DeclusterResult
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    result = detect_model_events(series, quantile_threshold=0.0, recession_ratio=0.9)
    assert isinstance(result, DeclusterResult)


def test_detect_model_events_return_peak_groups_matches_actual_merges():
    """The key, direct validation: return_peak_groups=True must report
    EXACTLY the original, raw qualifying-peak indices that were merged
    into each final event -- not just the tallest, not just the
    boundary span -- verified against the precisely-known A/B/C
    scenario (A absorbs B; C stays independent)."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    result, peak_groups = detect_model_events(
        series, quantile_threshold=0.0, recession_ratio=0.9,
        return_peak_groups=True,
    )
    assert result.n_events == 2
    assert peak_groups == {0: [10, 20], 1: [30]}
    # cross-check internal consistency: every peak_groups member must
    # actually fall within its own reported event's span
    for event_id, members in peak_groups.items():
        for m in members:
            assert result.event_id[m] == event_id


def test_detect_model_events_return_peak_groups_covers_larger_merge():
    """A three-way merge (A absorbs B absorbs C, all one event) must
    report all three original peak indices under the single resulting
    event_id, not just the tallest."""
    control_x = [0, 10, 15, 20, 25, 30, 35]
    control_y = [1, 10, 4, 8, 4.5, 6, 1]
    series = _ramp_series(control_x, control_y, 40)
    result, peak_groups = detect_model_events(
        series, quantile_threshold=0.0, pot_ratio=0.3, recession_ratio=0.9,
        return_peak_groups=True,
    )
    assert result.n_events == 1
    assert set(peak_groups[0]) == {10, 20, 30}


# ---------------------------------------------------------------------------
# split_multi_peak_events
# ---------------------------------------------------------------------------

def _own(n, spans):
    """A model's own events as a DeclusterResult, one event per (start, end)."""
    event_id = np.full(n, -1, dtype=np.int64)
    for k, (start, end) in enumerate(spans):
        event_id[start:end + 1] = k
    lengths = np.array([e - s + 1 for s, e in spans], dtype=np.int64)
    return DeclusterResult(event_id=event_id, n_events=len(spans), event_lengths=lengths)


def _two_runs(n=60):
    mask = np.zeros(n, dtype=bool)
    mask[10:15] = True
    mask[25:30] = True
    return mask


def test_runs_join_when_a_majority_of_own_events_span_both():
    mask = _two_runs()
    one_flood, two_floods = [(8, 32)], [(8, 18), (22, 32)]
    own = {"a": _own(60, one_flood), "b": _own(60, one_flood), "c": _own(60, two_floods)}
    result = group_runs_by_own_events(mask, own)
    assert result.n_events == 1
    assert set(result.event_id[mask]) == {0}


def test_runs_stay_separate_without_a_majority():
    mask = _two_runs()
    one_flood, two_floods = [(8, 32)], [(8, 18), (22, 32)]
    own = {"a": _own(60, one_flood), "b": _own(60, two_floods), "c": _own(60, two_floods)}
    assert group_runs_by_own_events(mask, own).n_events == 2


def test_half_of_the_models_join_runs():
    # Joining requires at least half of the models: two of four suffice.
    mask = _two_runs()
    one_flood, two_floods = [(8, 32)], [(8, 18), (22, 32)]
    own = {k: _own(60, one_flood if k in "ab" else two_floods) for k in "abcd"}
    assert group_runs_by_own_events(mask, own).n_events == 1
    own = {k: _own(60, one_flood if k == "a" else two_floods) for k in "abcd"}
    assert group_runs_by_own_events(mask, own).n_events == 2


def test_upper_median_marks_high_flow_when_half_the_models_agree():
    from model_selection.event_detection.consensus import rank_consensus_regime
    n = 100
    low, high = np.arange(n, dtype=float), np.arange(n, dtype=float)[::-1]
    # Two of four models rank the first timestep at their top, two at their bottom.
    mask = rank_consensus_regime({"a": high, "b": high, "c": low, "d": low}, quantile_threshold=0.9)
    assert mask[0] and mask[-1]


def test_grouping_needs_no_separation_length():
    # Runs far apart join if most models see one flood; runs one step apart
    # stay separate if most models see two.
    mask = np.zeros(400, dtype=bool)
    mask[10:15] = True
    mask[300:305] = True
    own = {k: _own(400, [(5, 310)]) for k in "abc"}
    assert group_runs_by_own_events(mask, own).n_events == 1
    mask = np.zeros(60, dtype=bool)
    mask[10:15] = True
    mask[16:20] = True
    own = {k: _own(60, [(8, 15), (16, 22)]) for k in "abc"}
    assert group_runs_by_own_events(mask, own).n_events == 2


def test_grouping_is_transitive_and_marks_only_high_flow_timesteps():
    mask = np.zeros(80, dtype=bool)
    for start in (10, 20, 30):
        mask[start:start + 3] = True
    own = {k: _own(80, [(8, 35)]) for k in "abc"}
    result = group_runs_by_own_events(mask, own)
    assert result.n_events == 1
    assert np.all(result.event_id[~mask] == -1)
    assert result.event_lengths.tolist() == [9]


def test_grouping_with_no_high_flow_returns_empty():
    result = group_runs_by_own_events(np.zeros(50, dtype=bool), {})
    assert result.n_events == 0


def test_independent_event_summary_zero_events():
    empty = DeclusterResult(event_id=np.full(20, -1, dtype=np.int64), n_events=0,
                              event_lengths=np.array([], dtype=np.int64))
    summary = independent_event_summary(empty)
    assert summary["n_events"] == 0
    assert summary["median_event_length"] == 0.0


def test_find_candidate_peaks_independently_matches_the_orchestrator():
    """Stage 1, called directly: must find the same raw candidates
    detect_model_events itself would find internally."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    peaks = find_candidate_peaks(series, min_peak_separation=1)
    assert list(peaks) == [10, 20, 30]


def test_find_candidate_peaks_min_peak_separation_filters_directly():
    n = 100
    series = np.full(n, 1.0)
    series[40:60] = [1, 2, 5, 10, 20, 30, 25, 18, 12, 8, 6, 9, 14, 10, 6, 4, 3, 2, 1.5, 1]
    peaks_default = find_candidate_peaks(series, min_peak_separation=1)
    peaks_filtered = find_candidate_peaks(series, min_peak_separation=8)
    assert len(peaks_filtered) < len(peaks_default)


def test_qualify_peaks_independently_matches_the_orchestrator():
    """Stage 2, called directly on stage 1's own output: must produce
    the same qualifying subset detect_model_events finds internally."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    peaks = find_candidate_peaks(series, min_peak_separation=1)
    qualifying = qualify_peaks(series, peaks, quantile_threshold=0.0)
    assert list(qualifying) == [10, 20, 30]

    # a threshold just above B's own rank (~0.537) should exclude it specifically
    qualifying_strict = qualify_peaks(series, peaks, quantile_threshold=0.6)
    assert list(qualifying_strict) == [10, 30]


def test_group_peaks_by_independence_reproduces_known_ab_c_grouping():
    """Stage 3, called directly on stage 2's own output: must reproduce
    the precisely-known A/B/C grouping (A absorbs B; C independent)."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    qualifying = np.array([10, 20, 30])
    test_fn = make_pot_ratio_independence_test(pot_ratio=2 / 3)
    groups, tallest = group_peaks_by_independence(series, qualifying, test_fn)
    assert groups == [[10, 20], [30]]
    assert tallest == [10, 30]


def test_resolve_event_boundaries_reproduces_known_span_and_no_overlap():
    """Stage 4, called directly on stage 3's own output: must reproduce
    the correct, non-overlapping spans and cover every group member."""
    control_x = [0, 10, 15, 20, 25, 30, 40]
    control_y = [1, 10, 4, 5, 4.5, 8, 1]
    series = _ramp_series(control_x, control_y, 41)
    groups = [[10, 20], [30]]
    tallest = [10, 30]
    result = resolve_event_boundaries(series, groups, tallest, recession_ratio=0.9)
    assert result.n_events == 2
    assert result.event_id[10] == result.event_id[20] == 0
    assert result.event_id[30] == 1
    idx0 = np.where(result.event_id == 0)[0]
    idx1 = np.where(result.event_id == 1)[0]
    assert idx0.max() < idx1.min()


def test_four_stages_chained_manually_match_detect_model_events_directly():
    """The strongest possible test of the refactor's own claim (four
    genuinely separate, independently composable stages): manually
    chaining all four stage functions, exactly as detect_model_events'
    own orchestrator does internally, must produce byte-identical
    results to calling detect_model_events itself, on a busier, noisier
    series -- not just the small, hand-built A/B/C case."""
    rng = np.random.default_rng(7)
    n = 2000
    series = np.full(n, 3.0)
    for c in rng.choice(np.arange(100, n - 100), size=6, replace=False):
        width = rng.integers(15, 60)
        height = rng.uniform(10, 50)
        t = np.arange(min(width, n - c))
        end = min(c + width, n)
        series[c:end] = np.maximum(
            series[c:end], series[c] + height * np.exp(-((t[:end-c] - width/3) ** 2) / (2 * max(width/6, 1) ** 2))
        )
    series = series + rng.normal(scale=0.05, size=n)

    direct_result = detect_model_events(series, quantile_threshold=0.85,
                                          pot_ratio=2 / 3, recession_ratio=0.25, min_peak_separation=3)

    peaks = find_candidate_peaks(series, min_peak_separation=3)
    qualifying = qualify_peaks(series, peaks, quantile_threshold=0.85)
    test_fn = make_pot_ratio_independence_test(pot_ratio=2 / 3)
    groups, tallest = group_peaks_by_independence(series, qualifying, test_fn)
    manual_result = resolve_event_boundaries(series, groups, tallest, recession_ratio=0.25)

    assert np.array_equal(direct_result.event_id, manual_result.event_id)
    assert direct_result.n_events == manual_result.n_events


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_peak_qualification_on_a_reference_set_matches_the_series_itself():
    """The high-flow quantile is taken among the values present; with nothing
    missing the reference is the series and the result is unchanged."""
    rng = np.random.default_rng(8)
    series = rng.gamma(2.0, size=500)
    peaks = find_candidate_peaks(series)
    assert np.array_equal(qualify_peaks(series, peaks, 0.8, reference=series), qualify_peaks(series, peaks, 0.8))
