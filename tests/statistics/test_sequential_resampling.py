"""
Sequential resampling of tests made at a level below alpha (the escalation
family's Bonferroni level): the confidence-sequence method of Ding, Gandy and
Hahn (2020), started after the draws of a test at alpha and capped at the
draws that give the same relative Monte Carlo error at the level
(significance.base; docs/methods.md, Resamples per test).

Checked here: the helpers against their formulas; that the stopping rule's
decisions and draw counts follow from the exceedance count alone, on
Bernoulli exceedance streams whose exact p-value is known; and, on a real
escalation test, that a test made below alpha keeps the draws and result of
a test at alpha whenever it stops at the first check, draws on for a
departure, and never draws fewer than a test at alpha or more than the cap.
"""
import numpy as np
import pytest
from scipy import stats

from model_selection.statistics.significance.base import (
    draw_until_decided, resamples_at_level, resamples_for_relative_error, sequential_decision)

ALPHA, N0, RISK, CHUNK = 0.05, 1900, 1e-3, 128


def test_resamples_at_level_keeps_the_relative_error_of_the_draws_at_alpha():
    assert resamples_for_relative_error(ALPHA, 0.1) == N0
    assert resamples_at_level(ALPHA, N0, ALPHA) == N0
    assert resamples_at_level(ALPHA, N0, ALPHA / 16) == resamples_for_relative_error(ALPHA / 16, 0.1) == 31900
    # A level above alpha, or a coarse draw count, never gives fewer draws than at alpha.
    assert resamples_at_level(ALPHA, N0, 0.2) == N0
    assert resamples_at_level(ALPHA, 199, ALPHA / 10) > 199


def test_sequential_decision_follows_the_confidence_sequence_criterion():
    level = ALPHA / 16
    for n in (1900, 5000, 31900):
        for s in (0, 1, 3, int(level * n), int(3 * level * n), int(0.5 * n)):
            stop = (n + 1) * stats.binom.pmf(s, n, level) <= RISK
            d = sequential_decision(s, n, level, RISK)
            assert (d is not None) == stop
            if stop:
                assert d == (s / n < level)
    # At exactly the level the confidence set always contains it: never settled.
    assert sequential_decision(10, 1000, 0.01, RISK) is None


def _bernoulli_stream(p, rng):
    """draw_more for a test whose draws exceed the observed value with probability p."""
    def draw_more(k):
        e = (rng.random(k) < p).astype(float)
        return e, np.full(k, 7.0)      # the statistic's draws (1 = exceeds 0.5) and a companion array
    return draw_more


@pytest.mark.parametrize("ratio", [0.0, 0.3, 3.0, 20.0])
def test_draw_until_decided_settles_away_from_the_level_with_the_exact_decision(ratio):
    level = ALPHA / 16
    cap = resamples_at_level(ALPHA, N0, level)
    rng = np.random.default_rng(1)
    wrong, draws = 0, []
    for _ in range(40):
        more = _bernoulli_stream(ratio * level, rng)
        (null, companion), settled = draw_until_decided(0.5, more(N0), more, level, RISK, cap, CHUNK)
        assert null.size == companion.size and np.all(companion == 7.0)
        assert N0 <= null.size <= cap and (null.size - N0) % CHUNK == 0
        exceed = int(null.sum())
        decision = (sequential_decision(exceed, null.size, level, RISK) if settled
                    else (exceed + 1) / (null.size + 1) < level)
        wrong += decision != (ratio < 1)
        draws.append(null.size)
    assert wrong == 0
    if ratio >= 20:
        assert max(draws) == N0                      # a clear non-departure stops at the first check
    if ratio == 0.0:
        # A clear departure stops at the first checkpoint with (n + 1)(1 - level)^n <= risk.
        n = N0
        while (n + 1) * (1 - level) ** n > RISK:
            n += CHUNK
        assert set(draws) == {n}


def test_draw_until_decided_reaches_the_cap_at_the_level():
    level = ALPHA / 16
    cap = resamples_at_level(ALPHA, N0, level)
    rng = np.random.default_rng(2)
    more = _bernoulli_stream(level, rng)
    (null, _), settled = draw_until_decided(0.5, more(N0), more, level, RISK, cap, CHUNK)
    assert null.size == cap and not settled


def _escalation_data(departing: bool, seed: int, n: int = 400):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    candidate = rng.normal(size=n)
    effect = np.where(candidate > 0, 2.0 * candidate, 0.2 * candidate) if departing else 0.5 * candidate
    obs = effect + 0.8 * z + rng.normal(scale=0.5, size=n)
    return candidate, obs, z


def _run(test, data, seed, n_permutations, **kw):
    candidate, obs, z = data
    return test.run(candidate, obs, conditioning=z, time_positions=np.arange(candidate.size), min_temporal_lag=1,
                    n_permutations=n_permutations, rng=np.random.default_rng(seed), **kw)


def test_a_test_at_alpha_is_unchanged_by_the_level_argument():
    from model_selection.statistics.adequacy.linearity import LinearityTest
    test = LinearityTest(alpha=ALPHA, effect_size_floor=0.0)
    data = _escalation_data(False, 3)
    plain = _run(test, data, 5, 199)
    at_alpha = _run(test, data, 5, 199, level=ALPHA, resampling_risk=RISK)
    assert at_alpha.permutation_p_value == plain.permutation_p_value
    assert at_alpha.n_permutations == plain.n_permutations == 199
    assert at_alpha.level is None and at_alpha.settled is None


def test_a_test_below_alpha_keeps_the_fixed_result_when_it_stops_at_the_first_check():
    from model_selection.statistics.adequacy.linearity import LinearityTest
    test = LinearityTest(alpha=ALPHA, effect_size_floor=0.0)
    level = ALPHA / 16
    stopped_first = 0
    for seed in range(6):
        data = _escalation_data(False, 10 + seed)
        plain = _run(test, data, seed, 199)
        seq = _run(test, data, seed, 199, level=level, resampling_risk=RISK)
        assert 199 <= seq.n_permutations <= resamples_at_level(ALPHA, 199, level)
        assert seq.level == level
        if seq.n_permutations == 199:
            stopped_first += 1
            # The same draws: identical p-value and effect, decided at the level.
            assert seq.permutation_p_value == plain.permutation_p_value
            assert seq.bias_corrected_semipartial_effect_size == plain.bias_corrected_semipartial_effect_size
            assert seq.settled and not seq.reject_linearity
    assert stopped_first >= 4


def test_a_departure_below_alpha_draws_on_and_is_decided_at_the_level():
    from model_selection.statistics.adequacy.base import as_adequacy_test
    from model_selection.statistics.adequacy.linearity import LinearityTest
    level = ALPHA / 16
    test = LinearityTest(alpha=ALPHA, effect_size_floor=0.0)
    data = _escalation_data(True, 4)
    # 199 draws at alpha cannot settle a departure at alpha / 16: the cap is reached and the
    # test is decided by its p-value from all of the cap's draws.
    small = _run(test, data, 6, 199, level=level, resampling_risk=RISK)
    assert small.n_permutations == resamples_at_level(ALPHA, 199, level) and small.settled is False
    assert small.reject_linearity and small.permutation_p_value < level
    # With the default draws, the sequential test settles a clear departure before the cap.
    n = N0
    while (n + 1) * (1 - level) ** n > RISK:
        n += CHUNK
    full = _run(test, data, 6, N0, level=level, resampling_risk=RISK)
    assert full.settled and full.reject_linearity and full.n_permutations == n
    # The adequacy wrapper passes the level through and reports the draws used.
    outcome = as_adequacy_test(test)(*data, n_permutations=N0, rng=np.random.default_rng(6),
                                     time_positions=np.arange(data[0].size), min_temporal_lag=1,
                                     level=level, resampling_risk=RISK)
    assert outcome.n_draws == n and outcome.level == level and outcome.settled and not outcome.tier_adequate
