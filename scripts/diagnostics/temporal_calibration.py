"""
False-positive rate of the tests that resample in time, on autocorrelated
series satisfying each test's null hypothesis, with samples built as the
pooled pathway builds them.

Scenarios:

- "ar1": first-order autoregressive series with an e-folding time of 24 steps.
- "two_scale": the sum of a fast component (e-folding time 6 steps, 70% of
  the variance) and a slow component (e-folding time 100 steps, 30%), a
  simple analogue of quick and slow runoff. Its autocorrelation falls to 1/e
  after 12 steps but stays near 0.25 for tens of steps.

Samples: the linear tier and the nonlinear-dependence and
nonlinear-relation checks run on the full record, as in
the pooled pathway. The nonlinear tier, the effect-modification check and the CMI tier run, as in
the pooled pathway, on a subsample of a long record at a constant stride of
at least the dependence lag (the largest Politis decorrelation lag of the
simulated series), capped at 2,000 points. Rejections are counted at
alpha = 0.05 with no effect-size floor, so the rate measures the p-value
alone.

Usage: python -m scripts.diagnostics.temporal_calibration [--reps N] [--tests a,b] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal import lfilter

from model_selection.statistics.adequacy.linearity import LinearityTest
from model_selection.statistics.adequacy.specification import linear_specification_test
from model_selection.statistics.adequacy.state_independence import StateIndependenceTest
from model_selection.statistics.conditional_independence import CMITest, LinearResidualTest, NonlinearResidualTest
from model_selection.statistics.significance.base import permutation_p_value
from model_selection.statistics.significance.block_length import estimate_dependence
from model_selection.statistics.significance.resampling import grid_subsample

ALPHA = 0.05
SCENARIOS = ("ar1", "two_scale")
FULL_N = 3000        # full-record tests
LONG_N = 40000       # long record subsampled for the costlier tests
MAX_N = 2000         # the pooled pathway's cap


def _ar1(n: int, tau: float, rng: np.random.Generator) -> np.ndarray:
    phi = np.exp(-1.0 / tau)
    burn = int(10 * tau)
    x = lfilter([np.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal(n + burn))
    return x[burn:]


def series(scenario: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if scenario == "ar1":
        return _ar1(n, 24, rng)
    return np.sqrt(0.7) * _ar1(n, 6, rng) + np.sqrt(0.3) * _ar1(n, 100, rng)


def _null_triplet(scenario, n, rng, o_on_c=False):
    """(z, c, o) with c and o conditionally independent given z."""
    z = series(scenario, n, rng)
    c = 0.7 * z + 0.7 * series(scenario, n, rng)
    o = (0.5 * z + 0.5 * c if o_on_c else 0.7 * z) + 0.7 * series(scenario, n, rng)
    return z, c, o


def _lag(*xs) -> int:
    return max(estimate_dependence(x).decorrelation_lag for x in xs)


def _subsample(scenario, rng, o_on_c=False):
    """A long record subsampled as the pooled pathway does:
    (z, c, o, time positions, grid step, dependence lag)."""
    z, c, o = _null_triplet(scenario, LONG_N, rng, o_on_c)
    lag = _lag(z, c)
    idx, positions, step, _ = grid_subsample(np.arange(LONG_N), 1, lag, MAX_N, rng, at_lag_spacing=False)
    return z[idx], c[idx], o[idx], positions, step, lag


def _p(result) -> float:
    return permutation_p_value(float(result.observed[0]), result.null[:, 0])


# Each trial returns (p-value, block length) for one dataset satisfying the null.
def trial_linear(scenario, rng, n_perm):
    z, c, o = _null_triplet(scenario, FULL_N, rng)
    res = LinearResidualTest().run({"c": c}, o, z[:, None], n_perm, rng, np.arange(FULL_N), _lag(z, c),
                                   grid_step=1)
    return _p(res), res.block_length


def trial_nonlinear(scenario, rng, n_perm):
    z, c, o, t, step, lag = _subsample(scenario, rng)
    res = NonlinearResidualTest().run({"c": c}, o, z[:, None], n_perm, rng, t, lag, grid_step=step)
    return _p(res), res.block_length


def trial_cmi_unconditional(scenario, rng, n_perm):
    _, c, o, t, step, lag = _subsample(scenario, rng)
    res = CMITest(workers=1).run({"c": c}, o, None, n_perm, rng, t, lag, grid_step=step)
    return _p(res), res.block_length


def trial_check_a(scenario, rng, n_perm):
    z, c, o = _null_triplet(scenario, FULL_N, rng)
    out = linear_specification_test(np.column_stack([c, o]), z[:, None], ALPHA, 0.0, n_perm, rng,
                                    np.arange(FULL_N), _lag(z, c), grid_step=1)
    # Each column is tested at alpha / 2; report the equivalent single
    # p-value (twice the smallest) so the rate is at alpha.
    return min(1.0, 2 * out.p_value), out.block_length


def trial_check_b(scenario, rng, n_perm):
    z, c, o = _null_triplet(scenario, FULL_N, rng, o_on_c=True)
    res = LinearityTest(alpha=ALPHA, effect_size_floor=0.0).run(
        c, o, z[:, None], time_positions=np.arange(FULL_N), min_temporal_lag=_lag(z, c),
        n_permutations=n_perm, rng=rng, grid_step=1)
    return res.permutation_p_value, res.block_length


def trial_check_c(scenario, rng, n_perm):
    z, c, o, t, step, lag = _subsample(scenario, rng, o_on_c=True)
    res = StateIndependenceTest(alpha=ALPHA, effect_size_floor=0.0, max_n=MAX_N).run(
        c, o, z[:, None], time_positions=t, min_temporal_lag=lag, n_permutations=n_perm, rng=rng,
        grid_step=step)
    return res.permutation_p_value, res.block_length


TRIALS = {
    "linear tier": trial_linear,
    "nonlinear tier": trial_nonlinear,
    "CMI tier (Z empty)": trial_cmi_unconditional,
    "nonlinear dependence": trial_check_a,
    "nonlinear relation": trial_check_b,
    "effect modification": trial_check_c,
}


def run_cell(args):
    test, scenario, reps, n_perm, seed = args
    rng = np.random.default_rng(seed)
    p_values, blocks = [], []
    for _ in range(reps):
        p, b = TRIALS[test](scenario, rng, n_perm)
        p_values.append(p)
        blocks.append(b if b is not None else 0)
    return {"test": test, "scenario": scenario, "reps": reps,
            "false_positive_rate": float(np.mean(np.array(p_values) < ALPHA)),
            "median_block_length": float(np.median(blocks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--tests", default=",".join(TRIALS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cells = [(t, s, args.reps, args.n_perm, 1000 * i + j)
             for i, t in enumerate(args.tests.split(",")) for j, s in enumerate(SCENARIOS)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run_cell, cells))
    se = np.sqrt(ALPHA * (1 - ALPHA) / args.reps)
    print(f"False-positive rate at alpha = {ALPHA} ({args.reps} datasets per cell; "
          f"standard error at the nominal rate {se:.3f}); {time.time() - t0:.0f} s")
    print("| Test | Scenario | Block length (median, time steps) | Rate |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['test']} | {r['scenario']} | {r['median_block_length']:.0f} | {r['false_positive_rate']:.3f} |")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
