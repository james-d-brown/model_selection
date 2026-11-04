"""
Power of the linear tier's test: Rao's F test of Cohen's set correlation.

The linear tier's effect size, 1 - Wilks' lambda of the residuals of the
candidate and the observations on the selected set, is Cohen's partial set
correlation R^2 (Cohen, 1982). The power of Rao's (1973) F test of it
follows Cohen (1988, chapter 10):

  effect size (10.2.2)   f^2 = L^(-1/s) - 1, with L the population Wilks' lambda
  noncentrality (10.3.1) lambda = f^2 (u + v + 1)

with k_Y and k_X the dimensions of the observations and the candidate, k_C
the dimension of the partialled set and N the sample size:

  u = k_Y k_X
  s = sqrt((k_Y^2 k_X^2 - 4) / (k_Y^2 + k_X^2 - 5)), or 1 if k_Y^2 + k_X^2 <= 5
  m = N - k_C - (k_Y + k_X + 3) / 2
  v = m s + 1 - u / 2

The noncentral F distribution is evaluated exactly (scipy), as recommended
over approximations by Bradley et al. (1998). Cohen and Nee (1987) validated
power calculations for this test by Monte Carlo.

N is the effective sample size. The linear tier's test statistic is Pillai's
trace PT, the sum of the squared canonical correlations (Pillai, 1955). For N
independent samples whose observations have normal errors of constant
variance, its null mean is u / (N - 1 - k_C): the observations' orthonormal
residuals span a k_Y-dimensional subspace of the N - 1 - k_C residual
dimensions, every orientation equally likely, and a unit vector's squared
projection onto such a subspace has mean k_Y / (N - 1 - k_C) (Dasgupta and
Gupta, 2003). Autocorrelated samples carry less information than their
number, and the test's resampled null, which reproduces that dependence, is
wider than the independent one. Matching its mean gives the number of
independent samples with the same null, which is used as N (the first-moment
matching of Satterthwaite, 1946). For a single feature the test is the same
as Rao's F test, whose power is computed here; with several, Rao's test's
power is used for Pillai's trace too (docs/methods.md, Detectable contribution and
effective sample size).

See docs/methods.md, Samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq


@dataclass(frozen=True)
class RaoDegrees:
    u: float
    v: float
    s: float


def rao_degrees(n: float, k_y: int, k_x: int, k_c: int = 0) -> RaoDegrees:
    """Degrees of freedom of Rao's F for a partial set correlation."""
    u = float(k_y * k_x)
    denom = k_y ** 2 + k_x ** 2 - 5
    s = float(np.sqrt((k_y ** 2 * k_x ** 2 - 4) / denom)) if denom > 0 else 1.0
    m = n - k_c - (k_y + k_x + 3) / 2.0
    v = m * s + 1.0 - u / 2.0
    return RaoDegrees(u, v, s)


def set_correlation_power(n: float, k_y: int, k_x: int, k_c: int, f2: float, alpha: float) -> float:
    """Power of Rao's F test to detect a population effect f^2 at level alpha.
    Returns 0 when the test is undefined (v <= 0)."""
    d = rao_degrees(n, k_y, k_x, k_c)
    if d.v <= 0:
        return 0.0
    critical = stats.f.ppf(1.0 - alpha, d.u, d.v)
    return float(stats.ncf.sf(critical, d.u, d.v, f2 * (d.u + d.v + 1.0)))


def effective_sample_size(n: float, null_trace: np.ndarray, k_y: int, k_x: int, k_c: float = 0) -> float:
    """Number of independent samples whose null mean of Pillai's trace,
    k_y k_x / (N - 1 - k_c), equals the mean of the resampled null draws
    null_trace (module docstring), at most n."""
    draws = np.asarray(null_trace, dtype=float)
    mean = float(np.mean(draws)) if draws.size else 0.0
    if not np.isfinite(mean) or mean <= 0:
        return float(n)
    return float(min(n, 1.0 + k_c + k_y * k_x / mean))


def f2_to_r2(f2: float, k_y: int, k_x: int) -> float:
    """Set correlation R^2 = 1 - L implied by f^2 (inverse of equation 10.2.2)."""
    s = rao_degrees(10_000, k_y, k_x).s
    return float(1.0 - (1.0 + f2) ** (-s))


def minimum_detectable_r2(n: float, k_y: int, k_x: int, k_c: int, alpha: float, power: float = 0.80) -> float:
    """Smallest set-correlation R^2 detectable with the given power; 1.0 when
    no effect is detectable (undefined test or insufficient samples)."""
    if rao_degrees(n, k_y, k_x, k_c).v <= 0:
        return 1.0
    hi = 1.0
    while set_correlation_power(n, k_y, k_x, k_c, hi, alpha) < power:
        hi *= 2.0
        if hi > 1e6:
            return 1.0
    f2 = brentq(lambda f: set_correlation_power(n, k_y, k_x, k_c, f, alpha) - power, 1e-9, hi, xtol=1e-6)
    return f2_to_r2(f2, k_y, k_x)


def format_share(x: float) -> str:
    """A number to two significant figures in positional notation, so that
    small values are not rounded to zero (0.0001 -> "0.0001", 24.3 -> "24")."""
    if x == 0 or not np.isfinite(x):
        return f"{x:g}"
    return np.format_float_positional(float(x), precision=2, unique=False, fractional=False, trim="-")


def describe_detectable(r2: float, n_features: int = 1) -> dict:
    """The smallest detectable contribution in plain terms: the share of the
    remaining unexplained variation a formulation must account for to be
    selected with 80% power and, when the observations have a single
    feature, the corresponding reduction in root-mean-square error
    (rmse_reduction_pct is None otherwise)."""
    single = n_features == 1
    if r2 >= 1.0:
        return {"detectable_r2": 1.0, "unexplained_variation_pct": 100.0,
                "rmse_reduction_pct": 100.0 if single else None,
                "statement": "These data cannot support the test: no contribution, however large, can be detected."}
    pct = 100.0 * r2
    rmse = 100.0 * (1.0 - np.sqrt(1.0 - r2))
    rmse_clause = f" (a root-mean-square error {format_share(rmse)}% smaller)" if single else ""
    return {
        "detectable_r2": float(r2),
        "unexplained_variation_pct": float(pct),
        "rmse_reduction_pct": float(rmse) if single else None,
        "statement": (
            f"An additional formulation is detected with 80% power only if it accounts for at least {format_share(pct)}% of "
            "the variation in the observations, as analysed, that the formulations already selected leave "
            f"unexplained{rmse_clause}. Smaller contributions may exist but cannot be reliably detected with these data."
        ),
    }

