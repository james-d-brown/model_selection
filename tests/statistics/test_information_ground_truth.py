"""
Ground-truth validation of the nearest-neighbour MI/CMI estimators (Mesner
and Shalizi, 2021; Gao et al., 2017) against closed-form values for
multivariate Gaussians and for discrete and mixed data:

  Bivariate Gaussian, correlation rho:
      I(X;Y) = -0.5 * log(1 - rho^2)

  Trivariate Gaussian with a known partial-correlation structure, we validate:
    (a) I(X;Y|Z) ~ 0 when X,Y are independent given Z (constructed so Y is a
        function of Z plus independent noise, unrelated to X given Z)
    (b) I(X;Y|Z) > 0 and roughly tracks the analytic conditional-Gaussian MI
        when X carries real information about Y beyond Z.

These are sanity/ground-truth checks (not exhaustive statistical guarantees);
the estimators are asymptotically consistent but biased at finite n, so we
assert *closeness* within a tolerance appropriate to n and k, not exact
equality.
"""
import numpy as np
import pytest

from model_selection.statistics.association.information import knn_cmi, knn_mi


def analytic_gaussian_mi(rho: float) -> float:
    return -0.5 * np.log(1 - rho ** 2)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_knn_mi_bivariate_gaussian(rho):
    rng = np.random.default_rng(0)
    n = 4000
    cov = np.array([[1.0, rho], [rho, 1.0]])
    xy = rng.multivariate_normal([0, 0], cov, size=n)
    x, y = xy[:, 0], xy[:, 1]

    est = knn_mi(x, y, k=5)
    truth = analytic_gaussian_mi(rho)
    # The estimator has known small-sample bias; allow generous but meaningful tolerance
    tol = 0.05 if rho == 0.0 else 0.08 * max(truth, 1.0)
    assert abs(est - truth) < max(tol, 0.05), f"rho={rho}: est={est:.4f} truth={truth:.4f}"


def test_knn_mi_nonnegative_and_zero_for_independent():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    est = knn_mi(x, y, k=5)
    assert est > -0.05  # can be slightly negative due to finite-sample noise
    assert abs(est) < 0.05


def test_knn_cmi_zero_when_conditionally_independent():
    """Construct X, Z independent; Y = f(Z) + noise (independent of X).
    Then I(X;Y|Z) should be ~0, even though X and Y may show nonzero
    *unconditional* MI through no real channel (here they don't even share
    that, since X is fully independent of both Z and the noise in Y)."""
    rng = np.random.default_rng(2)
    n = 4000
    z = rng.normal(size=n)
    x = rng.normal(size=n)  # independent of everything
    y = 0.8 * z + rng.normal(scale=0.5, size=n)  # depends only on z

    cmi = knn_cmi(x, y, z, k=5)
    assert abs(cmi) < 0.06, f"expected ~0, got {cmi:.4f}"


def test_knn_cmi_positive_when_candidate_adds_information():
    """X adds real information about Y beyond Z: Y = Z + X + noise."""
    rng = np.random.default_rng(3)
    n = 4000
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    y = z + x + rng.normal(scale=0.3, size=n)

    cmi = knn_cmi(x, y, z, k=5)
    assert cmi > 0.3, f"expected clearly positive CMI, got {cmi:.4f}"


def test_knn_cmi_reduces_to_mi_with_empty_conditioning():
    rng = np.random.default_rng(4)
    n = 2000
    rho = 0.5
    cov = np.array([[1.0, rho], [rho, 1.0]])
    xy = rng.multivariate_normal([0, 0], cov, size=n)
    x, y = xy[:, 0], xy[:, 1]

    mi = knn_mi(x, y, k=5)
    cmi_empty = knn_cmi(x, y, None, k=5)
    assert abs(mi - cmi_empty) < 1e-9


def test_knn_mi_discrete_data():
    """X uniform on four values and Y = X: I(X;Y) = log 4. Every sample's
    neighbours coincide with it, so the estimator uses its log branch; the
    KSG estimator is undefined here without noise to break the ties."""
    rng = np.random.default_rng(5)
    x = rng.integers(0, 4, size=2000).astype(float)
    assert knn_mi(x, x.copy(), k=5) == pytest.approx(np.log(4), abs=0.02)


def test_knn_mi_mixed_data():
    """X uniform on {0, ..., m-1}, Y | X uniform on [X, X + 2]: I(X;Y) =
    log m - (m - 1) log 2 / m, the mixed example of Gao et al. (2017)."""
    rng = np.random.default_rng(6)
    m, n = 5, 4000
    x = rng.integers(0, m, size=n).astype(float)
    y = x + 2 * rng.random(n)
    truth = np.log(m) - (m - 1) * np.log(2) / m
    assert knn_mi(x, y, k=10) == pytest.approx(truth, abs=0.05)


def test_knn_cmi_zero_for_censored_conditionally_independent_data():
    """X and Y both depend on Z and are censored at their 0.9 quantiles, so a
    point mass holds 90% of each; given the uncensored Z they stay
    independent and the estimate is near zero."""
    rng = np.random.default_rng(7)
    n = 3000
    z = rng.normal(size=n)
    x, y = z + rng.normal(size=n), z + rng.normal(size=n)
    x, y = np.maximum(x, np.quantile(x, 0.9)), np.maximum(y, np.quantile(y, 0.9))
    assert abs(knn_cmi(x, y, z, k=30)) < 0.02
