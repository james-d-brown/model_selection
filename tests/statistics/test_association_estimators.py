"""
Tests of the association building blocks: residualization, canonical
correlations and partial canonical correlation (exact reduction to r^2),
and the distance-covariance helpers, cross-checked against the `dcor`
package where it is installed.
"""
import numpy as np
import pytest

try:
    import dcor as _dcor_reference
    HAVE_DCOR = True
except ImportError:
    HAVE_DCOR = False

requires_dcor = pytest.mark.skipif(not HAVE_DCOR, reason="dcor package not installed")

from model_selection.statistics.association.canonical import canonical_correlations, residualize_on_conditioning
from model_selection.statistics.association.distance import pairwise_euclidean_distances, u_center, u_product
from model_selection.statistics.conditional_independence.linear import partial_canonical_correlation


def PartialCCA(obs, conditioning):
    """Partial canonical correlation of a candidate with obs given conditioning."""
    obs_resid = residualize_on_conditioning(obs, conditioning)
    return lambda candidate: partial_canonical_correlation(
        residualize_on_conditioning(candidate, conditioning), obs_resid)




def test_residualize_on_conditioning_matches_manual_ols_scalar():
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    y = 3.0 * z + 1.0 + rng.normal(scale=0.05, size=n)
    resid = residualize_on_conditioning(y, z)
    # Manual OLS residual for the same regression, single column.
    design = np.column_stack([np.ones(n), z])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    expected = y - design @ coef
    np.testing.assert_allclose(resid.ravel(), expected, atol=1e-8)


def test_residualize_on_conditioning_none_is_mean_deviation():
    rng = np.random.default_rng(1)
    y = rng.normal(loc=5.0, size=300)
    resid = residualize_on_conditioning(y, None)
    np.testing.assert_allclose(resid.ravel(), y - y.mean(), atol=1e-10)


def test_canonical_correlations_scalar_case_matches_pearson_r():
    """canonical_correlations' documented contract requires ALREADY
    centered input (see its docstring) -- it deliberately does not
    re-center internally, since PartialCCA always feeds it output from
    residualize_on_conditioning, which is centered by construction. Center
    explicitly here to test the function's actual contract, rather than
    relying on raw random draws happening to have a sample mean close
    enough to zero (finite-n noise alone gives a small but genuinely
    nonzero mean, which would otherwise show up as a spurious mismatch
    against np.corrcoef, which DOES center internally)."""
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(scale=0.6, size=n)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    rhos = canonical_correlations(x_centered, y_centered)
    assert rhos.shape == (1,)
    expected_r = abs(np.corrcoef(x, y)[0, 1])
    assert abs(rhos[0] - expected_r) < 1e-9


def test_partial_cca_reduces_exactly_to_r_squared_unconditional():
    """THE core algebraic claim this whole design rests on: at p=q=1 with
    no conditioning, PartialCCA's output must equal r^2 EXACTLY (to
    floating-point precision), not approximately."""
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=n)
    y = 1.8 * x + rng.normal(scale=0.4, size=n)

    est = PartialCCA(obs=y, conditioning=None)
    value = est(x)

    expected_r2 = np.corrcoef(x, y)[0, 1] ** 2
    assert abs(value - expected_r2) < 1e-12


def test_partial_cca_reduces_exactly_to_r_squared_conditional():
    """SAME claim, now WITH a real conditioning set -- confirms the
    reduction holds after residualization too, not just unconditionally."""
    rng = np.random.default_rng(4)
    n = 1000
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(scale=0.8, size=n)
    y = 1.2 * x + 0.9 * z + rng.normal(scale=0.5, size=n)

    est = PartialCCA(obs=y, conditioning=z)
    value = est(x)

    # Manually compute partial r^2 via OLS residualization + Pearson r.
    x_resid = residualize_on_conditioning(x, z).ravel()
    y_resid = residualize_on_conditioning(y, z).ravel()
    expected_r2 = np.corrcoef(x_resid, y_resid)[0, 1] ** 2

    assert abs(value - expected_r2) < 1e-10


def test_partial_cca_is_sign_invariant():
    """Consistent with the unsigned-magnitude convention shared with CMI/
    PartialDistanceCorrelation: negating candidate must not change the
    returned value (r^2 is sign-invariant; a signed r would not be)."""
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    y = -1.5 * x + rng.normal(scale=0.3, size=n)

    est = PartialCCA(obs=y, conditioning=None)
    value_pos = est(x)
    value_neg = est(-x)
    assert abs(value_pos - value_neg) < 1e-8




def test_partial_cca_bounded_in_unit_interval():
    rng = np.random.default_rng(7)
    n = 600
    z = rng.normal(size=(n, 3))
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # genuinely unrelated
    est = PartialCCA(obs=y, conditioning=z)
    value = est(x)
    assert 0.0 <= value < 1.0 + 1e-9


def test_partial_cca_multivariate_detects_genuine_linear_association():
    """Multivariate candidate/obs (mimicking FPC score vectors) with a
    real, strong linear relationship should give a value comfortably
    above what pure noise produces at the same dimensionality."""
    rng = np.random.default_rng(8)
    n = 500
    z = rng.normal(size=(n, 2))
    candidate = rng.normal(size=(n, 3))
    # obs is a genuine linear combination of candidate's columns, plus
    # some z dependence and noise.
    B = rng.normal(size=(3, 2))
    obs = candidate @ B + 0.3 * (z @ rng.normal(size=(2, 2))) + rng.normal(scale=0.3, size=(n, 2))

    est_real = PartialCCA(obs=obs, conditioning=z)
    value_real = est_real(candidate)

    noise_candidate = rng.normal(size=(n, 3))  # unrelated to obs
    value_noise = est_real(noise_candidate)

    assert value_real > value_noise
    assert value_real > 0.3  # a real, substantial linear relationship was constructed


def test_partial_cca_weak_on_purely_nonlinear_relationship():
    """PartialCCA measures LINEAR association only -- a genuinely
    nonlinear-but-mean-zero-correlation relationship (y = x^2 - 1 with
    E[x]=0) should show WEAK linear association, consistent with why
    LinearityTest exists as a gatekeeper before trusting this estimator."""
    rng = np.random.default_rng(9)
    n = 2000
    x = rng.normal(size=n)
    y = x ** 2 - 1.0 + rng.normal(scale=0.2, size=n)  # symmetric parabola: near-zero linear corr by construction

    est = PartialCCA(obs=y, conditioning=None)
    value = est(x)
    assert value < 0.05  # should NOT detect this strong-but-nonlinear relationship


def test_partial_cca_handles_small_n_without_error():
    """Ridge-stabilized whitening should keep this well-defined even at
    very small n / near-collinear conditioning (an early greedy round with
    few events, or few already-selected models)."""
    rng = np.random.default_rng(10)
    n = 15
    z = rng.normal(size=(n, 2))
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    est = PartialCCA(obs=y, conditioning=z)
    value = est(x)
    assert np.isfinite(value)
    assert 0.0 <= value < 1.0 + 1e-9


# ---------------------------------------------------------------------------
# PartialDistanceCorrelation tests
# ---------------------------------------------------------------------------

def test_pairwise_euclidean_distances_matches_manual_case():
    X = np.array([[0.0], [3.0], [4.0]])
    D = pairwise_euclidean_distances(X)
    expected = np.array([[0, 3, 4], [3, 0, 1], [4, 1, 0]], dtype=np.float64)
    np.testing.assert_allclose(D, expected)


def test_u_center_has_zero_diagonal():
    rng = np.random.default_rng(20)
    a = pairwise_euclidean_distances(rng.normal(size=(20, 2)))
    A = u_center(a)
    np.testing.assert_allclose(np.diag(A), 0.0, atol=1e-10)


@requires_dcor
def test_u_center_and_u_product_match_dcor_unconditional_reference():
    """Direct, machine-precision cross-validation against dcor's own
    u_distance_correlation_sqr -- the strongest available check for this
    literature-specific construction."""
    rng = np.random.default_rng(1)
    n = 60
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(scale=0.7, size=n)

    A = u_center(pairwise_euclidean_distances(x))
    B = u_center(pairwise_euclidean_distances(y))
    mine = u_product(A, B) / np.sqrt(u_product(A, A) * u_product(B, B))
    reference = _dcor_reference.u_distance_correlation_sqr(x, y)
    assert abs(mine - reference) < 1e-10




























# ---------------------------------------------------------------------------
# AbsoluteValueEstimator tests
# ---------------------------------------------------------------------------









if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
