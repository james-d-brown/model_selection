"""Scores of curves with missing points: conditional expectation given the
observed points (PACE) under the fitted FPCA basis."""
import numpy as np

from model_selection.statistics.fpca.fpca import (
    conditional_scores, fit_functional_pca, phase_normalize, project, reconstruction_error,
)


def _low_rank_curves(n=200, n_grid=120, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_grid)
    modes = np.vstack([np.sin(np.pi * t), np.sin(2 * np.pi * t), np.cos(3 * np.pi * t)])
    scores = rng.normal(size=(n, 3)) * np.array([3.0, 1.5, 0.7])
    return 2.0 + scores @ modes + noise * rng.normal(size=(n, n_grid))


def test_noise_variance_is_the_unexplained_variance_per_remaining_dimension():
    X = _low_rank_curves(noise=0.05)
    basis = fit_functional_pca(X, n_components=3)
    assert abs(basis.noise_variance - 0.05 ** 2) < 0.3 * 0.05 ** 2


def test_complete_curves_score_as_their_projection_up_to_shrinkage():
    X = _low_rank_curves()
    basis = fit_functional_pca(X, n_components=3)
    proj, pace = project(X, basis), conditional_scores(X, basis)
    shrink = basis.explained_variance / (basis.explained_variance + basis.noise_variance)
    assert np.allclose(pace, proj * shrink, atol=1e-8)


def test_gapped_curves_recover_the_complete_scores():
    X = _low_rank_curves(n=300, seed=1)
    basis = fit_functional_pca(X[:200], n_components=3)
    test = X[200:]
    gapped = test.copy()
    rng = np.random.default_rng(2)
    for row in gapped:
        s = rng.integers(0, 100)
        row[s:s + 20] = np.nan
    truth, est = project(test, basis), conditional_scores(gapped, basis)
    for k in range(3):
        assert np.corrcoef(truth[:, k], est[:, k])[0, 1] > 0.98


def test_a_curve_without_observed_points_has_undefined_scores():
    X = _low_rank_curves()
    basis = fit_functional_pca(X, n_components=3)
    empty = np.full((1, X.shape[1]), np.nan)
    assert np.isnan(conditional_scores(empty, basis)).all()


def test_phase_normalization_never_fills_gaps():
    curve = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    out = phase_normalize(curve, n_grid=9)  # grid points at every half step
    assert np.isnan(out[3:6]).all() and np.isfinite(out[:3]).all() and np.isfinite(out[6:]).all()


def test_reconstruction_error_uses_observed_points_only():
    x = np.array([[1.0, 2.0, np.nan, 4.0]])
    err = reconstruction_error(x, np.array([[1.0, 2.0, 100.0, 4.0]]))
    assert err["rmse"].iloc[0] == 0.0 and err["r2"].iloc[0] == 1.0
