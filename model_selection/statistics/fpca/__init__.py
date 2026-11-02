"""Functional principal component analysis."""
from model_selection.statistics.fpca.fpca import (
    FPCAResult,
    fit_functional_pca,
    fit_regularized_functional_pca,
    n_components_for_variance,
    phase_normalize,
    project,
    reconstruct,
    reconstruction_error,
    roughness_penalty_matrix,
)

__all__ = [
    "FPCAResult", "fit_functional_pca", "fit_regularized_functional_pca", "n_components_for_variance",
    "phase_normalize", "project", "reconstruct", "reconstruction_error", "roughness_penalty_matrix",
]
