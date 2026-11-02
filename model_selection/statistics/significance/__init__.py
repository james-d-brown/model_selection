"""Resampling, p-values and bias correction shared by the significance and
adequacy tests."""
from model_selection.statistics.significance.base import (
    bias_correct_association, minimum_resamples, permutation_p_value,
)
from model_selection.statistics.significance.neighborhood import compute_neighbor_index, local_permutation
from model_selection.statistics.significance.resampling import grid_subsample, time_block_permutation

__all__ = [
    "bias_correct_association", "minimum_resamples", "permutation_p_value", "compute_neighbor_index", "local_permutation",
    "grid_subsample", "time_block_permutation",
]
