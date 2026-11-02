"""Regression of variables on the conditioning set, used to remove the
influence of the selected formulations before testing residual dependence."""
from model_selection.statistics.regression.kernel_residuals import (
    ConditioningRegression, feature_kernel, kernel_factor,
)
from model_selection.statistics.regression.linear import ols_fit

__all__ = [
    "ConditioningRegression", "feature_kernel", "kernel_factor", "ols_fit",
]
