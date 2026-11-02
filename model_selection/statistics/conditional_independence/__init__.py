"""
Tests of conditional independence between candidate formulations and the
observations given the selected formulations: the three tiers of the
selection cascade. Each test scores every remaining candidate and builds
its null distribution with one shared resampling, so the round's winner can
be calibrated against the maximum statistic across candidates.
"""
from model_selection.statistics.conditional_independence.base import (
    ConditionalIndependenceTest, JointTestResult,
)
from model_selection.statistics.conditional_independence.cmi import CMITest
from model_selection.statistics.conditional_independence.linear import LinearResidualTest
from model_selection.statistics.conditional_independence.nonlinear import NonlinearResidualTest

__all__ = [
    "ConditionalIndependenceTest", "JointTestResult", "CMITest", "LinearResidualTest", "NonlinearResidualTest",
]
