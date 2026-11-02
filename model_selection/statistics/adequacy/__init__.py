"""Tests of whether a simpler association measure's functional-form
assumption holds (the tiers of the selection cascade)."""
from model_selection.statistics.adequacy.base import (
    AdequacyPlotFunc,
    AdequacyTestFunc,
    AdequacyTestOutcome,
    FunctionalFormTest,
    FunctionalFormTestResult,
    as_adequacy_test,
)
from model_selection.statistics.adequacy.linearity import (
    LinearityTest,
    MultivariateLinearityTest,
    make_linearity_adequacy_test,
    make_multivariate_linearity_adequacy_test,
)
from model_selection.statistics.adequacy.state_independence import (
    StateIndependenceTest,
    make_state_independence_adequacy_test,
)

__all__ = [
    "AdequacyPlotFunc", "AdequacyTestFunc", "AdequacyTestOutcome",
    "FunctionalFormTest", "FunctionalFormTestResult", "as_adequacy_test",
    "LinearityTest", "MultivariateLinearityTest",
    "make_linearity_adequacy_test", "make_multivariate_linearity_adequacy_test",
    "StateIndependenceTest", "make_state_independence_adequacy_test",
]
