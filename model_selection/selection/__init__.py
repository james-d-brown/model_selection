"""The hierarchical model-selection framework: greedy forward selection by
conditional independence testing, optionally inside the linear -> nonlinear
-> CMI cascade."""
from model_selection.selection.cascade import TierAttempt, TieredSelectionResult, run_tiered_selection
from model_selection.selection.config import CASCADE, SUPPORTED_MEASURES, TIERS, AdequacyConfig, SelectionConfig
from model_selection.selection.greedy import SelectionResult, SelectionStep, gate, greedy_forward_selection
from model_selection.selection.stage import SelectionOutcome, run_selection_stage
from model_selection.selection.tiers import (
    AdequacyCheck, SelectionSample, Tier, build_cascade, build_tier, effect_size_floor_for_tier,
)

__all__ = [
    "TierAttempt", "TieredSelectionResult", "run_tiered_selection",
    "CASCADE", "SUPPORTED_MEASURES", "TIERS", "AdequacyConfig", "SelectionConfig",
    "SelectionResult", "SelectionStep", "gate", "greedy_forward_selection",
    "SelectionOutcome", "run_selection_stage",
    "AdequacyCheck", "SelectionSample", "Tier", "build_cascade", "build_tier", "effect_size_floor_for_tier",
]
