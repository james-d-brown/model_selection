"""Pipeline tests check behaviour, not production-scale Monte Carlo accuracy:
adequacy tests here use 99 resamples unless a test sets its own."""
import pytest

from model_selection.selection.config import AdequacyConfig


@pytest.fixture(autouse=True)
def fast_adequacy(monkeypatch):
    original = AdequacyConfig.resolved_n_permutations

    def _resolved(self):
        return self.n_permutations if self.n_permutations is not None else 99

    monkeypatch.setattr(AdequacyConfig, "resolved_n_permutations", property(_resolved))
    yield
    monkeypatch.setattr(AdequacyConfig, "resolved_n_permutations", original)
