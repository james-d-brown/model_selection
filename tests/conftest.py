"""Test-suite settings."""
import pytest


@pytest.fixture(autouse=True)
def _small_nonlinear_samples(monkeypatch):
    """The synthetic records are long (years of hourly steps); run the
    nonlinear tier on 2,000 of their points rather than the default cap, so
    that the suite stays fast. The default cap is checked on real records."""
    from model_selection.selection import config
    monkeypatch.setattr(config, "DEFAULT_NONLINEAR_MAX_N", 2000)
