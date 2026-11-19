"""Tests run the website on the study's models unless a test opts in.

results/web-models.json swaps the website's live models, and their weights are not in Git, so a
test that loads the scorer would otherwise need them. A test of the swap sets its own value.
"""

import pytest


@pytest.fixture(autouse=True)
def _study_models(monkeypatch):
    monkeypatch.setenv("DDF_WEB_MODELS", "p001")


@pytest.fixture(autouse=True)
def _no_outside_service(monkeypatch):
    """Tests never call the outside face check; a test that wants it sets the keys itself."""
    monkeypatch.delenv("STILL_CHECK_USER", raising=False)
    monkeypatch.delenv("STILL_CHECK_SECRET", raising=False)


@pytest.fixture(autouse=True)
def _processor_device(monkeypatch):
    """Every test starts on the processor, and the setting is restored after it. Two tests let scripts set
    DDF_SCORER_DEVICE=cuda; it leaked, so later website jobs loaded the description model on the GPU and
    outran their 10-second wait (found 11-20)."""
    monkeypatch.setenv("DDF_SCORER_DEVICE", "cpu")
