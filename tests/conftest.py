"""Tests run the website on the study's models unless a test opts in.

results/web-models.json swaps the website's live models, and their weights are not in Git, so a
test that loads the scorer would otherwise need them. A test of the swap sets its own value.
"""

import pytest


@pytest.fixture(autouse=True)
def _study_models(monkeypatch):
    monkeypatch.setenv("DDF_WEB_MODELS", "p001")
