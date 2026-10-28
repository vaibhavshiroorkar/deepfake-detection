"""Fixtures of the Evaluation Lab tests."""

import pytest


@pytest.fixture
def ledger(tmp_path):
    return tmp_path / "ledger" / "final_ledger.jsonl"


@pytest.fixture
def scores_dir(tmp_path):
    d = tmp_path / "scores"
    d.mkdir()
    return d
