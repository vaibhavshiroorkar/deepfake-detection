"""The optional second opinion: the outside service's AI-generated image model on one frame, only when asked."""

from datetime import date

import httpx
import pytest

from deepfake_detection.webapp import outside


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("STILL_CHECK_USER", "user")
    monkeypatch.setenv("STILL_CHECK_SECRET", "secret")


def _image(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"jpeg")
    return path


def _reply(payload, status=200):
    return lambda *args, **kwargs: httpx.Response(status, json=payload)


def test_it_is_off_without_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("STILL_CHECK_USER", raising=False)
    monkeypatch.delenv("STILL_CHECK_SECRET", raising=False)
    assert not outside.enabled()
    quota = outside.Quota(tmp_path / "usage.json")
    called = []
    out = outside.ask(
        _image(tmp_path), post=lambda *a, **k: called.append(1), quota=quota
    )
    assert "error" in out and not called


def test_a_score_comes_back_and_its_operations_are_counted(keys, tmp_path):
    quota = outside.Quota(tmp_path / "usage.json", today=lambda: date(2026, 11, 19))
    reply = {
        "status": "success",
        "request": {"operations": 5},
        "type": {"ai_generated": 0.73, "deepfake": 0.01},
    }
    out = outside.ask(_image(tmp_path), post=_reply(reply), quota=quota)
    assert out == {"ai_generated": 0.73, "deepfake": 0.01}
    assert quota.used() == {"day": 5, "month": 5}


def test_the_free_allowance_is_never_exceeded(keys, tmp_path):
    quota = outside.Quota(tmp_path / "usage.json", today=lambda: date(2026, 11, 19))
    quota.record(outside.DAY_LIMIT - 2)
    called = []
    out = outside.ask(
        _image(tmp_path), post=lambda *a, **k: called.append(1), quota=quota
    )
    assert "allowance" in out["error"] and not called
    tomorrow = outside.Quota(tmp_path / "usage.json", today=lambda: date(2026, 11, 20))
    assert tomorrow.room(outside.COST)  # a new day, same month


def test_a_failed_call_says_so_and_costs_nothing(keys, tmp_path):
    quota = outside.Quota(tmp_path / "usage.json", today=lambda: date(2026, 11, 19))
    out = outside.ask(
        _image(tmp_path),
        post=_reply({"status": "failure", "error": {"message": "bad key"}}),
        quota=quota,
    )
    assert "error" in out
    assert quota.used() == {"day": 0, "month": 0}

    def down(*args, **kwargs):
        raise httpx.ConnectError("no route")

    assert "error" in outside.ask(_image(tmp_path), post=down, quota=quota)
