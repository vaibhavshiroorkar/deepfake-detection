"""The web app's routes, with the model registry replaced by a stub.

Nothing here loads a checkpoint. The rules under test are the ones the server
owns: admin settings are enforced on every request, uploads are size-checked
and deleted, and the admin routes are off without a token.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from deepfake_detection.webapp import settings as settings_module
from deepfake_detection.webapp.engines import Analysis
from deepfake_detection.webapp.server import create_app


class StubRegistry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def analyze(
        self, path: Path, *, model, threshold, withhold_unfamiliar, default_threshold
    ):
        assert path.is_file()
        self.calls.append(
            {
                "path": path,
                "model": model,
                "threshold": threshold,
                "default": default_threshold,
                "withhold": withhold_unfamiliar,
            }
        )
        return Analysis(
            model=model,
            verdict="fake",
            probability=0.8,
            threshold=default_threshold if threshold is None else threshold,
            members={"a": 0.8},
            blockers=(),
            familiarity=None,
            withheld=False,
            faces=(),
            media_kind="video",
            seconds=0.1,
        )

    def status(self):
        return {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    path = tmp_path / "webapp.yaml"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", path)
    original_load, original_save = settings_module.load, settings_module.save
    monkeypatch.setattr(settings_module, "load", lambda p=path: original_load(p))
    monkeypatch.setattr(settings_module, "save", lambda s, p=path: original_save(s, p))
    monkeypatch.delenv("DDF_ADMIN_TOKEN", raising=False)
    registry = StubRegistry()
    app = create_app(registry)
    return TestClient(app), registry, monkeypatch


def upload(client, name="clip.mp4", data=b"not really a video", **fields):
    return client.post(
        "/api/analyze", files={"file": (name, data, "video/mp4")}, data=fields
    )


def test_the_page_and_its_assets_are_served(client):
    http, _, _ = client
    assert "Is this video a deepfake?" in http.get("/").text
    assert http.get("/static/app.js").status_code == 200
    assert http.get("/static/vendor/katex/katex.min.js").status_code == 200


def test_an_upload_is_scored_and_then_deleted(client):
    http, registry, _ = client
    response = upload(http, model="ensemble", threshold="0.2")
    assert response.status_code == 200, response.text
    assert response.json()["probability"] == 0.8
    call = registry.calls[0]
    assert call["threshold"] == 0.2
    assert not call["path"].exists()


def test_unsupported_and_empty_files_are_refused(client):
    http, registry, _ = client
    assert upload(http, name="notes.txt").status_code == 415
    assert upload(http, data=b"").status_code == 400
    assert registry.calls == []


def test_a_locked_threshold_overrides_the_user(client):
    http, registry, _ = client
    settings_module.save(
        settings_module.Settings(lock_threshold=True, default_threshold=0.3)
    )
    upload(http, threshold="0.9")
    assert registry.calls[0]["threshold"] is None
    assert registry.calls[0]["default"] == 0.3


def test_no_threshold_leaves_the_choice_to_the_registry(client):
    # The registry then gives an image or a sound file its own fitted cut-off.
    http, registry, _ = client
    upload(http)
    assert registry.calls[0]["threshold"] is None
    assert registry.calls[0]["default"] == 0.5


def test_a_disabled_model_cannot_be_requested(client):
    http, registry, _ = client
    settings_module.save(
        settings_module.Settings(enabled_models=("ensemble",), default_model="ensemble")
    )
    assert upload(http, model="multimodal").status_code == 400
    assert registry.calls == []


def test_the_upload_limit_is_enforced(client):
    http, registry, _ = client
    settings_module.save(settings_module.Settings(max_upload_mb=1))
    assert upload(http, data=b"x" * (2 << 20)).status_code == 413
    assert registry.calls == []


def test_admin_routes_are_off_without_a_token(client):
    http, _, _ = client
    assert http.get("/api/admin/settings").status_code == 404


def test_admin_routes_check_the_token(client):
    http, _, monkeypatch = client
    monkeypatch.setenv("DDF_ADMIN_TOKEN", "secret")
    assert http.get("/api/admin/settings").status_code == 401
    assert (
        http.get("/api/admin/settings", headers={"X-Admin-Token": "wrong"}).status_code
        == 401
    )
    good = {"X-Admin-Token": "secret"}
    assert http.get("/api/admin/settings", headers=good).status_code == 200
    saved = http.put(
        "/api/admin/settings",
        headers=good,
        json={"default_threshold": 0.2, "enabled_models": ["ensemble"]},
    )
    assert saved.status_code == 200, saved.text
    assert settings_module.load().default_threshold == 0.2


def test_invalid_admin_settings_are_rejected(client):
    http, _, monkeypatch = client
    monkeypatch.setenv("DDF_ADMIN_TOKEN", "secret")
    good = {"X-Admin-Token": "secret"}
    for body in (
        {"enabled_models": []},
        {"default_threshold": 1.5},
        {"default_model": "multimodal", "enabled_models": ["ensemble"]},
        {"no_such_setting": True},
    ):
        assert (
            http.put("/api/admin/settings", headers=good, json=body).status_code == 422
        )
