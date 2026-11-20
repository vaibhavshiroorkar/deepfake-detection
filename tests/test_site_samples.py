"""Dataset samples on the public site: the list, the previews and a check of the original file."""

import json
import shutil

import pytest
from fastapi.testclient import TestClient
from test_site import _clip, _FakeScorer, _Registry

from deepfake_detection.webapp import site
from deepfake_detection.webapp.server import create_app

ID = "a1b2c3d4e5f6"


@pytest.fixture(autouse=True)
def fresh_detector():
    """The detector is cached with its scorer; a stand-in scorer must not leak into other test files."""
    site.detector.cache_clear()
    yield
    site.detector.cache_clear()


@pytest.fixture
def samples(tmp_path, monkeypatch):
    """A one-sample catalogue whose original lives under a fake data folder."""
    data = tmp_path / "data"
    (data / "FakeAVCeleb").mkdir(parents=True)
    (data / "samples" / "previews").mkdir(parents=True)
    original = data / "FakeAVCeleb" / "clip.mp4"
    if shutil.which("ffmpeg"):
        shutil.copy(_clip(tmp_path), original)
    else:
        original.write_bytes(b"not a video")
    (data / "samples" / "previews" / f"{ID}.mp4").write_bytes(b"preview bytes")
    entry = {
        "id": ID,
        "dataset": "FakeAVCeleb",
        "kind": "Face swap",
        "label": "fake",
        "title": "Face swap 1",
        "source": "FakeAVCeleb/clip.mp4",
    }
    (data / "samples" / "catalogue.json").write_text(
        json.dumps([entry]), encoding="utf-8"
    )
    monkeypatch.setattr(site, "SAMPLE_DATA", data)
    monkeypatch.setattr(site, "SAMPLES", data / "samples")
    return original


def _client():
    return TestClient(create_app(_Registry()))


def test_the_list_names_each_sample_without_its_file_path(samples):
    body = _client().get("/api/dataset-samples").json()
    assert body["samples"] == [
        {
            "id": ID,
            "dataset": "FakeAVCeleb",
            "kind": "Face swap",
            "label": "fake",
            "title": "Face swap 1",
        }
    ]


def test_without_a_catalogue_the_list_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "SAMPLES", tmp_path / "none")
    assert _client().get("/api/dataset-samples").json() == {"samples": []}


def test_a_known_preview_is_served_as_video(samples):
    response = _client().get(f"/api/dataset-samples/{ID}/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"preview bytes"


@pytest.mark.parametrize("sample", ["ffffffffffff", "..%2Fcatalogue", "A1B2C3D4E5F6"])
def test_an_unknown_or_malformed_preview_is_not_found(samples, sample):
    assert _client().get(f"/api/dataset-samples/{sample}/preview").status_code == 404


def test_an_unknown_sample_cannot_be_checked(samples):
    response = _client().post("/api/check-sample", json={"id": "ffffffffffff"})
    assert response.status_code == 404
    assert response.json()["status"] == "unknown"


def test_a_sample_whose_source_leaves_the_data_folder_is_refused(
    samples, tmp_path, monkeypatch
):
    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"x")
    catalogue = site.SAMPLES / "catalogue.json"
    entries = json.loads(catalogue.read_text(encoding="utf-8"))
    entries[0]["source"] = "../secret.mp4"
    catalogue.write_text(json.dumps(entries), encoding="utf-8")
    response = _client().post("/api/check-sample", json={"id": ID})
    assert response.status_code == 404
    assert outside.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_sample_is_checked_on_a_copy_and_the_original_stays(samples, monkeypatch):
    fake = _FakeScorer()
    monkeypatch.setattr(site, "make_scorer", lambda: fake)
    client = _client()
    response = client.post("/api/check-sample", json={"id": ID})
    assert response.status_code == 202 and response.json()["status"] == "queued"
    site.detector().wait(response.json()["job"], timeout=10)
    body = client.get(f"/api/check/{response.json()['job']}").json()
    assert body["status"] == "done"
    assert fake.seen[0] != samples and not fake.seen[0].exists()
    assert samples.exists()  # the detector deletes what it is given; never the original


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_samples_and_uploads_share_one_hourly_limit(samples, monkeypatch, tmp_path):
    monkeypatch.setenv("DDF_UPLOAD_LIMIT", "1")
    monkeypatch.setattr(site, "make_scorer", _FakeScorer)
    client = _client()
    first = client.post(
        "/api/check",
        files={"file": ("clip.mp4", _clip(tmp_path).read_bytes(), "video/mp4")},
    )
    assert first.status_code == 202
    second = client.post("/api/check-sample", json={"id": ID})
    assert second.status_code == 429
