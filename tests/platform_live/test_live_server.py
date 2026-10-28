"""Live dashboard API: upload, queue, job states, cleanup, limits. Real files through the real pipeline; detector modules use a fake adapter."""
import time

import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.platform.dashboard import serve
from dfd.platform.modules import build_default_modules
from dfd.platform.modules.base import AdapterProvider
from dfd.platform.pipeline import planned_modules
from dfd.platform.provenance.modules import build_provenance_modules
from dfd.platform.schema import validate_report


class FakeAdapter:
    meta = {"checkpoint_id": "fake-ckpt", "files": {}, "tasks": {"T-V": {"threshold": {"value": 0.6}}, "T-I": {"threshold": {"value": 0.5}},
                                                                    "T-A": {"threshold": {"value": 0.5}}}}

    def analyse(self, media, progress=None):
        pm = {}
        if media.has_audio:
            pm["audio"] = ModalityScore(True, 0.2, 1)
        if media.media_type == "image":
            pm["image"] = ModalityScore(True, 0.3, 1, path="whole_image")
        if media.media_type == "video":
            pm["visual"] = ModalityScore(False, reason="no face found")
        return AdapterResult(pm, [], {"checkpoint_id": "fake-ckpt"}, 0.5, {})


def factory():
    return build_default_modules(AdapterProvider(adapter=FakeAdapter())) + planned_modules() + build_provenance_modules()


@pytest.fixture()
def live(tmp_path):
    app = serve.create_app(live=True, modules_factory=factory, upload_dir=tmp_path / "up", max_bytes=2_000_000)
    return app.test_client(), tmp_path / "up"


def wait(client, jid, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["state"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def post(client, path, name=None):
    with open(path, "rb") as fh:
        return client.post("/api/analyse", data={"file": (fh, name or path.name)}, content_type="multipart/form-data")


def test_mock_server_has_no_upload_and_says_so():
    c = serve.create_app().test_client()
    m = c.get("/api/mode").get_json()
    assert {k: m[k] for k in ("live", "mock_only")} == {"live": False, "mock_only": True}
    assert set(m) == {"live", "mock_only", "accept", "max_bytes"} and set(m["accept"]) == {"image", "video", "audio"}   # redesign: upload panel facts
    assert c.post("/api/analyse").status_code == 404


def test_live_mode_flag():
    c = serve.create_app(live=True, modules_factory=factory).test_client()
    m = c.get("/api/mode").get_json()
    assert {k: m[k] for k in ("live", "mock_only")} == {"live": True, "mock_only": False}
    assert set(m) == {"live", "mock_only", "accept", "max_bytes"} and set(m["accept"]) == {"image", "video", "audio"}   # redesign: upload panel facts


def test_upload_to_report_and_file_deleted(live, media):
    c, up = live
    r = post(c, media / "clip_audio.mp4")
    assert r.status_code == 202
    j = wait(c, r.get_json()["job_id"])
    assert j["state"] == "done" and j["error"] is None
    assert validate_report(j["report"]) == []
    assert j["report"]["source_summary"] == "real" and j["progress"]["completed"] == j["progress"]["total"] > 0
    assert list(up.glob("*")) == []                       # the upload is gone


@pytest.mark.parametrize("name,code", [("corrupt.png", "UNREADABLE_FILE"), ("empty.png", "EMPTY_FILE"), ("notes.txt", "UNSUPPORTED_MEDIA_TYPE")])
def test_refused_files_fail_with_code_and_are_deleted(live, media, name, code):
    c, up = live
    r = post(c, media / name)
    j = wait(c, r.get_json()["job_id"])
    assert j["state"] == "failed" and j["report"] is None and j["error"]["code"] == code
    assert list(up.glob("*")) == []


def test_no_file_unknown_job_and_too_large(live, tmp_path):
    c, _ = live
    assert c.post("/api/analyse", data={}, content_type="multipart/form-data").get_json()["error"]["code"] == "NO_FILE"
    assert c.get("/api/jobs/nope").status_code == 404
    big = tmp_path / "big.wav"
    big.write_bytes(b"0" * 3_000_000)
    r = post(c, big)
    assert r.status_code == 413 and r.get_json()["error"]["code"] == "FILE_TOO_LARGE"


def test_filename_cannot_escape_upload_dir(live, media):
    c, up = live
    r = post(c, media / "speech.wav", name="../../evil.wav")
    j = wait(c, r.get_json()["job_id"])
    assert j["state"] == "done" and j["report"]["input"]["filename"] == "evil.wav"
    assert list(up.glob("*")) == []


def test_queue_full(live, media, monkeypatch):
    c, _ = live
    store = c.application.config["JOB_STORE"]
    monkeypatch.setattr(store, "pending", lambda: serve.MAX_QUEUED)
    r = post(c, media / "speech.wav")
    assert r.status_code == 429 and r.get_json()["error"]["code"] == "QUEUE_FULL"
