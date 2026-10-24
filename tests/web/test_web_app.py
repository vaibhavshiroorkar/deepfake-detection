import io
import re
import time
from pathlib import Path

import pytest

from dfd.infer.mock_adapter import MockAdapter
from dfd.infer.real_adapter import NoCheckpointError
from dfd.infer.schema import validate_response
from dfd.web.app import create_app
from dfd.web import __main__ as web_main


@pytest.fixture
def client(tmp_path):
    app = create_app(MockAdapter(), upload_dir=tmp_path / "up", max_bytes=2 * 1024 * 1024)
    return app.test_client(), tmp_path / "up"


def post(c, path, name, route="/api/analyse"):
    with open(path, "rb") as fh:
        return c.post(route, data={"file": (io.BytesIO(fh.read()), name)}, content_type="multipart/form-data")


def test_page_shows_the_mock_banner_and_the_upload_form(client):
    c, _ = client
    html = c.get("/").get_data(as_text=True)
    assert 'id="mock-banner"' in html and "MOCK" in html
    assert 'type="file"' in html and 'id="go"' in html and 'data-source="mock"' in html


def test_page_for_a_real_source_has_no_static_mock_banner(tmp_path):
    class RealLike:
        inference_source = "real"
    html = create_app(RealLike(), upload_dir=tmp_path).test_client().get("/").get_data(as_text=True)
    assert 'id="mock-banner"' not in html and 'data-source="real"' in html


def test_script_shows_the_banner_for_every_non_real_result():
    js = (Path(__file__).parents[2] / "src/dfd/web/static/app.js").read_text(encoding="utf-8")
    assert 'result-mock-banner").hidden = resp.inference_source === "real"' in js
    assert ".innerHTML" not in js  # server text is always inserted with textContent


@pytest.mark.parametrize("name", ["img.png", "speech.wav", "clip_audio.mp4"])
def test_analyse_each_media_type(client, media, name):
    c, up = client
    r = post(c, media / name, name)
    body = r.get_json()
    assert r.status_code == 200 and validate_response(body) == []
    assert body["inference_source"] == "mock" and body["input"]["filename"] == name
    assert list(up.iterdir()) == []  # the upload is deleted after scoring


@pytest.mark.parametrize("name,status,code", [("notes.txt", 415, "UNSUPPORTED_MEDIA_TYPE"), ("corrupt.png", 400, "UNREADABLE_FILE"), ("empty.png", 400, "EMPTY_FILE")])
def test_errors_are_failed_responses_with_the_right_status(client, media, name, status, code):
    c, up = client
    r = post(c, media / name, name)
    body = r.get_json()
    assert r.status_code == status and validate_response(body) == []
    assert body["status"]["state"] == "failed" and body["errors"][0]["code"] == code
    assert list(up.iterdir()) == []


def test_missing_file_field(client):
    c, _ = client
    r = c.post("/api/analyse", data={}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 400 and body["errors"][0]["code"] == "NO_FILE" and validate_response(body) == []


def test_file_over_the_limit(client):
    c, _ = client
    r = c.post("/api/analyse", data={"file": (io.BytesIO(b"x" * (3 * 1024 * 1024)), "big.wav")}, content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 413 and body["errors"][0]["code"] == "FILE_TOO_LARGE" and validate_response(body) == []


def test_path_in_client_filename_is_stripped(client, media):
    c, _ = client
    for raw in ("../../evil/img.png", r"..\..\evil\img.png"):
        name = post(c, media / "img.png", raw).get_json()["input"]["filename"]
        assert "/" not in name and "\\" not in name and name.endswith("img.png")


def test_job_flow_with_progress_and_final_response(client, media):
    c, up = client
    r = post(c, media / "clip_audio.mp4", "clip.mp4", route="/api/jobs")
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{job_id}").get_json()
        if j["state"] == "done":
            break
        time.sleep(0.05)
    assert j["state"] == "done" and validate_response(j["response"]) == []
    assert j["response"]["inference_source"] == "mock"
    assert list(up.iterdir()) == []
    assert c.get("/api/jobs/unknown").status_code == 404


def test_job_with_a_bad_file_ends_as_a_failed_response(client, media):
    c, _ = client
    job_id = post(c, media / "corrupt.wav", "corrupt.wav", route="/api/jobs").get_json()["job_id"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{job_id}").get_json()
        if j["state"] == "done":
            break
        time.sleep(0.05)
    assert j["response"]["status"]["state"] == "failed" and validate_response(j["response"]) == []


def test_server_refuses_to_start_with_the_real_adapter_and_no_checkpoint(capsys):
    assert web_main.main(["--adapter", "real"]) == 2
    assert "no checkpoint" in capsys.readouterr().err
    with pytest.raises(NoCheckpointError):
        from dfd.infer.factory import make_adapter
        make_adapter("real", None)


def test_static_files_are_served(client):
    c, _ = client
    for f in ("app.js", "style.css"):
        assert c.get(f"/static/{f}").status_code == 200
    assert re.search(r"--mock", c.get("/static/style.css").get_data(as_text=True))
