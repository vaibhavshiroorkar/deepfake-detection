"""The public site (webapp/site.py): pages, numbers, clip rules, the upload check and the contact form."""

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from deepfake_detection.webapp import site
from deepfake_detection.webapp.server import create_app


class _Registry:
    """The site never touches the model registry; the old app's routes do."""


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("CONTACT_TO", raising=False)
    return TestClient(create_app(_Registry()))


@pytest.mark.parametrize("route", list(site.PAGES))
def test_every_page_renders_with_no_template_left_over(client, route):
    response = client.get(route)
    assert response.status_code == 200
    assert "{{" not in response.text and "{%" not in response.text
    assert "Deepfake Check" in response.text


def test_home_takes_its_numbers_from_numbers_json(client):
    text = client.get("/").text
    assert site.number("audit.fakeavceleb.raw.clip") in text
    assert site.seed_range("sync.test-identity-strict.seed{}.auc", 2) in text


def test_the_old_app_moved_to_classic(client):
    assert "Is this video a deepfake?" in client.get("/classic").text


def test_a_missing_number_fails_loudly():
    with pytest.raises(KeyError):
        site.number("no.such.key")


def test_seed_range_and_percent_formats():
    assert " to " in site.seed_range("sync.test-identity-strict.seed{}.auc")
    assert site.percent_range("h4.hifiavdf-seed{}.conformal_real_only.fpr").endswith("%")


@pytest.mark.parametrize("info, message", [
    ({"streams": [{"codec_type": "audio"}], "format": {"duration": "5"}}, "one video stream"),
    ({"streams": [{"codec_type": "video"}, {"codec_type": "video"}], "format": {"duration": "5"}}, "one video stream"),
    ({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}, {"codec_type": "audio"}], "format": {"duration": "5"}}, "more than one audio"),
    ({"streams": [{"codec_type": "video"}], "format": {"duration": "2.0"}}, "3 to 60 seconds"),
    ({"streams": [{"codec_type": "video"}], "format": {"duration": "61"}}, "3 to 60 seconds"),
    ({"streams": [{"codec_type": "video"}], "format": {}}, "3 to 60 seconds"),
])
def test_clip_rules_reject(info, message):
    with pytest.raises(site.ClipRejected, match=message):
        site.check_clip_rules(info)


def test_clip_rules_accept_one_video_and_one_audio():
    info = {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "4.2"}}
    assert site.check_clip_rules(info) == 4.2


def test_upload_rejects_a_wrong_extension(client):
    response = client.post("/api/check", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400 and response.json()["status"] == "rejected"


def test_upload_rejects_a_file_that_is_not_a_video(client):
    response = client.post("/api/check", files={"file": ("clip.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 400
    assert "could not be read" in response.json()["reason"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_valid_clip_is_answered_as_not_scored_and_deleted(client, tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25", "-f", "lavfi",
                    "-i", "sine=frequency=440", "-t", "4", "-shortest", "-pix_fmt", "yuv420p", str(clip)], check=True)
    seen = []
    real_probe = site.probe
    monkeypatch.setattr(site, "probe", lambda path: seen.append(path) or real_probe(path))
    response = client.post("/api/check", files={"file": ("clip.mp4", clip.read_bytes(), "video/mp4")})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "not_scored"
    assert body["duration"] == pytest.approx(4.0, abs=0.2)
    assert not seen[0].exists()          # the upload is deleted before the answer is sent


def test_contact_honeypot_pretends_success_without_sending(client, monkeypatch):
    monkeypatch.setattr(site, "send_contact", lambda *a: pytest.fail("a bot message was sent"))
    response = client.post("/contact", data={"name": "Bot", "email": "b@x.io", "message": "buy now buy now", "website": "x"})
    assert "your message was sent" in response.text


def test_contact_says_when_it_is_not_connected(client):
    response = client.post("/contact", data={"name": "Ana", "email": "ana@example.com", "message": "A question about H2."})
    assert "not connected yet" in response.text
    assert "A question about H2." in response.text      # the visitor's text is kept


def test_contact_rejects_a_bad_email(client):
    response = client.post("/contact", data={"name": "Ana", "email": "not-an-email", "message": "A question about H2."})
    assert "valid email" in response.text


def test_rate_limiter_allows_up_to_the_limit_per_hour():
    limiter = site.RateLimiter(2, window=3600)
    assert limiter.allow("a", now=0) and limiter.allow("a", now=1)
    assert not limiter.allow("a", now=2)
    assert limiter.allow("b", now=2)
    assert limiter.allow("a", now=3602)


@pytest.mark.parametrize("route", ["/", "/check", "/classic"])
def test_html_pages_tell_the_browser_to_revalidate(client, route):
    # The old app served "/" as a file with no Cache-Control, so browsers kept showing it from cache
    # after "/" became the new site. Every HTML page must make the browser check with the server.
    assert client.get(route).headers.get("cache-control") == "no-cache"
