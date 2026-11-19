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
    # Home speaks to visitors: how often it is right on clips it never saw. The dataset findings live on
    # the research page.
    home = client.get("/").text
    assert site.percent("web.outcomes.fakeavceleb.fake.caught") in home
    assert site.percent("web.outcomes.hifiavdf.real.flagged") in home
    assert site.number("audit.fakeavceleb.raw.clip") not in home
    research = client.get("/research").text
    assert site.number("audit.fakeavceleb.raw.clip") in research
    assert site.seed_range("sync.test-identity-strict.seed{}.auc", 2) in research


def test_the_old_app_moved_to_classic(client):
    assert "Is this video a deepfake?" in client.get("/classic").text


def test_a_missing_number_fails_loudly():
    with pytest.raises(KeyError):
        site.number("no.such.key")


def test_seed_range_and_percent_formats():
    assert " to " in site.seed_range("sync.test-identity-strict.seed{}.auc")
    assert site.percent_range("h4.hifiavdf-seed{}.conformal_real_only.fpr").endswith(
        "%"
    )


@pytest.mark.parametrize(
    "info, message",
    [
        (
            {"streams": [{"codec_type": "audio"}], "format": {"duration": "5"}},
            "one video stream",
        ),
        (
            {
                "streams": [{"codec_type": "video"}, {"codec_type": "video"}],
                "format": {"duration": "5"},
            },
            "one video stream",
        ),
        (
            {
                "streams": [
                    {"codec_type": "video"},
                    {"codec_type": "audio"},
                    {"codec_type": "audio"},
                ],
                "format": {"duration": "5"},
            },
            "more than one audio",
        ),
        (
            {"streams": [{"codec_type": "video"}], "format": {"duration": "2.0"}},
            "3 to 60 seconds",
        ),
        (
            {"streams": [{"codec_type": "video"}], "format": {"duration": "61"}},
            "3 to 60 seconds",
        ),
        ({"streams": [{"codec_type": "video"}], "format": {}}, "3 to 60 seconds"),
    ],
)
def test_clip_rules_reject(info, message):
    with pytest.raises(site.ClipRejected, match=message):
        site.check_clip_rules(info)


def test_clip_rules_accept_one_video_and_one_audio():
    info = {
        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        "format": {"duration": "4.2"},
    }
    assert site.check_clip_rules(info) == 4.2


def test_upload_rejects_a_wrong_extension(client):
    response = client.post(
        "/api/check", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400 and response.json()["status"] == "rejected"


def test_upload_rejects_a_file_that_is_not_a_video(client):
    response = client.post(
        "/api/check", files={"file": ("clip.mp4", b"not a video", "video/mp4")}
    )
    assert response.status_code == 400
    assert "could not be read" in response.json()["reason"]


class _FakeScorer:
    """Stands in for the P-001 scorer, so the site tests never load a model."""

    def __init__(self):
        self.seen = []

    def score(self, upload, workdir, progress=lambda stage: None, mode="video"):
        from deepfake_detection.webapp.scorer import CheckResult, ScoreReport

        self.seen.append(upload)
        self.modes = [*getattr(self, "modes", []), mode]
        progress("scoring")
        checks = {
            "face": CheckResult((0.9999,) * 3),
            "voice": CheckResult((0.01,) * 3),
            "sync": CheckResult(None, "no face in the mouth window"),
        }
        return [
            ScoreReport(checks, {"loudness_db": [-30.0] * 50, "windows": []}, 0.0, 3.0)
        ]


def _clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            "4",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    return clip


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_valid_clip_is_scored_through_a_job_and_deleted(monkeypatch, tmp_path):
    fake = _FakeScorer()
    monkeypatch.setattr(site, "make_scorer", lambda: fake)
    client = TestClient(create_app(_Registry()))
    response = client.post(
        "/api/check",
        files={"file": ("clip.mp4", _clip(tmp_path).read_bytes(), "video/mp4")},
    )
    assert response.status_code == 202 and response.json()["status"] == "queued"
    job = response.json()["job"]
    site.detector().wait(job, timeout=10)
    body = client.get(f"/api/check/{job}").json()
    assert body["status"] == "done" and body["outcome"] == "signs"
    assert body["checks"]["face"]["raised"] is True
    assert body["checks"]["sync"] == {
        "percentile": None,
        "score": None,
        "raised": False,
        "reason": "no face in the mouth window",
    }
    assert "5 in 100" in body["how_sure"]
    assert not fake.seen[0].exists()  # the upload is gone once the result can be read


def test_an_unknown_job_is_not_found(client):
    assert client.get("/api/check/" + "0" * 32).status_code == 404


def test_present_turns_percentiles_into_whole_numbers_below_100():
    outcome = {
        "kind": "cant_tell",
        "raised": [],
        "checks": {
            "face": {"percentile": 0.9995, "reason": None},
            "voice": {"percentile": 0.12, "reason": None},
            "sync": {"percentile": None, "reason": "x"},
        },
        "trace": {},
    }
    shown = site.present(outcome)
    assert (
        shown["checks"]["face"]["percentile"] == 99
        and shown["checks"]["voice"]["percentile"] == 12
    )
    assert shown["outcome"] == "cant_tell" and shown["summary"]


def test_contact_honeypot_pretends_success_without_sending(client, monkeypatch):
    monkeypatch.setattr(
        site, "send_contact", lambda *a: pytest.fail("a bot message was sent")
    )
    response = client.post(
        "/contact",
        data={
            "name": "Bot",
            "email": "b@x.io",
            "message": "buy now buy now",
            "website": "x",
        },
    )
    assert "your message was sent" in response.text


def test_contact_says_when_it_is_not_connected(client):
    response = client.post(
        "/contact",
        data={
            "name": "Ana",
            "email": "ana@example.com",
            "message": "A question about H2.",
        },
    )
    assert "not connected yet" in response.text
    assert "A question about H2." in response.text  # the visitor's text is kept


def test_contact_rejects_a_bad_email(client):
    response = client.post(
        "/contact",
        data={
            "name": "Ana",
            "email": "not-an-email",
            "message": "A question about H2.",
        },
    )
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


def test_present_names_every_raised_check_in_plain_english():
    checks = {
        n: {"percentile": 0.99, "reason": None} for n in ("face", "voice", "sync")
    }
    two = site.present(
        {"kind": "signs", "raised": ["face", "sync"], "checks": checks, "trace": {}}
    )
    assert two["summary"].startswith("The face and lip-sync checks scored")
    one = site.present(
        {"kind": "signs", "raised": ["voice"], "checks": checks, "trace": {}}
    )
    assert one["summary"].startswith("The voice check scored")


def test_the_upload_limit_can_be_raised_for_a_private_demo(monkeypatch):
    monkeypatch.setenv("DDF_UPLOAD_LIMIT", "500")
    assert site.upload_limit() == 500
    monkeypatch.delenv("DDF_UPLOAD_LIMIT")
    assert site.upload_limit() == site.UPLOAD_LIMIT


def test_present_lists_three_raised_checks_with_commas():
    checks = {
        n: {"percentile": 0.99, "reason": None} for n in ("face", "voice", "sync")
    }
    three = site.present(
        {
            "kind": "signs",
            "raised": ["face", "voice", "sync"],
            "checks": checks,
            "trace": {},
        }
    )
    assert three["summary"].startswith("The face, voice and lip-sync checks scored")


def test_present_says_which_part_of_a_long_video_decided():
    checks = {"face": {"percentile": 0.999, "reason": None}}
    shown = site.present(
        {
            "kind": "signs",
            "raised": ["face"],
            "checks": checks,
            "trace": {},
            "parts": 5,
            "part": {"start": 6.04, "end": 9.04},
        }
    )
    assert (
        shown["where"]
        == "The whole video was checked in 5 parts of 3 seconds. The most unusual part, 0:06 to 0:09, is shown."
    )
    one = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "parts": 1,
            "part": {"start": 0.52, "end": 3.52},
        }
    )
    assert (
        one["where"]
        == "The video has 3 seconds of sound, checked as one part (0:00 to 0:03)."
    )


def test_present_puts_a_file_label_first_in_plain_words():
    checks = {"face": {"percentile": 0.5, "reason": None}}
    shown = site.present(
        {
            "kind": "signs",
            "raised": ["file"],
            "checks": checks,
            "trace": {},
            "file_check": ["an AIGC label marking the file as AI-generated content"],
        }
    )
    assert shown["summary"].startswith("The file carries an AIGC label")
    assert shown["file_check"] == [
        "an AIGC label marking the file as AI-generated content"
    ]
    plain = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "file_check": [],
        }
    )
    assert plain["file_check"] == []


@pytest.mark.parametrize(
    ("suffix", "mode", "streams", "kind"),
    [
        (".mp4", "", ["video", "audio"], "video"),
        (".webm", "video", ["video", "audio"], "video"),
        (".webm", "voice", ["audio"], "voice"),
        (".webm", "", ["audio"], "voice"),
        (".wav", "", ["audio"], "voice"),
        (".mp3", "", ["audio"], "voice"),
        (".jpg", "", ["video"], "picture"),
        (".png", "picture", ["video"], "picture"),
    ],
)
def test_the_kind_of_input_comes_from_the_mode_the_name_and_the_streams(
    suffix, mode, streams, kind
):
    info = {"streams": [{"codec_type": s} for s in streams]}
    assert site.classify(suffix, mode, info) == kind


@pytest.mark.parametrize(
    ("suffix", "mode", "streams", "message"),
    [
        (".txt", "", ["video"], "Use"),
        (".wav", "", [], "no sound"),
        (".mp4", "video", ["audio"], "no picture"),
        (".jpg", "voice", ["video"], "no sound"),
    ],
)
def test_an_input_that_does_not_fit_its_kind_is_refused(suffix, mode, streams, message):
    with pytest.raises(site.ClipRejected, match=message):
        site.classify(suffix, mode, {"streams": [{"codec_type": s} for s in streams]})


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_voice_and_a_picture_become_clips_the_checks_can_read(tmp_path):
    wav, png = tmp_path / "voice.wav", tmp_path / "face.png"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300",
            "-t",
            "4",
            str(wav),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=161x121",
            "-frames:v",
            "1",
            str(png),
        ],
        check=True,
    )
    voice = site.to_clip(wav, "voice", tmp_path / "voice.mp4")
    assert site.check_clip_rules(site.probe(voice)) == pytest.approx(4.0, abs=0.1)
    picture = site.to_clip(png, "picture", tmp_path / "face.mp4")
    info = site.probe(picture)
    assert [s["codec_type"] for s in info["streams"]] == ["video"]
    assert site.check_clip_rules(info) >= 3.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_recorded_voice_is_scored_as_a_voice(monkeypatch, tmp_path):
    fake = _FakeScorer()
    monkeypatch.setattr(site, "make_scorer", lambda: fake)
    site.detector.cache_clear()
    wav = tmp_path / "voice.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300",
            "-t",
            "4",
            str(wav),
        ],
        check=True,
    )
    client = TestClient(create_app(_Registry()))
    response = client.post(
        "/api/check",
        files={"file": ("recording.wav", wav.read_bytes(), "audio/wav")},
        data={"mode": "voice"},
    )
    assert response.status_code == 202
    site.detector().wait(response.json()["job"], timeout=10)
    assert fake.modes == ["voice"]
    site.detector.cache_clear()


def test_a_picture_and_a_voice_say_which_check_ran():
    checks = {"face": {"percentile": 0.5, "reason": None}}
    picture = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "input": "picture",
            "parts": 1,
            "part": {"start": 0.0, "end": 3.0},
        }
    )
    assert picture["where"] == "A picture, checked by the face check only."
    voice = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "input": "voice",
            "parts": 2,
            "part": {"start": 3.0, "end": 6.0},
        }
    )
    assert voice["where"].startswith(
        "A voice recording, checked by the voice check only."
    )
    assert "whole recording was checked" in voice["where"]
    assert "video" not in voice["where"]


def test_a_picture_is_refused_when_there_is_no_picture_calibration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(site, "PICTURE_CALIBRATION", tmp_path / "missing.json")
    site.picture_calibration.cache_clear()
    with pytest.raises(site.ClipRejected, match="Pictures cannot be checked"):
        site.require_calibration("picture")
    site.require_calibration("voice")  # voices and videos use the video calibration
    site.picture_calibration.cache_clear()


def test_how_sure_for_a_picture_speaks_of_pictures():
    if site.picture_calibration() is None:
        pytest.skip("no picture calibration built")
    assert "pictures" in site.how_sure("picture")


def test_present_passes_the_timeline_and_the_lines_through():
    checks = {"face": {"percentile": 0.5, "reason": None}}
    timeline = [
        {
            "start": 0.0,
            "end": 3.0,
            "reading": 0.5,
            "checks": {"face": 0.5, "voice": None, "sync": None},
        }
    ]
    shown = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "timeline": timeline,
            "lines": {"flag": 0.982, "lower": 0.928},
        }
    )
    assert shown["timeline"] == timeline
    assert shown["lines"] == {"flag": 0.982, "lower": 0.928}
    explained = site.present(
        {
            "kind": "no_signs",
            "raised": [],
            "checks": checks,
            "trace": {},
            "explain": {"frames": [], "summary": "s"},
        }
    )
    assert explained["explain"] == {"frames": [], "summary": "s"}
    assert shown["checks"]["face"]["score"] == 0.5


def test_the_detector_lives_on_the_check_page_not_the_home_page(client):
    home = client.get("/").text
    assert "data-detector" not in home
    assert 'href="/check"' in home
    assert "data-detector" in client.get("/check").text


def test_home_rates_come_from_the_lines_in_force(monkeypatch):
    """The home page must quote the calibration the site runs on, not the study's numbers."""
    from types import SimpleNamespace

    meta = {
        "outcomes": {
            "FakeAVCeleb": {
                "real": {"no_signs": 10},
                "fake": {"signs": 3, "no_signs": 1},
            },
            "DFDC": {
                "real": {"signs": 1, "no_signs": 9},
                "fake": {"signs": 1, "no_signs": 9},
            },
            "HiFi-AVDF": {
                "real": {"signs": 2, "no_signs": 8},
                "fake": {"signs": 6, "no_signs": 4},
            },
        }
    }
    monkeypatch.setattr(site, "calibration", lambda: SimpleNamespace(meta=meta))
    assert site.live_rates() == {
        "home_caught": "75%",
        "other_caught": ("10%", "60%"),
        "real_flagged": ("0%", "20%"),
        "real_flagged_mean": "10",
        "all_caught": "42%",
        "all_flagged": "10%",
        "fake_clips": "24",
        "real_clips": "30",
    }


def test_home_page_quotes_no_study_auc():
    """The study's fusion AUC (98 to 99%) read as the site's accuracy; the site does not run that model."""
    text = (site.TEMPLATES / "home.html").read_text(encoding="utf-8")
    assert "auc" not in text.lower()
    assert "live().all_caught" in text


def test_each_check_carries_its_explanation_and_measured_catch_rates(monkeypatch):
    from types import SimpleNamespace

    meta = {
        "outcomes": {},
        "raised_by": {
            "FakeAVCeleb": {"fakes": 4, "face": 2, "voice": 1, "sync": 0},
            "HiFi-AVDF": {"fakes": 10, "face": 0, "voice": 7, "sync": 1},
        },
    }
    monkeypatch.setattr(site, "calibration", lambda: SimpleNamespace(meta=meta))
    about = site.check_info()
    assert set(about) == {"face", "gen", "voice", "sync", "file"}
    for check in ("face", "gen", "voice", "sync", "file"):
        assert about[check]["what"] and about[check]["misses"]
    assert about["voice"]["caught"] == [["HiFi-AVDF", "70%"], ["FakeAVCeleb", "25%"]]
    assert about["file"]["caught"] == []


def test_an_older_calibration_without_counts_gives_no_rates(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        site, "calibration", lambda: SimpleNamespace(meta={"outcomes": {}})
    )
    assert site.check_info()["face"]["caught"] == []
