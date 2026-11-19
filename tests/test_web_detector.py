import threading
from pathlib import Path

import pytest

from deepfake_detection.webapp import calibration as cal
from deepfake_detection.webapp.detector import Detector
from deepfake_detection.webapp.scorer import CheckResult, ClipRefused, ScoreReport


@pytest.fixture(autouse=True)
def _no_file_check(monkeypatch):
    """These tests are about the queue; the file check (ffprobe) has its own tests."""
    from deepfake_detection.webapp import provenance

    monkeypatch.setattr(provenance, "read", lambda path: [])


def _calibration() -> cal.Calibration:
    ref = cal.Reference.from_scores(
        [i / 10 for i in range(-50, 50)], [1] * 100
    )  # logits -5 to 4.9
    combined = cal.Reference.from_scores([i / 1000 for i in range(1000)], [1] * 1000)
    return cal.Calibration(
        {"face": ref, "voice": ref, "sync": ref},
        threshold=0.97,
        lower=0.8,
        meta={"outcomes": {}},
        combined=combined,
    )


class FakeScorer:
    """Returns fixed check probabilities; records the files it saw; can block or fail."""

    def __init__(
        self,
        probabilities=None,
        error=None,
        gate: threading.Event | None = None,
        parts=None,
    ):
        self.probabilities = probabilities or {
            "face": (0.5, 0.5, 0.5),
            "voice": (0.5, 0.5, 0.5),
            "sync": None,
        }
        self.error, self.gate, self.seen = error, gate, []
        self.parts = parts  # a list of probability dicts, one per 3-second part

    def score(
        self, upload: Path, workdir: Path, progress=lambda stage: None, mode="video"
    ) -> list[ScoreReport]:
        self.seen.append((upload.exists(), workdir))
        (workdir / "neutral.mp4").write_bytes(b"x")
        progress("scoring")
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        reports = []
        for i, probabilities in enumerate(self.parts or [self.probabilities]):
            checks = {
                k: CheckResult(v, None if v else "not checked")
                for k, v in probabilities.items()
            }
            reports.append(
                ScoreReport(
                    checks,
                    {"loudness_db": [], "windows": []},
                    start=3.0 * i,
                    end=3.0 * i + 3,
                    explain={"frames": [{"time": 3.0 * i + 1}], "summary": f"part {i}"},
                )
            )
        return reports


def _upload(tmp_path: Path, name: str = "clip.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"video")
    return path


def _finished(detector: Detector, job: str) -> dict:
    detector.wait(job, timeout=5)
    return detector.status(job)


def test_a_finished_job_reports_the_outcome_and_deletes_every_file(tmp_path) -> None:
    scorer = FakeScorer(
        {"face": (0.999, 0.999, 0.999), "voice": (0.5,) * 3, "sync": None}
    )
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    upload = _upload(tmp_path)
    status = _finished(detector, detector.submit(upload))
    assert status["status"] == "done"
    assert status["outcome"]["kind"] == "signs"
    assert status["outcome"]["raised"] == ["face"]
    assert status["outcome"]["checks"]["sync"] == {
        "percentile": None,
        "score": None,
        "reason": "not checked",
    }
    assert not upload.exists()
    assert not scorer.seen[0][1].exists()


def test_a_refused_clip_becomes_cant_check_with_its_reason(tmp_path) -> None:
    scorer = FakeScorer(error=ClipRefused("under 3 seconds of sound"))
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    upload = _upload(tmp_path)
    status = _finished(detector, detector.submit(upload))
    assert status["outcome"]["kind"] == "cant_check"
    assert status["outcome"]["reason"] == "under 3 seconds of sound"
    assert not upload.exists()


def test_an_internal_error_is_reported_without_details_and_still_deletes(
    tmp_path,
) -> None:
    scorer = FakeScorer(error=RuntimeError("secret path C:/x"))
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    upload = _upload(tmp_path)
    status = _finished(detector, detector.submit(upload))
    assert status["status"] == "error"
    assert "secret" not in status["reason"]
    assert not upload.exists()
    assert not scorer.seen[0][1].exists()


def test_one_job_runs_and_three_wait_then_the_detector_is_busy(tmp_path) -> None:
    gate = threading.Event()
    scorer = FakeScorer(gate=gate)
    detector = Detector(
        lambda: scorer, _calibration(), workroot=tmp_path / "work", waiting=3
    )
    jobs = [detector.submit(_upload(tmp_path, f"c{i}.mp4")) for i in range(4)]
    busy = _upload(tmp_path, "c4.mp4")
    with pytest.raises(Detector.Busy):
        detector.submit(busy)
    assert not busy.exists()
    gate.set()
    assert all(_finished(detector, job)["status"] == "done" for job in jobs)


def test_an_unknown_job_has_no_status(tmp_path) -> None:
    detector = Detector(
        lambda: FakeScorer(), _calibration(), workroot=tmp_path / "work"
    )
    assert detector.status("nope") is None


def test_warm_loads_the_scorer_once_before_any_upload(tmp_path) -> None:
    made = []
    detector = Detector(
        lambda: made.append(1) or FakeScorer(),
        _calibration(),
        workroot=tmp_path / "work",
    )
    detector.warm()
    detector.warm()
    assert made == [1]
    assert _finished(detector, detector.submit(_upload(tmp_path)))["status"] == "done"
    assert made == [1]


def test_a_tampered_avhubert_file_is_refused_before_it_is_unpickled(tmp_path) -> None:
    from deepfake_detection.webapp.scorer import verified_avhubert

    fake = tmp_path / "base_vox_iter5.pt"
    fake.write_bytes(b"not the released checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        verified_avhubert(fake)


def test_a_long_video_is_judged_by_its_most_unusual_part(tmp_path) -> None:
    calm = {"face": (0.5,) * 3, "voice": (0.5,) * 3, "sync": None}
    odd = {"face": (0.999,) * 3, "voice": (0.5,) * 3, "sync": None}
    scorer = FakeScorer(parts=[calm, calm, odd, calm])
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "signs" and outcome["raised"] == ["face"]
    assert outcome["part"] == {"start": 6.0, "end": 9.0}
    assert outcome["parts"] == 4


def test_a_video_where_no_part_can_be_checked_says_why(tmp_path) -> None:
    empty = {"face": None, "voice": None, "sync": None}
    detector = Detector(
        lambda: FakeScorer(parts=[empty, empty]),
        _calibration(),
        workroot=tmp_path / "work",
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "cant_check" and "not checked" in outcome["reason"]


def test_an_ai_label_in_the_file_decides_and_is_shown_first(tmp_path) -> None:
    calm = {"face": (0.5,) * 3, "voice": (0.5,) * 3, "sync": None}
    detector = Detector(
        lambda: FakeScorer(calm),
        _calibration(),
        workroot=tmp_path / "work",
        read_labels=lambda path: [
            "an AIGC label marking the file as AI-generated content"
        ],
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "signs"
    assert outcome["raised"][0] == "file"
    assert outcome["file_check"] == [
        "an AIGC label marking the file as AI-generated content"
    ]


def test_no_label_changes_nothing(tmp_path) -> None:
    calm = {"face": (0.5,) * 3, "voice": (0.5,) * 3, "sync": None}
    detector = Detector(
        lambda: FakeScorer(calm),
        _calibration(),
        workroot=tmp_path / "work",
        read_labels=lambda path: [],
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] != "signs" and outcome["file_check"] == []


def test_a_picture_is_judged_against_the_picture_calibration(tmp_path) -> None:
    still = {"face": (0.999,) * 3, "voice": None, "sync": None}
    lenient = cal.Calibration(
        {"face": cal.Reference.from_scores([10.0, 20.0], [1, 1])},
        threshold=0.97,
        lower=0.8,
    )
    detector = Detector(
        lambda: FakeScorer(still),
        _calibration(),
        workroot=tmp_path / "work",
        read_labels=lambda p: [],
        calibrations={"picture": lenient},
    )
    picture = _finished(
        detector, detector.submit(_upload(tmp_path, "a.mp4"), mode="picture")
    )["outcome"]
    video = _finished(detector, detector.submit(_upload(tmp_path, "b.mp4")))["outcome"]
    assert picture["kind"] == "no_signs" and video["kind"] == "signs"


def test_the_outcome_carries_every_part_and_the_lines_it_was_judged_by(
    tmp_path,
) -> None:
    calm = {"face": (0.5,) * 3, "voice": (0.5,) * 3, "sync": None}
    odd = {"face": (0.999,) * 3, "voice": (0.5,) * 3, "sync": None}
    calibration = _calibration()
    detector = Detector(
        lambda: FakeScorer(parts=[calm, odd, calm]),
        calibration,
        workroot=tmp_path / "work",
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    timeline = outcome["timeline"]
    assert [(p["start"], p["end"]) for p in timeline] == [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 9.0),
    ]
    assert timeline[1]["reading"] > timeline[0]["reading"]
    assert timeline[1]["checks"]["face"] > timeline[0]["checks"]["face"]
    assert timeline[0]["checks"]["sync"] is None
    flag, lower = calibration.limits(3)
    # The page shows one scale: 0 real, 0.5 neutral, 1 fake; flagged from 0.6, can't tell from 0.4.
    assert outcome["lines"] == {"flag": 0.6, "lower": 0.4}
    assert outcome["raw_lines"] == {"flag": flag, "lower": lower}
    assert 0.6 <= timeline[1]["reading"] <= 1.0
    assert timeline[0]["reading"] < 0.4
    assert outcome["checks"]["face"]["score"] >= 0.6


def test_the_outcome_explains_the_part_where_the_face_scored_highest(tmp_path) -> None:
    calm = {"face": (0.5,) * 3, "voice": (0.5,) * 3, "sync": None}
    odd = {"face": (0.999,) * 3, "voice": (0.5,) * 3, "sync": None}
    detector = Detector(
        lambda: FakeScorer(parts=[calm, odd, calm]),
        _calibration(),
        workroot=tmp_path / "work",
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["explain"]["summary"] == "part 1"
    assert outcome["explain"]["frames"][0]["time"] == 4.0


def test_cant_check_names_each_reason_once(tmp_path) -> None:
    """The generated-video check reads the face crops, so with no face it repeats the face's reason (found 11-19)."""
    scorer = FakeScorer({"face": None, "gen": None, "voice": None, "sync": None})
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    status = _finished(detector, detector.submit(_upload(tmp_path)))
    assert status["outcome"]["kind"] == "cant_check"
    assert status["outcome"]["reason"].count("not checked") == 1


def test_the_outside_face_score_replaces_ours_when_it_answers(tmp_path) -> None:
    asked = []

    def opinion(upload, time):
        asked.append((upload.exists(), time))
        return {"ai_generated": 0.97, "deepfake": 0.02}

    low = {
        "face": (0.01,) * 3,
        "voice": (0.5,) * 3,
        "sync": None,
    }  # our face model says real
    scorer = FakeScorer(low)
    detector = Detector(
        lambda: scorer, _calibration(), workroot=tmp_path / "work", face_opinion=opinion
    )
    upload = _upload(tmp_path)
    status = _finished(detector, detector.submit(upload))
    outcome = status["outcome"]
    assert outcome["kind"] == "signs" and outcome["raised"] == ["face"]
    assert outcome["checks"]["face"]["score"] >= 0.6
    # one still from the first part, asked while our models score, before the upload is deleted
    assert asked == [(True, 1.5)]
    assert not upload.exists()


def test_our_face_model_answers_when_the_outside_check_cannot(tmp_path) -> None:
    high = {"face": (0.999,) * 3, "voice": (0.5,) * 3, "sync": None}
    detector = Detector(
        lambda: FakeScorer(high),
        _calibration(),
        workroot=tmp_path / "work",
        face_opinion=lambda upload, time: {"error": "allowance used up"},
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "signs" and outcome["raised"] == ["face"]


def test_a_voice_recording_is_never_sent_out(tmp_path) -> None:
    asked = []
    detector = Detector(
        lambda: FakeScorer(),
        _calibration(),
        workroot=tmp_path / "work",
        face_opinion=lambda upload, time: asked.append(1) or {"error": "x"},
    )
    _finished(detector, detector.submit(_upload(tmp_path), mode="voice"))
    assert asked == []


class _GenScorer:
    """One part whose generated-video check scored `gen` and guessed `guess`."""

    def __init__(self, gen: float, guess: dict) -> None:
        self.gen, self.guess = gen, guess

    def score(self, upload, workdir, progress=lambda stage: None, mode="video"):
        checks = {
            "face": CheckResult((0.5,) * 3),
            "voice": CheckResult((0.5,) * 3),
            "sync": CheckResult(None, "not checked"),
            "gen": CheckResult((self.gen,), detail={"generator": self.guess}),
        }
        return [ScoreReport(checks, {"loudness_db": [], "windows": []})]


def _with_gen() -> cal.Calibration:
    calibration = _calibration()
    calibration.references["gen"] = calibration.references["face"]
    return calibration


def test_the_generator_is_named_only_when_the_video_looks_generated(tmp_path) -> None:
    guess = {"name": "sora2", "share": 0.8, "sure": True}
    flagged = Detector(
        lambda: _GenScorer(0.999, guess), _with_gen(), workroot=tmp_path / "a"
    )
    outcome = _finished(flagged, flagged.submit(_upload(tmp_path, "a.mp4")))["outcome"]
    assert outcome["generator"] == {
        "suspected": True,
        "source": "model",
        "name": "sora2",
        "share": 0.8,
        "sure": True,
    }
    calm = Detector(
        lambda: _GenScorer(0.01, guess), _with_gen(), workroot=tmp_path / "b"
    )
    outcome = _finished(calm, calm.submit(_upload(tmp_path, "b.mp4")))["outcome"]
    assert outcome["generator"] == {"suspected": False}


def test_a_label_that_names_the_tool_wins(tmp_path) -> None:
    label = 'an AIGC label marking the file as AI-generated content, which names the tool "dreamina"'
    detector = Detector(
        lambda: _GenScorer(0.01, None),
        _with_gen(),
        workroot=tmp_path / "w",
        read_labels=lambda path: [label],
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["generator"] == {
        "suspected": True,
        "source": "label",
        "name": "dreamina",
        "share": None,
        "sure": True,
    }


def test_a_description_tops_the_answer_and_a_failure_never_breaks_it(tmp_path) -> None:
    seen = []

    def describe(upload, time, explain):
        seen.append((upload.exists(), time, explain.get("summary")))
        return "A woman talks in a kitchen. The heatmap covers her mouth."

    detector = Detector(
        lambda: FakeScorer(), _calibration(), workroot=tmp_path / "a", describe=describe
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path, "a.mp4")))[
        "outcome"
    ]
    assert (
        outcome["description"]
        == "A woman talks in a kitchen. The heatmap covers her mouth."
    )
    assert seen == [(True, 1.0, "part 0")]

    voice = _finished(
        detector, detector.submit(_upload(tmp_path, "b.mp4"), mode="voice")
    )["outcome"]
    assert "description" not in voice and len(seen) == 1

    def broken(upload, time, explain):
        raise RuntimeError("model missing")

    failing = Detector(
        lambda: FakeScorer(), _calibration(), workroot=tmp_path / "c", describe=broken
    )
    status = _finished(failing, failing.submit(_upload(tmp_path, "c.mp4")))
    assert status["status"] == "done" and "description" not in status["outcome"]


def test_a_labelled_file_is_not_sent_out(tmp_path) -> None:
    asked = []
    detector = Detector(
        lambda: FakeScorer(),
        _calibration(),
        workroot=tmp_path / "w",
        read_labels=lambda path: [
            "a C2PA content credential declaring AI-generated media"
        ],
        face_opinion=lambda upload, time: asked.append(1) or {"error": "x"},
    )
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "signs" and asked == []
