import threading
from pathlib import Path

import pytest

from deepfake_detection.webapp import calibration as cal
from deepfake_detection.webapp.detector import Detector
from deepfake_detection.webapp.scorer import CheckResult, ClipRefused, ScoreReport


def _calibration() -> cal.Calibration:
    ref = cal.Reference.from_scores([i / 10 for i in range(-50, 50)], [1] * 100)   # logits -5 to 4.9
    combined = cal.Reference.from_scores([i / 1000 for i in range(1000)], [1] * 1000)
    return cal.Calibration({"face": ref, "voice": ref, "sync": ref}, threshold=0.97, lower=0.8,
                           meta={"outcomes": {}}, combined=combined)


class FakeScorer:
    """Returns fixed check probabilities; records the files it saw; can block or fail."""

    def __init__(self, probabilities=None, error=None, gate: threading.Event | None = None, parts=None):
        self.probabilities = probabilities or {"face": (0.5, 0.5, 0.5), "voice": (0.5, 0.5, 0.5), "sync": None}
        self.error, self.gate, self.seen = error, gate, []
        self.parts = parts            # a list of probability dicts, one per 3-second part

    def score(self, upload: Path, workdir: Path, progress=lambda stage: None) -> list[ScoreReport]:
        self.seen.append((upload.exists(), workdir))
        (workdir / "neutral.mp4").write_bytes(b"x")
        progress("scoring")
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        reports = []
        for i, probabilities in enumerate(self.parts or [self.probabilities]):
            checks = {k: CheckResult(v, None if v else "not checked") for k, v in probabilities.items()}
            reports.append(ScoreReport(checks, {"loudness_db": [], "windows": []}, start=3.0 * i, end=3.0 * i + 3))
        return reports


def _upload(tmp_path: Path, name: str = "clip.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"video")
    return path


def _finished(detector: Detector, job: str) -> dict:
    detector.wait(job, timeout=5)
    return detector.status(job)


def test_a_finished_job_reports_the_outcome_and_deletes_every_file(tmp_path) -> None:
    scorer = FakeScorer({"face": (0.999, 0.999, 0.999), "voice": (0.5,) * 3, "sync": None})
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work")
    upload = _upload(tmp_path)
    status = _finished(detector, detector.submit(upload))
    assert status["status"] == "done"
    assert status["outcome"]["kind"] == "signs"
    assert status["outcome"]["raised"] == ["face"]
    assert status["outcome"]["checks"]["sync"] == {"percentile": None, "reason": "not checked"}
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


def test_an_internal_error_is_reported_without_details_and_still_deletes(tmp_path) -> None:
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
    detector = Detector(lambda: scorer, _calibration(), workroot=tmp_path / "work", waiting=3)
    jobs = [detector.submit(_upload(tmp_path, f"c{i}.mp4")) for i in range(4)]
    busy = _upload(tmp_path, "c4.mp4")
    with pytest.raises(Detector.Busy):
        detector.submit(busy)
    assert not busy.exists()
    gate.set()
    assert all(_finished(detector, job)["status"] == "done" for job in jobs)


def test_an_unknown_job_has_no_status(tmp_path) -> None:
    detector = Detector(lambda: FakeScorer(), _calibration(), workroot=tmp_path / "work")
    assert detector.status("nope") is None


def test_warm_loads_the_scorer_once_before_any_upload(tmp_path) -> None:
    made = []
    detector = Detector(lambda: made.append(1) or FakeScorer(), _calibration(), workroot=tmp_path / "work")
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
    detector = Detector(lambda: FakeScorer(parts=[empty, empty]), _calibration(), workroot=tmp_path / "work")
    outcome = _finished(detector, detector.submit(_upload(tmp_path)))["outcome"]
    assert outcome["kind"] == "cant_check" and "not checked" in outcome["reason"]
