"""The upload queue: one clip scored at a time, a few waiting, every file deleted after.

A visitor's upload becomes a job. One worker thread scores jobs in order, so the
models are loaded once and two uploads never compete for the CPU. The page polls
the job's status, which names the stage it is in. The upload and every derived
file are deleted before the result becomes visible, and on any error.
"""

from __future__ import annotations

import logging
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from deepfake_detection.webapp import provenance
from deepfake_detection.webapp.calibration import (
    SHOWN_FLAG,
    SHOWN_LOWER,
    Calibration,
    display_score,
)
from deepfake_detection.webapp.scorer import ClipRefused

log = logging.getLogger(__name__)
KEEP_SECONDS = 15 * 60  # a finished job's result stays readable this long


class Detector:
    class Busy(RuntimeError):
        """Too many uploads are waiting."""

    def __init__(
        self,
        make_scorer: Callable[[], object],
        calibration: Calibration,
        *,
        workroot: Path | None = None,
        waiting: int = 3,
        read_labels: Callable[[Path], list[str]] | None = None,
        calibrations: dict[str, Calibration] | None = None,
    ) -> None:
        self._make_scorer, self._scorer = make_scorer, None
        self._read_labels = read_labels or provenance.read
        # An input kind with its own reference: a still picture is not judged against moving video.
        self._calibrations = calibrations or {}
        self.calibration = calibration
        self._workroot = Path(workroot or tempfile.gettempdir()) / "deepfake-check"
        self._capacity = 1 + waiting
        self._jobs: dict[str, dict] = {}
        self._done: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, Path, str]] = queue.Queue()
        threading.Thread(target=self._work, name="detector", daemon=True).start()

    def warm(self) -> None:
        """Load the models now, so the first upload is not slow. Safe to call more than once."""
        with self._load_lock:
            if self._scorer is None:
                self._scorer = self._make_scorer()

    def submit(self, upload: Path, mode: str = "video") -> str:
        """Queue an upload the server has already checked against the clip rules. Takes ownership of the file."""
        with self._lock:
            self._expire()
            active = sum(
                job["status"] in ("waiting", "running") for job in self._jobs.values()
            )
            if active >= self._capacity:
                upload.unlink(missing_ok=True)
                raise Detector.Busy("busy")
            job = uuid.uuid4().hex
            self._jobs[job] = {
                "status": "waiting",
                "stage": "received",
                "finished": None,
            }
            self._done[job] = threading.Event()
        self._queue.put((job, upload, mode))
        return job

    def status(self, job: str) -> dict | None:
        with self._lock:
            state = self._jobs.get(job)
            if state is None:
                return None
            ahead = (
                sum(
                    1
                    for other, s in self._jobs.items()
                    if s["status"] == "waiting" and other < job
                )
                if state["status"] == "waiting"
                else 0
            )
            return {k: v for k, v in state.items() if k != "finished"} | (
                {"ahead": ahead} if ahead else {}
            )

    def wait(self, job: str, timeout: float | None = None) -> bool:
        event = self._done.get(job)
        return event.wait(timeout) if event else False

    def _set(self, job: str, **values) -> None:
        with self._lock:
            self._jobs[job].update(values)

    def _expire(self) -> None:
        now = time.monotonic()
        for job in [
            j
            for j, s in self._jobs.items()
            if s["finished"] and now - s["finished"] > KEEP_SECONDS
        ]:
            del self._jobs[job]
            self._done.pop(job, None)

    def _work(self) -> None:
        while True:
            job, upload, mode = self._queue.get()
            self._set(job, status="running", stage="cleaning")
            workdir = self._workroot / job
            result: dict
            try:
                workdir.mkdir(parents=True, exist_ok=True)
                self._set(job, stage="file check")
                labels = self._read_labels(upload)
                self.warm()
                report = self._scorer.score(
                    upload,
                    workdir,
                    progress=lambda stage, job=job: self._set(job, stage=stage),
                    mode=mode,
                )
                result = {
                    "status": "done",
                    "outcome": self._outcome(report, labels, mode) | {"input": mode},
                }
            except ClipRefused as error:
                result = {
                    "status": "done",
                    "outcome": {"kind": "cant_check", "reason": str(error)},
                }
            except Exception:  # noqa: BLE001 - any failure must still delete the files and answer the page
                log.exception("scoring job %s failed", job)
                result = {
                    "status": "error",
                    "reason": "The check failed on our side. Please try again later.",
                }
            finally:
                upload.unlink(missing_ok=True)
                shutil.rmtree(workdir, ignore_errors=True)
            self._set(job, **result, stage="done", finished=time.monotonic())
            self._done[job].set()

    def _outcome(self, reports, labels: list[str], mode: str = "video") -> dict:
        """One outcome from the file check and the reports of every part; the part that decided supplies the details.

        An AI label in the file is decisive on its own: the answer is "signs", with "file" named first.
        """
        calibration = self._calibrations.get(mode, self.calibration)
        decision = calibration.decide_parts([r.scores() for r in reports])
        checked = sum(p is not None for p in decision.parts)
        flag, lower = calibration.limits(max(checked, 1))
        # Every part's scores, so the page can colour a timeline and show the score under the cursor.
        timeline = []
        for r, reading in zip(reports, decision.parts, strict=True):
            per_model = calibration.percentiles(r.scores())
            timeline.append(
                {
                    "start": r.start,
                    "end": r.end,
                    "reading": display_score(reading, flag, lower),
                    "checks": {
                        k: display_score(v, flag, lower) for k, v in per_model.items()
                    },
                }
            )
        report = reports[decision.part if decision.part is not None else 0]
        checks = {
            name: {
                "percentile": decision.percentiles[name],
                "score": display_score(decision.percentiles[name], flag, lower),
                "reason": result.reason,
            }
            for name, result in report.checks.items()
        }
        outcome = {
            "kind": decision.kind,
            "combined": decision.combined,
            "raised": decision.raised,
            "checks": checks,
            "trace": report.trace,
            "parts": len(reports),
            "part": {"start": report.start, "end": report.end},
            "file_check": labels,
            "timeline": timeline,
            "lines": {"flag": SHOWN_FLAG, "lower": SHOWN_LOWER},
            "raw_lines": {"flag": flag, "lower": lower},
            "explain": _face_explanation(reports),
        }
        if labels:
            outcome["kind"], outcome["raised"] = "signs", ["file", *decision.raised]
        if decision.kind == "cant_check":
            outcome["reason"] = (
                "Neither the face nor the voice could be checked: "
                + "; ".join(r for r in (c["reason"] for c in checks.values()) if r)
            )
        return outcome


def _face_explanation(reports) -> dict:
    """The heatmaps of the part where the face model scored highest: the face evidence the visitor should see."""
    scored = [
        r
        for r in reports
        if r.explain and r.checks.get("face") and r.checks["face"].score is not None
    ]
    if not scored:
        return {}
    return max(scored, key=lambda r: r.checks["face"].score).explain
