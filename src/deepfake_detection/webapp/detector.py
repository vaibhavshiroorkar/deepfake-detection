"""The upload queue: one clip scored at a time, a few waiting, every file deleted after.

A visitor's upload becomes a job. One worker thread scores jobs in order, so the
models are loaded once and two uploads never compete for the CPU. The page polls
the job's status, which names the stage it is in. The upload and every derived
file are deleted before the result becomes visible, and on any error.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from deepfake_detection.webapp import describe as describe_module
from deepfake_detection.webapp import provenance
from deepfake_detection.webapp.calibration import (
    SHOWN_FLAG,
    SHOWN_LOWER,
    Calibration,
    display_open,
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
        describe: Callable[[Path, float, dict], str | None] | None = None,
    ) -> None:
        self._make_scorer, self._scorer = make_scorer, None
        # (upload, time, explain) -> a plain description of the still and the heatmap, or None
        self._describe = describe or describe_module.describe
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
                outcome = self._outcome(report, labels, mode) | {"input": mode}
                if mode != "voice":
                    self._set(job, stage="describing")
                    # The description names the mapped area as suspicious, so it only gets the map
                    # when the answer flags the video; otherwise it describes the scene alone.
                    flagged = outcome.get("kind") == "signs"
                    text = self._describe_safely(
                        upload,
                        _describe_time(report),
                        (outcome.get("explain") or {}) if flagged else {},
                    )
                    if text:
                        outcome["description"] = text
                result = {"status": "done", "outcome": outcome}
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

    def _describe_safely(self, upload: Path, time: float, explain: dict) -> str | None:
        """The description is a courtesy: a failure is logged and the answer goes out without it."""
        try:
            return self._describe(upload, time, explain)
        except Exception:  # noqa: BLE001 - the checks' answer must not depend on the describer
            log.exception("describing the still failed")
            return None

    def _outcome(
        self,
        reports,
        labels: list[str],
        mode: str = "video",
    ) -> dict:
        """One outcome from the file check and the reports of every part; the part that decided supplies the details.

        An AI label in the file is decisive on its own: the answer is "signs", with "file" named first.
        """
        calibration = self._calibrations.get(mode, self.calibration)
        parts = [r.scores() for r in reports]
        decision = calibration.decide_parts(parts)
        checked = sum(p is not None for p in decision.parts)
        flag, lower = calibration.limits(max(checked, 1))  # each check's own bar
        read_flag, read_lower = calibration.reading_limits(
            max(checked, 1), decision.fused
        )  # the score that decides
        # The evidence sum is open-ended, so it is shown on a curve; a single check's percentile is not.
        shown = display_open if decision.fused else display_score
        # Every part's scores, so the page can colour a timeline and show the score under the cursor.
        timeline = []
        for r, scores, reading in zip(reports, parts, decision.parts, strict=True):
            per_model = calibration.percentiles(scores)
            timeline.append(
                {
                    "start": r.start,
                    "end": r.end,
                    "reading": shown(reading, read_flag, read_lower),
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
            # The main meter: the evidence sum of all checks on the page's scale.
            "overall": shown(decision.combined, read_flag, read_lower),
            "raw_lines": {"flag": flag, "lower": lower},
            "explain": _face_explanation(reports),
        }
        _highest_check_decides(outcome)
        if labels:
            outcome["kind"], outcome["raised"] = (
                "signs",
                ["file", *[c for c in outcome["raised"] if c != "file"]],
            )
        outcome["generator"] = _generator(reports, labels, outcome)
        if outcome["kind"] == "cant_check":
            outcome["reason"] = (
                "Neither the face nor the voice could be checked: "
                + "; ".join(
                    dict.fromkeys(
                        r for r in (c["reason"] for c in checks.values()) if r
                    )
                )
            )
        return outcome


# Below the flag line the overall is a weighted average of the checks (Vaibhav, 11-20). Lip sync counts half:
# it is the weakest check in the study. The weights are chosen, not fitted.
WEIGHTS = {"face": 1.0, "gen": 1.0, "voice": 1.0, "sync": 0.5}


def _reading(scores: dict[str, float]) -> float:
    """One number for a set of check scores: the highest when it reaches the flag line, else the weighted average."""
    top = max(scores.values())
    if top >= SHOWN_FLAG:
        return top
    weight = sum(WEIGHTS.get(k, 1.0) for k in scores)
    return round(sum(WEIGHTS.get(k, 1.0) * s for k, s in scores.items()) / weight, 3)


def _highest_check_decides(outcome: dict) -> None:
    """One check at the flag line flags on its own, and the overall shows it; otherwise the overall is the
    weighted average of the checks, so a single middling check among real-looking ones reads real.

    The evidence sum let several real-looking checks add up (three near 0.38 reached 0.64 and "signs"); it
    caught a few more fakes on unseen data (32% against 29% for the highest check, scripts/fusion_test.py),
    but an answer a viewer cannot trace to one check was judged worse (Vaibhav, 11-20). A plain average would
    hide one-sided fakes, such as a cloned voice on a real face, so the highest check still decides at the line.
    Timeline parts follow the same rule.
    """
    for part in outcome.get("timeline", []):
        own = {k: v for k, v in part["checks"].items() if v is not None}
        if own:
            part["reading"] = _reading(own)
    scores = {
        k: c["score"]
        for k, c in outcome["checks"].items()
        if c.get("score") is not None
    }
    if not scores or outcome["kind"] == "cant_check":
        return
    overall = _reading(scores)
    outcome["overall"] = overall
    outcome["raised"] = [k for k, s in scores.items() if s >= SHOWN_FLAG]
    outcome["kind"] = (
        "signs"
        if overall >= SHOWN_FLAG
        else "cant_tell"
        if overall >= SHOWN_LOWER
        else "no_signs"
    )


def _generator(reports, labels: list[str], outcome: dict) -> dict:
    """Which tool may have made the upload: the one a file label names, else the generated-video check's
    closest match among the generators it learned, given only when the upload looks generated."""
    for label in labels:
        named = re.search(r'"([^"]+)"', label)
        return {
            "suspected": True,
            "source": "label",
            "name": named.group(1) if named else None,
            "share": None,
            "sure": named is not None,
        }
    gen = outcome["checks"].get("gen") or {}
    looks = "gen" in outcome["raised"] or (gen.get("score") or 0.0) >= SHOWN_LOWER
    if not looks:
        return {"suspected": False}
    scored = [
        r
        for r in reports
        if r.checks.get("gen") is not None and r.checks["gen"].score is not None
    ]
    best = max(scored, key=lambda r: r.checks["gen"].score, default=None)
    guess = (best.checks["gen"].detail or {}).get("generator") if best else None
    if not guess:
        return {
            "suspected": True,
            "source": "model",
            "name": None,
            "share": None,
            "sure": False,
        }
    return {"suspected": True, "source": "model", **guess}


def _describe_time(reports) -> float:
    """Where to take the still to describe: the face frame our face model cited first,
    else the middle of the first part."""
    frames = (_face_explanation(reports) or {}).get("frames") or []
    if frames:
        return float(frames[0]["time"])
    return (reports[0].start + reports[0].end) / 2


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
