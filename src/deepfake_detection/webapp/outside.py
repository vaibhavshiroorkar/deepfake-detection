"""The face check's first source: an outside detection service on one still of the upload.

When the service is set up and today's allowance has room, the face check reads its scores for one still
(face swap and AI-made image); otherwise the site's own face model answers. Nothing else of the upload is
sent. The allowance file keeps use under the plan's daily and monthly limits, with a margin.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from datetime import date
from pathlib import Path

import httpx

API = "https://still-check.example/1.0/check.json"
MODELS = "genai,deepfake"
COST = 10  # operations per still for the two models (measured 11-19)
DAY_LIMIT, MONTH_LIMIT = 480, 1900  # plan: 500 a day, 2,000 a month
PROJECT_ROOT = Path(__file__).resolve().parents[3]
USAGE = PROJECT_ROOT / "runs" / "outside" / "usage.json"
log = logging.getLogger(__name__)


def credentials() -> tuple[str, str] | None:
    user, secret = (
        os.environ.get(k, "").strip()
        for k in ("STILL_CHECK_USER", "STILL_CHECK_SECRET")
    )
    return (user, secret) if user and secret else None


def enabled() -> bool:
    return credentials() is not None


class Quota:
    """Operations used today and this month, in a small JSON file, so a restart does not reset them."""

    _lock = threading.Lock()

    def __init__(self, path: Path = USAGE, today=date.today) -> None:
        self.path, self.today = path, today

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def used(self) -> dict[str, int]:
        data, now = self._load(), self.today()
        return {
            "day": data.get("day_ops", 0) if data.get("day") == now.isoformat() else 0,
            "month": data.get("month_ops", 0)
            if data.get("month") == now.strftime("%Y-%m")
            else 0,
        }

    def room(self, ops: int) -> bool:
        used = self.used()
        return used["day"] + ops <= DAY_LIMIT and used["month"] + ops <= MONTH_LIMIT

    def record(self, ops: int) -> None:
        with self._lock:
            used, now = self.used(), self.today()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {
                        "day": now.isoformat(),
                        "day_ops": used["day"] + ops,
                        "month": now.strftime("%Y-%m"),
                        "month_ops": used["month"] + ops,
                    }
                ),
                encoding="utf-8",
            )


def ask(image: Path, post=httpx.post, quota: Quota | None = None) -> dict:
    """{"ai_generated": 0..1, "deepfake": 0..1} for one still, or {"error": reason} (the site then falls back)."""
    quota = quota or Quota()
    keys = credentials()
    if keys is None:
        return {"error": "not set up"}
    if not quota.room(COST):
        return {"error": "allowance used up"}
    try:
        with open(image, "rb") as f:
            response = post(
                API,
                data={"models": MODELS, "api_user": keys[0], "api_secret": keys[1]},
                files={"media": ("frame.jpg", f, "image/jpeg")},
                timeout=30,
            )
        reply = response.json()
    except (httpx.HTTPError, ValueError) as error:
        log.warning("outside face check failed: %s", error)
        return {"error": "unreachable"}
    if reply.get("status") != "success":
        log.warning("outside face check refused: %s", reply.get("error"))
        return {"error": "refused"}
    quota.record(int(reply.get("request", {}).get("operations", COST)))
    kinds = reply.get("type", {})
    return {
        "ai_generated": float(kinds.get("ai_generated", 0.0)),
        "deepfake": float(kinds.get("deepfake", 0.0)),
    }


def frame(upload: Path, time: float, target: Path) -> Path:
    """One still of the upload at `time` seconds, as JPEG."""
    run = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{max(time, 0.0):.2f}",
            "-i",
            str(upload),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(target),
        ],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if run.returncode != 0 or not target.is_file():
        raise RuntimeError("could not take a still from the upload")
    return target


def face_opinion(upload: Path, time: float) -> dict:
    """The detector's hook: a still at `time`, sent once, then deleted."""
    if not enabled():
        return {"error": "not set up"}
    target = upload.with_name(upload.stem + "-still.jpg")
    try:
        return ask(frame(upload, time, target))
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        log.warning("outside face check skipped: %s", error)
        return {"error": "no still"}
    finally:
        target.unlink(missing_ok=True)
