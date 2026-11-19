"""An optional second opinion: the outside service's AI-generated image model on one frame, only when the visitor asks.

The site's own four checks decide the answer. This is shown beside them, labelled as an outside service
whose accuracy on our test sets is not yet measured. One still frame leaves the computer, and only for a
visitor who ticked the box; the page says so before they upload.

The free plan allows 2,000 operations a month and 500 a day; the AI-generated model costs 5 per image
(measured 11-19). The quota file keeps the site under both limits, with a margin.
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
MODEL = "genai"
COST = 5  # operations per image for the AI-generated model
DAY_LIMIT, MONTH_LIMIT = 480, 1900  # free plan: 500 a day, 2,000 a month
SERVICE = "the outside service"
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

    def left_today(self) -> int:
        used = self.used()
        return max(0, min(DAY_LIMIT - used["day"], MONTH_LIMIT - used["month"]) // COST)


def ask(image: Path, post=httpx.post, quota: Quota | None = None) -> dict:
    """{"ai_generated": 0..1, "service": ...} for one image, or {"error": reason} in plain words."""
    quota = quota or Quota()
    keys = credentials()
    if keys is None:
        return {"error": "The second opinion is not set up on this server."}
    if not quota.room(COST):
        return {
            "error": "Today's free allowance for second opinions is used up. Try again tomorrow."
        }
    try:
        with open(image, "rb") as f:
            response = post(
                API,
                data={"models": MODEL, "api_user": keys[0], "api_secret": keys[1]},
                files={"media": ("frame.jpg", f, "image/jpeg")},
                timeout=30,
            )
        reply = response.json()
    except (httpx.HTTPError, ValueError) as error:
        log.warning("second opinion failed: %s", error)
        return {"error": "The second opinion could not be reached."}
    if reply.get("status") != "success":
        log.warning("second opinion refused: %s", reply.get("error"))
        return {"error": "The second opinion could not be reached."}
    quota.record(int(reply.get("request", {}).get("operations", COST)))
    return {"ai_generated": float(reply["type"]["ai_generated"]), "service": SERVICE}


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


def second_opinion(upload: Path, time: float) -> dict:
    """The detector's hook: a still at `time`, sent once, then deleted."""
    target = upload.with_name(upload.stem + "-second.jpg")
    try:
        return ask(frame(upload, time, target))
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        log.warning("second opinion skipped: %s", error)
        return {"error": "No still could be taken from this file."}
    finally:
        target.unlink(missing_ok=True)
