"""The still check, the face check's first source, on one still of the upload.

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

COST = 10  # operations per still for the two models (measured 11-19)
DAY_LIMIT, MONTH_LIMIT = 480, 1900  # plan: 500 a day, 2,000 a month
PROJECT_ROOT = Path(__file__).resolve().parents[3]
USAGE = PROJECT_ROOT / "runs" / "still-check" / "usage.json"
CACHE = PROJECT_ROOT / "runs" / "still-check" / "cache.json"
CACHE_SIZE = 5000  # answers kept, newest first; one entry is about 80 bytes
TIMEOUT = 12  # seconds; past this our face model answers instead
MAX_SIDE = 1024  # pixels; a smaller still uploads faster and costs the same
log = logging.getLogger(__name__)


def settings() -> dict[str, str] | None:
    """The service's address, models and keys, from the environment (.env); None when any is missing."""
    names = {
        "url": "STILL_CHECK_URL",
        "models": "STILL_CHECK_MODELS",
        "user": "STILL_CHECK_USER",
        "secret": "STILL_CHECK_SECRET",
    }
    found = {k: os.environ.get(v, "").strip() for k, v in names.items()}
    return found if all(found.values()) else None


def enabled() -> bool:
    return settings() is not None


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


class Cache:
    """Answers by file fingerprint, so the same file is never paid for twice (re-uploads, demos, retries)."""

    _lock = threading.Lock()

    def __init__(self, path: Path = CACHE) -> None:
        self.path = path

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def get(self, key: str) -> dict | None:
        return self._load().get(key)

    def put(self, key: str, answer: dict) -> None:
        with self._lock:
            entries = self._load()
            entries.pop(key, None)
            entries = {key: answer, **dict(list(entries.items())[: CACHE_SIZE - 1])}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(entries), encoding="utf-8")


def fingerprint(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


_client: httpx.Client | None = None


def _post(*args, **kwargs):
    """One kept-alive connection for every call, instead of a new handshake each time."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=TIMEOUT)
    return _client.post(*args, **kwargs)


def ask(
    image: Path,
    post=_post,
    quota: Quota | None = None,
    cache: Cache | None = None,
    key: str | None = None,
) -> dict:
    """{"ai_generated": 0..1, "deepfake": 0..1} for one still, or {"error": reason} (the site then falls back).

    With `key` (the upload's fingerprint) a cached answer is returned without a call or any cost.
    """
    quota, cache = quota or Quota(), cache or Cache()
    if key is not None and (known := cache.get(key)) is not None:
        return known
    keys = settings()
    if keys is None:
        return {"error": "not set up"}
    if not quota.room(COST):
        return {"error": "allowance used up"}
    try:
        with open(image, "rb") as f:
            response = post(
                keys["url"],
                data={
                    "models": keys["models"],
                    "api_user": keys["user"],
                    "api_secret": keys["secret"],
                },
                files={"media": ("frame.jpg", f, "image/jpeg")},
                timeout=TIMEOUT,
            )
        reply = response.json()
    except (httpx.HTTPError, ValueError) as error:
        log.warning("The still check failed: %s", error)
        return {"error": "unreachable"}
    if reply.get("status") != "success":
        log.warning("The still check refused: %s", reply.get("error"))
        return {"error": "refused"}
    quota.record(int(reply.get("request", {}).get("operations", COST)))
    kinds = reply.get("type", {})
    answer = {
        "ai_generated": float(kinds.get("ai_generated", 0.0)),
        "deepfake": float(kinds.get("deepfake", 0.0)),
    }
    if key is not None:
        cache.put(key, answer)
    return answer


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
            "-vf",
            f"scale='min({MAX_SIDE},iw)':-2",
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
        key = fingerprint(upload)
        if (known := Cache().get(key)) is not None:
            return known  # no still, no call
        return ask(frame(upload, time, target), key=key)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        log.warning("The still check skipped: %s", error)
        return {"error": "no still"}
    finally:
        target.unlink(missing_ok=True)
