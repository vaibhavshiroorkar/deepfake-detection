"""The public site: pages rendered from templates, with every number read from numbers.json.

The design is `docs/DESIGN.md`; the structure is the 2026-11-17 website design spec. The P-001
detector is not connected yet, so an upload is checked against the clip rules, deleted, and
answered honestly as "not scored yet" rather than with an invented result.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from deepfake_detection.paths import PROJECT_ROOT

NUMBERS = PROJECT_ROOT / "results" / "numbers.json"
TEMPLATES = Path(__file__).parent / "templates" / "site"
SEEDS = (17, 29, 43)

MAX_UPLOAD_MB = 90
MIN_SECONDS, MAX_SECONDS = 3.0, 60.0
EXTENSIONS = {".mp4", ".mov", ".webm"}
CONTACT_LIMIT = 5                    # messages per address per hour
RESEND_URL = "https://api.resend.com/emails"

PAGES = {
    "/": ("home.html", "Check a video"),
    "/check": ("check.html", "Check a video"),
    "/how-it-works": ("how.html", "How it works"),
    "/research": ("research.html", "Research"),
    "/about": ("about.html", "About us"),
    "/contact": ("contact.html", "Contact"),
}


@lru_cache(maxsize=1)
def _numbers(path: Path = NUMBERS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(key: str, digits: int = 3) -> str:
    """One value from numbers.json, formatted. A missing key raises: a page must not invent one."""
    value = _numbers()[key]["value"]
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def seed_range(pattern: str, digits: int = 3) -> str:
    """'0.780 to 0.810' over the three seeds of a key pattern with a {} for the seed."""
    values = [_numbers()[pattern.format(seed)]["value"] for seed in SEEDS]
    low, high = min(values), max(values)
    return f"{low:.{digits}f}" if low == high else f"{low:.{digits}f} to {high:.{digits}f}"


def percent(key: str) -> str:
    """A rate as a whole percentage: 0.057 -> '6%'."""
    return f"{round(_numbers()[key]['value'] * 100)}%"


def percent_range(pattern: str) -> str:
    values = [round(_numbers()[pattern.format(seed)]["value"] * 100) for seed in SEEDS]
    low, high = min(values), max(values)
    return f"{low}%" if low == high else f"{low} to {high}%"


def _environment() -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(),
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.globals.update(n=number, seeds=seed_range, pct=percent, pct_range=percent_range)
    return env


def ceilings() -> list[dict]:
    """File-cue ceiling per corpus, raw and after cleaning, for the dumbbell chart."""
    rows = [("FakeAVCeleb", "fakeavceleb"), ("HiFi-AVDF", "hifiavdf"), ("LAV-DF", "lavdf"), ("DFDC", "dfdc")]
    return [{"name": name, "raw": _numbers()[f"audit.{key}.raw.clip"]["value"],
             "clean": _numbers()[f"audit.{key}.neutral.clip"]["value"]} for name, key in rows]


# Upload checks ------------------------------------------------------------------------------------

class ClipRejected(ValueError):
    """The upload breaks a clip rule; the message is shown to the visitor."""


def probe(path: Path) -> dict:
    """Streams and duration from ffprobe, bounded in time so a hostile file cannot hang the server."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=20, check=False)
    if result.returncode != 0:
        raise ClipRejected("This file could not be read as a video.")
    return json.loads(result.stdout or "{}")


def check_clip_rules(info: dict) -> float:
    """Apply the clip rules to ffprobe's output; return the duration in seconds."""
    kinds = [s.get("codec_type") for s in info.get("streams", [])]
    if kinds.count("video") != 1:
        raise ClipRejected("The file needs exactly one video stream.")
    if kinds.count("audio") > 1:
        raise ClipRejected("The file has more than one audio track.")
    try:
        duration = float(info.get("format", {}).get("duration", "nan"))
    except ValueError as error:
        raise ClipRejected("The video's length could not be read.") from error
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise ClipRejected(f"The video must be {MIN_SECONDS:.0f} to {MAX_SECONDS:.0f} seconds long; this one is {duration:.1f} s.")
    return duration


# Contact ------------------------------------------------------------------------------------------

EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RateLimiter:
    """At most `limit` events per key in a sliding hour. In memory: the site runs as one process."""

    def __init__(self, limit: int, window: float = 3600.0) -> None:
        self.limit, self.window = limit, window
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            events = self._events[key]
            while events and now - events[0] > self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


def send_contact(name: str, email: str, message: str) -> None:
    """Send through Resend. Raises RuntimeError if the form is not configured or Resend refuses."""
    key, to = os.environ.get("RESEND_API_KEY"), os.environ.get("CONTACT_TO")
    if not key or not to:
        raise RuntimeError("not configured")
    sender = os.environ.get("CONTACT_FROM", "Deepfake Check <onboarding@resend.dev>")
    response = httpx.post(RESEND_URL, timeout=15, headers={"Authorization": f"Bearer {key}"}, json={
        "from": sender, "to": [to], "reply_to": email,
        "subject": f"Deepfake Check: message from {name}",
        "text": f"From: {name} <{email}>\n\n{message}",
    })
    if response.status_code >= 300:
        raise RuntimeError(f"Resend answered {response.status_code}")


def register(app: FastAPI) -> None:
    env = _environment()
    contact_limit = RateLimiter(CONTACT_LIMIT)

    @app.middleware("http")
    async def revalidate_html(request: Request, call_next):
        # The old "/" was a static file with no Cache-Control, and browsers kept showing it from cache
        # after "/" became this site. Pages change with every result, so the browser always asks.
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def render(template: str, title: str, path: str, **extra) -> HTMLResponse:
        page = env.get_template(template).render(title=title, path=path, ceilings=ceilings(), **extra)
        return HTMLResponse(page)

    def page_for(template: str, title: str, route: str):
        # A factory, not default arguments: FastAPI would read those as query parameters.
        def page() -> HTMLResponse:
            return render(template, title, route)
        return page

    for route, (template, title) in PAGES.items():
        app.add_api_route(route, page_for(template, title, route), methods=["GET"],
                          response_class=HTMLResponse, include_in_schema=False)

    @app.post("/api/check")
    async def check(file: UploadFile = File(...)) -> JSONResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in EXTENSIONS:
            return JSONResponse({"status": "rejected", "reason": "Use an MP4, MOV or WebM file."}, status_code=400)
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = Path(handle.name)
        try:
            size = 0
            with handle:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        return JSONResponse({"status": "rejected", "reason": f"The file is over {MAX_UPLOAD_MB} MB."},
                                            status_code=413)
                    handle.write(chunk)
            duration = check_clip_rules(probe(path))
        except ClipRejected as error:
            return JSONResponse({"status": "rejected", "reason": str(error)}, status_code=400)
        except subprocess.TimeoutExpired:
            return JSONResponse({"status": "rejected", "reason": "Reading the file took too long."}, status_code=400)
        finally:
            path.unlink(missing_ok=True)
        return JSONResponse({
            "status": "not_scored",
            "duration": round(duration, 1),
            "reason": "The clip passed the upload checks and was deleted. The P-001 detector is being "
                      "connected, so it was not scored yet.",
        })

    @app.post("/contact", response_class=HTMLResponse)
    def contact(request: Request, name: str = Form(""), email: str = Form(""), message: str = Form(""),
                website: str = Form("")) -> HTMLResponse:
        title, path = PAGES["/contact"][1], "/contact"
        if website:                      # a hidden field only bots fill in
            return render("contact.html", title, path, sent=True)
        name, email, message = name.strip()[:120], email.strip()[:200], message.strip()[:4000]
        if not name or not EMAIL.match(email) or len(message) < 10:
            return render("contact.html", title, path, error="Please give your name, a valid email and a message.",
                          form={"name": name, "email": email, "message": message})
        client = request.client.host if request.client else "unknown"
        if not contact_limit.allow(client):
            return render("contact.html", title, path, error="Too many messages from here. Try again in an hour.")
        try:
            send_contact(name, email, message)
        except (RuntimeError, httpx.HTTPError) as error:
            reason = "The form is not connected yet." if str(error) == "not configured" else "Sending failed."
            return render("contact.html", title, path, error=f"{reason} Please try again later.",
                          form={"name": name, "email": email, "message": message})
        return render("contact.html", title, path, sent=True)
