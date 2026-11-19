"""The public site: pages rendered from templates, with every number read from numbers.json.

The design is `docs/DESIGN.md`; the structure is the 2026-11-17 website design spec. An upload
is checked against the clip rules, then queued for the P-001 detector (`detector.py`), which
scores it with the four checks (`scorer.py`), places the scores among real calibration clips
(`calibration.py`) and deletes every file before the result can be read.
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
from deepfake_detection.webapp import outside
from deepfake_detection.webapp.detector import Detector

NUMBERS = PROJECT_ROOT / "results" / "numbers.json"
TEMPLATES = Path(__file__).parent / "templates" / "site"
STATIC_SITE = Path(__file__).parent / "static" / "site"
SEEDS = (17, 29, 43)

MAX_UPLOAD_MB = 90
MIN_SECONDS, MAX_SECONDS = 3.0, 60.0
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS
MODES = ("video", "voice", "picture")
PICTURE_SECONDS = (
    3.2  # a still is looped into a clip just over the 3 s the cleaning keeps
)
CONTACT_LIMIT = 5  # messages per address per hour
UPLOAD_LIMIT = 10  # uploads per address per hour


def upload_limit() -> int:
    """Uploads per address per hour. DDF_UPLOAD_LIMIT raises it for a private demo, where every request can
    arrive through one local proxy and so share one address."""
    return int(os.environ.get("DDF_UPLOAD_LIMIT", UPLOAD_LIMIT))


CALIBRATION = PROJECT_ROOT / "results" / "web-calibration.json"
PICTURE_CALIBRATION = PROJECT_ROOT / "results" / "web-calibration-picture.json"
# Which face and voice models are live, and the calibration files measured with them. Without the file,
# or with DDF_WEB_MODELS=p001, the site runs the study's models and the two files above.
WEB_MODELS = PROJECT_ROOT / "results" / "web-models.json"
RESEND_URL = "https://api.resend.com/emails"

PAGES = {
    "/": ("home.html", "Home"),
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
    return (
        f"{low:.{digits}f}" if low == high else f"{low:.{digits}f} to {high:.{digits}f}"
    )


def percent(key: str) -> str:
    """A rate as a whole percentage: 0.057 -> '6%'."""
    return f"{round(_numbers()[key]['value'] * 100)}%"


def percent_range(pattern: str) -> str:
    values = [round(_numbers()[pattern.format(seed)]["value"] * 100) for seed in SEEDS]
    low, high = min(values), max(values)
    return f"{low}%" if low == high else f"{low} to {high}%"


def live_rates() -> dict:
    """Catch and false-alarm rates of the lines in force (results/web-models.json may swap models).

    FakeAVCeleb is the corpus the study's models trained on; the others are the unseen sources.
    """
    rates = {}
    pooled = {"caught": [0, 0], "flagged": [0, 0]}  # [signs, clips] over every corpus
    for corpus, sides in calibration().meta["outcomes"].items():
        for side, name in (("fake", "caught"), ("real", "flagged")):
            counts = sides.get(side, {})
            if sum(counts.values()):
                rates[corpus, name] = counts.get("signs", 0) / sum(counts.values())
                pooled[name][0] += counts.get("signs", 0)
                pooled[name][1] += sum(counts.values())

    def span(values: list[float]) -> tuple[str, str]:
        return f"{min(values):.0%}", f"{max(values):.0%}"

    flagged = [v for (c, n), v in rates.items() if n == "flagged"]
    return {
        "home_caught": f"{rates['FakeAVCeleb', 'caught']:.0%}",
        "other_caught": span(
            [v for (c, n), v in rates.items() if n == "caught" and c != "FakeAVCeleb"]
        ),
        "real_flagged": span(flagged),
        "real_flagged_mean": f"{100 * sum(flagged) / len(flagged):.0f}",
        "all_caught": f"{pooled['caught'][0] / pooled['caught'][1]:.0%}",
        "all_flagged": f"{pooled['flagged'][0] / pooled['flagged'][1]:.0%}",
        "fake_clips": f"{pooled['caught'][1]:,}",
        "real_clips": f"{pooled['flagged'][1]:,}",
    }


# What each row on the result card means, shown when a visitor opens it. "misses" comes from the
# measured catch rates (results/web-calibration-v2.json, raised_by) and the 11-19 model test.
CHECK_ABOUT = {
    "face": {
        "what": "Looks at 16 face crops from the part for traces a face swap leaves.",
        "high": "The face looks more unusual than almost all real faces we tested.",
        "misses": "Fully generated videos (such as Veo or Sora), which have no pasted face.",
    },
    "gen": {
        "what": "Looks at the face for signs that the whole video was made by an AI video generator, such as Veo, Sora or Kling.",
        "high": "The face looks more like generated video than almost all real faces we tested.",
        "misses": "Many videos from generators it has not learned, and face swaps (the face check covers those).",
    },
    "voice": {
        "what": "Listens to the speech for signs of a cloned or generated voice.",
        "high": "The voice sounds more unusual than almost all real voices we tested.",
        "misses": "Videos where only the face is fake and the voice is real.",
    },
    "sync": {
        "what": "Checks whether the lips move with the words. It learned from real clips only.",
        "high": "Lips and words fit together worse than in almost all real clips.",
        "misses": "Most fakes from sources it has not seen; it is the weakest of the three.",
    },
    "file": {
        "what": "Reads labels that AI tools write into a file on purpose, such as a C2PA content credential.",
        "high": "the file says it was made by AI, which is strong evidence on its own.",
        "misses": "Most fakes: uploading or re-saving a file removes its labels, so no label proves nothing.",
    },
}


def check_info() -> dict:
    """Each check's explanation, with the share of fakes it flagged on each test set (highest first)."""
    counts = calibration().meta.get("raised_by", {})
    about = {}
    for check, text in CHECK_ABOUT.items():
        rates = [
            (c[check] / c["fakes"], corpus)
            for corpus, c in counts.items()
            if check in c and c["fakes"]
        ]
        about[check] = text | {
            "caught": [
                [corpus, f"{rate:.0%}"] for rate, corpus in sorted(rates, reverse=True)
            ]
        }
    return about


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.update(
        n=number,
        seeds=seed_range,
        pct=percent,
        pct_range=percent_range,
        live=live_rates,
    )
    return env


def ceilings() -> list[dict]:
    """File-cue ceiling per corpus, raw and after cleaning, for the dumbbell chart."""
    rows = [
        ("FakeAVCeleb", "fakeavceleb"),
        ("HiFi-AVDF", "hifiavdf"),
        ("LAV-DF", "lavdf"),
        ("DFDC", "dfdc"),
    ]
    return [
        {
            "name": name,
            "raw": _numbers()[f"audit.{key}.raw.clip"]["value"],
            "clean": _numbers()[f"audit.{key}.neutral.clip"]["value"],
        }
        for name, key in rows
    ]


# Upload checks ------------------------------------------------------------------------------------


class ClipRejected(ValueError):
    """The upload breaks a clip rule; the message is shown to the visitor."""


def probe(path: Path) -> dict:
    """Streams and duration from ffprobe, bounded in time so a hostile file cannot hang the server."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
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
        raise ClipRejected(
            f"The video must be {MIN_SECONDS:.0f} to {MAX_SECONDS:.0f} seconds long; this one is {duration:.1f} s."
        )
    return duration


def classify(suffix: str, mode: str, info: dict) -> str:
    """What kind of input this is: a video, a voice recording or a picture.

    The page says which when it records (`mode`); a dropped file is judged by its name, and a WebM file,
    which can hold sound alone, by its streams. An input that lacks what its kind needs is refused.
    """
    kinds = [s.get("codec_type") for s in info.get("streams", [])]
    if suffix not in EXTENSIONS:
        raise ClipRejected(
            "Use a video (MP4, MOV, WebM), a voice recording (WAV, MP3, M4A, OGG) or a picture (JPG, PNG)."
        )
    if mode in MODES:
        kind = mode
    elif suffix in IMAGE_EXTENSIONS:
        kind = "picture"
    elif suffix in VIDEO_EXTENSIONS and "video" in kinds:
        kind = "video"
    else:
        kind = "voice"
    if kind == "voice" and "audio" not in kinds:
        raise ClipRejected("This recording has no sound.")
    if kind in ("video", "picture") and "video" not in kinds:
        raise ClipRejected("This file has no picture.")
    return kind


def to_clip(source: Path, kind: str, target: Path) -> Path:
    """A clip the checks can read: a voice over plain black frames, a picture held still, or a recording re-encoded.

    The black frames hold no face, so only the voice check runs on a voice; a still has no sound, so only the
    face check runs on a picture. Browser recordings are re-encoded because their WebM files often carry no length.
    """
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25"]
    if kind == "voice":
        command = [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x320:r=25",
            "-i",
            str(source),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-shortest",
            *common,
            "-c:a",
            "aac",
        ]
    elif kind == "picture":
        command = [
            "-loop",
            "1",
            "-i",
            str(source),
            "-t",
            str(PICTURE_SECONDS),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            *common,
            "-an",
        ]
    else:
        command = ["-i", str(source), *common, "-c:a", "aac"]
    run = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-nostdin", *command, str(target)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if run.returncode != 0 or not target.is_file():
        raise ClipRejected("This file could not be converted for checking.")
    return target


# Detector -----------------------------------------------------------------------------------------


def web_models() -> dict | None:
    """The live model choice (results/web-models.json), or None for the study's models."""
    if os.environ.get("DDF_WEB_MODELS") == "p001" or not WEB_MODELS.exists():
        return None
    return json.loads(WEB_MODELS.read_text(encoding="utf-8"))


def make_scorer():
    """The scorer. Loading takes about a minute: on the first upload, or at start with --preload-detector.

    CPU by default, the path checked against the study's scores; DDF_SCORER_DEVICE=cuda for a faster demo.
    With results/web-models.json the face and voice models it names replace the study's.
    """
    from deepfake_detection.webapp.scorer import Scorer

    device = os.environ.get("DDF_SCORER_DEVICE", "cpu")
    scorer = Scorer(device=device)
    choice = web_models()
    if choice is not None:
        from deepfake_detection.webapp.candidates import website_models

        scorer.replace = website_models(choice, PROJECT_ROOT, device)
    return scorer


def _calibration_file(kind: str) -> Path:
    choice = web_models()
    if choice is None:
        return PICTURE_CALIBRATION if kind == "picture" else CALIBRATION
    return PROJECT_ROOT / choice["calibration"][kind]


@lru_cache(maxsize=1)
def calibration():
    from deepfake_detection.webapp.calibration import Calibration

    return Calibration.from_json(_calibration_file("video").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def picture_calibration():
    """The face check's lines for still pictures (scripts/build_picture_calibration.py), or None if not built."""
    from deepfake_detection.webapp.calibration import Calibration

    path = _calibration_file("picture")
    if not path.exists():
        return None
    return Calibration.from_json(path.read_text(encoding="utf-8"))


def require_calibration(kind: str) -> None:
    """A picture needs its own calibration: judged against moving video, a real face looks unusual."""
    if kind == "picture" and picture_calibration() is None:
        raise ClipRejected(
            "Pictures cannot be checked yet: their calibration has not been built."
        )


@lru_cache(maxsize=1)
def detector():
    # Looked up at call time, so a test can replace make_scorer before the first job.
    pictures = picture_calibration()
    return Detector(
        lambda: make_scorer(),
        calibration(),
        calibrations={"picture": pictures} if pictures else None,
    )


CHECK_NAMES = {
    "face": "face",
    "gen": "generated-video",
    "voice": "voice",
    "sync": "lip-sync",
}


def how_sure(kind: str = "video") -> str:
    """The false-alarm and catch rates on the calibration sets, from the calibration file itself."""
    source = picture_calibration() if kind == "picture" else calibration()
    if source is None:
        return ""
    caught = []
    for corpus, sides in source.meta["outcomes"].items():
        fakes = sides.get("fake", {})
        if sum(fakes.values()):
            caught.append((fakes.get("signs", 0) / sum(fakes.values()), corpus))
    low, high = min(caught), max(caught)
    if kind == "picture":
        return (
            f"On still pictures from the four test sets, about 5 in 100 real pictures get this flag. The same "
            f"threshold catches {low[0]:.0%} of fake pictures on {low[1]} and {high[0]:.0%} on {high[1]}. The face "
            f"check learned from video, so treat a picture's answer as a rough guide."
        )
    return (
        f"On the four test sets we calibrated on, about 5 in 100 real clips get this flag. The same "
        f"threshold catches {low[0]:.0%} of fakes on {low[1]} and {high[0]:.0%} on {high[1]}. For a video from anywhere else, 5 in 100 is an estimate."
    )


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def where(outcome: dict) -> str:
    """Which part of the video the details come from, in one sentence; empty before any part was checked."""
    part, parts, kind = (
        outcome.get("part"),
        outcome.get("parts"),
        outcome.get("input", "video"),
    )
    if kind == "picture":
        return "A picture, checked by the face check only."
    if not part or not parts:
        return ""
    if kind == "voice":
        return "A voice recording, checked by the voice check only. " + where(
            outcome | {"input": "video"}
        ).replace("video", "recording")
    span = f"{_clock(part['start'])} to {_clock(part['end'])}"
    if parts == 1:
        return f"The video has 3 seconds of sound, checked as one part ({span})."
    return f"The whole video was checked in {parts} parts of 3 seconds. The most unusual part, {span}, is shown."


def present(outcome: dict) -> dict:
    """The detector's outcome in the shape the page draws: whole-number percentiles and plain sentences."""
    checks = {
        name: {
            "percentile": None
            if c["percentile"] is None
            else min(99, int(c["percentile"] * 100)),
            # The detector sends the 0 (real) / 0.5 (neutral) / 1 (fake) scale; older outcomes only a percentile.
            "score": c["score"]
            if "score" in c
            else (None if c["percentile"] is None else round(c["percentile"], 3)),
            "raised": name in outcome.get("raised", []),
            "reason": c["reason"],
        }
        for name, c in outcome.get("checks", {}).items()
    }
    kind = outcome["kind"]
    labels = outcome.get("file_check") or []
    if labels:
        summary = (
            f"The file carries {' and '.join(labels)}. AI tools write such labels on purpose, "
            "so this is strong evidence on its own."
        )
        checks_raised = [n for n in outcome["raised"] if n != "file"]
        if checks_raised:
            summary += (
                " The "
                + " and ".join(CHECK_NAMES[n] for n in checks_raised)
                + " checks also flagged it."
            )
    elif kind == "signs":
        words = [CHECK_NAMES[n] for n in outcome["raised"]]
        names = (
            words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
        )
        noun = "checks" if len(outcome["raised"]) > 1 else "check"
        summary = f"The {names} {noun} scored higher than almost every real clip we calibrated on."
    elif kind == "cant_tell":
        summary = "One check scored higher than most real clips, but not high enough to flag the video."
    elif kind == "no_signs":
        summary = "No check scored unusually high compared with real clips. That is not proof the video is real."
    else:
        summary = outcome.get("reason", "The video could not be checked.")
    return {
        "status": "done",
        "outcome": kind,
        "summary": summary,
        "where": where(outcome),
        "file_check": labels,
        "checks": checks,
        "timeline": outcome.get("timeline", []),
        "explain": outcome.get("explain", {}),
        "lines": outcome.get("lines"),
        "about": check_info(),
        "outside": outcome.get("outside"),
        "trace": outcome.get("trace", {}),
        "how_sure": how_sure(outcome.get("input", "video"))
        if kind != "cant_check"
        else "",
    }


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
    response = httpx.post(
        RESEND_URL,
        timeout=15,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "from": sender,
            "to": [to],
            "reply_to": email,
            "subject": f"Deepfake Check: message from {name}",
            "text": f"From: {name} <{email}>\n\n{message}",
        },
    )
    if response.status_code >= 300:
        raise RuntimeError(f"Resend answered {response.status_code}")


def render_page(template: str, title: str, path: str, **extra) -> str:
    """One page as HTML; the server sends it, and scripts/export_site.py writes it for the static frontend."""
    return (
        _environment()
        .get_template(template)
        .render(title=title, path=path, ceilings=ceilings(), **extra)
    )


def register(app: FastAPI) -> None:
    contact_limit = RateLimiter(CONTACT_LIMIT)
    uploads = RateLimiter(upload_limit())

    @app.middleware("http")
    async def revalidate_html(request: Request, call_next):
        # The old "/" was a static file with no Cache-Control, and browsers kept showing it from cache
        # after "/" became this site. Pages change with every result, so the browser always asks.
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def render(template: str, title: str, path: str, **extra) -> HTMLResponse:
        return HTMLResponse(render_page(template, title, path, **extra))

    def page_for(template: str, title: str, route: str):
        # A factory, not default arguments: FastAPI would read those as query parameters.
        def page() -> HTMLResponse:
            return render(template, title, route)

        return page

    for route, (template, title) in PAGES.items():
        app.add_api_route(
            route,
            page_for(template, title, route),
            methods=["GET"],
            response_class=HTMLResponse,
            include_in_schema=False,
        )

    @app.post("/api/check")
    async def check(
        request: Request,
        file: UploadFile = File(...),
        mode: str = Form(""),
        second: str = Form(""),
    ) -> JSONResponse:
        client = request.client.host if request.client else "unknown"
        if not uploads.allow(client):
            return JSONResponse(
                {
                    "status": "rejected",
                    "reason": "Too many videos from here. Try again in an hour.",
                },
                status_code=429,
            )
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in EXTENSIONS:
            return JSONResponse(
                {
                    "status": "rejected",
                    "reason": "Use a video (MP4, MOV, WebM), a voice recording "
                    "(WAV, MP3, M4A, OGG) or a picture (JPG, PNG).",
                },
                status_code=400,
            )
        converted = None
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        path = Path(handle.name)
        try:
            size = 0
            with handle:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        return JSONResponse(
                            {
                                "status": "rejected",
                                "reason": f"The file is over {MAX_UPLOAD_MB} MB.",
                            },
                            status_code=413,
                        )
                    handle.write(chunk)
            kind = classify(suffix, mode, probe(path))
            require_calibration(kind)
            if kind != "video" or suffix == ".webm":
                converted = to_clip(path, kind, path.with_name(path.stem + "-clip.mp4"))
                path.unlink(missing_ok=True)
                path, converted = converted, None
            check_clip_rules(probe(path))
            job = detector().submit(
                path, mode=kind, second=second == "1" and outside.enabled()
            )  # the detector owns the file from here and deletes it
        except ClipRejected as error:
            path.unlink(missing_ok=True)
            if converted is not None:
                converted.unlink(missing_ok=True)
            return JSONResponse(
                {"status": "rejected", "reason": str(error)}, status_code=400
            )
        except subprocess.TimeoutExpired:
            path.unlink(missing_ok=True)
            return JSONResponse(
                {"status": "rejected", "reason": "Reading the file took too long."},
                status_code=400,
            )
        except Detector.Busy:  # the detector already deleted the file
            return JSONResponse(
                {
                    "status": "busy",
                    "reason": "The detector is busy. Try again in a minute.",
                },
                status_code=503,
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return JSONResponse({"status": "queued", "job": job}, status_code=202)

    @app.get("/api/outside")
    def outside_status() -> JSONResponse:
        """Whether the page may offer the second opinion, and how many are left today."""
        on = outside.enabled()
        return JSONResponse(
            {"enabled": on, "left_today": outside.Quota().left_today() if on else 0}
        )

    @app.get("/api/check/{job}")
    def check_status(job: str) -> JSONResponse:
        state = detector().status(job) if re.fullmatch(r"[0-9a-f]{32}", job) else None
        if state is None:
            return JSONResponse(
                {
                    "status": "unknown",
                    "reason": "This check has expired or never existed.",
                },
                status_code=404,
            )
        if state["status"] == "done":
            return JSONResponse(present(state["outcome"]))
        return JSONResponse(state)

    @app.post("/contact", response_class=HTMLResponse)
    def contact(
        request: Request,
        name: str = Form(""),
        email: str = Form(""),
        message: str = Form(""),
        website: str = Form(""),
    ) -> HTMLResponse:
        title, path = PAGES["/contact"][1], "/contact"
        if website:  # a hidden field only bots fill in
            return render("contact.html", title, path, sent=True)
        name, email, message = (
            name.strip()[:120],
            email.strip()[:200],
            message.strip()[:4000],
        )
        if not name or not EMAIL.match(email) or len(message) < 10:
            return render(
                "contact.html",
                title,
                path,
                error="Please give your name, a valid email and a message.",
                form={"name": name, "email": email, "message": message},
            )
        client = request.client.host if request.client else "unknown"
        if not contact_limit.allow(client):
            return render(
                "contact.html",
                title,
                path,
                error="Too many messages from here. Try again in an hour.",
            )
        try:
            send_contact(name, email, message)
        except (RuntimeError, httpx.HTTPError) as error:
            reason = (
                "The form is not connected yet."
                if str(error) == "not configured"
                else "Sending failed."
            )
            return render(
                "contact.html",
                title,
                path,
                error=f"{reason} Please try again later.",
                form={"name": name, "email": email, "message": message},
            )
        return render("contact.html", title, path, sent=True)
