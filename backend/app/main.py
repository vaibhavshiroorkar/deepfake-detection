"""FastAPI entrypoint for the deepfake detection service."""
from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from .auth import Identity, resolve_identity
from .db import insert_scan, is_configured as db_is_configured
from .detectors import analyze_audio, analyze_image, analyze_text, analyze_video

log = logging.getLogger("veritas.main")

MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB (HF Spaces gateway caps larger uploads)
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_TEXT_CHARS = 50_000

# Rate limits. Anonymous users share an IP-based bucket. Signed-in users
# get a separate higher-ceiling bucket keyed on user id.
ANON_LIMIT = os.getenv("VERITAS_ANON_RATE", "10/minute")
AUTH_LIMIT = os.getenv("VERITAS_AUTH_RATE", "30/minute")

# Origins are read from the comma-separated ALLOWED_ORIGINS env var.
# Default stays permissive to preserve backward compatibility. Flip
# VERITAS_STRICT_CORS=1 and the wildcard is refused at startup, forcing
# an explicit origin list. The loud warning stays on either way so the
# wide-open default doesn't quietly ship to prod.
_RAW_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()
_STRICT_CORS = os.getenv("VERITAS_STRICT_CORS", "0") == "1"

_ALLOWED_ORIGINS: list[str]
if _RAW_ORIGINS == "*":
    if _STRICT_CORS:
        log.error(
            "VERITAS_STRICT_CORS=1 but ALLOWED_ORIGINS is still '*'. "
            "Refusing all origins. Set ALLOWED_ORIGINS to your frontend URL."
        )
        _ALLOWED_ORIGINS = []
    else:
        log.warning(
            "CORS is wide open (ALLOWED_ORIGINS=*). Safe for dev. "
            "Set ALLOWED_ORIGINS to your production frontend URL(s) and "
            "VERITAS_STRICT_CORS=1 for prod."
        )
        _ALLOWED_ORIGINS = ["*"]
else:
    _ALLOWED_ORIGINS = [o.strip() for o in _RAW_ORIGINS.split(",") if o.strip()]


def _rate_limit_key(request: Request) -> str:
    ident = getattr(request.state, "identity", None)
    if ident and getattr(ident, "user_id", None):
        return f"user:{ident.user_id}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


app = FastAPI(
    title="Veritas Deepfake Detection",
    description="Forensic signal extraction for images, video, audio, and text.",
    version="1.0.0",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                "Too many requests. Please wait a moment before trying again."
            ),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "db": db_is_configured()}


async def _persist(result: dict, ident: Identity, kind: str, filename: str | None) -> dict:
    if not ident.user_id or not db_is_configured():
        return result
    row = {
        "user_id": ident.user_id,
        "api_key_id": ident.api_key_id,
        "kind": kind,
        "filename": filename,
        "suspicion": float(result.get("suspicion", 0.0)),
        "verdict": str(result.get("verdict", "")),
        "confidence": float(result.get("confidence", 0.0)),
        "signals": result.get("signals", []),
        # Strip heatmaps before persisting. They are heavy and reproducible.
        "result": {k: v for k, v in result.items() if k != "heatmaps"},
    }
    scan_id = await insert_scan(row)
    if scan_id:
        result["id"] = scan_id
    return result


def _detect_limit(request: Request) -> str:
    ident = getattr(request.state, "identity", None)
    return AUTH_LIMIT if (ident and getattr(ident, "user_id", None)) else ANON_LIMIT


@app.post("/api/detect/image")
@limiter.limit(_detect_limit)
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
    request.state.identity = ident
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit.")
    try:
        result = analyze_image(data, filename=file.filename or "image")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze image: {e}")
    return await _persist(result, ident, "image", file.filename)


@app.post("/api/detect/video")
@limiter.limit(_detect_limit)
async def detect_video(
    request: Request,
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
    request.state.identity = ident
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, f"Video exceeds {MAX_VIDEO_BYTES // (1024*1024)} MB limit.")
    try:
        result = analyze_video(data, filename=file.filename or "video")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze video: {e}")
    return await _persist(result, ident, "video", file.filename)


@app.post("/api/detect/audio")
@limiter.limit(_detect_limit)
async def detect_audio(
    request: Request,
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
    request.state.identity = ident
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Audio exceeds {MAX_AUDIO_BYTES // (1024*1024)} MB limit.")
    try:
        result = analyze_audio(data, filename=file.filename or "audio")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze audio: {e}")
    return await _persist(result, ident, "audio", file.filename)


@app.post("/api/detect/text")
@limiter.limit(_detect_limit)
async def detect_text(
    request: Request,
    payload: TextPayload,
    ident: Identity = Depends(resolve_identity),
):
    request.state.identity = ident
    try:
        result = analyze_text(payload.text)
    except Exception as e:
        raise HTTPException(422, f"Could not analyze text: {e}")
    return await _persist(result, ident, "text", None)
