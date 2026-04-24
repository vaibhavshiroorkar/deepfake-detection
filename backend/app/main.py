"""FastAPI entrypoint for the deepfake detection service."""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import Identity, resolve_identity
from .db import insert_scan, is_configured as db_is_configured
from .detectors import analyze_audio, analyze_image, analyze_text, analyze_video

MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB (HF Spaces gateway caps larger uploads)
MAX_AUDIO_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_TEXT_CHARS = 50_000

# Comma-separated list, or "*" for any (default during dev).
_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app = FastAPI(
    title="Veritas — Deepfake Detection",
    description="Forensic signal extraction for images, video, audio, and text.",
    version="1.0.0",
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
        # Strip heatmaps before persisting — they are heavy and reproducible.
        "result": {k: v for k, v in result.items() if k != "heatmaps"},
    }
    scan_id = await insert_scan(row)
    if scan_id:
        result["id"] = scan_id
    return result


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
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
async def detect_video(
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
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
async def detect_audio(
    file: UploadFile = File(...),
    ident: Identity = Depends(resolve_identity),
):
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
async def detect_text(
    payload: TextPayload,
    ident: Identity = Depends(resolve_identity),
):
    try:
        result = analyze_text(payload.text)
    except Exception as e:
        raise HTTPException(422, f"Could not analyze text: {e}")
    return await _persist(result, ident, "text", None)
