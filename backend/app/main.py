"""FastAPI entrypoint for the deepfake detection service."""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .detectors import analyze_audio, analyze_image, analyze_text, analyze_video

MAX_IMAGE_BYTES = 25 * 1024 * 1024   # 25 MB
MAX_VIDEO_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_TEXT_CHARS = 50_000

app = FastAPI(
    title="Veritas — Deepfake Detection",
    description="Forensic signal extraction for images, video, and text.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB limit.")
    try:
        return analyze_image(data, filename=file.filename or "image")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze image: {e}")


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(413, f"Video exceeds {MAX_VIDEO_BYTES // (1024*1024)} MB limit.")
    try:
        return analyze_video(data, filename=file.filename or "video")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze video: {e}")


@app.post("/api/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, f"Audio exceeds {MAX_AUDIO_BYTES // (1024*1024)} MB limit.")
    try:
        return analyze_audio(data, filename=file.filename or "audio")
    except Exception as e:
        raise HTTPException(422, f"Could not analyze audio: {e}")


@app.post("/api/detect/text")
async def detect_text(payload: TextPayload):
    try:
        return analyze_text(payload.text)
    except Exception as e:
        raise HTTPException(422, f"Could not analyze text: {e}")
