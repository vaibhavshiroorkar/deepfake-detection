"""The inference library: one function, file in, response out (D-027).

    analyse_file(path, filename, adapter) -> dict   # the schema-version-1 response

The library inspects and validates the input (type, readability, duration, resolution, audio track), calls the adapter,
fuses the modalities and builds the response. It imports nothing from the web code. Fusion rule used now (D-040 item 6):
an image or audio file gets the score of its only modality; a video gets NO fused score (decision "undetermined") until
the combining rule O-009 is decided, and its visual and audio scores are returned separately.
"""
from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from dfd.infer.adapters import Adapter, Media
from dfd.infer.schema import SCHEMA_VERSION, SCORE_MEANING

log = logging.getLogger("dfd.infer")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
SHORT_AUDIO_S = 1.0
LOW_RES_PX = 64


class InputError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def media_type_of(filename: str) -> Optional[str]:
    ext = Path(filename).suffix.lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else "audio" if ext in AUDIO_EXT else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _ffprobe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise InputError("UNREADABLE_FILE", "the file could not be read as media: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:200])
    return json.loads(r.stdout or "{}")


def inspect(path: Path, filename: str) -> Media:
    """Validate the file and describe it. Raises InputError with a response error code."""
    mt = media_type_of(filename)
    if not path.is_file():
        raise InputError("NO_FILE", "no file was received")
    size = path.stat().st_size
    if size == 0:
        raise InputError("EMPTY_FILE", "the file is empty")
    if mt is None:
        raise InputError("UNSUPPORTED_MEDIA_TYPE", f"unsupported file type {Path(filename).suffix or '(none)'}; accepted: "
                         + ", ".join(sorted(IMAGE_EXT | VIDEO_EXT | AUDIO_EXT)))
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    sha = _sha256(path)
    if mt == "image":
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise InputError("UNREADABLE_FILE", "the image could not be decoded")
        h, w = img.shape[:2]
        return Media(path, filename, mt, mime, size, sha, None, False, w, h)
    info = _ffprobe(path)
    streams = info.get("streams", [])
    a = [s for s in streams if s.get("codec_type") == "audio"]
    v = [s for s in streams if s.get("codec_type") == "video"]
    dur = info.get("format", {}).get("duration")
    dur = float(dur) if dur not in (None, "N/A") else None
    if mt == "audio":
        if not a:
            raise InputError("UNREADABLE_FILE", "the file has no audio stream")
        return Media(path, filename, mt, mime, size, sha, dur, True, None, None)
    if not v:
        raise InputError("UNREADABLE_FILE", "the file has no video stream")
    return Media(path, filename, mt, mime, size, sha, dur, bool(a), int(v[0].get("width", 0)) or None, int(v[0].get("height", 0)) or None)


def _input_block(filename: str, size: int = 0, sha: str = "", mt: Optional[str] = None, mime: str = "", dur=None, frames=None) -> dict:
    return {"filename": filename, "media_type": mt, "mime_type": mime, "size_bytes": size, "sha256": sha,
            "duration_s": None if dur is None else round(dur, 3), "frames_used": frames}


def failed_response(source: str, filename: str, code: str, message: str, request_id: Optional[str] = None,
                    media: Optional[Media] = None, stages: Optional[list] = None, t0: Optional[float] = None) -> dict:
    inp = _input_block(filename) if media is None else _input_block(filename, media.size_bytes, media.sha256, media.media_type, media.mime_type, media.duration_s)
    return {"schema_version": SCHEMA_VERSION, "request_id": request_id or str(uuid.uuid4()), "input": inp, "inference_source": source,
            "model": None, "result": None, "score_meaning": SCORE_MEANING,
            "status": {"state": "failed", "processing_time_ms": round(((time.perf_counter() - t0) if t0 else 0.0) * 1000, 1), "stages": stages or []},
            "warnings": [], "errors": [{"code": code, "message": message}]}


def analyse_file(path: str | Path, filename: str, adapter: Adapter, progress: Optional[Callable[[str], None]] = None,
                 request_id: Optional[str] = None) -> dict:
    t0 = time.perf_counter()
    tick = progress or (lambda stage: None)
    rid = request_id or str(uuid.uuid4())
    src = adapter.inference_source
    stages: list[dict] = []
    path = Path(path)

    t = time.perf_counter()
    tick("validate")
    try:
        media = inspect(path, filename)
    except InputError as e:
        stages.append({"name": "validate", "ms": round((time.perf_counter() - t) * 1000, 1)})
        return failed_response(src, filename, e.code, e.message, rid, None, stages, t0)
    stages.append({"name": "validate", "ms": round((time.perf_counter() - t) * 1000, 1)})
    stages.append({"name": "decode", "ms": 0.0})  # decoding for scoring happens inside the adapter

    warnings: list[dict] = []
    if media.media_type in ("image", "video") and media.width and media.height and min(media.width, media.height) < LOW_RES_PX:
        warnings.append({"code": "LOW_RESOLUTION", "message": f"the smallest side is {min(media.width, media.height)} px; faces may be too small to score reliably.",
                         "modality": "image" if media.media_type == "image" else "visual"})
    if media.media_type == "audio" or (media.media_type == "video" and media.has_audio):
        if media.duration_s is not None and media.duration_s < SHORT_AUDIO_S:
            warnings.append({"code": "SHORT_AUDIO", "message": f"the audio is {media.duration_s:.2f} s long; scores on very short audio are unreliable.", "modality": "audio"})

    try:
        res = adapter.analyse(media, progress)
    except Exception as e:  # never turn an adapter failure into a score
        log.exception("adapter failed")
        return failed_response(src, filename, "ADAPTER_FAILED", f"{type(e).__name__}: {e}", rid, media, stages, t0)
    for name in ("detect", "score", "calibrate"):
        stages.append({"name": name, "ms": round(res.stage_ms.get(name, 0.0), 1)})
    warnings += res.warnings

    t = time.perf_counter()
    tick("fuse")
    per = {k: {"available": m.available, "score": m.score, "n_units": m.n_units, **({"reason": m.reason} if m.reason else {})}
           for k, m in res.per_modality.items()}
    required_missing = any((not m.available) and not (k == "audio" and media.media_type == "video" and not media.has_audio)
                           for k, m in res.per_modality.items())
    if media.media_type == "video":
        fused = {"score": None, "decision": "undetermined", "reason": "no combining rule for visual and audio scores is decided yet (O-009); see per_modality"}
    else:
        m = next(iter(res.per_modality.values()))
        if m.available and m.score is not None:
            fused = {"score": m.score, "decision": "fake" if m.score >= res.threshold else "real"}
        else:
            fused = {"score": None, "decision": "undetermined", "reason": m.reason or "no score available"}
    stages.append({"name": "fuse", "ms": round((time.perf_counter() - t) * 1000, 1)})
    frames = None
    if media.media_type == "image":
        frames = 1
    elif media.media_type == "video":
        frames = per.get("visual", {}).get("n_units")
    state = "partial" if required_missing else "ok"
    return {"schema_version": SCHEMA_VERSION, "request_id": rid,
            "input": _input_block(filename, media.size_bytes, media.sha256, media.media_type, media.mime_type, media.duration_s, frames),
            "inference_source": src, "model": res.model, "result": {"fused": fused, "per_modality": per},
            "score_meaning": SCORE_MEANING,
            "status": {"state": state, "processing_time_ms": round((time.perf_counter() - t0) * 1000, 1), "stages": stages},
            "warnings": warnings, "errors": []}


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Score one image, video or audio file and print the D-027 response as JSON.")
    ap.add_argument("file")
    ap.add_argument("--adapter", choices=["mock", "real"], required=True, help="mock results are not real and carry a MOCK source")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)
    from dfd.infer.factory import make_adapter
    adapter = make_adapter(a.adapter, a.checkpoint)
    out = analyse_file(a.file, Path(a.file).name, adapter)
    sys.stdout.write(json.dumps(out, indent=1) + "\n")
    return 0 if out["status"]["state"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
