"""
Video forensics. Sample frames at intervals, run image analysis on each,
and look for temporal inconsistency — deepfakes tend to flicker on
high-frequency facial detail and show jitter in crop boundaries.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .image import analyze_image


def _sample_frames(path: str, n: int = 8) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        # Stream with unknown length — pull up to n frames linearly.
        frames = []
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((i / fps, frame))
        cap.release()
        return frames

    indices = np.linspace(0, max(0, total - 1), num=n, dtype=int)
    frames: list[tuple[float, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((float(idx) / fps, frame))
    cap.release()
    return frames


def _temporal_flicker(frames: list[np.ndarray]) -> float:
    """High-frequency luminance delta between adjacent samples.
    Unnaturally large swings in mid-frequency edges are a common
    signature of frame-by-frame face generation."""
    if len(frames) < 2:
        return 0.0
    deltas = []
    prev = None
    for frame in frames:
        small = cv2.resize(frame, (160, 160))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        if prev is not None:
            deltas.append(float(np.abs(edges - prev).mean()))
        prev = edges
    if not deltas:
        return 0.0
    # Normalize: typical natural video sits under ~6. Deepfakes spike.
    return float(np.clip((np.std(deltas) - 1.0) / 6.0, 0.0, 1.0))


def analyze_video(data: bytes, filename: str = "video") -> dict[str, Any]:
    # cv2 needs a path; write to a temp file.
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        tmp_path = f.name

    try:
        sampled = _sample_frames(tmp_path, n=8)
        if not sampled:
            return {
                "kind": "video",
                "filename": filename,
                "error": "Could not decode video. Try MP4, WebM, or MOV.",
                "suspicion": 0.0,
                "verdict": "unreadable",
                "signals": [],
            }

        per_frame = []
        frame_images = []
        for (t, frame) in sampled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            import io as _io
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=92)
            r = analyze_image(buf.getvalue(), filename=f"frame@{t:.1f}s")
            per_frame.append({"timestamp": round(t, 2), "suspicion": r["suspicion"], "verdict": r["verdict"]})
            frame_images.append(frame)

        flicker = _temporal_flicker(frame_images)
        avg = float(np.mean([p["suspicion"] for p in per_frame]))
        peak = float(np.max([p["suspicion"] for p in per_frame]))

        score = float(np.clip(0.5 * avg + 0.3 * peak + 0.2 * flicker, 0.0, 1.0))

        if score < 0.3:
            label = "likely authentic"
        elif score < 0.55:
            label = "inconclusive"
        elif score < 0.75:
            label = "likely manipulated"
        else:
            label = "highly likely manipulated"

        cap = cv2.VideoCapture(tmp_path)
        duration = 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps:
            duration = frames_total / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "kind": "video",
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "dimensions": {"width": width, "height": height},
            "suspicion": score,
            "verdict": label,
            "confidence": round(abs(score - 0.5) * 2, 3),
            "signals": [
                {
                    "name": "Temporal flicker",
                    "score": round(flicker, 3),
                    "detail": (
                        "Frame-to-frame high-frequency energy is stable."
                        if flicker < 0.35
                        else "Edges oscillate between frames — typical of per-frame face synthesis."
                    ),
                },
                {
                    "name": "Frame suspicion (average)",
                    "score": round(avg, 3),
                    "detail": f"Mean suspicion across {len(per_frame)} sampled frames.",
                },
                {
                    "name": "Frame suspicion (peak)",
                    "score": round(peak, 3),
                    "detail": "Highest individual frame suspicion. A single strong frame matters.",
                },
            ],
            "timeline": per_frame,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
