"""
Video forensics — Phase 2.

Primary signal: **Temporal Transformer** built on DINOv2 frame embeddings.
For each sampled frame, DINOv2 extracts a [CLS] embedding; the sequence
of embeddings is then fed through a Transformer encoder that can detect
temporal inconsistencies — flickering faces, drifting boundaries, and
jittery lighting that appear across frames but not within any single one.

Also runs per-frame DINOv2 classification for the timeline view, and
keeps the Phase 1 temporal flicker heuristic as supporting evidence.
"""
from __future__ import annotations

import io as _io
import os
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .image import analyze_image


# ---------------------------------------------------------------------------
# Video I/O helpers (unchanged)
# ---------------------------------------------------------------------------

def _open_capture(path: str) -> tuple[cv2.VideoCapture | None, dict]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, {"fps": fps, "total": total, "width": width, "height": height}


def _sample_frames(cap: cv2.VideoCapture, meta: dict, n: int = 8) -> list[tuple[float, np.ndarray]]:
    fps = meta["fps"]
    total = meta["total"]
    if total <= 0:
        frames = []
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((i / fps, frame))
        return frames

    indices = np.linspace(0, max(0, total - 1), num=n, dtype=int)
    frames: list[tuple[float, np.ndarray]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((float(idx) / fps, frame))
    return frames


def _temporal_flicker(frames: list[np.ndarray]) -> float:
    """Phase 1 heuristic: high-frequency luminance delta between adjacent samples."""
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
    return float(np.clip((np.std(deltas) - 1.0) / 6.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Phase 2: DINOv2 embeddings + temporal transformer
# ---------------------------------------------------------------------------

def _extract_dinov2_embeddings(
    pil_frames: list[Image.Image],
) -> np.ndarray | None:
    """Extract DINOv2 [CLS] embeddings for each frame.

    Returns (N, 1024) numpy array, or None if DINOv2 is unavailable.
    """
    try:
        from ..models import get_dinov2_classifier, get_device
        from .dinov2_head import extract_cls_embedding

        model, processor = get_dinov2_classifier()
        device = get_device()

        embeddings = []
        for pil_img in pil_frames:
            emb = extract_cls_embedding(pil_img, model, processor, device)
            embeddings.append(emb)

        return np.stack(embeddings, axis=0)  # (N, 1024)
    except Exception:
        return None


def _temporal_transformer_signal(
    frame_embeddings: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """Run the temporal transformer on frame embeddings.

    Returns (p_fake, p_real, per_frame_scores) or None if unavailable.
    """
    try:
        from ..models import get_device, get_temporal_transformer
        from .temporal_transformer import predict_video

        model = get_temporal_transformer()
        device = get_device()
        return predict_video(frame_embeddings, model, device)
    except Exception:
        return None


def _per_frame_dinov2_scores(
    pil_frames: list[Image.Image],
) -> list[float] | None:
    """Classify each frame individually with DINOv2.

    Returns a list of P(fake) scores, or None if unavailable.
    """
    try:
        from ..models import get_dinov2_classifier, get_device
        from .dinov2_head import predict_image

        model, processor = get_dinov2_classifier()
        device = get_device()

        scores = []
        for pil_img in pil_frames:
            p_fake, _ = predict_image(pil_img, model, processor, device)
            scores.append(p_fake)
        return scores
    except Exception:
        return None


def analyze_video(data: bytes, filename: str = "video") -> dict[str, Any]:
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        tmp_path = f.name

    try:
        cap, meta = _open_capture(tmp_path)
        if cap is None:
            return {
                "kind": "video",
                "filename": filename,
                "error": "Could not decode video. Try MP4, WebM, or MOV.",
                "suspicion": 0.0,
                "verdict": "unreadable",
                "signals": [],
            }
        try:
            sampled = _sample_frames(cap, meta, n=8)
        finally:
            cap.release()
        if not sampled:
            return {
                "kind": "video",
                "filename": filename,
                "error": "Video opened but no frames could be read.",
                "suspicion": 0.0,
                "verdict": "unreadable",
                "signals": [],
            }

        # Convert frames to PIL for model input
        pil_frames = []
        frame_images = []
        timestamps = []
        for (t, frame) in sampled:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil_frames.append(pil)
            frame_images.append(frame)
            timestamps.append(t)

        signals = []

        # --- Phase 2: Temporal transformer ---
        frame_embeddings = _extract_dinov2_embeddings(pil_frames)
        temporal_result = None
        temporal_per_frame = None
        if frame_embeddings is not None:
            temporal_result = _temporal_transformer_signal(frame_embeddings)

        if temporal_result is not None:
            t_p_fake, t_p_real, temporal_per_frame = temporal_result
            score_temporal = float(np.clip(t_p_fake, 0.0, 1.0))
            signals.append({
                "name": "Temporal coherence",
                "score": round(score_temporal, 3),
                "detail": (
                    f"Temporal transformer: P(manipulated) = {t_p_fake:.3f}. "
                    + (
                        "Strong temporal inconsistencies detected across frames."
                        if score_temporal > 0.6
                        else "Cross-frame temporal patterns do not indicate manipulation."
                        if score_temporal < 0.35
                        else "Mild temporal irregularities, which could be compression or manipulation."
                    )
                ),
            })

        # --- Phase 2: Per-frame DINOv2 classification ---
        dinov2_scores = _per_frame_dinov2_scores(pil_frames)

        # --- Phase 1 fallback: per-frame image analysis ---
        per_frame = []
        if dinov2_scores is not None:
            for i, (t, score_val) in enumerate(zip(timestamps, dinov2_scores)):
                if score_val < 0.30:
                    verdict = "likely authentic"
                elif score_val < 0.50:
                    verdict = "inconclusive"
                elif score_val < 0.72:
                    verdict = "likely AI-generated or manipulated"
                else:
                    verdict = "highly likely AI-generated or manipulated"
                per_frame.append({
                    "timestamp": round(t, 2),
                    "suspicion": round(score_val, 3),
                    "verdict": verdict,
                })
        else:
            # Fall back to full image analysis per frame
            for (t, frame) in sampled:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                buf = _io.BytesIO()
                pil.save(buf, format="JPEG", quality=92)
                r = analyze_image(buf.getvalue(), filename=f"frame@{t:.1f}s", with_heatmaps=False)
                per_frame.append({
                    "timestamp": round(t, 2),
                    "suspicion": r["suspicion"],
                    "verdict": r["verdict"],
                })

        # --- Phase 1: Temporal flicker heuristic ---
        flicker = _temporal_flicker(frame_images)
        signals.append({
            "name": "Temporal flicker",
            "score": round(flicker, 3),
            "detail": (
                "Frame-to-frame high-frequency energy is stable."
                if flicker < 0.35
                else "Edges oscillate between frames, typical of per-frame face synthesis."
            ),
        })

        # Frame-level aggregate signals
        avg = float(np.mean([p["suspicion"] for p in per_frame]))
        peak = float(np.max([p["suspicion"] for p in per_frame]))

        signals.append({
            "name": "Frame suspicion (average)",
            "score": round(avg, 3),
            "detail": f"Mean suspicion across {len(per_frame)} sampled frames.",
        })
        signals.append({
            "name": "Frame suspicion (peak)",
            "score": round(peak, 3),
            "detail": "Highest individual frame suspicion. A single strong frame matters.",
        })

        # --- Final verdict ---
        if temporal_result is not None:
            # Phase 2: temporal transformer gets significant weight
            t_score = temporal_result[0]
            score = float(np.clip(
                0.35 * t_score + 0.30 * avg + 0.20 * peak + 0.15 * flicker,
                0.0, 1.0,
            ))
        else:
            # Phase 1 fallback
            score = float(np.clip(
                0.5 * avg + 0.3 * peak + 0.2 * flicker,
                0.0, 1.0,
            ))

        if score < 0.3:
            label = "likely authentic"
        elif score < 0.55:
            label = "inconclusive"
        elif score < 0.75:
            label = "likely manipulated"
        else:
            label = "highly likely manipulated"

        fps = meta["fps"]
        frames_total = meta["total"]
        duration = (frames_total / fps) if fps and frames_total > 0 else 0.0
        width = meta["width"]
        height = meta["height"]

        return {
            "kind": "video",
            "filename": filename,
            "duration_seconds": round(duration, 2),
            "dimensions": {"width": width, "height": height},
            "suspicion": score,
            "verdict": label,
            "confidence": round(abs(score - 0.5) * 2, 3),
            "signals": signals,
            "timeline": per_frame,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
