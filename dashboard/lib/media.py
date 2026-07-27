"""Media decoding for the dashboard: frames at timestamps, audio, face detection.

Frame decoding and the Streamlit-cached MTCNN loader live here; the actual
per-step ops (detect/align/crop/mouth, audio decode/window, timestamps) come from
preprocessing.ops so the dashboard and the real pipeline run the SAME code.
NEVER writes data/processed/.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2

from preprocessing.ops import faces as _faces, audio as _audio
from preprocessing.ops.constants import AUDIO_SR, FRAME_SIZE, MOUTH_SIZE

# Re-exported so pages/tests can keep importing them from dashboard.lib.media.
sample_timestamps = _audio.sample_timestamps


def get_detector():
    import streamlit as st

    @st.cache_resource(show_spinner="Loading MTCNN face detector...")
    def _load():
        import torch
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MTCNN(keep_all=False, device=device), device

    return _load()


def frame_meta(video_path: str) -> tuple[float, float]:
    """Return (duration_sec, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    return duration, fps


def decode_frames(video_path: str, timestamps: np.ndarray) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, bgr = cap.read()
        if not ok:
            frames.append(np.zeros((FRAME_SIZE, FRAME_SIZE, 3), np.uint8))
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def decode_audio(video_path: str) -> tuple[np.ndarray, int]:
    """([channels, samples] float32, native_sr) — thin wrapper over ops.audio.decode."""
    return _audio.decode(str(video_path))


def detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float,
                    align: bool = True):
    """(face_224_rgb, detected). The visual/emotion face path."""
    face, _mouth, detected = _faces.detect_align_crop(
        frame_rgb, detector, conf_thresh=conf_thresh, margin=margin, align=align)
    return face, detected


def detect_face_and_mouth(frame_rgb, detector, conf_thresh: float, margin: float,
                          mouth_size: int = MOUTH_SIZE, align: bool = True,
                          align_inset: float = 0.0):
    """(face_224_rgb, mouth_96_rgb, detected) from one detect call.

    The face crop feeds the visual + emotion streams; the mouth crop is a
    PARALLEL output for the lip-sync stream (Stage 4), derived from MTCNN's two
    mouth-corner landmarks. `margin` pads the bbox crop, `align_inset` insets the
    alignment template — only one applies, depending on `align`.
    """
    return _faces.detect_align_crop(
        frame_rgb, detector, conf_thresh=conf_thresh, margin=margin,
        align=align, align_inset=align_inset, mouth_size=mouth_size)
