"""Media decoding for the dashboard: frames at timestamps, audio, face detection.

Mirrors preprocessing/extract_clip.py's decode logic (av for audio, OpenCV for
frames) but returns in-memory arrays and NEVER writes data/processed/.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import av

from preprocessing.crop_faces import crop_and_resize_face

AUDIO_SR = 16000
FRAME_SIZE = 224


def sample_timestamps(duration_sec: float, num_frames: int, window_sec: float) -> np.ndarray:
    margin = window_sec / 2
    return np.linspace(margin, max(duration_sec - margin, margin), num_frames)


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
    container = av.open(str(video_path))
    if not container.streams.audio:
        container.close()
        return np.zeros((1, 0), np.float32), AUDIO_SR
    stream = container.streams.audio[0]
    native_sr = stream.rate
    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()            # [channels, samples] or [samples]
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks.append(arr.astype(np.float32))
    container.close()
    if not chunks:
        return np.zeros((1, 0), np.float32), int(native_sr)
    wav = np.concatenate(chunks, axis=1)
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav / np.iinfo(wav.dtype).max
    return wav.astype(np.float32), int(native_sr)


def detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float):
    box, prob = detector.detect(frame_rgb)
    if box is not None and prob is not None and prob[0] is not None and prob[0] >= conf_thresh:
        crop = crop_and_resize_face(frame_rgb, box[0], (FRAME_SIZE, FRAME_SIZE),
                                    margin_percentage=margin)
        if crop is not None:
            return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), True  # crop_* returns BGR
    return cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC), False
