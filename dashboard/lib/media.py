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


def detect_face_and_mouth(frame_rgb, detector, conf_thresh: float, margin: float,
                          mouth_size: int = 96):
    """Detect once; return (face_224_rgb, mouth_96_rgb, detected).

    The face crop feeds the visual + emotion streams; the mouth crop is a
    PARALLEL output for the lip-sync stream — derived from MTCNN's two mouth-
    corner landmarks, not by chopping the face crop. Both come from one detect
    call so the two branches stay aligned.
    """
    boxes, probs, points = detector.detect(frame_rgb, landmarks=True)
    ok = (boxes is not None and probs is not None
          and probs[0] is not None and probs[0] >= conf_thresh)
    if not ok:
        face = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)
        mouth = cv2.resize(frame_rgb, (mouth_size, mouth_size), interpolation=cv2.INTER_CUBIC)
        return face, mouth, False

    crop = crop_and_resize_face(frame_rgb, boxes[0], (FRAME_SIZE, FRAME_SIZE),
                                margin_percentage=margin)
    face = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop is not None
            else cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC))

    # Mouth region from the two mouth-corner landmarks (points 3 and 4).
    ml, mr = points[0][3], points[0][4]
    cx, cy = (ml[0] + mr[0]) / 2.0, (ml[1] + mr[1]) / 2.0
    corner_dist = float(np.hypot(mr[0] - ml[0], mr[1] - ml[1]))
    half = max(corner_dist * 0.9, 20.0)          # square half-size around the mouth
    h_img, w_img = frame_rgb.shape[:2]
    x1, x2 = int(max(0, cx - half)), int(min(w_img, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h_img, cy + half))
    region = frame_rgb[y1:y2, x1:x2]
    mouth = (cv2.resize(region, (mouth_size, mouth_size), interpolation=cv2.INTER_CUBIC)
             if region.size else cv2.resize(face, (mouth_size, mouth_size)))
    return face, mouth, True
