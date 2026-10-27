"""Media decoding for the dashboard: frames at timestamps, audio, face detection.

Frame decoding and the Streamlit-cached detector loader live here; the actual
per-step ops (detect/crop/mouth, audio decode/window, timestamps) come from
preprocessing.ops so the dashboard and the real pipeline run the SAME code.
Never writes into data/ and never writes the cache.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from deepfake_detection import media_kind
from deepfake_detection.preprocessing.ops import audio as _audio
from deepfake_detection.preprocessing.ops import constants as _constants
from deepfake_detection.preprocessing.ops import detectors as _detectors
from deepfake_detection.preprocessing.ops import faces as _faces
from deepfake_detection.preprocessing.ops.constants import (
    FRAME_SIZE,
    MOUTH_SIZE,
)
from deepfake_detection.views.playback import playable_video_bytes

# Re-exported so pages/tests can keep importing them from deepfake_detection.dashboard.lib.media.
sample_timestamps = _audio.sample_timestamps
AUDIO_SR = _constants.AUDIO_SR
DETECTOR_NAMES = _detectors.DETECTOR_NAMES
DEFAULT_DETECTOR = _detectors.DEFAULT_DETECTOR


def get_detector(name: str = _detectors.DEFAULT_DETECTOR):
    """The named detector, loaded once per session. Returns (detector, device)."""
    import streamlit as st

    @st.cache_resource(show_spinner="Loading face detector...")
    def _load(detector_name: str):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        det = _detectors.build(detector_name, device=device)
        return det, det.device

    return _load(name)


def cached_playable_video(video_path) -> tuple[bytes, str | None]:
    """playable_video_bytes memoized on (path, mtime, size): the page-facing entry.

    Re-encoding a clip costs a few hundred milliseconds, which is fine once but
    not on every Streamlit rerun (each slider nudge triggers one).
    """
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=32)
    def _load(path: str, mtime_ns: int, size: int):
        return playable_video_bytes(path)

    stat = Path(video_path).stat()
    return _load(str(video_path), stat.st_mtime_ns, stat.st_size)


def frame_meta(video_path: str) -> tuple[float, float]:
    """Return (duration_sec, fps)."""
    kind = media_kind.classify(video_path)
    if kind == media_kind.IMAGE:
        return 0.0, 0.0
    if kind == media_kind.AUDIO:
        waveform, rate = decode_audio(video_path)
        return (waveform.shape[-1] / rate if rate else 0.0), 0.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    return duration, fps


def decode_frames(video_path: str, timestamps: np.ndarray) -> list[np.ndarray]:
    if media_kind.classify(video_path) == media_kind.IMAGE:
        image = cv2.imread(str(video_path))
        if image is None:
            raise ValueError(f"Cannot decode image: {video_path}")
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
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
    """([channels, samples] float32, native_sr). Thin wrapper over ops.audio.decode."""
    return _audio.decode(str(video_path))


def cached_decode_frames(video_path, timestamps) -> list[np.ndarray]:
    """decode_frames memoized on (path, mtime, timestamps).

    Seeking and decoding 16 frames costs a few hundred milliseconds. Streamlit
    reruns the whole script on every widget interaction, including one that
    touches nothing on this path (a checkbox in another step, closing the clip
    picker), so without this the page pays for the decode again each time.
    """
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=8)
    def _load(path: str, mtime_ns: int, size: int, ts: tuple[float, ...]):
        return decode_frames(path, np.asarray(ts, dtype=float))

    stat = Path(video_path).stat()
    return _load(
        str(video_path),
        stat.st_mtime_ns,
        stat.st_size,
        tuple(float(t) for t in timestamps),
    )


def cached_face_mouth(
    video_path,
    timestamps,
    conf_thresh: float,
    margin: float,
    mouth_size: int = MOUTH_SIZE,
    detector: str = _detectors.DEFAULT_DETECTOR,
):
    """(faces, mouths, n_detected, detect_ms), memoized on the detector settings.

    Detection over 16 frames is the slowest thing either the Preprocessing page
    or a stream page does, and it is pure with respect to (clip, timestamps,
    conf, margin, detector). Cached under those, so only a change that can alter
    the crops pays for it.

    detect_ms is measured inside the memoized body and cached with the crops, so
    a cache hit reports the time detection really took rather than the ~0 ms the
    cache hit itself cost. It covers the detect+crop calls only, not the frame
    decode, which is the same work whichever detector is loaded.
    """
    import streamlit as st

    @st.cache_data(show_spinner="Detecting faces...", max_entries=8)
    def _run(
        path: str,
        mtime_ns: int,
        size: int,
        ts: tuple[float, ...],
        conf: float,
        mrg: float,
        msize: int,
        det_name: str,
    ):
        det, _device = get_detector(det_name)
        frames = cached_decode_frames(path, ts)
        faces, mouths, found = [], [], 0
        t0 = perf_counter()
        for frame in frames:
            face, mouth, detected = detect_face_and_mouth(frame, det, conf, mrg, msize)
            faces.append(face)
            mouths.append(mouth)
            found += int(detected)
        detect_ms = (perf_counter() - t0) * 1000.0
        return faces, mouths, found, detect_ms

    stat = Path(video_path).stat()
    return _run(
        str(video_path),
        stat.st_mtime_ns,
        stat.st_size,
        tuple(float(t) for t in timestamps),
        float(conf_thresh),
        float(margin),
        int(mouth_size),
        str(detector),
    )


def detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float):
    """(face_224_rgb, detected). The visual/emotion face path."""
    face, _mouth, detected = _faces.detect_crop(
        frame_rgb, detector, conf_thresh=conf_thresh, margin=margin
    )
    return face, detected


def detect_face_and_mouth(
    frame_rgb, detector, conf_thresh: float, margin: float, mouth_size: int = MOUTH_SIZE
):
    """(face_224_rgb, mouth_96_rgb, detected) from one detect call.

    The face crop feeds the visual + emotion streams; the mouth crop is a
    PARALLEL output for the lip-sync stream (Stage 4), derived from the
    detector's two mouth-corner landmarks. `margin` pads the bbox crop.
    """
    return _faces.detect_crop(
        frame_rgb,
        detector,
        conf_thresh=conf_thresh,
        margin=margin,
        mouth_size=mouth_size,
    )
