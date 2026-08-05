"""Media decoding for the dashboard: frames at timestamps, audio, face detection.

Frame decoding and the Streamlit-cached detector loader live here; the actual
per-step ops (detect/crop/mouth, audio decode/window, timestamps) come from
preprocessing.ops so the dashboard and the real pipeline run the SAME code.
NEVER writes data/processed/.

Also home to the clip player's re-encode step (playable_video_bytes): OpenCV and
PyAV decode a far wider range of codecs than a browser will play, so a clip the
rest of this module handles fine can still arrive at st.video as a blank player.
"""
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import av
import cv2

from preprocessing.ops import faces as _faces, audio as _audio, detectors as _detectors
from preprocessing.ops.constants import AUDIO_SR, FRAME_SIZE, MOUTH_SIZE

# Re-exported so pages/tests can keep importing them from dashboard.lib.media.
sample_timestamps = _audio.sample_timestamps
DETECTOR_NAMES = _detectors.DETECTOR_NAMES
DEFAULT_DETECTOR = _detectors.DEFAULT_DETECTOR

# Video codecs that every current browser can decode natively. FakeAVCeleb is
# NOT uniformly one of them: the wav2lip generator wrote its output with an
# MPEG-4 Part 2 encoder, so every FakeVideo-FakeAudio clip (and some
# FakeVideo-RealAudio ones) is `mpeg4`, which no browser will play. Those are
# exactly the lip-sync forgeries this project is built around, so the clip
# player has to re-encode rather than shrug.
BROWSER_VIDEO_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})


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


def video_codec(video_path) -> str | None:
    """Name of the file's video codec, or None if it carries no video stream."""
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return None
        return container.streams.video[0].codec_context.name


def transcode_to_h264(video_path) -> bytes:
    """Re-encode a clip to browser-playable H.264/AAC MP4 and return its bytes.

    Goes through PyAV rather than a system ffmpeg binary, for the same reason the
    audio path does (see preprocessing/ops/audio.decode): PyAV bundles ffmpeg's
    libraries, so this needs no external install.

    Written to a real temp file rather than a BytesIO because `+faststart` moves
    the moov atom to the front in a second pass, which needs a seekable,
    re-openable output. Video and audio are encoded in separate passes over the
    input, which keeps the muxer's interleaving simple and preserves duration.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "playable.mp4"
        with av.open(str(video_path)) as inp:
            if not inp.streams.video:
                raise ValueError(f"{video_path} has no video stream to re-encode")
            src_v = inp.streams.video[0]
            src_a = inp.streams.audio[0] if inp.streams.audio else None

            out = av.open(str(out_path), mode="w", format="mp4",
                          options={"movflags": "+faststart"})
            dst_v = out.add_stream("libx264", rate=src_v.average_rate or 25)
            dst_v.width, dst_v.height = src_v.width, src_v.height
            dst_v.pix_fmt = "yuv420p"          # the only pix_fmt browsers all decode
            dst_v.options = {"crf": "23", "preset": "veryfast"}

            dst_a = resampler = None
            if src_a is not None:
                dst_a = out.add_stream("aac", rate=src_a.rate)
                # Some clips carry mp3 audio, whose frame format the AAC encoder
                # will not take directly.
                resampler = av.AudioResampler(format=dst_a.format.name,
                                              layout=dst_a.layout.name, rate=dst_a.rate)

            for frame in inp.decode(video=0):
                for packet in dst_v.encode(frame):
                    out.mux(packet)
            for packet in dst_v.encode():
                out.mux(packet)

            if dst_a is not None:
                inp.seek(0)
                for frame in inp.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        for packet in dst_a.encode(resampled):
                            out.mux(packet)
                for packet in dst_a.encode():
                    out.mux(packet)
            out.close()

        return out_path.read_bytes()


def playable_video_bytes(video_path) -> tuple[bytes, str | None]:
    """(mp4 bytes st.video can play, the codec that forced a re-encode or None).

    A clip already in a browser codec is handed over untouched; anything else is
    re-encoded, and the second element names what it was so the caller can say
    so rather than silently serving different pixels than are on disk.
    """
    codec = video_codec(video_path)
    if codec in BROWSER_VIDEO_CODECS:
        return Path(video_path).read_bytes(), None
    return transcode_to_h264(video_path), codec


def cached_playable_video(video_path) -> tuple[bytes, str | None]:
    """playable_video_bytes memoized on (path, mtime, size): the page-facing entry.

    Re-encoding a clip costs a few hundred milliseconds, which is fine once but
    not on every Streamlit rerun (each slider nudge triggers one).
    """
    import streamlit as st

    @st.cache_data(show_spinner=False, max_entries=32)
    def _load(path: str, _mtime: float, _size: int):
        return playable_video_bytes(path)

    stat = Path(video_path).stat()
    return _load(str(video_path), stat.st_mtime, stat.st_size)


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
    def _load(path: str, _mtime: float, ts: tuple[float, ...]):
        return decode_frames(path, np.asarray(ts, dtype=float))

    stat = Path(video_path).stat()
    return _load(str(video_path), stat.st_mtime, tuple(float(t) for t in timestamps))


def cached_face_mouth(video_path, timestamps, conf_thresh: float, margin: float,
                      mouth_size: int = MOUTH_SIZE,
                      detector: str = _detectors.DEFAULT_DETECTOR):
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
    def _run(path: str, _mtime: float, ts: tuple[float, ...], conf: float, mrg: float,
             msize: int, det_name: str):
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
    return _run(str(video_path), stat.st_mtime, tuple(float(t) for t in timestamps),
                float(conf_thresh), float(margin), int(mouth_size), str(detector))


def detect_and_crop(frame_rgb, detector, conf_thresh: float, margin: float):
    """(face_224_rgb, detected). The visual/emotion face path."""
    face, _mouth, detected = _faces.detect_crop(
        frame_rgb, detector, conf_thresh=conf_thresh, margin=margin)
    return face, detected


def detect_face_and_mouth(frame_rgb, detector, conf_thresh: float, margin: float,
                          mouth_size: int = MOUTH_SIZE):
    """(face_224_rgb, mouth_96_rgb, detected) from one detect call.

    The face crop feeds the visual + emotion streams; the mouth crop is a
    PARALLEL output for the lip-sync stream (Stage 4), derived from the
    detector's two mouth-corner landmarks. `margin` pads the bbox crop.
    """
    return _faces.detect_crop(
        frame_rgb, detector, conf_thresh=conf_thresh, margin=margin,
        mouth_size=mouth_size)
