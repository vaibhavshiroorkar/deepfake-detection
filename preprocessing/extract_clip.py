"""
Stage 1 - per-clip extraction: N=16 face crops + their aligned audio windows.

This is the "frame <-> audio-window sync" mechanism referenced in
docs/stage-1-plan.md. For one video we:
  1. Decode the audio track and measure its leading silence (FakeAVCeleb's known
     shortcut bug — fake-audio clips carry extra silence at t=0). Sampling starts
     PAST that silence so a model can't cheat on it.
  2. Sample 16 frame timestamps, evenly spaced across the remaining duration.
  3. At each timestamp, decode that video frame, detect the face (MTCNN or
     YuNet, see preprocessing/ops/detectors.py) and CROP it with a margin-padded
     bounding box (preprocessing/ops/faces.py), then crop+resize to 224x224.
  4. At each timestamp, cut a fixed-duration (default 0.35s) audio window
     CENTERED on that timestamp from the clip's audio track.
  5. Cache both to disk under data/processed/<clip_id>/ so re-running a stage
     doesn't redo face detection (slow) or audio decoding.

Frame <-> audio alignment is done by TIMESTAMP, not sample index: frame i's audio
window is [t_i - window/2, t_i + window/2] in seconds. Both modalities use the
SAME timestamps (with the same leading-silence offset), so they stay aligned.

The per-step ops live in preprocessing/ops/ and are shared with the dashboard —
there is one implementation of each, not two.
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# Make `from preprocessing.xxx import ...` resolve whether this file is run as
# `python preprocessing/extract_clip.py` or `python -m preprocessing.extract_clip`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
    import torch
    from preprocessing.ops import faces as F, audio as A, detectors as D
    from preprocessing.ops.constants import (
        NUM_FRAMES, FRAME_SIZE, MOUTH_SIZE, AUDIO_SR, AUDIO_WINDOW_SEC, PIPELINE_VERSION,
    )
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: uv sync --extra cpu (or --extra cu130 for GPU), see README.md")
    sys.exit(1)

# Leading-silence gate: bins this many dB below the clip's peak count as silence.
SILENCE_TOP_DB = 30.0

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_detector = None       # lazy singleton -- loading detector weights per-clip would be very slow
_detector_key = None   # (name, device) the singleton was built for


def _get_detector(name: str, device: str):
    global _detector, _detector_key
    if _detector is None or _detector_key != (name, device):
        _detector = D.build(name, device=device)
        _detector_key = (name, device)
    return _detector


def _version_stamp(detector: str) -> str:
    """What goes in version.txt. Crops depend on the detector as much as on the
    pipeline version, so both are stamped."""
    return f"{PIPELINE_VERSION}:{detector}"


def _cache_valid(out_dir: Path, detector: str) -> bool:
    """A cache hit needs every array AND a stamp matching this version+detector,
    so caches written by an older pipeline, or by the other detector, are
    re-extracted rather than silently reused.

    A bare "4" counts as MTCNN's. Every such cache predates the second detector,
    so MTCNN is the only thing that can have written it, and reading it that way
    saves re-extracting the whole set for a rename.
    """
    needed = ["frames.npy", "audio.npy", "timestamps.npy", "version.txt"]
    if not all((out_dir / n).exists() for n in needed):
        return False
    try:
        stamp = (out_dir / "version.txt").read_text().strip()
    except OSError:
        return False
    if stamp == str(PIPELINE_VERSION):
        stamp = _version_stamp(D.MTCNN_NAME)
    return stamp == _version_stamp(detector)


def extract_clip(video_path: Path, clip_id: str, force: bool = False,
                 device: str | None = None,
                 conf_thresh: float = 0.90, margin: float = 0.20,
                 detector: str = D.DEFAULT_DETECTOR) -> dict:
    """
    Extract and cache 16 face crops + aligned audio windows for one clip.

    detector: "mtcnn" or "yunet" (preprocessing/ops/detectors.py). It is stamped
    into the cache, so the two never read each other's crops.

    device: force the detector onto "cpu" or "cuda". Default None auto-selects
    CUDA if available. Pre-caching (preprocessing/precache.py) passes "cpu" so
    multiple worker processes don't contend over the single GPU. YuNet is
    CPU-only and ignores this.

    Returns a dict: {"frames": [N,224,224,3] uint8, "audio": [N, window_samples]
    float32, "timestamps": [N] float, "num_faces_detected": int,
    "leading_silence_sec": float, "detect_ms": float}. On a cache hit only the
    arrays and "cached" are present -- nothing was detected, so there is no
    honest time to report.

    Raises RuntimeError with a clear message on any I/O or detection failure --
    caller (batch driver) decides whether to skip and log, or abort.
    """
    out_dir = PROCESSED_DIR / clip_id
    frames_path = out_dir / "frames.npy"
    audio_path = out_dir / "audio.npy"
    ts_path = out_dir / "timestamps.npy"

    if not force and _cache_valid(out_dir, detector):
        return {
            "frames": np.load(frames_path),
            "audio": np.load(audio_path),
            "timestamps": np.load(ts_path),
            "cached": True,
        }

    try:
        # --- Audio first: decode, downmix, resample, measure leading silence. ---
        raw2d, native_sr = A.decode(str(video_path))
        if raw2d.size == 0:
            raise RuntimeError(f"No audio stream / zero samples in {video_path}")
        waveform = A.resample(A.downmix(raw2d), native_sr, AUDIO_SR)
        sr = AUDIO_SR
        leading_silence = A.leading_silence_sec(waveform, sr, top_db=SILENCE_TOP_DB)

        # --- Video metadata + silence-aware frame timestamps. ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Bad video metadata (fps={fps}, frames={total_frames}) for {video_path}")
        duration_sec = total_frames / fps

        timestamps = A.sample_timestamps(duration_sec, NUM_FRAMES, AUDIO_WINDOW_SEC,
                                         start_offset=leading_silence)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        det = _get_detector(detector, device)

        # --- Per-frame detect + crop. ---
        # detect_ms times the detect+crop calls only. Frame seeking and decoding
        # are the same work whichever detector is loaded, so folding them in
        # would flatter the slow one.
        frame_crops = []
        num_faces_detected = 0
        detect_ms = 0.0
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame at t={t:.2f}s in {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = perf_counter()
            face, _mouth, detected = F.detect_crop(
                frame_rgb, det, conf_thresh=conf_thresh, margin=margin,
                size=FRAME_SIZE, mouth_size=MOUTH_SIZE)
            detect_ms += (perf_counter() - t0) * 1000.0
            frame_crops.append(face)
            num_faces_detected += int(detected)
        cap.release()

        # --- Audio windows aligned to the same timestamps. ---
        audio_windows = A.extract_windows(waveform, sr, timestamps, AUDIO_WINDOW_SEC)

        frames_arr = np.stack(frame_crops).astype(np.uint8)      # [N, 224, 224, 3]
        audio_arr = audio_windows.astype(np.float32)             # [N, window_samples]

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(frames_path, frames_arr)
        np.save(audio_path, audio_arr)
        np.save(ts_path, timestamps)
        (out_dir / "version.txt").write_text(_version_stamp(detector))

        return {
            "frames": frames_arr,
            "audio": audio_arr,
            "timestamps": timestamps,
            "num_faces_detected": num_faces_detected,
            "leading_silence_sec": leading_silence,
            "detector": detector,
            "detect_ms": detect_ms,
            "cached": False,
        }
    except Exception as e:
        raise RuntimeError(f"extract_clip failed for {video_path} (clip_id={clip_id}): {e}") from e
