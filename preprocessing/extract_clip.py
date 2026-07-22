"""
Stage 1 - per-clip extraction: N=16 face crops + their aligned audio windows.

This is the "frame <-> audio-window sync" mechanism referenced in
docs/stage-1-plan.md. For one video we:
  1. Sample 16 frame timestamps, evenly spaced across the clip duration.
  2. At each timestamp, decode that video frame, detect the face (MTCNN,
     reusing preprocessing/crop_faces.py's crop logic), and crop+resize to
     224x224.
  3. At each timestamp, cut a fixed-duration (default 0.35s) audio window
     CENTERED on that timestamp from the clip's audio track.
  4. Cache both to disk under data/processed/<clip_id>/ so re-running a
     stage doesn't redo face detection (slow) or audio decoding.

Frame <-> audio alignment is done by TIMESTAMP, not sample index: frame i's
audio window is [frame_time_i - window/2, frame_time_i + window/2] in
seconds, converted to sample indices using the audio's own sample rate. This
is what "audio_manifest.csv" records per frame, so any later stage can trace
a frame back to exactly which audio samples it's paired with.

We use PyAV (the `av` package) instead of torchaudio/ffmpeg-python for audio
decoding because no system ffmpeg binary is on PATH in this environment;
PyAV ships its own bundled ffmpeg libraries, so it works without one.
"""
import sys
from pathlib import Path
import numpy as np

# Make `from preprocessing.xxx import ...` resolve whether this file is run as
# `python preprocessing/extract_clip.py` (script dir on path) or
# `python -m preprocessing.extract_clip` (repo root on path). We add the repo
# root (this file's parent's parent) to sys.path so the package is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2
    import av
    import librosa
    import torch
    from facenet_pytorch import MTCNN
    from preprocessing.crop_faces import crop_and_resize_face
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: uv sync --extra cpu (or --extra cu130 for GPU), see README.md")
    sys.exit(1)

NUM_FRAMES = 16
FRAME_SIZE = 224
AUDIO_WINDOW_SEC = 0.35
AUDIO_SR = 16000  # target sample rate every audio window is resampled to

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

_mtcnn = None          # lazy singleton -- loading MTCNN weights per-clip would be very slow
_mtcnn_device = None   # remember which device the singleton was built on


def _get_mtcnn(device: str) -> "MTCNN":
    global _mtcnn, _mtcnn_device
    if _mtcnn is None or _mtcnn_device != device:
        _mtcnn = MTCNN(keep_all=False, device=device)  # keep_all=False: just the most confident face
        _mtcnn_device = device
    return _mtcnn


def _sample_timestamps(duration_sec: float, num_frames: int) -> np.ndarray:
    """Evenly spaced timestamps inside the clip, avoiding the exact 0 and duration edges."""
    # linspace over an inset range so the first/last window (+/- half the
    # audio window) doesn't fall outside the clip's audio track.
    margin = AUDIO_WINDOW_SEC / 2
    return np.linspace(margin, max(duration_sec - margin, margin), num_frames)


def _decode_audio(video_path: Path) -> tuple[np.ndarray, int]:
    """Decode the full audio track of an mp4 to a mono float32 waveform at AUDIO_SR."""
    container = av.open(str(video_path))
    if not container.streams.audio:
        raise ValueError(f"No audio stream in {video_path}")
    stream = container.streams.audio[0]
    native_sr = stream.rate

    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()  # shape [channels, samples] or [samples] depending on layout
        if arr.ndim == 2:
            arr = arr.mean(axis=0)  # downmix to mono
        chunks.append(arr.astype(np.float32))
    container.close()

    if not chunks:
        raise ValueError(f"Audio stream in {video_path} decoded to zero samples")

    waveform = np.concatenate(chunks)
    # Normalize int PCM to [-1, 1] if PyAV handed us integer samples.
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform / np.iinfo(waveform.dtype).max

    if native_sr != AUDIO_SR:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=AUDIO_SR)

    return waveform, AUDIO_SR


def extract_clip(video_path: Path, clip_id: str, force: bool = False,
                 device: str | None = None) -> dict:
    """
    Extract and cache 16 face crops + aligned audio windows for one clip.

    device: force MTCNN onto "cpu" or "cuda". Default None auto-selects CUDA if
    available. Pre-caching (preprocessing/precache.py) passes "cpu" so multiple
    worker processes don't contend over the single GPU.

    Returns a dict: {"frames": [N,224,224,3] uint8, "audio": [N, window_samples]
    float32, "timestamps": [N] float, "num_faces_detected": int}

    Raises RuntimeError with a clear message on any I/O or detection failure
    -- caller (batch driver) decides whether to skip and log, or abort.
    """
    out_dir = PROCESSED_DIR / clip_id
    frames_path = out_dir / "frames.npy"
    audio_path = out_dir / "audio.npy"
    ts_path = out_dir / "timestamps.npy"

    if not force and frames_path.exists() and audio_path.exists() and ts_path.exists():
        return {
            "frames": np.load(frames_path),
            "audio": np.load(audio_path),
            "timestamps": np.load(ts_path),
            "cached": True,
        }

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Bad video metadata (fps={fps}, frames={total_frames}) for {video_path}")
        duration_sec = total_frames / fps

        timestamps = _sample_timestamps(duration_sec, NUM_FRAMES)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = _get_mtcnn(device)

        frame_crops = []
        num_faces_detected = 0
        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame at t={t:.2f}s in {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            box, prob = mtcnn.detect(frame_rgb)
            if box is not None and prob is not None and prob[0] is not None:
                crop = crop_and_resize_face(frame_rgb, box[0], target_size=(FRAME_SIZE, FRAME_SIZE))
                if crop is not None:
                    # crop_and_resize_face returns BGR (it converts for cv2.imwrite);
                    # convert back to RGB so cached frames are in a consistent, documented color order.
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    frame_crops.append(crop)
                    num_faces_detected += 1
                    continue
            # No face found at this timestamp: fall back to a plain resize of
            # the full frame rather than dropping it, so every clip still
            # yields exactly NUM_FRAMES (shape must stay fixed for batching).
            frame_crops.append(cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC))
        cap.release()

        waveform, sr = _decode_audio(video_path)
        window_samples = int(AUDIO_WINDOW_SEC * sr)
        audio_windows = []
        for t in timestamps:
            center = int(t * sr)
            start = max(0, center - window_samples // 2)
            end = start + window_samples
            if end > len(waveform):
                end = len(waveform)
                start = max(0, end - window_samples)
            window = waveform[start:end]
            if len(window) < window_samples:
                window = np.pad(window, (0, window_samples - len(window)))
            audio_windows.append(window)

        frames_arr = np.stack(frame_crops).astype(np.uint8)          # [N, 224, 224, 3]
        audio_arr = np.stack(audio_windows).astype(np.float32)       # [N, window_samples]

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(frames_path, frames_arr)
        np.save(audio_path, audio_arr)
        np.save(ts_path, timestamps)

        return {
            "frames": frames_arr,
            "audio": audio_arr,
            "timestamps": timestamps,
            "num_faces_detected": num_faces_detected,
            "cached": False,
        }
    except Exception as e:
        raise RuntimeError(f"extract_clip failed for {video_path} (clip_id={clip_id}): {e}") from e
