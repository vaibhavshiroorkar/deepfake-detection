"""Audio preprocessing steps. Mono waveform float32 [n] unless noted.

Pure NumPy, librosa and PyAV. This is the single implementation of
decode, downmix, resample and window shared by the pipeline and the dashboard.

`leading_silence_sec` measures FakeAVCeleb's known leading-silence shortcut:
fake-audio clips carry extra silence at t=0. `sample_timestamps` takes a
start_offset so frame and audio sampling can begin past that silence, which
neutralizes the shortcut while keeping the two modalities aligned.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import AUDIO_SR


def sample_timestamps(
    duration_sec: float,
    num_frames: int,
    window_sec: float,
    start_offset: float = 0.0,
) -> np.ndarray:
    """Evenly spaced frame timestamps, inset so each +/- window/2 stays in-clip."""
    margin = window_sec / 2
    low = max(start_offset + margin, margin)
    high = max(duration_sec - margin, low)
    return np.linspace(low, high, num_frames)


def decode(video_path: str) -> tuple[np.ndarray, int]:
    """Decode the full audio track -> ([channels, samples] float32, native_sr).

    Empty or absent audio yields a (1, 0) array. PyAV bundles ffmpeg, so no
    system ffmpeg binary is needed.
    """
    import av

    container = av.open(str(video_path))
    if not container.streams.audio:
        container.close()
        return np.zeros((1, 0), np.float32), AUDIO_SR
    stream = container.streams.audio[0]
    native_sr = int(stream.rate)
    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks.append(arr)
    container.close()
    if not chunks:
        return np.zeros((1, 0), np.float32), native_sr
    waveform = np.concatenate(chunks, axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform / np.iinfo(waveform.dtype).max
    return waveform.astype(np.float32), native_sr


def downmix(waveform_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(waveform_2d, dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or waveform.size == 0:
        return waveform.astype(np.float32)
    import librosa

    return librosa.resample(
        waveform.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    )


def trim_leading_silence(
    waveform: np.ndarray, sr: int, top_db: float = 30.0
) -> tuple[np.ndarray, float]:
    """Trim near-silent edges. Returns (trimmed, leading_sec)."""
    if waveform.size == 0:
        return waveform, 0.0
    import librosa

    trimmed, index = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed, float(index[0] / sr)


def leading_silence_sec(waveform: np.ndarray, sr: int, top_db: float = 30.0) -> float:
    """Seconds of leading silence, the FakeAVCeleb shortcut measurement."""
    return trim_leading_silence(waveform, sr, top_db)[1]


def extract_windows(
    waveform: np.ndarray,
    sr: int,
    timestamps: Sequence[float],
    window_sec: float,
) -> np.ndarray:
    """One window of window_sec centered on each timestamp -> [N, win] float32.

    Windows are clamped to the waveform and zero-padded if short, so the shape is
    always [N, int(window_sec * sr)].
    """
    win = int(window_sec * sr)
    windows = []
    for timestamp in timestamps:
        center = int(timestamp * sr)
        start = max(0, center - win // 2)
        end = start + win
        if end > len(waveform):
            end = len(waveform)
            start = max(0, end - win)
        window = waveform[start:end]
        if len(window) < win:
            window = np.pad(window, (0, win - len(window)))
        windows.append(window)
    if not windows:
        return np.zeros((0, win), np.float32)
    return np.stack(windows).astype(np.float32)
