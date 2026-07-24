"""MAIN audio preprocessing steps. Mono waveform float32 [n] unless noted.

Pure NumPy/librosa/PyAV — no Streamlit, no disk writes. This is the single
implementation of decode/downmix/resample/window shared by the real pipeline
and the dashboard (the window logic previously lived twice).

SOTA note: `leading_silence_sec` measures FakeAVCeleb's known leading-silence
shortcut (fake-audio clips carry extra silence at t=0). `sample_timestamps`
takes a start_offset so frame+audio sampling can begin PAST that silence,
neutralizing the shortcut while keeping frame↔audio alignment. See
docs/preprocessing.md and PROJECT_OVERVIEW.md §6.
"""
import av
import librosa
import numpy as np

from preprocessing.ops.constants import AUDIO_SR


def sample_timestamps(duration_sec: float, num_frames: int, window_sec: float,
                      start_offset: float = 0.0) -> np.ndarray:
    """Evenly-spaced frame timestamps, inset so each ±window/2 stays in-clip.

    start_offset pushes the first sample past leading silence; frames AND audio
    windows use the same timestamps, so the two modalities stay aligned.
    """
    margin = window_sec / 2
    lo = max(start_offset + margin, margin)
    hi = max(duration_sec - margin, lo)
    return np.linspace(lo, hi, num_frames)


def decode(video_path: str) -> tuple[np.ndarray, int]:
    """Decode the full audio track. Returns ([channels, samples] float32, native_sr).

    Empty/absent audio yields a (1, 0) array. PyAV bundles ffmpeg, so no system
    ffmpeg binary is needed.
    """
    container = av.open(str(video_path))
    if not container.streams.audio:
        container.close()
        return np.zeros((1, 0), np.float32), AUDIO_SR
    stream = container.streams.audio[0]
    native_sr = int(stream.rate)
    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()            # [channels, samples] or [samples]
        if arr.ndim == 1:
            arr = arr[None, :]
        chunks.append(arr)
    container.close()
    if not chunks:
        return np.zeros((1, 0), np.float32), native_sr
    wav = np.concatenate(chunks, axis=1)
    if np.issubdtype(wav.dtype, np.integer):    # normalize int PCM to [-1, 1]
        wav = wav / np.iinfo(wav.dtype).max
    return wav.astype(np.float32), native_sr


def downmix(wav_2d) -> np.ndarray:
    arr = np.asarray(wav_2d, dtype=np.float32)
    return arr.mean(axis=0) if arr.ndim == 2 else arr


def resample(wav, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or wav.size == 0:
        return wav.astype(np.float32)
    return librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def trim_leading_silence(wav, sr: int, top_db: float = 30.0):
    """Trim near-silent edges (librosa energy gate). Returns (trimmed, leading_sec)."""
    if wav.size == 0:
        return wav, 0.0
    trimmed, index = librosa.effects.trim(wav, top_db=top_db)
    return trimmed, float(index[0] / sr)


def leading_silence_sec(wav, sr: int, top_db: float = 30.0) -> float:
    """Seconds of leading silence — the FakeAVCeleb shortcut measurement."""
    return trim_leading_silence(wav, sr, top_db)[1]


def extract_windows(wav, sr: int, timestamps, window_sec: float) -> np.ndarray:
    """One window of `window_sec` centered on each timestamp → [N, win] float32.

    Windows are clamped to the waveform and zero-padded if short, so the shape
    is always [N, int(window_sec*sr)].
    """
    win = int(window_sec * sr)
    out = []
    for t in timestamps:
        center = int(t * sr)
        start = max(0, center - win // 2)
        end = start + win
        if end > len(wav):
            end = len(wav)
            start = max(0, end - win)
        w = wav[start:end]
        if len(w) < win:
            w = np.pad(w, (0, win - len(w)))
        out.append(w)
    return np.stack(out).astype(np.float32) if out else np.zeros((0, win), np.float32)
