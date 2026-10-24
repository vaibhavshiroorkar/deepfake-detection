"""Audio extraction from videos and loading of standalone audio files.

Everything is returned as mono float32 at `sr` Hz (default 16 kHz), values in
[-1, 1]. Audio of the lengths used here (seconds to a few minutes) is small
(16 kHz float32 = 64 kB per second), so it is held in memory, unlike video.

    extract_audio(video)   ffmpeg subprocess, raw f32le on a pipe. A video with
                           no audio stream gives an empty array (FF++ videos have none).
    load_audio(path)       FLAC/WAV via soundfile when the file is already at `sr`;
                           otherwise decoded and resampled by ffmpeg.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

DEFAULT_SR = 16000
WINDOW_SAMPLES = 64_600     # the first 4 seconds (the AASIST window), the audio input of every speech encoder of the sweep


def window_first_repeat(wav: np.ndarray, n: int = WINDOW_SAMPLES) -> np.ndarray:
    """First n samples; a shorter clip is tiled until it fills the window; an empty clip becomes silence. float32."""
    if wav.size == 0:
        return np.zeros(n, np.float32)
    if wav.size < n:
        wav = np.tile(wav, -(-n // wav.size))
    return wav[:n].astype(np.float32)


def has_audio_stream(path: str | Path) -> bool:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() != ""


def extract_audio(video: str | Path, sr: int = DEFAULT_SR, threads: int = 1,
                  check_stream: bool = False) -> np.ndarray:
    """Audio track of a video as mono float32 at `sr`. Empty array if there is no audio.

    check_stream=True runs ffprobe first (one more process start); by default the
    decode is run directly and an empty output means no audio (or no decodable audio).
    """
    if check_stream and not has_audio_stream(video):
        return np.zeros(0, np.float32)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", str(threads), "-i", str(video),
           "-vn", "-sn", "-map", "0:a:0?", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        if b"does not contain any stream" in r.stderr:  # no audio track: empty, not an error
            return np.zeros(0, np.float32)
        raise RuntimeError(f"ffmpeg failed on {video}: {r.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(r.stdout, np.float32).copy()


def extract_audio_to_wav(video: str | Path, out_wav: str | Path, sr: int = DEFAULT_SR) -> bool:
    """Write the audio of a video as 16-bit mono WAV. Returns False (no file) if there is no audio."""
    x = extract_audio(video, sr)
    if x.size == 0:
        return False
    import soundfile as sf
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), x, sr, subtype="PCM_16")
    return True


def load_audio(path: str | Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """Load a standalone audio file (FLAC, WAV, ...) as mono float32 at `sr`."""
    import soundfile as sf
    try:
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return extract_audio(path, sr)  # formats libsndfile cannot read (e.g. mp3 in old builds)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if file_sr == sr:
        return np.ascontiguousarray(x)
    return extract_audio(path, sr)  # resample with ffmpeg's resampler
