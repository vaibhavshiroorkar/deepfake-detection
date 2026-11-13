"""Make real and fake clips leave the encoder looking the same.

FakeAVCeleb shows why this exists. Its real clips are H.264; 92 percent of its
face-swapped clips are MPEG-4 Part 2. Clips with generated audio start with
about 30 ms of digital silence and end with about 1.2 s of it, while real clips
have neither. Each of those cues alone separates the classes at 0.97 AUC or
better (results/shortcut-audit). A detector trained on the raw files can learn
the pipeline instead of the manipulation.

The fix treats every clip identically:

1. Find the first and last audio sample above -60 dBFS.
2. Cut video and audio together at those two instants, so lip-sync is kept.
3. Re-encode with one codec, frame rate, pixel format and audio format.
4. Normalise loudness, and strip container metadata.

The cut offset is returned so that labels with timestamps (LAV-DF's forged
segments) can be shifted by the same amount.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
SILENCE_DBFS = -60.0


@dataclass(frozen=True, slots=True)
class NeutralSettings:
    fps: int = 25
    crf: int = 23
    preset: str = "medium"
    audio_bitrate: str = "64k"
    audio_rate: int = SAMPLE_RATE
    loudness_lufs: float = -23.0
    silence_dbfs: float = SILENCE_DBFS
    # Never cut a clip below this; a clip that is mostly silence keeps its length.
    minimum_seconds: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}


def silence_bounds(
    wave: np.ndarray, rate: int = SAMPLE_RATE, threshold_dbfs: float = SILENCE_DBFS
) -> tuple[float, float]:
    """Seconds of digital silence at the start and at the end of a mono wave."""
    if wave.size == 0:
        return 0.0, 0.0
    loud = np.flatnonzero(np.abs(wave) > 10 ** (threshold_dbfs / 20))
    if loud.size == 0:
        return 0.0, 0.0
    return float(loud[0]) / rate, float(wave.size - 1 - loud[-1]) / rate


def cut_window(
    duration: float, leading: float, trailing: float, minimum: float
) -> tuple[float, float]:
    """Start and end of the kept span, falling back to the whole clip if too short."""
    start, end = leading, duration - trailing
    if end - start < minimum:
        return 0.0, duration
    return start, end


def ffmpeg_command(
    source: Path, target: Path, start: float, end: float, has_audio: bool, settings: NeutralSettings
) -> list[str]:
    # One input seek before -i cuts both streams at the same instant.
    command = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-ss", f"{start:.4f}", "-to", f"{end:.4f}", "-i", str(source),
        "-map_metadata", "-1", "-map", "0:v:0",
        "-vf", f"fps={settings.fps},format=yuv420p",
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
    ]
    if has_audio:
        command += [
            "-map", "0:a:0",
            "-af", f"loudnorm=I={settings.loudness_lufs}:TP=-2:LRA=11",
            "-c:a", "aac", "-b:a", settings.audio_bitrate,
            "-ar", str(settings.audio_rate), "-ac", "1",
        ]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(target)]
    return command


def _read_audio(path: Path) -> np.ndarray:
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0?", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        capture_output=True, check=False,
    )
    return np.frombuffer(run.stdout, np.float32) if run.returncode == 0 else np.zeros(0, np.float32)


def _duration(path: Path) -> float:
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(run.stdout)["format"]["duration"])


def neutralize_clip(source: Path, target: Path, settings: NeutralSettings) -> dict[str, object]:
    """Write the neutral copy and report what was cut."""
    target.parent.mkdir(parents=True, exist_ok=True)
    wave = _read_audio(source)
    duration = _duration(source)
    leading, trailing = silence_bounds(wave, SAMPLE_RATE, settings.silence_dbfs)
    start, end = cut_window(duration, leading, trailing, settings.minimum_seconds)
    run = subprocess.run(
        ffmpeg_command(source, target, start, end, wave.size > 0, settings),
        capture_output=True, check=False,
    )
    if run.returncode != 0 or not target.is_file():
        return {"error": run.stderr.decode(errors="replace")[:200]}
    return {
        "offset_sec": round(start, 4),
        "kept_sec": round(end - start, 4),
        "cut_leading_sec": round(start, 4),
        "cut_trailing_sec": round(duration - end, 4),
        "has_audio": wave.size > 0,
    }
