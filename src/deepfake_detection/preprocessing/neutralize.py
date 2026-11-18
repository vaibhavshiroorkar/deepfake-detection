"""Make real and fake clips leave the encoder looking the same.

FakeAVCeleb shows why this exists. Its real clips are H.264; 92 percent of its
face-swapped clips are MPEG-4 Part 2. Its generated-voice clips carry MP3 audio,
start with about 30 ms of digital silence and end with about 1.2 s of it. Each
of those cues alone separates the classes at 0.97 AUC or better
(results/shortcut-audit). A detector trained on the raw files can learn the
pipeline instead of the manipulation.

Every clip is treated identically:

1. Decode the audio once, to 16 kHz mono samples.
2. Scale the samples to a fixed RMS level. This is a plain gain in numpy, with
   no filter look-ahead, so no filter can add a clip-dependent delay. Scaling
   comes first so that "silence" means the same thing in every clip.
3. Find the first and last sample above -60 dBFS. Round the start up to a
   whole video frame, and keep exactly `fixed_seconds` from there, so every
   clip has the same length. Generated speech is shorter than real speech, so
   duration alone was a cue. A clip too short for the fixed span is refused and
   counted.
4. Add dither, quiet noise at -66 dBFS, seeded by the file name. Speech
   synthesizers write exact digital zeros in their pauses; microphones never
   do. Dither removes the zeros without hiding that a pause is clean.
5. Encode the video for that span (H.264, 25 fps, no B-frames) and mux it with
   the cut audio as AAC. Strip container metadata.

Version 2 detected silence before scaling, so quiet tails crossed the
threshold differently by class, and it kept each clip's natural length.
Version 1 cut with an ffmpeg input seek and normalized with `loudnorm`. That
re-muxed each source's AAC priming differently: face-swapped clips gained 28 ms
of leading silence that real clips did not, so the shortcut came back
(results/shortcut-audit/fakeavceleb-neutral-v1). Version 2 never lets ffmpeg
touch the audio timeline before encoding.

The cut offset is returned so that labels with timestamps (LAV-DF's forged
segments) can be shifted by the same amount.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# A hostile or broken upload must not hold the only scoring worker forever (docs/SECURITY.md).
FFMPEG_TIMEOUT = 300  # seconds; the website's longest input is a 60 s video

SAMPLE_RATE = 16_000
SILENCE_DBFS = -60.0
VERSION = 4


@dataclass(frozen=True, slots=True)
class NeutralSettings:
    fps: int = 25
    crf: int = 23
    preset: str = "medium"
    audio_bitrate: str = "64k"
    audio_rate: int = SAMPLE_RATE
    loudness_dbfs: float = -23.0
    dither_dbfs: float = -66.0
    # One frame size for a whole corpus, "WxH", stretched (not padded). None keeps
    # the native size, which is right when a corpus is already uniform.
    size: str | None = None
    silence_dbfs: float = SILENCE_DBFS
    # Every neutral clip is exactly this long; shorter clips are refused.
    fixed_seconds: float = 3.0
    # None keeps v4's constant quality (crf). A value such as "160k" encodes the video at that
    # constant bitrate instead (P-001 amendment A18, exploratory).
    video_bitrate: str | None = None
    version: int = VERSION

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__slots__}


def silence_bounds(
    wave_samples: np.ndarray,
    rate: int = SAMPLE_RATE,
    threshold_dbfs: float = SILENCE_DBFS,
) -> tuple[float, float]:
    """Seconds of digital silence at the start and at the end of a mono wave."""
    if wave_samples.size == 0:
        return 0.0, 0.0
    loud = np.flatnonzero(np.abs(wave_samples) > 10 ** (threshold_dbfs / 20))
    if loud.size == 0:
        return 0.0, 0.0
    return float(loud[0]) / rate, float(wave_samples.size - 1 - loud[-1]) / rate


def cut_window(
    duration: float, leading: float, trailing: float, fixed: float, fps: int = 25
) -> tuple[float, float] | None:
    """A span of exactly `fixed` seconds starting on the first whole frame of sound.

    Returns None when the sounding part of the clip is shorter than `fixed`.
    The start rounds up, so no leading silence is kept.
    """
    start = math.ceil(leading * fps - 1e-9) / fps
    if start + fixed > duration - trailing + 1e-9:
        return None
    return start, start + fixed


def sound_windows(
    duration: float, leading: float, trailing: float, fixed: float, fps: int = 25
) -> list[tuple[float, float]]:
    """Back-to-back spans of `fixed` seconds that together cover all the sound.

    The first span is `cut_window`'s. When the sound does not divide evenly, the
    last span ends where the sound ends, overlapping the one before it, so no
    second of sound is left out. Empty when the sound is shorter than `fixed`.
    """
    first = cut_window(duration, leading, trailing, fixed, fps)
    if first is None:
        return []
    end_of_sound = duration - trailing
    starts = [first[0]]
    while starts[-1] + 2 * fixed <= end_of_sound + 1e-9:
        starts.append(round(starts[-1] + fixed, 6))
    last = math.floor((end_of_sound - fixed) * fps + 1e-9) / fps
    if last > starts[-1] + 1e-9:
        starts.append(last)
    return [(s, round(s + fixed, 6)) for s in starts]


def normalize_level(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale to a fixed RMS level. A plain gain: no look-ahead, no delay."""
    rms = (
        float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if samples.size
        else 0.0
    )
    if rms <= 0:
        return samples.astype(np.float32)
    return np.clip(samples * (10 ** (target_dbfs / 20) / rms), -1.0, 1.0).astype(
        np.float32
    )


def dither(samples: np.ndarray, level_dbfs: float, seed_text: str) -> np.ndarray:
    """Add Gaussian noise at a fixed RMS level, the same recipe for every clip."""
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
    noise = np.random.default_rng(seed).normal(
        0.0, 10 ** (level_dbfs / 20), samples.shape
    )
    return (samples + noise).astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _video_filter(settings: NeutralSettings) -> str:
    steps = []
    if settings.size:
        width, height = (int(v) for v in settings.size.lower().split("x"))
        steps += [f"scale={width}:{height}:flags=bicubic", "setsar=1"]
    return ",".join([*steps, f"fps={settings.fps}", "format=yuv420p"])


def _rate_control(settings: NeutralSettings) -> list[str]:
    if settings.video_bitrate is None:
        return ["-crf", str(settings.crf)]
    rate = settings.video_bitrate
    return [
        "-b:v",
        rate,
        "-minrate",
        rate,
        "-maxrate",
        rate,
        "-bufsize",
        rate,
        "-x264-params",
        "nal-hrd=cbr",
    ]


def ffmpeg_command(
    source: Path,
    target: Path,
    start: float,
    end: float,
    audio_wav: Path | None,
    settings: NeutralSettings,
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{start:.4f}",
        "-to",
        f"{end:.4f}",
        "-i",
        str(source),
    ]
    if audio_wav is not None:
        command += ["-i", str(audio_wav)]
    command += [
        "-map_metadata",
        "-1",
        "-map",
        "0:v:0",
        "-vf",
        _video_filter(settings),
        "-c:v",
        "libx264",
        "-preset",
        settings.preset,
        *_rate_control(settings),
        "-bf",
        "0",
    ]
    if audio_wav is not None:
        command += [
            "-map",
            "1:a:0",
            "-c:a",
            "aac",
            "-b:a",
            settings.audio_bitrate,
            "-shortest",
        ]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", str(target)]
    return command


def _read_audio(path: Path) -> np.ndarray:
    run = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0?",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
        timeout=FFMPEG_TIMEOUT,
    )
    return (
        np.frombuffer(run.stdout, np.float32)
        if run.returncode == 0
        else np.zeros(0, np.float32)
    )


def _duration(path: Path) -> float:
    run = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=FFMPEG_TIMEOUT,
    )
    return float(json.loads(run.stdout)["format"]["duration"])


def plan_windows(source: Path, settings: NeutralSettings) -> list[tuple[float, float]]:
    """Every `fixed_seconds` span of `source` that holds sound, from `sound_windows`."""
    audio = normalize_level(_read_audio(source), settings.loudness_dbfs)
    leading, trailing = silence_bounds(audio, SAMPLE_RATE, settings.silence_dbfs)
    return sound_windows(
        _duration(source), leading, trailing, settings.fixed_seconds, settings.fps
    )


def neutralize_clip(
    source: Path, target: Path, settings: NeutralSettings, start: float | None = None
) -> dict[str, object]:
    """Write the neutral copy and report what was cut.

    `start` picks the span (one from `plan_windows`); None keeps the first span of sound.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    audio = normalize_level(_read_audio(source), settings.loudness_dbfs)
    duration = _duration(source)
    if start is None:
        leading, trailing = silence_bounds(audio, SAMPLE_RATE, settings.silence_dbfs)
        window = cut_window(
            duration, leading, trailing, settings.fixed_seconds, settings.fps
        )
    else:
        window = (start, start + settings.fixed_seconds)
    if window is None:
        return {"error": f"too short: under {settings.fixed_seconds} s of sound"}
    start, end = window
    with tempfile.TemporaryDirectory() as scratch:
        wav = None
        if audio.size:
            span = audio[round(start * SAMPLE_RATE) : round(end * SAMPLE_RATE)]
            wav = Path(scratch) / "audio.wav"
            # Scale the kept span again so every clip has the same level.
            level = normalize_level(span, settings.loudness_dbfs)
            write_wav(
                wav, dither(level, settings.dither_dbfs, source.name), SAMPLE_RATE
            )
        run = subprocess.run(
            ffmpeg_command(source, target, start, end, wav, settings),
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT,
        )
    if run.returncode != 0 or not target.is_file():
        return {"error": run.stderr.decode(errors="replace")[:200]}
    return {
        "offset_sec": round(start, 4),
        "kept_sec": round(end - start, 4),
        "cut_leading_sec": round(start, 4),
        "cut_trailing_sec": round(duration - end, 4),
        "has_audio": audio.size > 0,
    }
