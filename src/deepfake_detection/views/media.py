from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np

from .preprocessor import MediaInfo


class FFmpegMediaDecoder:
    def probe(self, path: Path) -> MediaInfo:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"Media has no video stream: {path}")
            video = container.streams.video[0]
            if video.duration is not None and video.time_base is not None:
                duration = float(video.duration * video.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                raise ValueError(f"Media duration is unavailable: {path}")
            video_fps = float(video.average_rate) if video.average_rate else 0.0
            audio_present = bool(container.streams.audio)
            audio_duration = 0.0
            if audio_present:
                audio = container.streams.audio[0]
                if audio.duration is not None and audio.time_base is not None:
                    audio_duration = float(audio.duration * audio.time_base)
                else:
                    audio_duration = duration
        return MediaInfo(
            duration_sec=duration,
            video_fps=video_fps,
            audio_duration_sec=audio_duration,
            audio_present=audio_present,
        )

    def read_frames(
        self,
        path: Path,
        timestamps_sec: tuple[float, ...],
    ) -> tuple[np.ndarray, ...]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frames: list[np.ndarray] = []
        try:
            for timestamp in timestamps_sec:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1_000)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"Cannot decode frame at {timestamp:.3f} seconds from {path}"
                    )
                frames.append(frame)
        finally:
            capture.release()
        return tuple(frames)

    def read_audio(
        self,
        path: Path,
        *,
        start_sec: float,
        duration_sec: float,
        sample_rate: int,
    ) -> np.ndarray:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to decode audio")
        # The command uses a fixed argument list and never invokes a shell.
        process = subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-ss",
                str(start_sec),
                "-i",
                str(path),
                "-t",
                str(duration_sec),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        expected = round(duration_sec * sample_rate)
        waveform = np.frombuffer(process.stdout, dtype="<f4").astype(np.float32)
        if len(waveform) < expected:
            waveform = np.pad(waveform, (0, expected - len(waveform)))
        return waveform[:expected]
