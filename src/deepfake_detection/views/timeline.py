from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ViewConfig:
    visual_frames: int = 16
    visual_height: int = 224
    visual_width: int = 224
    audio_seconds: float = 4.0
    sync_seconds: float = 2.0
    sync_fps: int = 25
    sync_height: int = 112
    sync_width: int = 112
    sync_max_offset_seconds: float = 0.32
    sample_rate: int = 16_000
    eval_overlap: float = 0.5
    crop_margin: float = 0.20
    detector_confidence: float = 0.80
    detector: Literal["mtcnn", "yunet"] = "mtcnn"
    detector_model_sha256: str | None = None
    remove_leading_silence: bool = True
    mouth_crop_mode: Literal["box", "landmark"] = "box"
    track_association: Literal["greedy_iou", "constant_velocity"] = "greedy_iou"
    track_max_gap: int = 0

    def __post_init__(self) -> None:
        if self.visual_frames <= 0 or self.sync_fps <= 0 or self.sample_rate <= 0:
            raise ValueError("Frame and sample counts must be positive")
        if self.audio_seconds <= 0 or self.sync_seconds <= 0:
            raise ValueError("Window durations must be positive")
        if self.sync_max_offset_seconds < 0:
            raise ValueError("Maximum sync offset cannot be negative")
        if not 0 <= self.eval_overlap < 1:
            raise ValueError("Evaluation overlap must be in [0, 1)")
        if self.mouth_crop_mode not in {"box", "landmark"}:
            raise ValueError("Mouth crop mode must be 'box' or 'landmark'")
        if self.track_association not in {"greedy_iou", "constant_velocity"}:
            raise ValueError(
                "Track association must be 'greedy_iou' or 'constant_velocity'"
            )
        if self.detector not in {"mtcnn", "yunet"}:
            raise ValueError("Detector must be 'mtcnn' or 'yunet'")
        if self.detector_model_sha256 is not None:
            digest = self.detector_model_sha256
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("Detector model hash must be a lowercase SHA-256")
        if self.track_max_gap < 0:
            raise ValueError("Track gap cannot be negative")


@dataclass(frozen=True, slots=True)
class SyncWindow:
    start_sec: float
    video_timestamps_sec: tuple[float, ...]
    audio_start_sample: int
    audio_sample_count: int


def uniform_timestamps(
    *,
    duration_sec: float,
    count: int,
    start_sec: float = 0.0,
) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("Timestamp count must be positive")
    if duration_sec <= start_sec or start_sec < 0:
        raise ValueError("Sampling interval must have positive duration")
    step = (duration_sec - start_sec) / count
    return tuple(start_sec + (index + 0.5) * step for index in range(count))


def sliding_window_starts(
    *,
    duration_sec: float,
    window_sec: float,
    overlap: float,
) -> tuple[float, ...]:
    if duration_sec <= 0 or window_sec <= 0:
        raise ValueError("Durations must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be in [0, 1)")
    if duration_sec <= window_sec:
        return (0.0,)

    final_start = duration_sec - window_sec
    stride = window_sec * (1 - overlap)
    starts: list[float] = []
    current = 0.0
    while current <= final_start + 1e-9:
        starts.append(round(current, 9))
        current += stride
    if abs(starts[-1] - final_start) > 1e-9:
        starts.append(final_start)
    return tuple(starts)


def make_sync_window(*, start_sec: float, config: ViewConfig) -> SyncWindow:
    if start_sec < 0:
        raise ValueError("Sync window start cannot be negative")
    frame_count = round(config.sync_seconds * config.sync_fps)
    frame_period = 1.0 / config.sync_fps
    timestamps = tuple(
        start_sec + (index + 0.5) * frame_period for index in range(frame_count)
    )
    return SyncWindow(
        start_sec=start_sec,
        video_timestamps_sec=timestamps,
        audio_start_sample=round(start_sec * config.sample_rate),
        audio_sample_count=round(config.sync_seconds * config.sample_rate),
    )
