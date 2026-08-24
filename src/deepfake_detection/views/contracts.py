from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class QualityReport:
    face_coverage: float
    stable_face_track: bool
    audio_present: bool
    audio_clipped: bool
    av_duration_delta_sec: float
    sync_duration_sufficient: bool = True
    landmark_coverage: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.face_coverage <= 1:
            raise ValueError("Face coverage must be in [0, 1]")
        if self.av_duration_delta_sec < 0:
            raise ValueError("Duration difference cannot be negative")
        if not 0 <= self.landmark_coverage <= 1:
            raise ValueError("Landmark coverage must be in [0, 1]")

    def full_fusion_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if not self.audio_present:
            blockers.append("missing_audio")
        if not self.stable_face_track:
            blockers.append("unstable_face_track")
        if self.face_coverage < 0.80:
            blockers.append("low_face_coverage")
        if self.av_duration_delta_sec > 0.25:
            blockers.append("av_duration_mismatch")
        if not self.sync_duration_sufficient:
            blockers.append("insufficient_sync_duration")
        if self.landmark_coverage < 1.0:
            blockers.append("missing_face_landmarks")
        return tuple(blockers)


@dataclass(frozen=True, slots=True)
class PreparedClip:
    clip_id: str
    visual_view: Any | None
    audio_view: Any | None
    sync_video_view: Any | None
    sync_audio_view: Any | None
    quality: QualityReport
    preprocessing_fingerprint: str
    sync_audio_context: Any | None = None
    preprocessing_config_hash: str = ""
