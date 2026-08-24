from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from deepfake_detection.data.manifest import ClipRecord

from .alignment import aligned_lower_face
from .cache import cache_fingerprint, preprocessing_config_hash
from .contracts import PreparedClip, QualityReport
from .timeline import ViewConfig, make_sync_window, uniform_timestamps
from .tracking import Box, Detection, TrackSelection, select_primary_track


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_sec: float
    video_fps: float
    audio_duration_sec: float
    audio_present: bool


class MediaDecoder(Protocol):
    def probe(self, path: Path) -> MediaInfo: ...

    def read_frames(
        self, path: Path, timestamps_sec: tuple[float, ...]
    ) -> tuple[np.ndarray, ...]: ...

    def read_audio(
        self,
        path: Path,
        *,
        start_sec: float,
        duration_sec: float,
        sample_rate: int,
    ) -> np.ndarray: ...


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]: ...


def _pad_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(flattened) >= length:
        return flattened[:length]
    return np.pad(flattened, (0, length - len(flattened)))


def _normalize_and_pad(
    values: np.ndarray,
    *,
    length: int,
    valid_samples: int,
) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = min(len(flattened), length, max(0, valid_samples))
    output = np.zeros(length, dtype=np.float32)
    if valid == 0:
        return output
    content = flattened[:valid]
    standard_deviation = float(content.std())
    if standard_deviation > 1e-7:
        output[:valid] = (content - float(content.mean())) / standard_deviation
    return output


def _expanded_square(
    box: Box, frame: np.ndarray, margin: float
) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    box_width = box.right - box.left
    box_height = box.bottom - box.top
    side = max(box_width, box_height) * (1 + 2 * margin)
    center_x = (box.left + box.right) / 2
    center_y = (box.top + box.bottom) / 2
    left = max(0, round(center_x - side / 2))
    top = max(0, round(center_y - side / 2))
    right = min(width, round(center_x + side / 2))
    bottom = min(height, round(center_y + side / 2))
    if right <= left or bottom <= top:
        raise ValueError("Face box has no pixels after clamping")
    return left, top, right, bottom


def _normalize_image(
    image: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (rgb - mean) / standard_deviation
    return normalized.transpose(2, 0, 1)


def _filled_detections(
    selection: TrackSelection,
    frame_count: int,
) -> tuple[Detection, ...]:
    known = dict(zip(selection.frame_indices, selection.detections, strict=True))
    return tuple(
        known[min(known, key=lambda candidate: abs(candidate - index))]
        for index in range(frame_count)
    )


def _face_view(
    frames: tuple[np.ndarray, ...],
    selection: TrackSelection,
    *,
    height: int,
    width: int,
    margin: float,
) -> np.ndarray | None:
    if not selection.stable:
        return None
    crops: list[np.ndarray] = []
    for frame, detection in zip(
        frames, _filled_detections(selection, len(frames)), strict=True
    ):
        left, top, right, bottom = _expanded_square(detection.box, frame, margin)
        crops.append(
            _normalize_image(frame[top:bottom, left:right], height=height, width=width)
        )
    return np.stack(crops)


def _mouth_view(
    frames: tuple[np.ndarray, ...],
    selection: TrackSelection,
    *,
    height: int,
    width: int,
    mode: str,
) -> tuple[np.ndarray | None, float]:
    if not selection.stable:
        return None, 0.0 if mode == "landmark" else 1.0
    filled = _filled_detections(selection, len(frames))
    if mode == "landmark":
        aligned: list[np.ndarray] = []
        valid_landmarks = 0
        for frame, detection in zip(frames, filled, strict=True):
            if detection.landmarks is None:
                continue
            try:
                aligned.append(
                    aligned_lower_face(
                        frame,
                        detection.landmarks,
                        height=height,
                        width=width,
                    )
                )
            except ValueError:
                continue
            valid_landmarks += 1
        coverage = valid_landmarks / len(frames) if frames else 0.0
        if valid_landmarks != len(frames):
            return None, coverage
        return np.stack(aligned), coverage

    crops: list[np.ndarray] = []
    for frame, detection in zip(frames, filled, strict=True):
        box = detection.box
        mouth = Box(
            box.left,
            box.top + 0.52 * (box.bottom - box.top),
            box.right,
            box.bottom,
        )
        left, top, right, bottom = _expanded_square(mouth, frame, 0.10)
        crops.append(
            _normalize_image(frame[top:bottom, left:right], height=height, width=width)
        )
    return np.stack(crops), 1.0


class Preprocessor:
    def __init__(
        self,
        *,
        decoder: MediaDecoder,
        detector: FaceDetector,
        config: ViewConfig,
        code_version: str,
    ) -> None:
        self.decoder = decoder
        self.detector = detector
        self.config = config
        self.code_version = code_version

    def _track(self, frames: tuple[np.ndarray, ...]) -> TrackSelection:
        detections = tuple(self.detector.detect(frame) for frame in frames)
        return select_primary_track(detections, min_iou=0.30)

    def prepare(self, record: ClipRecord, media_path: Path) -> PreparedClip:
        info = self.decoder.probe(media_path)
        if info.duration_sec <= 0:
            raise ValueError("Video duration must be positive")
        content_start = (
            min(record.leading_silence_sec, info.duration_sec)
            if self.config.remove_leading_silence
            else 0.0
        )
        if content_start >= info.duration_sec:
            content_start = 0.0

        visual_timestamps = uniform_timestamps(
            duration_sec=info.duration_sec,
            count=self.config.visual_frames,
            start_sec=content_start,
        )
        sync_start = min(
            content_start,
            max(0.0, info.duration_sec - self.config.sync_seconds),
        )
        context_lower = content_start + self.config.sync_max_offset_seconds
        context_upper = min(
            info.duration_sec - self.config.sync_seconds,
            info.audio_duration_sec
            - self.config.sync_seconds
            - self.config.sync_max_offset_seconds,
        )
        context_available = info.audio_present and context_lower <= context_upper
        if context_available:
            sync_start = context_lower
        sync_window = make_sync_window(start_sec=sync_start, config=self.config)
        final_video_timestamp = max(
            0.0,
            info.duration_sec - 0.5 / self.config.sync_fps,
        )
        sync_timestamps = tuple(
            min(timestamp, final_video_timestamp)
            for timestamp in sync_window.video_timestamps_sec
        )
        visual_frames = self.decoder.read_frames(media_path, visual_timestamps)
        sync_frames = self.decoder.read_frames(media_path, sync_timestamps)
        if len(visual_frames) != self.config.visual_frames:
            raise ValueError("Decoder returned the wrong number of visual frames")
        if len(sync_frames) != len(sync_window.video_timestamps_sec):
            raise ValueError("Decoder returned the wrong number of sync frames")

        visual_track = self._track(visual_frames)
        sync_track = self._track(sync_frames)
        visual_view = _face_view(
            visual_frames,
            visual_track,
            height=self.config.visual_height,
            width=self.config.visual_width,
            margin=self.config.crop_margin,
        )
        sync_video, landmark_coverage = _mouth_view(
            sync_frames,
            sync_track,
            height=self.config.sync_height,
            width=self.config.sync_width,
            mode=self.config.mouth_crop_mode,
        )

        audio_view = None
        sync_audio = None
        sync_audio_context = None
        audio_clipped = False
        if info.audio_present:
            audio_start = min(
                content_start,
                max(0.0, info.audio_duration_sec - self.config.audio_seconds),
            )
            audio_raw = self.decoder.read_audio(
                media_path,
                start_sec=audio_start,
                duration_sec=self.config.audio_seconds,
                sample_rate=self.config.sample_rate,
            )
            audio_valid_samples = round(
                min(
                    self.config.audio_seconds,
                    max(0.0, info.audio_duration_sec - audio_start),
                )
                * self.config.sample_rate
            )
            audio_view = _normalize_and_pad(
                audio_raw,
                length=round(self.config.audio_seconds * self.config.sample_rate),
                valid_samples=audio_valid_samples,
            )
            sync_raw = self.decoder.read_audio(
                media_path,
                start_sec=sync_start,
                duration_sec=self.config.sync_seconds,
                sample_rate=self.config.sample_rate,
            )
            sync_valid_samples = round(
                min(
                    self.config.sync_seconds,
                    max(0.0, info.audio_duration_sec - sync_start),
                )
                * self.config.sample_rate
            )
            sync_audio = _normalize_and_pad(
                sync_raw,
                length=sync_window.audio_sample_count,
                valid_samples=sync_valid_samples,
            )
            if context_available:
                context_duration = (
                    self.config.sync_seconds + 2 * self.config.sync_max_offset_seconds
                )
                sync_audio_context = _pad_or_trim(
                    self.decoder.read_audio(
                        media_path,
                        start_sec=sync_start - self.config.sync_max_offset_seconds,
                        duration_sec=context_duration,
                        sample_rate=self.config.sample_rate,
                    ),
                    round(context_duration * self.config.sample_rate),
                )
            audio_clipped = bool(np.max(np.abs(audio_raw), initial=0.0) >= 0.999)

        quality = QualityReport(
            face_coverage=min(visual_track.coverage, sync_track.coverage),
            stable_face_track=visual_track.stable and sync_track.stable,
            audio_present=info.audio_present,
            audio_clipped=audio_clipped,
            av_duration_delta_sec=(
                abs(info.duration_sec - info.audio_duration_sec)
                if info.audio_present
                else 0.0
            ),
            sync_duration_sufficient=(
                info.duration_sec >= self.config.sync_seconds
                and (
                    not info.audio_present
                    or info.audio_duration_sec >= self.config.sync_seconds
                )
            ),
            landmark_coverage=landmark_coverage,
        )
        return PreparedClip(
            clip_id=record.clip_id,
            visual_view=visual_view,
            audio_view=audio_view,
            sync_video_view=sync_video,
            sync_audio_view=sync_audio,
            quality=quality,
            preprocessing_fingerprint=cache_fingerprint(
                media_path,
                dataset=record.dataset,
                config=self.config,
                code_version=self.code_version,
                leading_silence_sec=record.leading_silence_sec,
            ),
            sync_audio_context=sync_audio_context,
            preprocessing_config_hash=preprocessing_config_hash(
                config=self.config,
                code_version=self.code_version,
            ),
        )
