from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


def _require_finite(name: str, *values: float) -> None:
    if not all(isfinite(value) for value in values):
        raise ValueError(f"{name} values must be finite")


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        _require_finite("Box", self.left, self.top, self.right, self.bottom)

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)

    def iou(self, other: Box) -> float:
        intersection = Box(
            max(self.left, other.left),
            max(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
        ).area
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        _require_finite("Point", self.x, self.y)


@dataclass(frozen=True, slots=True)
class Landmarks5:
    eye_left: Point
    eye_right: Point
    nose: Point
    mouth_left: Point
    mouth_right: Point

    def __post_init__(self) -> None:
        points = (
            self.eye_left,
            self.eye_right,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        )
        if not all(isinstance(point, Point) for point in points):
            raise TypeError("Landmark values must be Point instances")
        if self.eye_left.x > self.eye_right.x:
            left, right = self.eye_right, self.eye_left
            object.__setattr__(self, "eye_left", left)
            object.__setattr__(self, "eye_right", right)
        if self.mouth_left.x > self.mouth_right.x:
            left, right = self.mouth_right, self.mouth_left
            object.__setattr__(self, "mouth_left", left)
            object.__setattr__(self, "mouth_right", right)


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    confidence: float
    landmarks: Landmarks5 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            raise TypeError("Detection box must be a Box")
        if self.box.right <= self.box.left or self.box.bottom <= self.box.top:
            raise ValueError("Detection box must have positive width and height")
        _require_finite("Detection confidence", self.confidence)
        if not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be in [0, 1]")
        if self.landmarks is not None and not isinstance(self.landmarks, Landmarks5):
            raise TypeError("Detection landmarks must be Landmarks5 or None")


@dataclass(frozen=True, slots=True)
class TrackSelection:
    frame_indices: tuple[int, ...]
    detections: tuple[Detection, ...]
    coverage: float
    stable: bool


_Track = list[tuple[int, Detection]]


def _predicted_box(track: _Track, frame_index: int) -> Box:
    last_frame, last_detection = track[-1]
    if len(track) < 2:
        return last_detection.box
    previous_frame, previous_detection = track[-2]
    observation_elapsed = last_frame - previous_frame
    prediction_elapsed = frame_index - last_frame
    previous = previous_detection.box
    last = last_detection.box

    def predict(previous_value: float, last_value: float) -> float:
        velocity = (last_value - previous_value) / observation_elapsed
        return last_value + velocity * prediction_elapsed

    return Box(
        predict(previous.left, last.left),
        predict(previous.top, last.top),
        predict(previous.right, last.right),
        predict(previous.bottom, last.bottom),
    )


def _constant_velocity_tracks(
    frames: tuple[tuple[Detection, ...], ...],
    *,
    min_iou: float,
    max_gap: int,
) -> list[_Track]:
    tracks: list[_Track] = []
    for frame_index, detections in enumerate(frames):
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            missing_frames = frame_index - track[-1][0] - 1
            if missing_frames > max_gap:
                continue
            prediction = _predicted_box(track, frame_index)
            for detection_index, detection in enumerate(detections):
                overlap = prediction.iou(detection.box)
                if overlap >= min_iou:
                    candidates.append((-overlap, track_index, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            tracks[track_index].append((frame_index, detections[detection_index]))
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                tracks.append([(frame_index, detection)])
    return tracks


def select_primary_track(
    frames: tuple[tuple[Detection, ...], ...],
    *,
    min_iou: float,
    association: Literal["greedy_iou", "constant_velocity"] = "greedy_iou",
    max_gap: int = 0,
    min_coverage: float = 0.80,
    min_dominance_ratio: float = 1.25,
) -> TrackSelection:
    if not 0 <= min_iou <= 1:
        raise ValueError("Minimum IoU must be in [0, 1]")
    if association not in {"greedy_iou", "constant_velocity"}:
        raise ValueError("Association must be 'greedy_iou' or 'constant_velocity'")
    if max_gap < 0:
        raise ValueError("Maximum track gap cannot be negative")
    if association == "constant_velocity":
        tracks = _constant_velocity_tracks(
            frames,
            min_iou=min_iou,
            max_gap=max_gap,
        )
    else:
        tracks = []
        for frame_index, detections in enumerate(frames):
            unused = set(range(len(detections)))
            for track in sorted(tracks, key=len, reverse=True):
                candidates = [
                    (track[-1][1].box.iou(detections[index].box), index)
                    for index in unused
                ]
                if not candidates:
                    continue
                overlap, index = max(candidates)
                if overlap >= min_iou:
                    track.append((frame_index, detections[index]))
                    unused.remove(index)
            for index in sorted(unused):
                tracks.append([(frame_index, detections[index])])

    if not tracks:
        return TrackSelection((), (), 0.0, False)
    ranked = sorted(
        tracks,
        key=lambda track: (
            len(track),
            sum(item.confidence for _, item in track) / len(track),
        ),
        reverse=True,
    )
    primary = ranked[0]
    coverage = len(primary) / len(frames) if frames else 0.0
    second_length = len(ranked[1]) if len(ranked) > 1 else 0
    dominance = len(primary) / second_length if second_length else float("inf")
    stable = coverage >= min_coverage and dominance >= min_dominance_ratio
    return TrackSelection(
        frame_indices=tuple(index for index, _ in primary),
        detections=tuple(detection for _, detection in primary),
        coverage=coverage,
        stable=stable,
    )
