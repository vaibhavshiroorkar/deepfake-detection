from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

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
class Detection:
    box: Box
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackSelection:
    frame_indices: tuple[int, ...]
    detections: tuple[Detection, ...]
    coverage: float
    stable: bool


def select_primary_track(
    frames: tuple[tuple[Detection, ...], ...],
    *,
    min_iou: float,
    min_coverage: float = 0.80,
    min_dominance_ratio: float = 1.25,
) -> TrackSelection:
    if not 0 <= min_iou <= 1:
        raise ValueError("Minimum IoU must be in [0, 1]")
    tracks: list[list[tuple[int, Detection]]] = []
    for frame_index, detections in enumerate(frames):
        unused = set(range(len(detections)))
        for track in sorted(tracks, key=len, reverse=True):
            candidates = [
                (track[-1][1].box.iou(detections[index].box), index) for index in unused
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
