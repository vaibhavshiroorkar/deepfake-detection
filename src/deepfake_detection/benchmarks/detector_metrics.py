from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from deepfake_detection.evaluation.bootstrap import BootstrapInterval
from deepfake_detection.views.tracking import (
    Detection,
    Landmarks5,
    Point,
    select_primary_track,
)

from .detector_annotations import AnnotationAudit, FaceAnnotation, FrameAnnotation
from .detector_sample import SplitRole, _validate_sha256

FROZEN_DETECTOR_RULE_REVISION = "detector-selection-v1"
TARGET_IOU_THRESHOLD = 0.50
MAX_FALSE_DETECTIONS_PER_FRAME = 0.10
RECALL_REJECTION_MARGIN = 0.01
LANDMARK_NME_REJECTION_MARGIN = 0.01
TRACK_ERROR_REJECTION_MARGIN_PER_1000 = 1.0
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260825

EvidenceScope = Literal["research_evidence", "software_fixture_only"]
Association = Literal["greedy_iou", "constant_velocity"]


def _looks_absolute(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _required_string(name: str, value: str, *, path_free: bool = False) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be blank")
    if value != value.strip():
        raise ValueError(f"{name} must be canonical without outer whitespace")
    if path_free and _looks_absolute(value):
        raise ValueError(f"{name} cannot contain a private path")


def _finite(name: str, value: float, *, nonnegative: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_runtime(value: Any, *, name: str = "runtime_snapshot") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        _finite(name, value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("runtime_snapshot keys must be nonblank strings")
            _validate_runtime(item, name=f"runtime_snapshot.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_runtime(item, name=f"runtime_snapshot[{index}]")
        return
    raise TypeError("runtime_snapshot must contain JSON-compatible values")


def _validate_runtime_snapshot(snapshot: Mapping[str, Any]) -> None:
    required = {
        "started_at_utc",
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "packages",
        "cpu",
        "gpu",
        "gpu_memory_mib",
        "available_memory_mib",
        "ffmpeg_version",
    }
    missing = required - set(snapshot)
    if missing:
        raise ValueError(
            "runtime snapshot is missing required fields: " + ", ".join(sorted(missing))
        )
    for name in ("started_at_utc", "git_commit", "python_version", "platform", "cpu"):
        value = snapshot[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"runtime snapshot {name} must be a nonblank string")
    if not isinstance(snapshot["git_dirty"], bool):
        raise ValueError("runtime snapshot git_dirty must be a boolean")
    packages = snapshot["packages"]
    if not isinstance(packages, Mapping) or not all(
        isinstance(name, str)
        and bool(name)
        and (version is None or isinstance(version, str))
        for name, version in packages.items()
    ):
        raise ValueError("runtime snapshot packages must map names to versions")
    for name in ("gpu", "ffmpeg_version"):
        value = snapshot[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"runtime snapshot {name} must be a string or null")
    for name in ("gpu_memory_mib", "available_memory_mib"):
        value = snapshot[name]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(
                f"runtime snapshot {name} must be a nonnegative integer or null"
            )
    _validate_runtime(snapshot)


def _point_mapping(point: Point) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y)}


def _landmark_mapping(landmarks: Landmarks5 | None) -> dict[str, object] | None:
    if landmarks is None:
        return None
    return {
        "eye_left": _point_mapping(landmarks.eye_left),
        "eye_right": _point_mapping(landmarks.eye_right),
        "nose": _point_mapping(landmarks.nose),
        "mouth_left": _point_mapping(landmarks.mouth_left),
        "mouth_right": _point_mapping(landmarks.mouth_right),
    }


def _detection_key(detection: Detection) -> tuple[object, ...]:
    landmarks = _landmark_mapping(detection.landmarks)
    return (
        -detection.confidence,
        detection.box.left,
        detection.box.top,
        detection.box.right,
        detection.box.bottom,
        json.dumps(landmarks, sort_keys=True, separators=(",", ":")),
    )


@dataclass(frozen=True, slots=True)
class CandidateFrame:
    frame_id: str
    clip_id: str
    timestamp_sec: float
    frame_sha256: str
    source_hash: str
    split_role: SplitRole
    detections: tuple[Detection, ...]
    latency_ms: float
    detector_revision: str
    model_sha256: str
    device: str
    thread_count: int

    def __post_init__(self) -> None:
        _required_string("frame_id", self.frame_id, path_free=True)
        _required_string("clip_id", self.clip_id, path_free=True)
        _required_string("detector_revision", self.detector_revision, path_free=True)
        _required_string("device", self.device, path_free=True)
        _validate_sha256("frame_sha256", self.frame_sha256)
        _validate_sha256("source_hash", self.source_hash)
        _validate_sha256("model_sha256", self.model_sha256)
        if self.split_role not in {"calibration", "comparison"}:
            raise ValueError("split_role must be calibration or comparison")
        _finite("timestamp_sec", self.timestamp_sec, nonnegative=True)
        _finite("latency_ms", self.latency_ms, nonnegative=True)
        if not isinstance(self.thread_count, int) or isinstance(
            self.thread_count, bool
        ):
            raise TypeError("thread_count must be an integer")
        if self.thread_count <= 0:
            raise ValueError("thread_count must be positive")
        if not isinstance(self.detections, tuple) or not all(
            isinstance(detection, Detection) for detection in self.detections
        ):
            raise TypeError("detections must be a tuple of Detection values")
        object.__setattr__(
            self, "detections", tuple(sorted(self.detections, key=_detection_key))
        )


@dataclass(frozen=True, slots=True)
class DetectorMetrics:
    target_recall: float
    false_detections_per_frame: float
    non_target_detections_per_frame: float
    non_target_candidate_count: int
    landmark_nme: float | None
    landmark_coverage: float
    aligned_mouth_jitter: float | None

    def __post_init__(self) -> None:
        for name in (
            "target_recall",
            "false_detections_per_frame",
            "non_target_detections_per_frame",
            "landmark_coverage",
        ):
            _finite(name, getattr(self, name), nonnegative=True)
        for name in ("landmark_nme", "aligned_mouth_jitter"):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value, nonnegative=True)
        if self.target_recall > 1 or self.landmark_coverage > 1:
            raise ValueError("Recall and coverage values must be in [0, 1]")
        if (
            not isinstance(self.non_target_candidate_count, int)
            or isinstance(self.non_target_candidate_count, bool)
            or self.non_target_candidate_count < 0
        ):
            raise ValueError("non_target_candidate_count must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class TrackerMetrics:
    association: Association
    stable_track_coverage: float
    abstention_rate: float
    target_track_errors: int
    tracked_frames: int
    target_track_errors_per_1000: float

    def __post_init__(self) -> None:
        if self.association not in {"greedy_iou", "constant_velocity"}:
            raise ValueError("Unknown tracker association")
        for name in (
            "stable_track_coverage",
            "abstention_rate",
            "target_track_errors_per_1000",
        ):
            _finite(name, getattr(self, name), nonnegative=True)
        if self.stable_track_coverage > 1 or self.abstention_rate > 1:
            raise ValueError("Tracker coverage values must be in [0, 1]")
        for name in ("target_track_errors", "tracked_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.target_track_errors < 0 or self.tracked_frames < 0:
            raise ValueError("Tracker counts must be nonnegative")
        if self.target_track_errors > self.tracked_frames:
            raise ValueError("Target-track errors cannot exceed tracked frames")
        if self.abstention_rate != 1 - self.stable_track_coverage:
            raise ValueError("Abstention and stable coverage must be exact complements")
        expected_rate = (
            1000 * self.target_track_errors / self.tracked_frames
            if self.tracked_frames
            else 0.0
        )
        if self.target_track_errors_per_1000 != expected_rate:
            raise ValueError("Target-track error rate must exactly match its counts")


@dataclass(frozen=True, slots=True)
class DetectorLatency:
    timed_frames: int
    median_ms: float
    p95_ms: float
    throughput_fps: float
    device: str
    thread_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.timed_frames, int) or isinstance(
            self.timed_frames, bool
        ):
            raise TypeError("timed_frames must be an integer")
        if self.timed_frames <= 0:
            raise ValueError("Latency requires at least one timed frame")
        for name in ("median_ms", "p95_ms", "throughput_fps"):
            _finite(name, getattr(self, name), nonnegative=True)
        if self.p95_ms < self.median_ms:
            raise ValueError("p95_ms cannot be lower than median_ms")
        _required_string("device", self.device, path_free=True)
        if not isinstance(self.thread_count, int) or isinstance(
            self.thread_count, bool
        ):
            raise TypeError("thread_count must be an integer")
        if self.thread_count <= 0:
            raise ValueError("thread_count must be positive")


def _report_point_estimates(
    metrics: DetectorMetrics,
    trackers: Sequence[TrackerMetrics],
    latency: DetectorLatency,
) -> dict[str, float]:
    values: dict[str, float] = {
        "target_recall": metrics.target_recall,
        "false_detections_per_frame": metrics.false_detections_per_frame,
        "non_target_detections_per_frame": metrics.non_target_detections_per_frame,
        "non_target_candidate_count": float(metrics.non_target_candidate_count),
        "landmark_coverage": metrics.landmark_coverage,
        "latency.median_ms": latency.median_ms,
        "latency.p95_ms": latency.p95_ms,
        "latency.throughput_fps": latency.throughput_fps,
    }
    if metrics.landmark_nme is not None:
        values["landmark_nme"] = metrics.landmark_nme
    if metrics.aligned_mouth_jitter is not None:
        values["aligned_mouth_jitter"] = metrics.aligned_mouth_jitter
    for tracker in trackers:
        prefix = tracker.association
        values[f"{prefix}.stable_track_coverage"] = tracker.stable_track_coverage
        values[f"{prefix}.abstention_rate"] = tracker.abstention_rate
        values[f"{prefix}.target_track_errors_per_1000"] = (
            tracker.target_track_errors_per_1000
        )
    return values


@dataclass(frozen=True, slots=True)
class DetectorBenchmarkReport:
    detector_name: str
    detector_revision: str
    model_sha256: str
    threshold: float
    collection_threshold: float
    evidence_scope: EvidenceScope
    rule_revision: str
    frame_count: int
    source_count: int
    metrics: DetectorMetrics
    trackers: tuple[TrackerMetrics, ...]
    latency: DetectorLatency
    intervals: dict[str, BootstrapInterval]
    runtime_snapshot: dict[str, Any]
    raw_results_sha256: str
    evaluation_set_sha256: str
    annotation_audit_validated: bool
    bootstrap_samples: int = BOOTSTRAP_SAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        _required_string("detector_name", self.detector_name, path_free=True)
        _required_string("detector_revision", self.detector_revision, path_free=True)
        _validate_sha256("model_sha256", self.model_sha256)
        _validate_sha256("raw_results_sha256", self.raw_results_sha256)
        _validate_sha256("evaluation_set_sha256", self.evaluation_set_sha256)
        for name in ("threshold", "collection_threshold"):
            value = getattr(self, name)
            _finite(name, value)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.evidence_scope not in {"research_evidence", "software_fixture_only"}:
            raise ValueError("Unknown detector evidence scope")
        if not isinstance(self.annotation_audit_validated, bool):
            raise TypeError("annotation_audit_validated must be a boolean")
        if (
            self.evidence_scope == "research_evidence"
            and not self.annotation_audit_validated
        ):
            raise ValueError("Research evidence requires a valid annotation audit")
        if (
            self.evidence_scope == "research_evidence"
            and self.latency.device.casefold() != "cpu"
        ):
            raise ValueError("Research evidence requires CPU runtime metadata")
        if self.rule_revision != FROZEN_DETECTOR_RULE_REVISION:
            raise ValueError("Report does not use the frozen detector rule revision")
        if self.frame_count <= 0 or self.source_count <= 0:
            raise ValueError("Report frame and source counts must be positive")
        if self.frame_count != self.latency.timed_frames:
            raise ValueError("Report frame_count must match latency timed_frames")
        if self.bootstrap_samples != BOOTSTRAP_SAMPLES:
            raise ValueError("Detector reports require 1,000 fixed source bootstraps")
        if self.bootstrap_seed != BOOTSTRAP_SEED:
            raise ValueError("Detector reports require the fixed bootstrap seed")
        if len(self.trackers) != 2 or {tracker.association for tracker in self.trackers} != {
            "greedy_iou",
            "constant_velocity",
        }:
            raise ValueError("Report must contain both frozen tracker associations")
        point_estimates = _report_point_estimates(
            self.metrics,
            self.trackers,
            self.latency,
        )
        if set(self.intervals) != set(point_estimates):
            raise ValueError("Report interval keys must exactly match defined metrics")
        for name, interval in self.intervals.items():
            if not isinstance(interval, BootstrapInterval):
                raise TypeError("Report intervals must be BootstrapInterval values")
            if (
                not isinstance(interval.successful_samples, int)
                or isinstance(interval.successful_samples, bool)
                or interval.successful_samples != BOOTSTRAP_SAMPLES
            ):
                raise ValueError(
                    "Every report interval requires 1,000 source resamples"
                )
            for field in ("estimate", "lower", "upper"):
                _finite(f"interval {name}.{field}", getattr(interval, field))
            if interval.lower > interval.upper:
                raise ValueError("Report interval bounds must be ordered")
            if interval.estimate != point_estimates[name]:
                raise ValueError(
                    "Report interval estimate differs from its point metric"
                )
        _validate_runtime_snapshot(self.runtime_snapshot)


@dataclass(frozen=True, slots=True)
class DetectorDecision:
    selected_detector: str | None
    selected_association: Association | None
    eligible_detectors: tuple[str, ...]
    rejected: dict[str, tuple[str, ...]]
    downstream_tie_candidates: tuple[str, ...]
    reason: str
    rule_revision: str


def _candidate_frame_mapping(record: CandidateFrame) -> dict[str, object]:
    return {
        "frame_id": record.frame_id,
        "clip_id": record.clip_id,
        "timestamp_sec": float(record.timestamp_sec),
        "frame_sha256": record.frame_sha256,
        "source_hash": record.source_hash,
        "split_role": record.split_role,
        "detections": [
            {
                "box": {
                    "left": float(detection.box.left),
                    "top": float(detection.box.top),
                    "right": float(detection.box.right),
                    "bottom": float(detection.box.bottom),
                },
                "score": float(detection.confidence),
                "landmarks": _landmark_mapping(detection.landmarks),
            }
            for detection in record.detections
        ],
        "latency_ms": float(record.latency_ms),
        "detector_revision": record.detector_revision,
        "model_sha256": record.model_sha256,
        "device": record.device,
        "thread_count": record.thread_count,
    }


def _candidate_jsonl_bytes(records: Sequence[CandidateFrame]) -> bytes:
    ordered = sorted(records, key=lambda row: row.frame_id)
    text = "".join(
        json.dumps(
            _candidate_frame_mapping(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in ordered
    )
    return text.encode("utf-8")


def _face_annotation_mapping(face: FaceAnnotation) -> dict[str, object]:
    return {
        "box": {
            "left": float(face.box.left),
            "top": float(face.box.top),
            "right": float(face.box.right),
            "bottom": float(face.box.bottom),
        },
        "target": face.target,
        "landmarks": _landmark_mapping(face.landmarks),
    }


def _evaluation_set_hash(
    records: Sequence[CandidateFrame],
    annotations: Mapping[str, FrameAnnotation],
) -> str:
    rows = [
        {
            "frame_id": record.frame_id,
            "clip_id": record.clip_id,
            "timestamp_sec": float(record.timestamp_sec),
            "frame_sha256": record.frame_sha256,
            "source_hash": record.source_hash,
            "split_role": record.split_role,
            "gold": {
                "faces": sorted(
                    (
                        _face_annotation_mapping(face)
                        for face in annotations[record.frame_id].faces
                    ),
                    key=lambda face: json.dumps(
                        face, sort_keys=True, separators=(",", ":")
                    ),
                ),
                "no_suitable_target": annotations[record.frame_id].no_suitable_target,
                "pose": annotations[record.frame_id].pose,
                "lighting": annotations[record.frame_id].lighting,
                "multi_person": annotations[record.frame_id].multi_person,
            },
        }
        for record in sorted(records, key=lambda row: row.frame_id)
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _annotation_map(
    annotations: Sequence[FrameAnnotation],
) -> dict[str, FrameAnnotation]:
    rows: dict[str, FrameAnnotation] = {}
    for annotation in annotations:
        if annotation.frame_id in rows:
            raise ValueError("Evaluation requires one resolved annotation per frame")
        rows[annotation.frame_id] = annotation
    return rows


def _validate_records(
    records: Sequence[CandidateFrame],
    annotations: Mapping[str, FrameAnnotation],
) -> tuple[CandidateFrame, ...]:
    rows = tuple(records)
    if not rows:
        raise ValueError("Detector evaluation requires candidate records")
    frame_ids: set[str] = set()
    source_roles: dict[str, set[str]] = defaultdict(set)
    metadata = {
        (row.detector_revision, row.model_sha256, row.device, row.thread_count)
        for row in rows
    }
    if len(metadata) != 1:
        raise ValueError("Detector candidate metadata is inconsistent")
    for row in rows:
        if row.frame_id in frame_ids:
            raise ValueError("Detector candidate frame identifiers must be unique")
        frame_ids.add(row.frame_id)
        source_roles[row.source_hash].add(row.split_role)
        annotation = annotations.get(row.frame_id)
        if annotation is None:
            raise ValueError(f"Missing resolved annotation for {row.frame_id}")
        if annotation.frame_sha256 != row.frame_sha256:
            raise ValueError(
                f"Detector candidate frame hash differs for {row.frame_id}"
            )
    if set(annotations) != frame_ids:
        raise ValueError("Candidate and resolved annotation frame sets differ")
    if any(len(roles) != 1 for roles in source_roles.values()):
        raise ValueError("Calibration and comparison source identities overlap")
    return tuple(sorted(rows, key=lambda row: row.frame_id))


def _maximum_iou_assignment(
    detections: Sequence[Detection],
    faces: Sequence[FaceAnnotation],
) -> tuple[tuple[int, int, float], ...]:
    if not detections or not faces:
        return ()
    overlaps = np.asarray(
        [[detection.box.iou(face.box) for detection in detections] for face in faces],
        dtype=np.float64,
    )
    face_count, detection_count = overlaps.shape
    costs = np.zeros((face_count, detection_count + face_count), dtype=np.float64)
    costs[:, :detection_count] = -overlaps
    tie_unit = np.finfo(np.float64).eps
    for face_index, detection_index in np.ndindex(overlaps.shape):
        tie_rank = abs(face_index - detection_index) + (
            detection_index / (detection_count + 1)
        )
        costs[face_index, detection_index] += tie_unit * tie_rank
    face_indices, column_indices = linear_sum_assignment(costs)
    pairs = sorted(
        (
            (detection_index, face_index)
            for face_index, detection_index in zip(
                face_indices.tolist(), column_indices.tolist(), strict=True
            )
            if (
                detection_index < detection_count
                and overlaps[face_index, detection_index] >= TARGET_IOU_THRESHOLD
            )
        ),
        key=lambda pair: pair[0],
    )
    return tuple(
        (
            detection_index,
            face_index,
            float(overlaps[face_index, detection_index]),
        )
        for detection_index, face_index in pairs
    )


def _target_face(annotation: FrameAnnotation) -> FaceAnnotation | None:
    return next((face for face in annotation.faces if face.target), None)


def _target_detection(
    detections: Sequence[Detection], annotation: FrameAnnotation
) -> tuple[Detection, float] | None:
    target = _target_face(annotation)
    if target is None or not detections:
        return None
    ranked = sorted(
        (
            (-detection.box.iou(target.box), index, detection)
            for index, detection in enumerate(detections)
        )
    )
    negative_iou, _, detection = ranked[0]
    return detection, -negative_iou


def _points(landmarks: Landmarks5) -> tuple[Point, ...]:
    return (
        landmarks.eye_left,
        landmarks.eye_right,
        landmarks.nose,
        landmarks.mouth_left,
        landmarks.mouth_right,
    )


def _landmark_error(detected: Landmarks5, annotated: Landmarks5) -> float:
    inter_eye = math.hypot(
        annotated.eye_left.x - annotated.eye_right.x,
        annotated.eye_left.y - annotated.eye_right.y,
    )
    if inter_eye <= 0:
        raise ValueError("Annotated target has degenerate inter-eye distance")
    distances = tuple(
        math.hypot(left.x - right.x, left.y - right.y)
        for left, right in zip(_points(detected), _points(annotated), strict=True)
    )
    return sum(distances) / len(distances) / inter_eye


@dataclass(frozen=True, slots=True)
class _FrameCounts:
    frames: int = 1
    target_frames: int = 0
    target_hits: int = 0
    false_detections: int = 0
    non_target_detections: int = 0
    matched_targets: int = 0
    landmark_targets: int = 0
    landmark_error_sum: float = 0.0


def _frame_counts(
    record: CandidateFrame,
    annotation: FrameAnnotation,
    threshold: float,
) -> _FrameCounts:
    detections = tuple(
        detection
        for detection in record.detections
        if detection.confidence >= threshold
    )
    assignment = _maximum_iou_assignment(detections, annotation.faces)
    valid_matches = tuple(
        match for match in assignment if match[2] >= TARGET_IOU_THRESHOLD
    )
    matched_detection_ids = {match[0] for match in valid_matches}
    non_target = sum(
        not annotation.faces[face_index].target for _, face_index, _ in valid_matches
    )
    target = _target_face(annotation)
    if target is None:
        return _FrameCounts(
            false_detections=len(detections) - len(matched_detection_ids),
            non_target_detections=non_target,
        )
    result = _target_detection(detections, annotation)
    hit = result is not None and result[1] >= TARGET_IOU_THRESHOLD
    if not hit or result is None:
        return _FrameCounts(
            target_frames=1,
            false_detections=len(detections) - len(matched_detection_ids),
            non_target_detections=non_target,
        )
    detection = result[0]
    has_landmarks = detection.landmarks is not None
    landmark_error = 0.0
    if has_landmarks:
        if target.landmarks is None:
            raise ValueError("Resolved target annotation is missing landmarks")
        landmark_error = _landmark_error(detection.landmarks, target.landmarks)
    return _FrameCounts(
        target_frames=1,
        target_hits=1,
        false_detections=len(detections) - len(matched_detection_ids),
        non_target_detections=non_target,
        matched_targets=1,
        landmark_targets=int(has_landmarks),
        landmark_error_sum=landmark_error,
    )


def _sum_frame_counts(values: Sequence[_FrameCounts]) -> _FrameCounts:
    return _FrameCounts(
        frames=sum(value.frames for value in values),
        target_frames=sum(value.target_frames for value in values),
        target_hits=sum(value.target_hits for value in values),
        false_detections=sum(value.false_detections for value in values),
        non_target_detections=sum(value.non_target_detections for value in values),
        matched_targets=sum(value.matched_targets for value in values),
        landmark_targets=sum(value.landmark_targets for value in values),
        landmark_error_sum=sum(value.landmark_error_sum for value in values),
    )


def calibrate_detector_threshold(
    records: Sequence[CandidateFrame],
    annotations: Sequence[FrameAnnotation],
    *,
    max_false_detections_per_frame: float = MAX_FALSE_DETECTIONS_PER_FRAME,
) -> float:
    if max_false_detections_per_frame != MAX_FALSE_DETECTIONS_PER_FRAME:
        raise ValueError("Cannot change the frozen false-detection calibration rule")
    annotation_by_id = _annotation_map(annotations)
    rows = _validate_records(records, annotation_by_id)
    return _calibrate_validated(rows, annotation_by_id)


def _calibrate_validated(
    rows: Sequence[CandidateFrame],
    annotation_by_id: Mapping[str, FrameAnnotation],
) -> float:
    calibration = tuple(row for row in rows if row.split_role == "calibration")
    if not calibration:
        raise ValueError("Threshold calibration requires calibration sources")
    thresholds = sorted(
        {
            1.0,
            *(
                detection.confidence
                for row in calibration
                for detection in row.detections
            ),
        }
    )
    candidates: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        counts = _sum_frame_counts(
            tuple(
                _frame_counts(row, annotation_by_id[row.frame_id], threshold)
                for row in calibration
            )
        )
        recall = (
            counts.target_hits / counts.target_frames if counts.target_frames else 0.0
        )
        false_rate = counts.false_detections / counts.frames
        if false_rate <= MAX_FALSE_DETECTIONS_PER_FRAME:
            candidates.append((recall, -false_rate, threshold))
    if not candidates:
        raise ValueError(
            "No detector threshold satisfies the frozen false-detection rule"
        )
    return max(candidates)[2]


@dataclass(frozen=True, slots=True)
class _TrackerCounts:
    stable_frames: int
    total_frames: int
    errors: int
    tracked_frames: int


def _tracker_counts(
    records: Sequence[CandidateFrame],
    annotations: Mapping[str, FrameAnnotation],
    threshold: float,
    association: Association,
) -> _TrackerCounts:
    by_clip: dict[str, list[CandidateFrame]] = defaultdict(list)
    for record in records:
        by_clip[record.clip_id].append(record)
    stable_frames = 0
    total_frames = 0
    errors = 0
    tracked_frames = 0
    for clip_id in sorted(by_clip):
        clip = sorted(
            by_clip[clip_id], key=lambda row: (row.timestamp_sec, row.frame_id)
        )
        filtered = tuple(
            tuple(
                detection
                for detection in row.detections
                if detection.confidence >= threshold
            )
            for row in clip
        )
        selection = select_primary_track(
            filtered,
            min_iou=0.30,
            association=association,
            max_gap=1 if association == "constant_velocity" else 0,
        )
        total_frames += len(clip)
        if not selection.stable:
            continue
        stable_frames += len(selection.frame_indices)
        identity_states: list[bool] = []
        for frame_index, detection in zip(
            selection.frame_indices, selection.detections, strict=True
        ):
            tracked_frames += 1
            target = _target_face(annotations[clip[frame_index].frame_id])
            identity_states.append(
                target is not None
                and detection.box.iou(target.box) >= TARGET_IOU_THRESHOLD
            )
        if identity_states:
            errors += int(not identity_states[0])
            errors += sum(
                current != previous
                for previous, current in zip(
                    identity_states, identity_states[1:], strict=False
                )
            )
    return _TrackerCounts(stable_frames, total_frames, errors, tracked_frames)


def _mouth_residual(
    detection: Detection,
    target: FaceAnnotation,
) -> tuple[float, float] | None:
    if detection.landmarks is None or target.landmarks is None:
        return None
    from deepfake_detection.views.alignment import similarity_transform

    matrix = similarity_transform(target.landmarks, output_size=(224, 224))

    def transform(point: Point) -> np.ndarray:
        return matrix @ np.asarray((point.x, point.y, 1.0), dtype=np.float64)

    detected_center = Point(
        (detection.landmarks.mouth_left.x + detection.landmarks.mouth_right.x) / 2,
        (detection.landmarks.mouth_left.y + detection.landmarks.mouth_right.y) / 2,
    )
    target_center = Point(
        (target.landmarks.mouth_left.x + target.landmarks.mouth_right.x) / 2,
        (target.landmarks.mouth_left.y + target.landmarks.mouth_right.y) / 2,
    )
    canonical_eye_distance = float(
        np.linalg.norm(
            transform(target.landmarks.eye_left) - transform(target.landmarks.eye_right)
        )
    )
    if canonical_eye_distance <= 0:
        raise ValueError("Annotated target has degenerate aligned inter-eye distance")
    residual = (
        transform(detected_center) - transform(target_center)
    ) / canonical_eye_distance
    return float(residual[0]), float(residual[1])


@dataclass(frozen=True, slots=True)
class _JitterCounts:
    total: float
    pairs: int


def _jitter_counts(
    records: Sequence[CandidateFrame],
    annotations: Mapping[str, FrameAnnotation],
    threshold: float,
) -> _JitterCounts:
    by_clip: dict[str, list[CandidateFrame]] = defaultdict(list)
    for record in records:
        by_clip[record.clip_id].append(record)
    total = 0.0
    pairs = 0
    for clip_id in sorted(by_clip):
        previous: tuple[float, float] | None = None
        clip = sorted(
            by_clip[clip_id], key=lambda row: (row.timestamp_sec, row.frame_id)
        )
        for record in clip:
            detections = tuple(
                detection
                for detection in record.detections
                if detection.confidence >= threshold
            )
            annotation = annotations[record.frame_id]
            target = _target_face(annotation)
            result = _target_detection(detections, annotation)
            current = None
            if (
                target is not None
                and result is not None
                and result[1] >= TARGET_IOU_THRESHOLD
            ):
                current = _mouth_residual(result[0], target)
            if previous is not None and current is not None:
                total += math.hypot(current[0] - previous[0], current[1] - previous[1])
                pairs += 1
            previous = current
    return _JitterCounts(total, pairs)


@dataclass(frozen=True, slots=True)
class _SourceSummary:
    frames: _FrameCounts
    greedy: _TrackerCounts
    motion: _TrackerCounts
    jitter: _JitterCounts
    latencies: tuple[float, ...]


def _source_summary(
    records: Sequence[CandidateFrame],
    annotations: Mapping[str, FrameAnnotation],
    threshold: float,
) -> _SourceSummary:
    return _SourceSummary(
        frames=_sum_frame_counts(
            tuple(
                _frame_counts(record, annotations[record.frame_id], threshold)
                for record in records
            )
        ),
        greedy=_tracker_counts(records, annotations, threshold, "greedy_iou"),
        motion=_tracker_counts(records, annotations, threshold, "constant_velocity"),
        jitter=_jitter_counts(records, annotations, threshold),
        latencies=tuple(record.latency_ms for record in records),
    )


def _aggregate_summaries(summaries: Sequence[_SourceSummary]) -> _SourceSummary:
    return _SourceSummary(
        frames=_sum_frame_counts(tuple(summary.frames for summary in summaries)),
        greedy=_TrackerCounts(
            stable_frames=sum(summary.greedy.stable_frames for summary in summaries),
            total_frames=sum(summary.greedy.total_frames for summary in summaries),
            errors=sum(summary.greedy.errors for summary in summaries),
            tracked_frames=sum(summary.greedy.tracked_frames for summary in summaries),
        ),
        motion=_TrackerCounts(
            stable_frames=sum(summary.motion.stable_frames for summary in summaries),
            total_frames=sum(summary.motion.total_frames for summary in summaries),
            errors=sum(summary.motion.errors for summary in summaries),
            tracked_frames=sum(summary.motion.tracked_frames for summary in summaries),
        ),
        jitter=_JitterCounts(
            total=sum(summary.jitter.total for summary in summaries),
            pairs=sum(summary.jitter.pairs for summary in summaries),
        ),
        latencies=tuple(
            latency for summary in summaries for latency in summary.latencies
        ),
    )


def _metric_values(summary: _SourceSummary) -> dict[str, float | None]:
    frames = summary.frames
    latencies = np.asarray(summary.latencies, dtype=np.float64)
    total_latency_seconds = float(latencies.sum()) / 1000
    values: dict[str, float | None] = {
        "target_recall": frames.target_hits / frames.target_frames
        if frames.target_frames
        else 0.0,
        "false_detections_per_frame": frames.false_detections / frames.frames,
        "non_target_detections_per_frame": frames.non_target_detections / frames.frames,
        "non_target_candidate_count": float(frames.non_target_detections),
        "landmark_nme": (
            frames.landmark_error_sum / frames.landmark_targets
            if frames.landmark_targets
            else None
        ),
        "landmark_coverage": (
            frames.landmark_targets / frames.matched_targets
            if frames.matched_targets
            else 0.0
        ),
        "aligned_mouth_jitter": (
            summary.jitter.total / summary.jitter.pairs
            if summary.jitter.pairs
            else None
        ),
        "latency.median_ms": float(np.median(latencies)),
        "latency.p95_ms": float(np.quantile(latencies, 0.95)),
        "latency.throughput_fps": (
            len(latencies) / total_latency_seconds if total_latency_seconds else 0.0
        ),
    }
    for prefix, tracker in (
        ("greedy_iou", summary.greedy),
        ("constant_velocity", summary.motion),
    ):
        stable = (
            tracker.stable_frames / tracker.total_frames
            if tracker.total_frames
            else 0.0
        )
        errors = (
            1000 * tracker.errors / tracker.tracked_frames
            if tracker.tracked_frames
            else 0.0
        )
        values[f"{prefix}.stable_track_coverage"] = stable
        values[f"{prefix}.abstention_rate"] = 1 - stable
        values[f"{prefix}.target_track_errors_per_1000"] = errors
    return values


def _tracker_metrics(
    association: Association, counts: _TrackerCounts
) -> TrackerMetrics:
    coverage = (
        counts.stable_frames / counts.total_frames if counts.total_frames else 0.0
    )
    errors = (
        1000 * counts.errors / counts.tracked_frames if counts.tracked_frames else 0.0
    )
    return TrackerMetrics(
        association=association,
        stable_track_coverage=coverage,
        abstention_rate=1 - coverage,
        target_track_errors=counts.errors,
        tracked_frames=counts.tracked_frames,
        target_track_errors_per_1000=errors,
    )


def _bootstrap_intervals(
    summaries: Sequence[_SourceSummary],
    *,
    samples: int,
    seed: int,
) -> dict[str, BootstrapInterval]:
    if samples != BOOTSTRAP_SAMPLES:
        raise ValueError("Detector evaluation requires 1,000 fixed source bootstraps")
    estimate = _metric_values(_aggregate_summaries(summaries))
    values: dict[str, list[float]] = {
        name: [] for name, value in estimate.items() if value is not None
    }
    generator = np.random.default_rng(seed)
    for _ in range(samples):
        indexes = generator.integers(0, len(summaries), size=len(summaries))
        metrics = _metric_values(
            _aggregate_summaries(tuple(summaries[int(index)] for index in indexes))
        )
        for name in values:
            value = metrics[name]
            if value is None:
                raise ValueError(
                    f"Metric {name} has an undefined fixed bootstrap resample"
                )
            values[name].append(value)
    intervals: dict[str, BootstrapInterval] = {}
    for name, samples_for_metric in values.items():
        intervals[name] = BootstrapInterval(
            estimate=float(estimate[name]),
            lower=float(np.quantile(samples_for_metric, 0.025)),
            upper=float(np.quantile(samples_for_metric, 0.975)),
            successful_samples=len(samples_for_metric),
        )
    return intervals


def evaluate_detector(
    records: Sequence[CandidateFrame],
    annotations: Sequence[FrameAnnotation],
    *,
    threshold: float,
    detector_name: str,
    runtime_snapshot: Mapping[str, Any],
    annotation_audit: AnnotationAudit | None = None,
    evidence_scope: EvidenceScope = "research_evidence",
    collection_threshold: float = 0.0,
    rule_revision: str = FROZEN_DETECTOR_RULE_REVISION,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> DetectorBenchmarkReport:
    if rule_revision != FROZEN_DETECTOR_RULE_REVISION:
        raise ValueError("Cannot change the frozen detector rule revision")
    if bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise ValueError("Detector evaluation requires 1,000 fixed source bootstraps")
    if bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("Detector evaluation requires the fixed bootstrap seed")
    runtime_dict = dict(runtime_snapshot)
    _validate_runtime_snapshot(runtime_dict)
    for name, value in (
        ("threshold", threshold),
        ("collection_threshold", collection_threshold),
    ):
        _finite(name, value)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    annotation_by_id = _annotation_map(annotations)
    rows = _validate_records(records, annotation_by_id)
    calibrated_threshold = _calibrate_validated(rows, annotation_by_id)
    if threshold != calibrated_threshold:
        raise ValueError(
            "Supplied threshold differs from the frozen calibrated threshold"
        )
    if threshold < collection_threshold:
        raise ValueError("Frozen threshold cannot be below the collection threshold")
    audit_validated = False
    if annotation_audit is not None:
        if (
            not isinstance(annotation_audit, AnnotationAudit)
            or not annotation_audit.valid
        ):
            raise ValueError("Detector evaluation requires a valid annotation audit")
        role_sources = {
            role: {row.source_hash for row in rows if row.split_role == role}
            for role in ("calibration", "comparison")
        }
        if (
            annotation_audit.frame_count != len(rows)
            or annotation_audit.source_count
            != len(role_sources["calibration"] | role_sources["comparison"])
            or annotation_audit.calibration_source_count
            != len(role_sources["calibration"])
            or annotation_audit.comparison_source_count
            != len(role_sources["comparison"])
            or annotation_audit.double_review_completed
            != annotation_audit.double_review_required
        ):
            raise ValueError(
                "Annotation audit does not match detector candidate records"
            )
        audit_validated = True
    if evidence_scope == "research_evidence" and not audit_validated:
        raise ValueError("Research evidence requires a valid annotation audit")
    comparison = tuple(row for row in rows if row.split_role == "comparison")
    if not comparison:
        raise ValueError("Detector evaluation requires comparison sources")
    grouped: dict[str, list[CandidateFrame]] = defaultdict(list)
    for row in comparison:
        grouped[row.source_hash].append(row)
    summaries = tuple(
        _source_summary(grouped[source], annotation_by_id, threshold)
        for source in sorted(grouped)
    )
    aggregate = _aggregate_summaries(summaries)
    values = _metric_values(aggregate)
    metrics = DetectorMetrics(
        target_recall=float(values["target_recall"]),
        false_detections_per_frame=float(values["false_detections_per_frame"]),
        non_target_detections_per_frame=float(
            values["non_target_detections_per_frame"]
        ),
        non_target_candidate_count=aggregate.frames.non_target_detections,
        landmark_nme=values["landmark_nme"],
        landmark_coverage=float(values["landmark_coverage"]),
        aligned_mouth_jitter=values["aligned_mouth_jitter"],
    )
    latencies = np.asarray(aggregate.latencies, dtype=np.float64)
    total_seconds = float(latencies.sum()) / 1000
    metadata = next(
        iter(
            {
                (row.detector_revision, row.model_sha256, row.device, row.thread_count)
                for row in rows
            }
        )
    )
    detector_revision, model_sha256, device, thread_count = metadata
    raw_bytes = _candidate_jsonl_bytes(rows)
    return DetectorBenchmarkReport(
        detector_name=detector_name,
        detector_revision=detector_revision,
        model_sha256=model_sha256,
        threshold=threshold,
        collection_threshold=collection_threshold,
        evidence_scope=evidence_scope,
        rule_revision=rule_revision,
        frame_count=len(comparison),
        source_count=len(grouped),
        metrics=metrics,
        trackers=(
            _tracker_metrics("greedy_iou", aggregate.greedy),
            _tracker_metrics("constant_velocity", aggregate.motion),
        ),
        latency=DetectorLatency(
            timed_frames=len(comparison),
            median_ms=float(np.median(latencies)),
            p95_ms=float(np.quantile(latencies, 0.95)),
            throughput_fps=len(comparison) / total_seconds if total_seconds else 0.0,
            device=device,
            thread_count=thread_count,
        ),
        intervals=_bootstrap_intervals(
            summaries,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        runtime_snapshot=runtime_dict,
        raw_results_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        evaluation_set_sha256=_evaluation_set_hash(rows, annotation_by_id),
        annotation_audit_validated=audit_validated,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _best_tracker(report: DetectorBenchmarkReport) -> TrackerMetrics:
    order = {"greedy_iou": 0, "constant_velocity": 1}
    usable_trackers = tuple(
        tracker for tracker in report.trackers if tracker.tracked_frames > 0
    )
    return min(
        usable_trackers or report.trackers,
        key=lambda tracker: (
            tracker.target_track_errors_per_1000,
            -tracker.stable_track_coverage,
            order[tracker.association],
        ),
    )


def _comparison_runtime_key(report: DetectorBenchmarkReport) -> str:
    names = ("git_commit", "python_version", "platform", "packages", "cpu")
    values = {name: report.runtime_snapshot[name] for name in names}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _within_frozen_margin(difference: float, margin: float) -> bool:
    return difference < margin or math.isclose(
        difference,
        margin,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def compare_detectors(
    reports: Sequence[DetectorBenchmarkReport],
    *,
    downstream_validation: Mapping[str, float] | None = None,
    rule_revision: str = FROZEN_DETECTOR_RULE_REVISION,
) -> DetectorDecision:
    if rule_revision != FROZEN_DETECTOR_RULE_REVISION:
        raise ValueError("Cannot change the frozen detector rule revision")
    rows = tuple(reports)
    if not rows:
        raise ValueError("Detector comparison requires reports")
    names = tuple(report.detector_name for report in rows)
    if len(set(names)) != len(names):
        raise ValueError("Detector report names must be unique")
    if any(report.rule_revision != rule_revision for report in rows):
        raise ValueError("Detector reports use inconsistent frozen rules")
    scopes = {report.evidence_scope for report in rows}
    if scopes == {"software_fixture_only"}:
        return DetectorDecision(
            selected_detector=None,
            selected_association=None,
            eligible_detectors=tuple(sorted(names)),
            rejected={},
            downstream_tie_candidates=(),
            reason="software_fixture_only",
            rule_revision=rule_revision,
        )
    if scopes != {"research_evidence"}:
        raise ValueError("Cannot mix fixture and research detector evidence")
    if any(report.latency.device.lower() != "cpu" for report in rows):
        raise ValueError("Frozen detector selection requires CPU latency reports")
    if len({report.latency.thread_count for report in rows}) != 1:
        raise ValueError("Detector reports use different CPU thread counts")
    if len({str(report.runtime_snapshot.get("cpu")) for report in rows}) != 1:
        raise ValueError("Detector reports use different CPU hardware")
    if len({_comparison_runtime_key(report) for report in rows}) != 1:
        raise ValueError("Detector reports use different runtime environments")
    if len({report.evaluation_set_sha256 for report in rows}) != 1:
        raise ValueError("Detector reports use different comparison frames")
    if any(report.metrics.landmark_nme is None for report in rows):
        raise ValueError("Detector selection requires landmark NME for every candidate")

    best_recall = max(report.metrics.target_recall for report in rows)
    recall_eligible = tuple(
        report
        for report in rows
        if _within_frozen_margin(
            best_recall - report.metrics.target_recall,
            RECALL_REJECTION_MARGIN,
        )
    )
    best_nme = min(float(report.metrics.landmark_nme) for report in recall_eligible)
    best_track_error = min(
        _best_tracker(report).target_track_errors_per_1000 for report in recall_eligible
    )
    nme_eligible = tuple(
        report
        for report in recall_eligible
        if _within_frozen_margin(
            float(report.metrics.landmark_nme) - best_nme,
            LANDMARK_NME_REJECTION_MARGIN,
        )
    )
    eligible = tuple(
        report
        for report in nme_eligible
        if _within_frozen_margin(
            _best_tracker(report).target_track_errors_per_1000 - best_track_error,
            TRACK_ERROR_REJECTION_MARGIN_PER_1000,
        )
    )
    eligible_names = tuple(sorted(report.detector_name for report in eligible))
    rejected: dict[str, tuple[str, ...]] = {}
    for report in sorted(rows, key=lambda item: item.detector_name):
        reasons: list[str] = []
        if report not in recall_eligible:
            reasons.append("target_recall_margin")
        else:
            if report not in nme_eligible:
                reasons.append("landmark_nme_margin")
            track_difference = (
                _best_tracker(report).target_track_errors_per_1000 - best_track_error
            )
            if not _within_frozen_margin(
                track_difference,
                TRACK_ERROR_REJECTION_MARGIN_PER_1000,
            ):
                reasons.append("target_track_error_margin")
        if reasons:
            rejected[report.detector_name] = tuple(reasons)

    if not eligible:
        return DetectorDecision(
            selected_detector=None,
            selected_association=None,
            eligible_detectors=(),
            rejected=rejected,
            downstream_tie_candidates=(),
            reason="no_eligible_detector",
            rule_revision=rule_revision,
        )

    fastest = min(report.latency.median_ms for report in eligible)
    speed_ties = tuple(
        sorted(
            report.detector_name
            for report in eligible
            if report.latency.median_ms == fastest
        )
    )
    selected_name: str | None = speed_ties[0] if len(speed_ties) == 1 else None
    reason = "fastest_eligible_cpu"
    downstream_ties: tuple[str, ...] = ()
    if len(speed_ties) > 1:
        downstream_ties = speed_ties
        reason = "downstream_validation_required"
        if downstream_validation is not None and all(
            name in downstream_validation for name in speed_ties
        ):
            scores = {name: float(downstream_validation[name]) for name in speed_ties}
            for score in scores.values():
                _finite("downstream validation score", score)
            best_score = max(scores.values())
            winners = tuple(
                sorted(name for name, score in scores.items() if score == best_score)
            )
            if len(winners) == 1:
                selected_name = winners[0]
                downstream_ties = ()
                reason = "downstream_validation_tie_break"
    selected_report = next(
        (report for report in eligible if report.detector_name == selected_name), None
    )
    return DetectorDecision(
        selected_detector=selected_name,
        selected_association=(
            _best_tracker(selected_report).association
            if selected_report is not None
            else None
        ),
        eligible_detectors=eligible_names,
        rejected=rejected,
        downstream_tie_candidates=downstream_ties,
        reason=reason,
        rule_revision=rule_revision,
    )
