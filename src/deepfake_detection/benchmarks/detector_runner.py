from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from deepfake_detection.experiments.runtime import RuntimeSnapshot
from deepfake_detection.views.tracking import Box, Detection, Landmarks5, Point

from .detector_annotations import (
    FrameAnnotation,
    resolve_annotations,
    validate_annotations,
)
from .detector_metrics import (
    BOOTSTRAP_SEED,
    FROZEN_DETECTOR_RULE_REVISION,
    CandidateFrame,
    DetectorBenchmarkReport,
    EvidenceScope,
    _candidate_jsonl_bytes,
    calibrate_detector_threshold,
    evaluate_detector,
)
from .detector_sample import ReviewFrame, _validate_sha256


class DetectorBackend(Protocol):
    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]: ...

    def runtime_metadata(self) -> Mapping[str, object]: ...


def _backend_runtime_metadata(detector: DetectorBackend) -> tuple[str, int]:
    metadata = detector.runtime_metadata()
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "device",
        "thread_count",
    }:
        raise ValueError(
            "Detector runtime metadata must contain device and thread_count"
        )
    device = metadata["device"]
    thread_count = metadata["thread_count"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError("Detector runtime device must be a nonblank string")
    if not isinstance(thread_count, int) or isinstance(thread_count, bool):
        raise TypeError("Detector runtime thread_count must be an integer")
    if thread_count <= 0:
        raise ValueError("Detector runtime thread_count must be positive")
    return device, thread_count


def _frame_hash(frame: np.ndarray) -> str:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame_reader must return a NumPy array")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError(
            "Benchmark source frames must be nonempty three-channel images"
        )
    contiguous = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _read_verified_frame(
    row: ReviewFrame,
    frame_reader: Callable[[ReviewFrame], np.ndarray],
) -> np.ndarray:
    frame = frame_reader(row)
    digest = _frame_hash(frame)
    if digest != row.frame_sha256:
        raise ValueError(f"Source frame hash differs for {row.frame_id}")
    if frame.shape[:2] != (row.height, row.width):
        raise ValueError(f"Source frame dimensions differ for {row.frame_id}")
    return frame


def write_candidate_records(records: Sequence[CandidateFrame], path: Path) -> None:
    payload = _candidate_jsonl_bytes(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _required_mapping(row: object, name: str) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ValueError(f"{name} must be an object")
    return row


def _point_from_mapping(row: object) -> Point:
    values = _required_mapping(row, "Landmark point")
    if set(values) != {"x", "y"}:
        raise ValueError("Landmark point must contain x and y")
    return Point(float(values["x"]), float(values["y"]))


def _landmarks_from_mapping(row: object) -> Landmarks5 | None:
    if row is None:
        return None
    values = _required_mapping(row, "Landmarks")
    names = {"eye_left", "eye_right", "nose", "mouth_left", "mouth_right"}
    if set(values) != names:
        raise ValueError("Landmarks must contain the five canonical points")
    return Landmarks5(
        eye_left=_point_from_mapping(values["eye_left"]),
        eye_right=_point_from_mapping(values["eye_right"]),
        nose=_point_from_mapping(values["nose"]),
        mouth_left=_point_from_mapping(values["mouth_left"]),
        mouth_right=_point_from_mapping(values["mouth_right"]),
    )


def _detection_from_mapping(row: object) -> Detection:
    values = _required_mapping(row, "Detection")
    if set(values) != {"box", "score", "landmarks"}:
        raise ValueError("Detection contains unknown or missing fields")
    box = _required_mapping(values["box"], "Detection box")
    if set(box) != {"left", "top", "right", "bottom"}:
        raise ValueError("Detection box contains unknown or missing fields")
    return Detection(
        box=Box(
            float(box["left"]),
            float(box["top"]),
            float(box["right"]),
            float(box["bottom"]),
        ),
        confidence=float(values["score"]),
        landmarks=_landmarks_from_mapping(values["landmarks"]),
    )


def _required_string(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _candidate_from_mapping(row: object) -> CandidateFrame:
    values = _required_mapping(row, "Candidate frame")
    names = {
        "frame_id",
        "clip_id",
        "timestamp_sec",
        "frame_sha256",
        "source_hash",
        "split_role",
        "detections",
        "latency_ms",
        "detector_revision",
        "model_sha256",
        "device",
        "thread_count",
    }
    if set(values) != names:
        raise ValueError("Candidate frame contains unknown or missing fields")
    detections = values["detections"]
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    split_role = _required_string(values, "split_role")
    if split_role not in {"calibration", "comparison"}:
        raise ValueError("split_role must be calibration or comparison")
    thread_count = values["thread_count"]
    if not isinstance(thread_count, int) or isinstance(thread_count, bool):
        raise ValueError("thread_count must be an integer")
    return CandidateFrame(
        frame_id=_required_string(values, "frame_id"),
        clip_id=_required_string(values, "clip_id"),
        timestamp_sec=float(values["timestamp_sec"]),
        frame_sha256=_required_string(values, "frame_sha256"),
        source_hash=_required_string(values, "source_hash"),
        split_role=split_role,
        detections=tuple(_detection_from_mapping(item) for item in detections),
        latency_ms=float(values["latency_ms"]),
        detector_revision=_required_string(values, "detector_revision"),
        model_sha256=_required_string(values, "model_sha256"),
        device=_required_string(values, "device"),
        thread_count=thread_count,
    )


def read_candidate_records(path: Path) -> tuple[CandidateFrame, ...]:
    records: list[CandidateFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Candidate JSONL line {line_number} is blank")
            try:
                records.append(_candidate_from_mapping(json.loads(line)))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid candidate JSONL line {line_number}: {error}"
                ) from error
    if not records:
        raise ValueError("Candidate JSONL cannot be empty")
    ordered = tuple(sorted(records, key=lambda row: row.frame_id))
    if tuple(records) != ordered:
        raise ValueError("Candidate JSONL rows must use canonical frame order")
    if len({record.frame_id for record in records}) != len(records):
        raise ValueError("Candidate JSONL frame identifiers must be unique")
    return ordered


def validate_candidate_artifact(path: Path) -> tuple[CandidateFrame, ...]:
    """Validate the exact path-free candidate JSONL schema before upload."""

    records = read_candidate_records(path)
    for record in records:
        for name, value in (
            ("frame_id", record.frame_id),
            ("clip_id", record.clip_id),
            ("detector_revision", record.detector_revision),
            ("device", record.device),
        ):
            if "/" in value or "\\" in value:
                raise ValueError(f"Candidate JSONL {name} must be path-free")
    canonical = _candidate_jsonl_bytes(records)
    if path.read_bytes() != canonical:
        raise ValueError("Candidate JSONL must use the exact canonical schema")
    return records


def _runtime_mapping(snapshot: RuntimeSnapshot | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(snapshot, RuntimeSnapshot):
        return snapshot.as_dict()
    if not isinstance(snapshot, Mapping):
        raise TypeError("runtime_snapshot must be RuntimeSnapshot or a mapping")
    return dict(snapshot)


def run_detector_benchmark(
    *,
    sample: Sequence[ReviewFrame],
    annotations: Sequence[FrameAnnotation],
    detector: DetectorBackend,
    detector_name: str,
    detector_revision: str,
    model_sha256: str,
    frame_reader: Callable[[ReviewFrame], np.ndarray],
    raw_output: Path,
    runtime_snapshot: RuntimeSnapshot | Mapping[str, Any],
    collection_threshold: float = 0.0,
    warmup_frames: int = 3,
    clock: Callable[[], float] = time.perf_counter,
    evidence_scope: EvidenceScope = "research_evidence",
    rule_revision: str = FROZEN_DETECTOR_RULE_REVISION,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    source_run_id: str = "software-fixture",
    environment_lock_sha256: str = "0" * 64,
) -> DetectorBenchmarkReport:
    sample_rows = tuple(sample)
    annotation_rows = tuple(annotations)
    audit = validate_annotations(sample_rows, annotation_rows)
    if not audit.valid:
        details = "; ".join(audit.errors)
        raise ValueError(f"Cannot benchmark an invalid annotation audit: {details}")
    resolved = resolve_annotations(sample_rows, annotation_rows)
    _validate_sha256("model_sha256", model_sha256)
    if not detector_name.strip() or not detector_revision.strip():
        raise ValueError("Detector name and revision must be nonblank")
    if not isinstance(warmup_frames, int) or isinstance(warmup_frames, bool):
        raise TypeError("warmup_frames must be an integer")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be nonnegative")
    if not math.isfinite(collection_threshold) or not 0 <= collection_threshold <= 1:
        raise ValueError("collection_threshold must be finite and in [0, 1]")
    backend_threshold = getattr(detector, "confidence", collection_threshold)
    if (
        isinstance(backend_threshold, bool)
        or not isinstance(backend_threshold, (int, float))
        or not math.isfinite(float(backend_threshold))
        or float(backend_threshold) != collection_threshold
    ):
        raise ValueError("Detector collection threshold differs from recorded metadata")
    if rule_revision != FROZEN_DETECTOR_RULE_REVISION:
        raise ValueError("Cannot change the frozen detector rule revision")

    ordered = tuple(sorted(sample_rows, key=lambda row: row.frame_id))
    for row in ordered[:warmup_frames]:
        frame = _read_verified_frame(row, frame_reader)
        detections = detector.detect(frame)
        if not isinstance(detections, tuple) or not all(
            isinstance(detection, Detection) for detection in detections
        ):
            raise TypeError("Detector must return a tuple of Detection values")

    device, thread_count = _backend_runtime_metadata(detector)
    records: list[CandidateFrame] = []
    for row in ordered:
        frame = _read_verified_frame(row, frame_reader)
        started = float(clock())
        detections = detector.detect(frame)
        finished = float(clock())
        if not isinstance(detections, tuple) or not all(
            isinstance(detection, Detection) for detection in detections
        ):
            raise TypeError("Detector must return a tuple of Detection values")
        records.append(
            CandidateFrame(
                frame_id=row.frame_id,
                clip_id=row.clip_id,
                timestamp_sec=row.timestamp_sec,
                frame_sha256=row.frame_sha256,
                source_hash=row.source_hash,
                split_role=row.split_role,
                detections=detections,
                latency_ms=(finished - started) * 1000,
                detector_revision=detector_revision,
                model_sha256=model_sha256,
                device=device,
                thread_count=thread_count,
            )
        )
    if _backend_runtime_metadata(detector) != (device, thread_count):
        raise ValueError("Detector runtime metadata changed during timed inference")
    candidate_rows = tuple(records)
    threshold = calibrate_detector_threshold(candidate_rows, resolved)
    effective_scope: EvidenceScope = (
        "software_fixture_only"
        if sample_rows
        and all(row.dataset.casefold() == "fixture" for row in sample_rows)
        else evidence_scope
    )
    report = evaluate_detector(
        candidate_rows,
        resolved,
        threshold=threshold,
        detector_name=detector_name,
        runtime_snapshot=_runtime_mapping(runtime_snapshot),
        annotation_audit=audit,
        evidence_scope=effective_scope,
        collection_threshold=collection_threshold,
        rule_revision=rule_revision,
        bootstrap_seed=bootstrap_seed,
        source_run_id=source_run_id,
        environment_lock_sha256=environment_lock_sha256,
    )
    write_candidate_records(candidate_rows, raw_output)
    output_hash = hashlib.sha256(raw_output.read_bytes()).hexdigest()
    if output_hash != report.raw_results_sha256:
        raise RuntimeError("Written detector candidates differ from evaluated records")
    return report
