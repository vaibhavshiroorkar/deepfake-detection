from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.benchmarks.detector_annotations import (
    FaceAnnotation,
    FrameAnnotation,
)
from deepfake_detection.benchmarks.detector_runner import (
    read_candidate_records,
    run_detector_benchmark,
    write_candidate_records,
)
from deepfake_detection.benchmarks.detector_sample import ReviewFrame
from deepfake_detection.experiments.runtime import RuntimeSnapshot
from deepfake_detection.views.tracking import Box, Detection, Landmarks5, Point


def _frame_hash(frame: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _landmarks() -> Landmarks5:
    return Landmarks5(
        eye_left=Point(4, 5),
        eye_right=Point(12, 5),
        nose=Point(8, 9),
        mouth_left=Point(5, 13),
        mouth_right=Point(11, 13),
    )


def _valid_review() -> tuple[
    tuple[ReviewFrame, ...],
    tuple[FrameAnnotation, ...],
    dict[str, np.ndarray],
]:
    sample: list[ReviewFrame] = []
    annotations: list[FrameAnnotation] = []
    frames: dict[str, np.ndarray] = {}
    for index in range(500):
        source_index = index // 5
        frame = np.zeros((20, 20, 3), dtype=np.uint16)
        frame[0, 0, 0] = index
        frame_hash = _frame_hash(frame)
        frame_id = f"frame-{index:03d}"
        row = ReviewFrame(
            frame_id=frame_id,
            dataset="fixture",
            clip_id=f"clip-{source_index:03d}",
            source_hash=hashlib.sha256(
                f"source-{source_index:03d}".encode()
            ).hexdigest(),
            timestamp_sec=float(index % 5),
            frame_sha256=frame_hash,
            width=20,
            height=20,
            split_role="calibration" if source_index < 20 else "comparison",
            double_review=index < 50,
            manipulation_type=f"manipulation-{source_index % 4}",
            method=f"method-{source_index % 4}",
            race=f"race-{source_index % 4}",
            gender=f"gender-{source_index % 2}",
        )
        review = FrameAnnotation(
            frame_id=frame_id,
            frame_sha256=frame_hash,
            reviewer_id="reviewer-a",
            faces=(FaceAnnotation(Box(1, 1, 16, 18), True, _landmarks()),),
            no_suitable_target=False,
            pose="frontal",
            lighting="even",
            multi_person=False,
        )
        sample.append(row)
        annotations.append(review)
        if index < 50:
            annotations.append(replace(review, reviewer_id="reviewer-b"))
        frames[frame_id] = frame
    return tuple(sample), tuple(annotations), frames


class _Detector:
    def __init__(self) -> None:
        self.calls = 0
        self.confidence = 0.0

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        self.calls += 1
        return (Detection(Box(1, 1, 16, 18), 0.8, _landmarks()),)


def _runtime() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        started_at_utc="2026-08-25T00:00:00+00:00",
        git_commit="a" * 40,
        git_dirty=False,
        python_version="3.13.0",
        platform="fixture-platform",
        packages={"opencv-python": "5.0.0.93"},
        cpu="fixture-cpu",
        gpu=None,
        gpu_memory_mib=None,
        available_memory_mib=1024,
        ffmpeg_version="fixture-ffmpeg",
    )


def test_runner_requires_a_valid_audit_before_detector_inference(
    tmp_path: Path,
) -> None:
    sample, annotations, frames = _valid_review()
    detector = _Detector()

    with pytest.raises(ValueError, match="invalid annotation audit"):
        run_detector_benchmark(
            sample=sample,
            annotations=annotations[:-1],
            detector=detector,
            detector_name="fixture-detector",
            detector_revision="revision-1",
            model_sha256="a" * 64,
            frame_reader=lambda row: frames[row.frame_id],
            raw_output=tmp_path / "raw.jsonl",
            runtime_snapshot=_runtime(),
            device="cpu",
            thread_count=2,
            evidence_scope="software_fixture_only",
        )

    assert detector.calls == 0
    assert not (tmp_path / "raw.jsonl").exists()


def test_runner_rejects_missing_independent_double_review(tmp_path: Path) -> None:
    sample, annotations, frames = _valid_review()
    incomplete = tuple(
        annotation
        for annotation in annotations
        if not (
            annotation.frame_id == "frame-000"
            and annotation.reviewer_id == "reviewer-b"
        )
    )
    detector = _Detector()

    with pytest.raises(ValueError, match="double review"):
        run_detector_benchmark(
            sample=sample,
            annotations=incomplete,
            detector=detector,
            detector_name="fixture-detector",
            detector_revision="revision-1",
            model_sha256="a" * 64,
            frame_reader=lambda row: frames[row.frame_id],
            raw_output=tmp_path / "raw.jsonl",
            runtime_snapshot=_runtime(),
            device="cpu",
            thread_count=2,
        )

    assert detector.calls == 0


def test_runner_warms_up_times_frames_and_writes_path_free_deterministic_jsonl(
    tmp_path: Path,
) -> None:
    sample, annotations, frames = _valid_review()
    detector = _Detector()
    elapsed = iter(index * 0.002 for index in range(1001))
    output = tmp_path / "raw.jsonl"

    report = run_detector_benchmark(
        sample=tuple(reversed(sample)),
        annotations=tuple(reversed(annotations)),
        detector=detector,
        detector_name="fixture-detector",
        detector_revision="revision-1",
        model_sha256="a" * 64,
        frame_reader=lambda row: frames[row.frame_id],
        raw_output=output,
        runtime_snapshot=_runtime(),
        device="cpu",
        thread_count=2,
        warmup_frames=2,
        clock=lambda: next(elapsed),
    )

    records = read_candidate_records(output)
    assert detector.calls == 502
    assert len(records) == 500
    assert records[0].frame_id == "frame-000"
    assert all(record.latency_ms == pytest.approx(2.0) for record in records)
    assert report.evidence_scope == "software_fixture_only"
    assert report.threshold == 0.8
    assert report.frame_count == 400
    assert report.latency.timed_frames == 400
    text = output.read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in text
    assert "frame_reader" not in text
    first_row = json.loads(text.splitlines()[0])
    assert first_row["detector_revision"] == "revision-1"
    assert first_row["model_sha256"] == "a" * 64
    assert first_row["frame_sha256"] == sample[0].frame_sha256
    assert first_row["split_role"] == "calibration"
    assert first_row["detections"][0]["landmarks"]["eye_left"] == {
        "x": 4,
        "y": 5,
    }

    second_output = tmp_path / "second.jsonl"
    write_candidate_records(tuple(reversed(records)), second_output)
    assert second_output.read_bytes() == output.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == report.raw_results_sha256


def test_runner_rejects_changed_source_pixels_and_metadata_before_output(
    tmp_path: Path,
) -> None:
    sample, annotations, frames = _valid_review()
    changed = dict(frames)
    changed[sample[0].frame_id] = np.ones((20, 20, 3), dtype=np.uint16)

    with pytest.raises(ValueError, match="frame hash"):
        run_detector_benchmark(
            sample=sample,
            annotations=annotations,
            detector=_Detector(),
            detector_name="fixture-detector",
            detector_revision="revision-1",
            model_sha256="a" * 64,
            frame_reader=lambda row: changed[row.frame_id],
            raw_output=tmp_path / "raw.jsonl",
            runtime_snapshot=_runtime(),
            device="cpu",
            thread_count=2,
            evidence_scope="software_fixture_only",
        )

    with pytest.raises(ValueError, match="model_sha256"):
        run_detector_benchmark(
            sample=sample,
            annotations=annotations,
            detector=_Detector(),
            detector_name="fixture-detector",
            detector_revision="revision-1",
            model_sha256="not-a-hash",
            frame_reader=lambda row: frames[row.frame_id],
            raw_output=tmp_path / "raw.jsonl",
            runtime_snapshot=_runtime(),
            device="cpu",
            thread_count=2,
            evidence_scope="software_fixture_only",
        )


def test_candidate_jsonl_rejects_private_paths_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    sample, annotations, frames = _valid_review()
    detector = _Detector()
    ticks = iter(index * 0.001 for index in range(1001))
    output = tmp_path / "raw.jsonl"
    run_detector_benchmark(
        sample=sample,
        annotations=annotations,
        detector=detector,
        detector_name="fixture-detector",
        detector_revision="revision-1",
        model_sha256="a" * 64,
        frame_reader=lambda row: frames[row.frame_id],
        raw_output=output,
        runtime_snapshot=_runtime(),
        device="cpu",
        thread_count=2,
        warmup_frames=0,
        clock=lambda: next(ticks),
        evidence_scope="software_fixture_only",
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    row["detector_revision"] = str(tmp_path.resolve())
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="private path"):
        read_candidate_records(output)

    row["detector_revision"] = "revision-1"
    row["latency_ms"] = float("nan")
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        read_candidate_records(output)


def test_runner_rejects_a_collection_threshold_metadata_mismatch(
    tmp_path: Path,
) -> None:
    sample, annotations, frames = _valid_review()
    detector = _Detector()
    detector.confidence = 0.2

    with pytest.raises(ValueError, match="collection threshold"):
        run_detector_benchmark(
            sample=sample,
            annotations=annotations,
            detector=detector,
            detector_name="fixture-detector",
            detector_revision="revision-1",
            model_sha256="a" * 64,
            frame_reader=lambda row: frames[row.frame_id],
            raw_output=tmp_path / "raw.jsonl",
            runtime_snapshot=_runtime(),
            device="cpu",
            thread_count=2,
            collection_threshold=0.1,
        )

    assert detector.calls == 0
