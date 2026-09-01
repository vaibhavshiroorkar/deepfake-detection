from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepfake_detection.benchmarks.detector_metrics import (
    BOOTSTRAP_SEED,
    FROZEN_DETECTOR_RULE_REVISION,
    CandidateFrame,
    DetectorBenchmarkReport,
    DetectorLatency,
    DetectorMetrics,
    TrackerMetrics,
)
from deepfake_detection.benchmarks.detector_runner import write_candidate_records
from deepfake_detection.evaluation.bootstrap import BootstrapInterval
from deepfake_detection.experiments import training_log
from deepfake_detection.experiments.training_log import (
    log_binary_training,
    log_fusion_training,
    log_sync_training,
)
from deepfake_detection.training.binary import (
    BinaryEpochRecord,
    BinaryTrainingHistory,
)
from deepfake_detection.training.sync import SyncEpochRecord, SyncTrainingHistory
from deepfake_detection.views.tracking import Box, Detection


@dataclass
class FakeLogger:
    params: list[dict[str, object]]
    metrics: list[tuple[dict[str, float], int | None]]
    artifacts: list[tuple[Path, str | None]]

    @property
    def run_id(self) -> str:
        return "run"

    def log_params(self, values: dict[str, object]) -> None:
        self.params.append(dict(values))

    def log_metrics(self, values: dict[str, float], *, step: int | None = None) -> None:
        self.metrics.append((dict(values), step))

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def log_dict(self, values: dict[str, object], artifact_file: str) -> None:
        del values, artifact_file


def _candidate() -> CandidateFrame:
    return CandidateFrame(
        frame_id="frame-1",
        clip_id="clip-1",
        timestamp_sec=0.5,
        frame_sha256="d" * 64,
        source_hash="e" * 64,
        split_role="comparison",
        detections=(Detection(Box(1, 1, 5, 5), 0.9),),
        latency_ms=1.0,
        detector_revision="fixture-revision",
        model_sha256="a" * 64,
        device="cpu",
        thread_count=1,
    )


def _detector_report(
    raw_results_sha256: str, *, name: str = "fixture"
) -> DetectorBenchmarkReport:
    metrics = DetectorMetrics(
        target_recall=1.0,
        false_detections_per_frame=0.0,
        non_target_detections_per_frame=0.0,
        non_target_candidate_count=0,
        landmark_nme=0.0,
        landmark_coverage=1.0,
        aligned_mouth_jitter=0.0,
    )
    trackers = tuple(
        TrackerMetrics(
            association=association,
            stable_track_coverage=1.0,
            abstention_rate=0.0,
            target_track_errors=0,
            tracked_frames=1,
            target_track_errors_per_1000=0.0,
        )
        for association in ("greedy_iou", "constant_velocity")
    )
    latency = DetectorLatency(
        timed_frames=1,
        median_ms=1.0,
        p95_ms=1.0,
        throughput_fps=1000.0,
        device="cpu",
        thread_count=1,
    )
    estimates = {
        "target_recall": 1.0,
        "false_detections_per_frame": 0.0,
        "non_target_detections_per_frame": 0.0,
        "non_target_candidate_count": 0.0,
        "landmark_nme": 0.0,
        "landmark_coverage": 1.0,
        "aligned_mouth_jitter": 0.0,
        "latency.median_ms": 1.0,
        "latency.p95_ms": 1.0,
        "latency.throughput_fps": 1000.0,
        "greedy_iou.stable_track_coverage": 1.0,
        "greedy_iou.abstention_rate": 0.0,
        "greedy_iou.target_track_errors_per_1000": 0.0,
        "constant_velocity.stable_track_coverage": 1.0,
        "constant_velocity.abstention_rate": 0.0,
        "constant_velocity.target_track_errors_per_1000": 0.0,
    }
    return DetectorBenchmarkReport(
        detector_name=name,
        detector_revision=f"{name}-revision",
        model_sha256="a" * 64,
        threshold=0.5,
        collection_threshold=0.0,
        evidence_scope="software_fixture_only",
        rule_revision=FROZEN_DETECTOR_RULE_REVISION,
        frame_count=1,
        comparison_clip_count=1,
        source_count=1,
        metrics=metrics,
        trackers=trackers,
        latency=latency,
        intervals={
            key: BootstrapInterval(value, value, value, 1000)
            for key, value in estimates.items()
        },
        runtime_snapshot={
            "started_at_utc": "2026-08-25T00:00:00+00:00",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python_version": "3.13.0",
            "platform": "fixture-platform",
            "packages": {"opencv-python": "5.0.0.93"},
            "cpu": "fixture-cpu",
            "gpu": None,
            "gpu_memory_mib": None,
            "available_memory_mib": 1024,
            "ffmpeg_version": "fixture-ffmpeg",
        },
        raw_results_sha256=raw_results_sha256,
        evaluation_set_sha256="c" * 64,
        split_hash="d" * 64,
        identity_strict_split_hash="b" * 64,
        reviewed_sample_sha256="e" * 64,
        annotation_audit_sha256="f" * 64,
        annotation_audit_validated=True,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


def _write_report(path: Path, report: DetectorBenchmarkReport) -> None:
    path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_log_binary_training_records_epoch_metrics_and_output_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "branch.pt"
    history_path = tmp_path / "history.json"
    checkpoint.write_bytes(b"checkpoint")
    history_path.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])
    history = BinaryTrainingHistory(
        epochs=(BinaryEpochRecord(1, 0.8, 0.6, 3, False),), best_epoch=1
    )

    log_binary_training(
        logger,
        history=history,
        configuration_hash="configuration-hash",
        checkpoint=checkpoint,
        history_path=history_path,
        elapsed_seconds=1.5,
        samples_per_second=12.5,
        peak_gpu_memory_mib=4096.0,
    )

    assert logger.params == [
        {
            "training.kind": "binary",
            "training.best_epoch": 1,
            "training.elapsed_seconds": 1.5,
            "configuration.sha256": "configuration-hash",
            "checkpoint.sha256": "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
        }
    ]
    assert logger.metrics == [
        (
            {
                "training.loss": 0.8,
                "validation.loss": 0.6,
                "optimizer.steps": 3.0,
                "stage.backbone_trainable": 0.0,
            },
            1,
        ),
        (
            {
                "training.samples_per_second": 12.5,
                "training.peak_gpu_memory_mib": 4096.0,
            },
            None,
        ),
    ]
    assert logger.artifacts == [(checkpoint, "checkpoints"), (history_path, "history")]


@pytest.mark.parametrize(
    ("samples_per_second", "peak_gpu_memory_mib"),
    ((0.0, 1.0), (1.0, 0.0)),
)
def test_log_binary_training_rejects_nonpositive_gpu_cost_metrics(
    tmp_path: Path,
    samples_per_second: float,
    peak_gpu_memory_mib: float,
) -> None:
    checkpoint = tmp_path / "branch.pt"
    history_path = tmp_path / "history.json"
    checkpoint.write_bytes(b"checkpoint")
    history_path.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])
    history = BinaryTrainingHistory(
        epochs=(BinaryEpochRecord(1, 0.8, 0.6, 3, False),), best_epoch=1
    )

    with pytest.raises(ValueError, match="positive"):
        log_binary_training(
            logger,
            history=history,
            configuration_hash="configuration-hash",
            checkpoint=checkpoint,
            history_path=history_path,
            elapsed_seconds=1.5,
            samples_per_second=samples_per_second,
            peak_gpu_memory_mib=peak_gpu_memory_mib,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []


def test_log_sync_training_rejects_nonfinite_metrics_before_logging(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sync.pt"
    history_path = tmp_path / "history.json"
    checkpoint.write_bytes(b"checkpoint")
    history_path.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])
    history = SyncTrainingHistory(
        epochs=(SyncEpochRecord(1, "heads", float("nan"), 0.6, 3),), best_epoch=1
    )

    with pytest.raises(ValueError, match="finite"):
        log_sync_training(
            logger,
            history=history,
            configuration_hash="configuration-hash",
            checkpoint=checkpoint,
            history_path=history_path,
            elapsed_seconds=1.5,
            samples_per_second=12.5,
            peak_gpu_memory_mib=4096.0,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []


def test_log_sync_training_records_stage_metrics_and_output_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "sync.pt"
    history_path = tmp_path / "history.json"
    checkpoint.write_bytes(b"checkpoint")
    history_path.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])
    history = SyncTrainingHistory(
        epochs=(SyncEpochRecord(1, "heads", 0.8, 0.6, 3),), best_epoch=1
    )

    log_sync_training(
        logger,
        history=history,
        configuration_hash="configuration-hash",
        checkpoint=checkpoint,
        history_path=history_path,
        elapsed_seconds=1.5,
        samples_per_second=12.5,
        peak_gpu_memory_mib=4096.0,
    )

    assert logger.metrics == [
        (
            {
                "training.loss": 0.8,
                "validation.loss": 0.6,
                "optimizer.steps": 3.0,
                "stage.heads": 1.0,
                "stage.upper": 0.0,
            },
            1,
        ),
        (
            {
                "training.samples_per_second": 12.5,
                "training.peak_gpu_memory_mib": 4096.0,
            },
            None,
        ),
    ]
    assert logger.artifacts == [(checkpoint, "checkpoints"), (history_path, "history")]


def test_log_fusion_training_records_provenance_and_output_artifacts(
    tmp_path: Path,
) -> None:
    model = tmp_path / "fusion.joblib"
    metadata = tmp_path / "fusion.json"
    model.write_bytes(b"model")
    metadata.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])

    log_fusion_training(
        logger,
        samples=8,
        branches=("visual", "audio", "sync"),
        model_kind="logistic",
        split_hash="split-hash",
        preprocessing_hash="preprocessing-hash",
        model_path=model,
        metadata_path=metadata,
    )

    assert logger.params == [
        {
            "fusion.samples": 8,
            "fusion.branches": "visual,audio,sync",
            "fusion.model_kind": "logistic",
            "fusion.split_hash": "split-hash",
            "fusion.preprocessing_hash": "preprocessing-hash",
            "fusion.model_sha256": "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4",
        }
    ]
    assert logger.artifacts == [(model, "models"), (metadata, "metadata")]


def test_detector_benchmark_logging_allows_only_path_free_evidence(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "aggregate-report.json"
    predictions = tmp_path / "predictions.jsonl"
    annotations = tmp_path / "private-annotations.jsonl"
    review_image = tmp_path / "private-review.jpg"
    for path in (annotations, review_image):
        path.write_text("{}", encoding="utf-8")
    write_candidate_records((_candidate(),), predictions)
    logger = FakeLogger([], [], [])
    report = _detector_report(hashlib.sha256(predictions.read_bytes()).hexdigest())
    _write_report(report_path, report)

    training_log.log_detector_benchmark(
        logger,
        report=report,
        report_path=report_path,
        predictions_path=predictions,
    )

    assert logger.artifacts == [
        (report_path, "detector/aggregate"),
        (predictions, "detector/predictions"),
    ]
    assert all(path not in {annotations, review_image} for path, _ in logger.artifacts)
    assert logger.params[0]["detector.evidence_scope"] == "software_fixture_only"


def test_detector_logging_rejects_a_valid_candidate_from_another_report(
    tmp_path: Path,
) -> None:
    expected_predictions = tmp_path / "expected.jsonl"
    other_predictions = tmp_path / "other.jsonl"
    report_path = tmp_path / "report.json"
    write_candidate_records((_candidate(),), expected_predictions)
    write_candidate_records((replace(_candidate(), latency_ms=2.0),), other_predictions)
    report = _detector_report(
        hashlib.sha256(expected_predictions.read_bytes()).hexdigest()
    )
    _write_report(report_path, report)
    logger = FakeLogger([], [], [])

    with pytest.raises(ValueError, match="raw-results hash"):
        training_log.log_detector_benchmark(
            logger,
            report=report,
            report_path=report_path,
            predictions_path=other_predictions,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []


def test_detector_logging_rejects_a_valid_report_from_another_run(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    write_candidate_records((_candidate(),), predictions)
    raw_hash = hashlib.sha256(predictions.read_bytes()).hexdigest()
    expected_report = _detector_report(raw_hash, name="expected")
    other_report = _detector_report(raw_hash, name="other")
    _write_report(report_path, other_report)
    logger = FakeLogger([], [], [])

    with pytest.raises(ValueError, match="report artifact"):
        training_log.log_detector_benchmark(
            logger,
            report=expected_report,
            report_path=report_path,
            predictions_path=predictions,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []


@pytest.mark.parametrize(
    "private_path",
    (
        "/private/review/frame.png",
        "private/review/frame.png",
        "C:\\private\\review\\frame.png",
        "private\\review\\frame.png",
    ),
)
def test_detector_logging_rejects_candidate_paths_before_upload(
    tmp_path: Path,
    private_path: str,
) -> None:
    report_path = tmp_path / "aggregate-report.json"
    predictions = tmp_path / "predictions.jsonl"
    report_path.write_text("{}", encoding="utf-8")
    write_candidate_records((_candidate(),), predictions)
    row = json.loads(predictions.read_text(encoding="utf-8"))
    row["clip_id"] = private_path
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    logger = FakeLogger([], [], [])

    with pytest.raises(ValueError, match="path-free"):
        training_log.log_detector_benchmark(
            logger,
            report=SimpleNamespace(),
            report_path=report_path,
            predictions_path=predictions,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []


def test_detector_logging_rejects_a_wrong_prediction_artifact(
    tmp_path: Path,
) -> None:
    wrong_file = tmp_path / "aggregate-report.json"
    wrong_file.write_text("{}", encoding="utf-8")
    logger = FakeLogger([], [], [])

    with pytest.raises(ValueError, match="candidate JSONL"):
        training_log.log_detector_benchmark(
            logger,
            report=SimpleNamespace(),
            report_path=wrong_file,
            predictions_path=wrong_file,
        )

    assert logger.params == []
    assert logger.metrics == []
    assert logger.artifacts == []
