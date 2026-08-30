from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from deepfake_detection.benchmarks.detector_metrics import CandidateFrame
from deepfake_detection.benchmarks.detector_runner import write_candidate_records
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
        )
    ]
    assert logger.artifacts == [(checkpoint, "checkpoints"), (history_path, "history")]


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
        )
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
    for path in (report_path, annotations, review_image):
        path.write_text("{}", encoding="utf-8")
    write_candidate_records((_candidate(),), predictions)
    logger = FakeLogger([], [], [])
    report = SimpleNamespace(
        detector_name="fixture",
        detector_revision="fixture-revision",
        model_sha256="a" * 64,
        rule_revision="detector-selection-v1",
        evidence_scope="software_fixture_only",
        raw_results_sha256="b" * 64,
        evaluation_set_sha256="c" * 64,
        split_hash="d" * 64,
        reviewed_sample_sha256="e" * 64,
        annotation_audit_sha256="f" * 64,
        comparison_clip_count=1,
        metrics=SimpleNamespace(
            target_recall=1.0,
            false_detections_per_frame=0.0,
            non_target_detections_per_frame=0.0,
            landmark_coverage=1.0,
            landmark_nme=0.0,
            aligned_mouth_jitter=0.0,
        ),
        latency=SimpleNamespace(
            median_ms=1.0,
            p95_ms=1.0,
            throughput_fps=1000.0,
        ),
    )

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
