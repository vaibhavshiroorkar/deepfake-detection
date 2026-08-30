from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

from deepfake_detection.benchmarks.detector_metrics import DetectorBenchmarkReport
from deepfake_detection.benchmarks.detector_runner import validate_candidate_artifact
from deepfake_detection.experiments.tracking import RunLogger
from deepfake_detection.training.binary import BinaryTrainingHistory
from deepfake_detection.training.sync import SyncTrainingHistory


def log_binary_training(
    logger: RunLogger,
    *,
    history: BinaryTrainingHistory,
    configuration_hash: str,
    checkpoint: Path,
    history_path: Path,
    elapsed_seconds: float,
) -> None:
    metrics = [
        (
            record.epoch,
            {
                "training.loss": record.train_loss,
                "validation.loss": record.validation_loss,
                "optimizer.steps": float(record.optimizer_steps),
                "stage.backbone_trainable": float(record.backbone_trainable),
            },
        )
        for record in history.epochs
    ]
    _log_training(
        logger,
        kind="binary",
        history_best_epoch=history.best_epoch,
        configuration_hash=configuration_hash,
        checkpoint=checkpoint,
        history_path=history_path,
        elapsed_seconds=elapsed_seconds,
        metrics=metrics,
    )


def log_sync_training(
    logger: RunLogger,
    *,
    history: SyncTrainingHistory,
    configuration_hash: str,
    checkpoint: Path,
    history_path: Path,
    elapsed_seconds: float,
) -> None:
    metrics = [
        (
            record.epoch,
            {
                "training.loss": record.train_loss,
                "validation.loss": record.validation_loss,
                "optimizer.steps": float(record.optimizer_steps),
                "stage.heads": float(record.stage == "heads"),
                "stage.upper": float(record.stage == "upper"),
            },
        )
        for record in history.epochs
    ]
    _log_training(
        logger,
        kind="sync",
        history_best_epoch=history.best_epoch,
        configuration_hash=configuration_hash,
        checkpoint=checkpoint,
        history_path=history_path,
        elapsed_seconds=elapsed_seconds,
        metrics=metrics,
    )


def log_fusion_training(
    logger: RunLogger,
    *,
    samples: int,
    branches: Sequence[str],
    model_kind: str,
    split_hash: str,
    preprocessing_hash: str,
    model_path: Path,
    metadata_path: Path,
) -> None:
    logger.log_params(
        {
            "fusion.samples": samples,
            "fusion.branches": ",".join(branches),
            "fusion.model_kind": model_kind,
            "fusion.split_hash": split_hash,
            "fusion.preprocessing_hash": preprocessing_hash,
            "fusion.model_sha256": _file_hash(model_path),
        }
    )
    logger.log_artifact(model_path, artifact_path="models")
    logger.log_artifact(metadata_path, artifact_path="metadata")


def log_detector_benchmark(
    logger: RunLogger,
    *,
    report: DetectorBenchmarkReport,
    report_path: Path,
    predictions_path: Path,
) -> None:
    validate_candidate_artifact(predictions_path)
    logger.log_params(
        {
            "detector.name": report.detector_name,
            "detector.revision": report.detector_revision,
            "detector.model_sha256": report.model_sha256,
            "detector.rule_revision": report.rule_revision,
            "detector.evidence_scope": report.evidence_scope,
            "detector.raw_results_sha256": report.raw_results_sha256,
            "detector.evaluation_set_sha256": report.evaluation_set_sha256,
            "detector.split_hash": report.split_hash,
            "detector.reviewed_sample_sha256": report.reviewed_sample_sha256,
            "detector.annotation_audit_sha256": report.annotation_audit_sha256,
        }
    )
    metrics = {
        "detector.target_recall": report.metrics.target_recall,
        "detector.false_detections_per_frame": (
            report.metrics.false_detections_per_frame
        ),
        "detector.non_target_detections_per_frame": (
            report.metrics.non_target_detections_per_frame
        ),
        "detector.landmark_coverage": report.metrics.landmark_coverage,
        "detector.latency_median_ms": report.latency.median_ms,
        "detector.latency_p95_ms": report.latency.p95_ms,
        "detector.throughput_fps": report.latency.throughput_fps,
        "detector.comparison_clips": float(report.comparison_clip_count),
    }
    if report.metrics.landmark_nme is not None:
        metrics["detector.landmark_nme"] = report.metrics.landmark_nme
    if report.metrics.aligned_mouth_jitter is not None:
        metrics["detector.aligned_mouth_jitter"] = report.metrics.aligned_mouth_jitter
    logger.log_metrics(metrics)
    logger.log_artifact(report_path, artifact_path="detector/aggregate")
    logger.log_artifact(predictions_path, artifact_path="detector/predictions")


def _log_training(
    logger: RunLogger,
    *,
    kind: str,
    history_best_epoch: int,
    configuration_hash: str,
    checkpoint: Path,
    history_path: Path,
    elapsed_seconds: float,
    metrics: Sequence[tuple[int, dict[str, float]]],
) -> None:
    _require_finite(elapsed_seconds)
    for _, values in metrics:
        _require_finite(*values.values())
    logger.log_params(
        {
            "training.kind": kind,
            "training.best_epoch": history_best_epoch,
            "training.elapsed_seconds": elapsed_seconds,
            "configuration.sha256": configuration_hash,
            "checkpoint.sha256": _file_hash(checkpoint),
        }
    )
    for epoch, values in metrics:
        logger.log_metrics(values, step=epoch)
    logger.log_artifact(checkpoint, artifact_path="checkpoints")
    logger.log_artifact(history_path, artifact_path="history")


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Training metrics must be finite")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
