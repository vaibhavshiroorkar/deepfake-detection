from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from deepfake_detection.digest import file_sha256
from deepfake_detection.experiments.tracking import RunLogger

if TYPE_CHECKING:
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
    samples_per_second: float,
    peak_gpu_memory_mib: float,
) -> None:
    metrics = [
        (
            record.epoch,
            {
                "training.loss": record.train_loss,
                "validation.loss": record.validation_loss,
                "optimizer.steps": float(record.optimizer_steps),
                "stage.backbone_trainable": float(record.backbone_trainable),
                **_finite_only({"validation.auc": record.validation_auc}),
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
        samples_per_second=samples_per_second,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
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
    samples_per_second: float,
    peak_gpu_memory_mib: float,
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
                **_finite_only(
                    {
                        "validation.misalignment_auc": (
                            record.validation_misalignment_auc
                        ),
                        "validation.offset_accuracy": record.validation_offset_accuracy,
                    }
                ),
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
        samples_per_second=samples_per_second,
        peak_gpu_memory_mib=peak_gpu_memory_mib,
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
            "fusion.model_sha256": file_sha256(model_path),
        }
    )
    logger.log_artifact(model_path, artifact_path="models")
    logger.log_artifact(metadata_path, artifact_path="metadata")


def _log_training(
    logger: RunLogger,
    *,
    kind: str,
    history_best_epoch: int,
    configuration_hash: str,
    checkpoint: Path,
    history_path: Path,
    elapsed_seconds: float,
    samples_per_second: float,
    peak_gpu_memory_mib: float,
    metrics: Sequence[tuple[int, dict[str, float]]],
) -> None:
    _require_positive_finite(
        elapsed_seconds,
        samples_per_second,
        peak_gpu_memory_mib,
    )
    for _, values in metrics:
        _require_finite(*values.values())
    logger.log_params(
        {
            "training.kind": kind,
            "training.best_epoch": history_best_epoch,
            "training.elapsed_seconds": elapsed_seconds,
            "configuration.sha256": configuration_hash,
            "checkpoint.sha256": file_sha256(checkpoint),
        }
    )
    for epoch, values in metrics:
        logger.log_metrics(values, step=epoch)
    logger.log_metrics(
        {
            "training.samples_per_second": samples_per_second,
            "training.peak_gpu_memory_mib": peak_gpu_memory_mib,
        }
    )
    logger.log_artifact(checkpoint, artifact_path="checkpoints")
    logger.log_artifact(history_path, artifact_path="history")


def _finite_only(values: dict[str, float]) -> dict[str, float]:
    """Drop a score that is undefined, such as AUC on a one-class split."""
    return {name: value for name, value in values.items() if math.isfinite(value)}


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Training metrics must be finite")


def _require_positive_finite(*values: float) -> None:
    _require_finite(*values)
    if not all(value > 0 for value in values):
        raise ValueError("Training cost metrics must be positive")
