from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

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
