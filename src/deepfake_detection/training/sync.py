from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.branches.sync_objective import (
    OFFSET_MILLISECONDS,
    sync_anomaly_logit,
)
from deepfake_detection.data.datasets import SyncBatch

from .engine import run_accumulated_epoch
from .losses import sync_training_loss
from .ranking import roc_auc
from .selection import BestEpoch
from .stages import apply_sync_training_stage


@dataclass(frozen=True, slots=True)
class SyncTrainingConfig:
    epochs: int = 12
    accumulation_steps: int = 4
    heads_epochs: int = 3
    early_stopping_patience: int = 3
    contrastive_weight: float = 0.1
    minimum_improvement: float = 1e-4

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.accumulation_steps <= 0:
            raise ValueError("Epochs and accumulation steps must be positive")
        if not 0 <= self.heads_epochs <= self.epochs:
            raise ValueError("Head-only epochs must be within the training run")
        if self.early_stopping_patience <= 0 or self.contrastive_weight < 0:
            raise ValueError("Patience must be positive and loss weights nonnegative")


@dataclass(frozen=True, slots=True)
class SyncEpochRecord:
    epoch: int
    stage: str
    train_loss: float
    validation_loss: float
    optimizer_steps: int
    # How well the anomaly score separates aligned windows from shifted or
    # mismatched ones. This is the selection score, and it is the same score
    # fusion reads. NaN in histories written before selection moved to it.
    validation_misalignment_auc: float = float("nan")
    # Accuracy of the offset head alone, logged separately because the
    # combined loss hid that it never left chance (F31).
    validation_offset_accuracy: float = float("nan")


@dataclass(frozen=True, slots=True)
class SyncTrainingHistory:
    epochs: tuple[SyncEpochRecord, ...]
    best_epoch: int


def _batch_loss(
    model: nn.Module,
    batch: SyncBatch,
    *,
    device: str,
    contrastive_weight: float,
) -> torch.Tensor:
    output = model(
        mouth_video=batch.mouth_video.to(device),
        waveform=batch.waveform.to(device),
    )
    return sync_training_loss(
        output,
        batch.offset_classes.to(device),
        contrastive_weight=contrastive_weight,
    ).total


def _validate(
    model: nn.Module,
    batches: Sequence[SyncBatch],
    *,
    device: str,
    contrastive_weight: float,
) -> tuple[float, float, float]:
    """Mean loss, misalignment ROC-AUC and offset accuracy on validation."""
    if not batches:
        raise ValueError("Validation batches cannot be empty")
    model.eval()
    aligned = OFFSET_MILLISECONDS.index(0)
    losses: list[float] = []
    scores: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in batches:
            output = model(
                mouth_video=batch.mouth_video.to(device),
                waveform=batch.waveform.to(device),
            )
            target = batch.offset_classes.to(device)
            losses.append(
                float(
                    sync_training_loss(
                        output, target, contrastive_weight=contrastive_weight
                    ).total
                )
            )
            scores.append(sync_anomaly_logit(output.offset_logits).cpu())
            predictions.append(output.offset_logits.argmax(dim=1).cpu())
            targets.append(target.cpu())
    target = torch.cat(targets)
    misaligned = (target != aligned).float()
    auc = roc_auc(torch.cat(scores), misaligned)
    accuracy = float((torch.cat(predictions) == target).float().mean())
    return sum(losses) / len(losses), auc, accuracy


def fit_sync_branch(
    *,
    model: nn.Module,
    train_batches: Sequence[SyncBatch],
    validation_batches: Sequence[SyncBatch],
    optimizer: Optimizer,
    config: SyncTrainingConfig,
    device: str,
) -> SyncTrainingHistory:
    model.to(device)
    records: list[SyncEpochRecord] = []
    selection = BestEpoch(
        patience=config.early_stopping_patience,
        minimum_improvement=config.minimum_improvement,
    )
    for epoch in range(config.epochs):
        stage = "heads" if epoch < config.heads_epochs else "upper"
        apply_sync_training_stage(model, stage)
        train = run_accumulated_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            accumulation_steps=config.accumulation_steps,
            loss_for_batch=lambda current, batch: _batch_loss(
                current,
                batch,
                device=device,
                contrastive_weight=config.contrastive_weight,
            ),
        )
        validation_loss, misalignment_auc, offset_accuracy = _validate(
            model,
            validation_batches,
            device=device,
            contrastive_weight=config.contrastive_weight,
        )
        records.append(
            SyncEpochRecord(
                epoch=epoch + 1,
                stage=stage,
                train_loss=train.mean_loss,
                validation_loss=validation_loss,
                optimizer_steps=train.optimizer_steps,
                validation_misalignment_auc=misalignment_auc,
                validation_offset_accuracy=offset_accuracy,
            )
        )
        if selection.update(epoch=epoch + 1, score=misalignment_auc, model=model):
            break
    best_epoch = selection.restore(model)
    return SyncTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)
