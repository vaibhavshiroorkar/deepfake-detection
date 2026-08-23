from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.data.datasets import SyncBatch

from .engine import run_accumulated_epoch
from .losses import sync_training_loss
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


def _validation_loss(
    model: nn.Module,
    batches: Sequence[SyncBatch],
    *,
    device: str,
    contrastive_weight: float,
) -> float:
    if not batches:
        raise ValueError("Validation batches cannot be empty")
    model.eval()
    with torch.inference_mode():
        losses = [
            float(
                _batch_loss(
                    model,
                    batch,
                    device=device,
                    contrastive_weight=contrastive_weight,
                )
            )
            for batch in batches
        ]
    return sum(losses) / len(losses)


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
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
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
        validation_loss = _validation_loss(
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
            )
        )
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Sync training did not produce a checkpoint candidate")
    model.load_state_dict(best_state)
    return SyncTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)
