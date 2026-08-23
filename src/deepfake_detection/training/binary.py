from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.data.datasets import BranchBatch

from .engine import run_accumulated_epoch


@dataclass(frozen=True, slots=True)
class BinaryTrainingConfig:
    epochs: int = 12
    accumulation_steps: int = 4
    freeze_epochs: int = 3
    early_stopping_patience: int = 3
    positive_weight: float = 1.0
    minimum_improvement: float = 1e-4

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.accumulation_steps <= 0:
            raise ValueError("Epochs and accumulation steps must be positive")
        if not 0 <= self.freeze_epochs <= self.epochs:
            raise ValueError("Freeze epochs must be within the training run")
        if self.early_stopping_patience <= 0 or self.positive_weight <= 0:
            raise ValueError("Patience and positive weight must be positive")


@dataclass(frozen=True, slots=True)
class BinaryEpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    optimizer_steps: int
    backbone_trainable: bool


@dataclass(frozen=True, slots=True)
class BinaryTrainingHistory:
    epochs: tuple[BinaryEpochRecord, ...]
    best_epoch: int


def _batch_loss(
    model: nn.Module,
    batch: BranchBatch,
    *,
    device: str,
    criterion: nn.Module,
) -> torch.Tensor:
    values = batch.values.to(device)
    labels = batch.labels.to(device)
    output = model(values)
    return criterion(output.logits, labels)


def _validation_loss(
    model: nn.Module,
    batches: Sequence[BranchBatch],
    *,
    device: str,
    criterion: nn.Module,
) -> float:
    if not batches:
        raise ValueError("Validation batches cannot be empty")
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in batches:
            losses.append(
                float(_batch_loss(model, batch, device=device, criterion=criterion))
            )
    return sum(losses) / len(losses)


def fit_binary_branch(
    *,
    model: nn.Module,
    train_batches: Sequence[BranchBatch],
    validation_batches: Sequence[BranchBatch],
    optimizer: Optimizer,
    config: BinaryTrainingConfig,
    device: str,
) -> BinaryTrainingHistory:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.positive_weight, device=device)
    )
    records: list[BinaryEpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(config.epochs):
        backbone_trainable = epoch >= config.freeze_epochs
        setter = getattr(model, "set_backbone_trainable", None)
        if setter is None:
            if config.freeze_epochs:
                raise ValueError("Model does not expose staged backbone control")
        else:
            setter(backbone_trainable)
        train = run_accumulated_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            accumulation_steps=config.accumulation_steps,
            loss_for_batch=lambda current, batch: _batch_loss(
                current,
                batch,
                device=device,
                criterion=criterion,
            ),
        )
        validation_loss = _validation_loss(
            model,
            validation_batches,
            device=device,
            criterion=criterion,
        )
        records.append(
            BinaryEpochRecord(
                epoch=epoch + 1,
                train_loss=train.mean_loss,
                validation_loss=validation_loss,
                optimizer_steps=train.optimizer_steps,
                backbone_trainable=backbone_trainable,
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
        raise RuntimeError("Training did not produce a checkpoint candidate")
    model.load_state_dict(best_state)
    return BinaryTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)
