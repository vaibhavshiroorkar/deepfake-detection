from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.data.datasets import BranchBatch

from .engine import run_accumulated_epoch
from .ranking import roc_auc
from .selection import BestEpoch


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
    # Selection score. NaN when the validation split holds one class. Histories
    # written before selection moved to AUC do not carry it.
    validation_auc: float = float("nan")


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
    output = _forward(model, batch, values)
    return criterion(output.logits, labels)


def _forward(model: nn.Module, batch: BranchBatch, values: torch.Tensor):
    from .input_contract import audio_forward

    return audio_forward(model, values, batch.valid_samples)


def _validate(
    model: nn.Module,
    batches: Sequence[BranchBatch],
    *,
    device: str,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Mean validation loss and ROC-AUC."""
    if not batches:
        raise ValueError("Validation batches cannot be empty")
    model.eval()
    losses: list[float] = []
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in batches:
            values = batch.values.to(device)
            target = batch.labels.to(device)
            logits = _forward(model, batch, values).logits
            losses.append(float(criterion(logits, target)))
            scores.append(logits.detach().cpu())
            labels.append(target.detach().cpu())
    return sum(losses) / len(losses), roc_auc(torch.cat(scores), torch.cat(labels))


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
    selection = BestEpoch(
        patience=config.early_stopping_patience,
        minimum_improvement=config.minimum_improvement,
    )
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
        validation_loss, validation_auc = _validate(
            model,
            validation_batches,
            device=device,
            criterion=criterion,
        )
        # Printed, not only logged: MLflow receives metrics when the run ends,
        # so without this an hour-long run is invisible until it finishes.
        print(
            f"epoch {epoch + 1}/{config.epochs}  train {train.mean_loss:.4f}  "
            f"val {validation_loss:.4f}  AUC {validation_auc:.4f}  "
            f"backbone={'on' if backbone_trainable else 'frozen'}",
            flush=True,
        )
        records.append(
            BinaryEpochRecord(
                epoch=epoch + 1,
                train_loss=train.mean_loss,
                validation_loss=validation_loss,
                optimizer_steps=train.optimizer_steps,
                backbone_trainable=backbone_trainable,
                validation_auc=validation_auc,
            )
        )
        if selection.update(epoch=epoch + 1, score=validation_auc, model=model):
            break
    best_epoch = selection.restore(model)
    return BinaryTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)
