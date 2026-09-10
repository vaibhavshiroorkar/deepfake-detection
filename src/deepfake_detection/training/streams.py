"""Training loop for the audiovisual cross-attention streams.

Mirrors `training/binary.py` rather than generalising it. The two differ in
their batch shape, one tensor against a video and audio pair, and forcing one
function to serve both would add an indirection layer to save thirty lines. The
repo already keeps `binary.py` and `sync.py` as parallel modules for the same
reason.

What is genuinely shared is `engine.run_accumulated_epoch`, used unchanged, and
the early-stopping and best-state-restore shape.

The one addition is `diagonal_mass` tracking per epoch. It is recorded, never
optimised. A model rewarded for producing diagonal attention would learn to
produce it without learning synchronisation, which would destroy the only
independent check the stream has. Watching it rise while the loss falls is
evidence the stream learned the real correspondence; a falling loss with flat
diagonal mass says it found a shortcut instead.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from deepfake_detection.data.datasets import AVPairBatch

from .engine import run_accumulated_epoch
from .ranking import roc_auc


@dataclass(frozen=True, slots=True)
class StreamTrainingConfig:
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
class StreamEpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    optimizer_steps: int
    encoders_trainable: bool
    validation_diagonal_mass: float
    # Selection reads this, not the loss. See training/ranking.py: the lowest
    # validation BCE picked an epoch-1 checkpoint that scored 0.4668 in-domain,
    # below chance, because BCE measures calibration and the objectives are
    # stated in ranking. Defaulted so existing history files still parse.
    validation_auc: float = float("nan")


@dataclass(frozen=True, slots=True)
class StreamTrainingHistory:
    epochs: tuple[StreamEpochRecord, ...]
    best_epoch: int


def _batch_loss(
    model: nn.Module,
    batch: AVPairBatch,
    *,
    device: str,
    criterion: nn.Module,
) -> torch.Tensor:
    output = model(video=batch.video.to(device), audio=batch.audio.to(device))
    return criterion(output.logit, batch.labels.to(device))


def _validate(
    model: nn.Module,
    batches: Sequence[AVPairBatch],
    *,
    device: str,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Mean validation loss, mean diagonal attention mass, and ROC-AUC."""
    if not batches:
        raise ValueError("Validation batches cannot be empty")
    model.eval()
    losses: list[float] = []
    masses: list[float] = []
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in batches:
            output = model(video=batch.video.to(device), audio=batch.audio.to(device))
            target = batch.labels.to(device)
            losses.append(float(criterion(output.logit, target)))
            masses.append(float(output.diagonal_mass.mean()))
            scores.append(output.logit.detach().cpu())
            labels.append(target.detach().cpu())
    return (
        sum(losses) / len(losses),
        sum(masses) / len(masses),
        roc_auc(torch.cat(scores), torch.cat(labels)),
    )


def parameter_groups(
    model: nn.Module,
    *,
    head_lr: float,
    encoder_lr: float,
    encoder_names: tuple[str, ...] = ("video_encoder", "audio_encoder"),
) -> list[dict]:
    """Split the model so pretrained encoders learn far slower than new heads.

    One learning rate across both is what wrecked the first real run: at a
    uniform 1e-4 the training loss fell to 0.039 while validation climbed to
    2.56, because 100M pretrained encoder parameters were being dragged around
    at the rate a randomly initialised attention head needs.

    `StreamConfig` has carried `lr_head` and `lr_backbone` since it was written
    for exactly this, and nothing had ever read them.
    """
    encoder_parameters = []
    head_parameters = []
    encoder_ids: set[int] = set()
    for name in encoder_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            encoder_ids.add(id(parameter))
            encoder_parameters.append(parameter)
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    ]
    groups = [{"params": head_parameters, "lr": head_lr}]
    # A frozen backbone yields no trainable parameters, and an optimizer group
    # with an empty list raises, so the group is only added when it has members.
    trainable = [p for p in encoder_parameters if p.requires_grad]
    if trainable:
        groups.append({"params": trainable, "lr": encoder_lr})
    return groups


def fit_visual_stream(
    *,
    model: nn.Module,
    train_batches: Sequence,
    validation_batches: Sequence,
    optimizer: Optimizer,
    config: StreamTrainingConfig,
    device: str,
) -> StreamTrainingHistory:
    """Train a `VisualStream`, which takes one tensor and returns a tuple.

    `fit_stream` cannot serve this: its batches carry a video and an audio
    tensor and its model returns a `StreamOutput`. The early stopping, staged
    freezing and best-state restore are the same, so only the forward call and
    the diagonal-mass field differ.
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    records: list[StreamEpochRecord] = []
    # Selection maximises AUC, so the sentinel is -inf rather than +inf. An
    # epoch whose AUC is NaN, which happens when the validation split holds one
    # class, never beats it and so is never selected.
    best_auc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    def batch_loss(current: nn.Module, batch) -> torch.Tensor:
        logit, _ = current(batch.values.to(device))
        return criterion(logit, batch.labels.to(device))

    for epoch in range(config.epochs):
        trainable = epoch >= config.freeze_epochs
        setter = getattr(model, "set_backbone_trainable", None)
        if setter is not None:
            setter(trainable)
        train = run_accumulated_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            accumulation_steps=config.accumulation_steps,
            loss_for_batch=batch_loss,
        )
        model.eval()
        losses: list[float] = []
        scores: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in validation_batches:
                logit, _ = model(batch.values.to(device))
                target = batch.labels.to(device)
                losses.append(float(criterion(logit, target)))
                scores.append(logit.detach().cpu())
                labels.append(target.detach().cpu())
        validation_loss = sum(losses) / len(losses)
        validation_auc = roc_auc(torch.cat(scores), torch.cat(labels))
        print(
            f"epoch {epoch + 1}/{config.epochs}  train {train.mean_loss:.4f}  "
            f"val {validation_loss:.4f}  AUC {validation_auc:.4f}  "
            f"backbone={'on' if trainable else 'frozen'}",
            flush=True,
        )
        records.append(
            StreamEpochRecord(
                epoch=epoch + 1,
                train_loss=train.mean_loss,
                validation_loss=validation_loss,
                optimizer_steps=train.optimizer_steps,
                encoders_trainable=trainable,
                # Not applicable: a visual stream makes no correspondence claim,
                # so there is no attention map to read a diagonal from.
                validation_diagonal_mass=0.0,
                validation_auc=validation_auc,
            )
        )
        if validation_auc > best_auc + config.minimum_improvement:
            best_auc, best_epoch, stale = validation_auc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint candidate")
    model.load_state_dict(best_state)
    return StreamTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)


def fit_stream(
    *,
    model: nn.Module,
    train_batches: Sequence[AVPairBatch],
    validation_batches: Sequence[AVPairBatch],
    optimizer: Optimizer,
    config: StreamTrainingConfig,
    device: str,
) -> StreamTrainingHistory:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.positive_weight, device=device)
    )
    records: list[StreamEpochRecord] = []
    # Maximised, so the sentinel is -inf. See training/ranking.py for why this
    # is AUC and not the loss.
    best_auc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(config.epochs):
        # Both encoders move together. Unfreezing a pretrained backbone into a
        # randomly initialised attention head wrecks it in the first few steps,
        # which is what the freeze schedule exists to prevent.
        encoders_trainable = epoch >= config.freeze_epochs
        setter = getattr(model, "set_backbone_trainable", None)
        if setter is None:
            if config.freeze_epochs:
                raise ValueError("Model does not expose staged encoder control")
        else:
            setter(encoders_trainable)

        train = run_accumulated_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            accumulation_steps=config.accumulation_steps,
            loss_for_batch=lambda current, batch: _batch_loss(
                current, batch, device=device, criterion=criterion
            ),
        )
        validation_loss, diagonal, validation_auc = _validate(
            model, validation_batches, device=device, criterion=criterion
        )
        # Printed, not just logged. MLflow only receives metrics when the run
        # finishes, so without this a multi-hour training is invisible until it
        # ends, and a stalled epoch looks the same as a slow one.
        print(
            f"epoch {epoch + 1}/{config.epochs}  "
            f"train {train.mean_loss:.4f}  val {validation_loss:.4f}  "
            f"AUC {validation_auc:.4f}  diag_mass {diagonal:.4f}  "
            f"encoders={'on' if encoders_trainable else 'frozen'}",
            flush=True,
        )
        records.append(
            StreamEpochRecord(
                epoch=epoch + 1,
                train_loss=train.mean_loss,
                validation_loss=validation_loss,
                optimizer_steps=train.optimizer_steps,
                encoders_trainable=encoders_trainable,
                validation_diagonal_mass=diagonal,
                validation_auc=validation_auc,
            )
        )
        if validation_auc > best_auc + config.minimum_improvement:
            best_auc = validation_auc
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
    return StreamTrainingHistory(epochs=tuple(records), best_epoch=best_epoch)
