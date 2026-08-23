from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class EpochStats:
    mean_loss: float
    batches: int
    optimizer_steps: int


def run_accumulated_epoch(
    *,
    model: nn.Module,
    batches: Sequence[Any],
    optimizer: Optimizer,
    accumulation_steps: int,
    loss_for_batch: Callable[[nn.Module, Any], Tensor],
) -> EpochStats:
    if accumulation_steps <= 0:
        raise ValueError("Accumulation steps must be positive")
    if not batches:
        raise ValueError("Training batches cannot be empty")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    optimizer_steps = 0
    batch_count = len(batches)
    for index, batch in enumerate(batches):
        group_start = (index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, batch_count - group_start)
        loss = loss_for_batch(model, batch)
        if loss.ndim != 0:
            raise ValueError("Loss function must return a scalar tensor")
        total_loss += float(loss.detach())
        (loss / group_size).backward()
        if (index + 1) % accumulation_steps == 0 or index + 1 == batch_count:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

    return EpochStats(
        mean_loss=total_loss / batch_count,
        batches=batch_count,
        optimizer_steps=optimizer_steps,
    )
