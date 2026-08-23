from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from deepfake_detection.branches.sync import SyncOutput
from deepfake_detection.branches.sync_objective import (
    MISMATCH_CLASS_INDEX,
    contrastive_alignment_loss,
)


@dataclass(frozen=True, slots=True)
class SyncLoss:
    total: Tensor
    offset: Tensor
    contrastive: Tensor


def sync_training_loss(
    output: SyncOutput,
    targets: Tensor,
    *,
    offset_weight: float = 1.0,
    contrastive_weight: float = 0.1,
    temperature: float = 0.07,
) -> SyncLoss:
    offset = functional.cross_entropy(output.offset_logits, targets)
    corresponding = targets != MISMATCH_CLASS_INDEX
    if int(corresponding.sum()) >= 2:
        contrastive = contrastive_alignment_loss(
            output.video_tokens[corresponding],
            output.audio_tokens[corresponding],
            temperature=temperature,
        )
    else:
        contrastive = torch.zeros((), device=output.offset_logits.device)
    return SyncLoss(
        total=offset_weight * offset + contrastive_weight * contrastive,
        offset=offset,
        contrastive=contrastive,
    )
