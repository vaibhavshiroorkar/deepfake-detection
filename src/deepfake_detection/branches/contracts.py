from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class BranchOutput:
    logits: Tensor
    embedding: Tensor
    token_count: int
