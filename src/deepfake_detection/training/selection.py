"""Which epoch to keep, and when to stop. Shared by every training loop.

Every loop selects on validation ROC-AUC. Selecting on loss is what kept a
checkpoint that scored 0.4668 in-domain (see `training/ranking.py`), and it is
how a ConvNeXt seed that never fine-tuned got shipped: its lowest loss came
before the backbone was unfrozen. One rule in one place keeps a new trainer
from quietly reintroducing the old one.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


class BestEpoch:
    """Keep the weights of the highest-scoring epoch and count stale epochs.

    An epoch whose score is NaN, which happens when the validation split holds
    one class, never counts as an improvement. The first epoch is still kept as
    a fallback, so a run whose score is undefined throughout ends with a
    checkpoint rather than none, and any defined score replaces it.
    """

    def __init__(self, *, patience: int, minimum_improvement: float = 1e-4) -> None:
        if patience <= 0:
            raise ValueError("Patience must be positive")
        self.patience = patience
        self.minimum_improvement = minimum_improvement
        self.best_score = -math.inf
        self.best_epoch = 0
        self._state: dict[str, torch.Tensor] | None = None
        self._stale = 0

    def update(self, *, epoch: int, score: float, model: nn.Module) -> bool:
        """Record one epoch. Returns True when training should stop."""
        if self._state is None:
            self._state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        if score > self.best_score + self.minimum_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self._state = copy.deepcopy(model.state_dict())
            self._stale = 0
            return False
        self._stale += 1
        return self._stale >= self.patience

    def restore(self, model: nn.Module) -> int:
        """Load the kept weights into `model` and return their epoch."""
        if self._state is None:
            raise RuntimeError("No epoch was recorded")
        model.load_state_dict(self._state)
        return self.best_epoch
