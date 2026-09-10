"""Validation ROC-AUC, used to choose which epoch to keep.

Early stopping on validation loss picked a checkpoint that scores below chance.
`visual-efficientnet` reached its lowest BCE at epoch 1, while the backbone was
still frozen, and that checkpoint measured 0.4668 ROC-AUC in-domain. Epochs 3
and 4 had a far worse loss and a trained backbone.

Loss and ranking come apart because BCE is a calibration measure. It punishes a
confident mistake very hard, so a model that ranks clips well but is
overconfident scores worse than one that hedges and ranks badly. Every objective
in this project is stated in ROC-AUC, so that is what selection has to use.

Loss is still recorded every epoch. It diagnoses a run that is diverging rather
than merely miscalibrated, which AUC alone would hide.
"""

from __future__ import annotations

import torch
from torch import Tensor


def roc_auc(scores: Tensor, labels: Tensor) -> float:
    """Rank-based ROC-AUC. Returns NaN when only one class is present.

    Computed from ranks rather than by sweeping thresholds, so it needs no
    sklearn import inside a training loop, and ties get their average rank
    instead of an arbitrary order.
    """
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length")
    if scores.numel() == 0:
        return float("nan")
    scores = scores.detach().flatten().to(torch.float64).cpu()
    labels = labels.detach().flatten().cpu()
    positive = labels > 0.5
    count_positive = int(positive.sum())
    count_negative = int(labels.numel() - count_positive)
    # One class means the metric is undefined. Say so rather than return 0.5,
    # which would read as a real measurement of a model at chance.
    if count_positive == 0 or count_negative == 0:
        return float("nan")

    order = scores.argsort()
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    # Average the ranks inside each tie group. An untrained model can emit the
    # same logit for every clip, and without this that case scores 1.0 or 0.0
    # depending on how argsort happened to break the ties.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, sorted_scores.numel() + 1):
        if index == sorted_scores.numel() or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    rank_sum = float(ranks[positive].sum())
    return (rank_sum - count_positive * (count_positive + 1) / 2) / (
        count_positive * count_negative
    )
