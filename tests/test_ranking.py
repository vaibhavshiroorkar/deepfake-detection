"""Validation ROC-AUC, the metric that selects a checkpoint.

Pinned against sklearn rather than trusted, because a wrong AUC here does not
crash: it silently keeps the wrong epoch, which is the failure this module was
written to fix.
"""

import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.training.ranking import roc_auc


def test_matches_sklearn_on_random_input() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    generator = torch.Generator().manual_seed(3)

    for _ in range(5):
        scores = torch.randn(200, generator=generator)
        labels = (torch.rand(200, generator=generator) > 0.7).float()

        assert roc_auc(scores, labels) == pytest.approx(
            sklearn.roc_auc_score(labels.numpy(), scores.numpy())
        )


def test_perfect_ranking_scores_one() -> None:
    assert roc_auc(torch.tensor([0.1, 0.2, 0.9, 1.0]), torch.tensor([0, 0, 1, 1])) == 1.0


def test_reversed_ranking_scores_zero() -> None:
    assert roc_auc(torch.tensor([0.9, 1.0, 0.1, 0.2]), torch.tensor([0, 0, 1, 1])) == 0.0


def test_all_scores_tied_is_chance_not_a_perfect_score() -> None:
    """An untrained model emits one logit for every clip. Without averaging the
    ranks inside a tie group this returns 1.0 or 0.0 on whichever order argsort
    happened to produce, which would read as a perfect model."""
    assert roc_auc(torch.zeros(10), torch.tensor([1.0, 0.0] * 5)) == 0.5


def test_one_class_is_undefined_not_chance() -> None:
    """Returning 0.5 would read as a measurement of a model at chance. MNW's
    fake-only lab set is the standing case."""
    import math

    assert math.isnan(roc_auc(torch.randn(10), torch.ones(10)))
    assert math.isnan(roc_auc(torch.randn(10), torch.zeros(10)))


def test_empty_input_is_undefined() -> None:
    import math

    assert math.isnan(roc_auc(torch.tensor([]), torch.tensor([])))


def test_mismatched_lengths_are_an_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        roc_auc(torch.randn(5), torch.ones(4))


def test_accepts_a_tensor_that_requires_grad() -> None:
    """Called on a validation logit inside a training loop, which may still
    carry grad depending on the caller's context."""
    scores = torch.randn(20, requires_grad=True)

    assert 0.0 <= roc_auc(scores, (torch.arange(20) % 2).float()) <= 1.0
