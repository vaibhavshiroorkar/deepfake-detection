import pytest

from deepfake_detection.evaluation.bootstrap import (
    PairedPrediction,
    bootstrap_binary_metrics,
    cluster_bootstrap_interval,
    paired_auc_difference,
)
from deepfake_detection.evaluation.metrics import EvaluationItem


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_identities() -> None:
    items = [
        EvaluationItem(0, 0.1, "id1", "real"),
        EvaluationItem(0, 0.3, "id1", "real"),
        EvaluationItem(1, 0.7, "id2", "fake"),
        EvaluationItem(1, 0.9, "id2", "fake"),
    ]

    def statistic(sample: list[EvaluationItem]) -> float:
        return sum(float(item.probability) for item in sample) / len(sample)

    first = cluster_bootstrap_interval(items, statistic, samples=100, seed=9)
    second = cluster_bootstrap_interval(items, statistic, samples=100, seed=9)

    assert first == second
    assert first.estimate == pytest.approx(0.5)
    assert first.lower <= first.estimate <= first.upper


def test_paired_auc_difference_uses_the_same_identity_resamples() -> None:
    predictions = [
        PairedPrediction(0, "real-1", 0.1, 0.9),
        PairedPrediction(0, "real-2", 0.2, 0.8),
        PairedPrediction(1, "fake-1", 0.8, 0.2),
        PairedPrediction(1, "fake-2", 0.9, 0.1),
    ]

    interval = paired_auc_difference(predictions, samples=200, seed=4)

    assert interval.estimate == 1.0
    assert interval.successful_samples > 0


def test_binary_metric_bootstrap_skips_resamples_without_both_classes() -> None:
    items = [
        EvaluationItem(0, 0.1, "real-1", "real"),
        EvaluationItem(0, 0.2, "real-2", "real"),
        EvaluationItem(1, 0.8, "fake-1", "fake"),
        EvaluationItem(1, 0.9, "fake-2", "fake"),
    ]

    intervals = bootstrap_binary_metrics(
        items,
        threshold=0.5,
        samples=200,
        seed=5,
    )

    assert intervals["roc_auc"].estimate == 1.0
    assert intervals["roc_auc"].successful_samples > 0
