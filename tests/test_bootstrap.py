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


def test_cluster_bootstrap_resamples_whole_identities_and_never_splits_one() -> None:
    """The test above claims this in its name and does not check it.

    Its statistic is a mean over probabilities, which is identical whether the
    resampling unit is the identity or the clip, so it cannot distinguish the
    two. This inspects the sample the statistic actually receives.
    """
    items = [
        EvaluationItem(0, 0.1, "id1", "real"),
        EvaluationItem(0, 0.3, "id1", "real"),
        EvaluationItem(1, 0.7, "id2", "fake"),
        EvaluationItem(1, 0.9, "id2", "fake"),
    ]
    seen: list[list[str]] = []

    def statistic(sample: list[EvaluationItem]) -> float:
        seen.append([str(item.probability) for item in sample])
        return 0.0

    cluster_bootstrap_interval(items, statistic, samples=50, seed=3)

    # Every resample is a whole number of identities, so its size is a multiple
    # of two here, and a clip never appears without its identity partner.
    for sample in seen:
        assert len(sample) == 4
        assert sample.count("0.1") == sample.count("0.3")
        assert sample.count("0.7") == sample.count("0.9")


def test_cluster_bootstrap_skips_single_class_draws_and_counts_them_out() -> None:
    """A draw that lands on one class is skipped, not scored as 0.5.

    Five scripts used to return 0.5 there, which is a value rather than an
    absence: it sits in the middle of the percentile distribution and pulls the
    interval toward chance.
    """
    items = [
        EvaluationItem(0, 0.1, "real-1", "real"),
        EvaluationItem(1, 0.9, "fake-1", "fake"),
    ]

    def auc_or_raise(sample: list[EvaluationItem]) -> float:
        if len({item.label for item in sample}) != 2:
            raise ValueError("one class")
        return 1.0

    interval = cluster_bootstrap_interval(items, auc_or_raise, samples=200, seed=7)

    # With two identities, half the draws pick the same one twice, so a real
    # count of survivors must be below the requested sample count. The old
    # implementation always reported `samples` because nothing was ever skipped.
    assert 0 < interval.successful_samples < 200
    assert interval.estimate == 1.0
    # Nothing scored 0.5, so no part of the interval is dragged to chance.
    assert interval.lower == 1.0
    assert interval.upper == 1.0


def test_cluster_bootstrap_raises_when_no_draw_has_both_classes() -> None:
    items = [
        EvaluationItem(0, 0.1, "only-identity", "real"),
        EvaluationItem(1, 0.9, "only-identity", "fake"),
    ]

    def needs_two_identities(sample: list[EvaluationItem]) -> float:
        raise ValueError("never valid")

    with pytest.raises(ValueError, match="No valid bootstrap samples"):
        cluster_bootstrap_interval(items, needs_two_identities, samples=10, seed=1)
