import pytest

from deepfake_detection.evaluation.metrics import (
    EvaluationItem,
    binary_metrics,
    evaluate_items,
    per_method_metrics,
    select_balanced_accuracy_threshold,
    subgroup_metrics,
)


def test_binary_metrics_include_discrimination_calibration_and_error_rates() -> None:
    report = binary_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.2, 0.8, 0.9],
        threshold=0.5,
        calibration_bins=2,
    )

    assert report.roc_auc == 1.0
    assert report.pr_auc == 1.0
    assert report.balanced_accuracy == 1.0
    assert report.f1 == 1.0
    assert report.fpr == 0.0
    assert report.fnr == 0.0
    assert report.brier == pytest.approx(0.025)
    assert report.expected_calibration_error == pytest.approx(0.15)


def test_evaluation_reports_abstentions_in_total_coverage() -> None:
    items = [
        EvaluationItem(0, 0.1, "id1", "real"),
        EvaluationItem(1, None, "id2", "wav2lip"),
        EvaluationItem(1, 0.9, "id3", "rtvc"),
    ]

    report = evaluate_items(items, threshold=0.5)

    assert report.total == 3
    assert report.scored == 2
    assert report.abstained == 1
    assert report.coverage == pytest.approx(2 / 3)


def test_per_method_metrics_compare_each_fake_method_with_real_clips() -> None:
    items = [
        EvaluationItem(0, 0.1, "real-1", "real"),
        EvaluationItem(0, 0.2, "real-2", "real"),
        EvaluationItem(1, 0.8, "fake-1", "wav2lip"),
        EvaluationItem(1, 0.9, "fake-2", "rtvc"),
    ]

    reports = per_method_metrics(items, threshold=0.5)

    assert set(reports) == {"rtvc", "wav2lip"}
    assert reports["wav2lip"].total == 3
    assert reports["rtvc"].metrics.roc_auc == 1.0


def test_subgroup_metrics_keep_group_specific_coverage() -> None:
    items = [
        EvaluationItem(0, 0.1, "a-real", "real", race="A"),
        EvaluationItem(1, 0.9, "a-fake", "wav2lip", race="A"),
        EvaluationItem(0, 0.2, "b-real", "real", race="B"),
        EvaluationItem(1, None, "b-fake", "rtvc", race="B"),
    ]

    reports = subgroup_metrics(items, attribute="race", threshold=0.5)

    assert reports["A"].coverage == 1.0
    assert reports["B"].coverage == 0.5


def test_threshold_selection_uses_validation_probabilities() -> None:
    selection = select_balanced_accuracy_threshold(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.4, 0.6, 0.9],
    )

    assert selection.threshold == pytest.approx(0.5)
    assert selection.balanced_accuracy == 1.0


def test_threshold_selection_rejects_single_class_input() -> None:
    with pytest.raises(ValueError, match="both classes"):
        select_balanced_accuracy_threshold(
            labels=[0, 0],
            probabilities=[0.1, 0.2],
        )
