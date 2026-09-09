"""Report shaping for `ddf evaluate branch`, including the fake-only case.

The single-class path is not a corner case here: MNW's lab half is 120
forgeries with no genuine video at all, so it is the shape the external
evaluation actually takes.
"""

import argparse
from pathlib import Path

import pytest

from deepfake_detection.cli import (
    _branch_evaluation_report,
    _flat_metrics,
    _grouped_rates,
)
from deepfake_detection.training.checkpoints import RunMetadata


class FakeState:
    def __init__(self) -> None:
        self.metadata = RunMetadata(
            run_id="run-1",
            branch="visual",
            git_commit="abc123",
            split_hash="split-hash",
            preprocessing_hash="prep-hash",
            config_hash="config-hash",
            seed=17,
        )


def arguments(checkpoint: Path, *, dataset: str, scope: str) -> argparse.Namespace:
    return argparse.Namespace(
        branch="visual",
        checkpoint=checkpoint,
        dataset=dataset,
        evidence_scope=scope,
        threshold=0.5,
    )


def row(label: int, probability: float, method: str) -> dict[str, object]:
    return {
        "clip_id": f"{method}-{label}-{probability}",
        "label": label,
        "probability": probability,
        "predicted": int(probability >= 0.5),
        "source": "source-1",
        "manipulation_type": (
            "FakeVideo-RealAudio" if label else "RealVideo-RealAudio"
        ),
        "method": method,
    }


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"fixture-checkpoint")
    return path


def test_both_classes_present_produces_ranking_metrics(checkpoint: Path) -> None:
    rows = [
        row(1, 0.9, "wav2lip"),
        row(1, 0.8, "wav2lip"),
        row(0, 0.1, "real"),
        row(0, 0.2, "real"),
    ]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="FakeAVCeleb", scope="development_validation"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["metrics"]["roc_auc"] == 1.0
    assert "single_class" not in report
    assert report["confusion"] == {
        "true_positive": 2,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
    }


def test_fake_only_set_reports_a_detection_rate_and_no_auc(checkpoint: Path) -> None:
    """MNW's lab half has no negatives, so ROC-AUC is undefined, not zero."""
    rows = [
        row(1, 0.9, "mnw-wav2lip"),
        row(1, 0.7, "mnw-vasa_1"),
        row(1, 0.3, "mnw-vasa_1"),
        row(1, 0.2, "mnw-diff2lip"),
    ]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="MNW", scope="external_mnw"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["metrics"] is None
    assert report["single_class"]["class"] == "fake"
    assert report["single_class"]["detection_rate"] == 0.5
    assert "ROC-AUC" in report["single_class"]["note"]


def test_real_only_set_reports_specificity(checkpoint: Path) -> None:
    rows = [row(0, 0.1, "real"), row(0, 0.9, "real")]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="Celeb-DF-v2", scope="generalization_celebdf"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["metrics"] is None
    assert report["single_class"]["class"] == "real"
    assert report["single_class"]["specificity"] == 0.5


def test_report_records_the_split_the_checkpoint_was_trained_on(
    checkpoint: Path,
) -> None:
    """A cross-dataset row must not claim the evaluation set's own split hash."""
    rows = [row(1, 0.9, "mnw-wav2lip"), row(0, 0.1, "real")]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="MNW", scope="external_mnw"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["trained_on_split_hash"] == "split-hash"
    assert report["dataset"] == "MNW"
    assert report["evidence_scope"] == "external_mnw"


def test_per_method_rates_separate_each_generator() -> None:
    """The per-generator hit rate is the headline number for unseen generators."""
    rows = [
        row(1, 0.9, "mnw-wav2lip"),
        row(1, 0.8, "mnw-wav2lip"),
        row(1, 0.1, "mnw-vasa_1"),
        row(1, 0.2, "mnw-vasa_1"),
    ]

    rates = _grouped_rates(rows, "method")

    assert rates["mnw-wav2lip"]["accuracy"] == 1.0
    assert rates["mnw-vasa_1"]["accuracy"] == 0.0
    assert rates["mnw-vasa_1"]["rows"] == 2


def test_flat_metrics_emits_the_single_class_rate_for_tracking(
    checkpoint: Path,
) -> None:
    rows = [row(1, 0.9, "mnw-wav2lip"), row(1, 0.1, "mnw-vasa_1")]
    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="MNW", scope="external_mnw"),
        rows,
        FakeState(),
        "prep-hash",
    )

    flat = dict(_flat_metrics(report))

    assert flat["detection_rate"] == 0.5
    assert flat["confusion.true_positive"] == 1
    # The note is prose, not a metric, and must not reach MLflow as one.
    assert "note" not in flat


def test_report_rejects_an_unknown_evidence_scope() -> None:
    from deepfake_detection.experiments.scopes import validate_evidence_scope

    with pytest.raises(ValueError, match="Unknown evidence_scope"):
        validate_evidence_scope("generalisation_celebdf")


def test_lopsided_class_balance_is_flagged_as_unreliable(checkpoint: Path) -> None:
    """84 fakes against 1 real gives a defined ROC-AUC that means nothing.

    MNW is exactly this shape, so the record has to say so rather than letting
    the number be quoted.
    """
    rows = [row(1, 0.9, "mnw-wav2lip") for _ in range(20)]
    rows.append(row(0, 0.1, "real"))

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="MNW", scope="external_mnw"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["class_balance"]["real"] == 1
    assert report["class_balance"]["ranking_metrics_reliable"] is False
    assert "must not be" in report["class_balance"]["note"]
    # The metric is still recorded, just marked. Deleting it would hide that a
    # lopsided evaluation was run at all.
    assert report["metrics"]["roc_auc"] is not None


def test_balanced_set_is_marked_reliable(checkpoint: Path) -> None:
    rows = [row(1, 0.9, "wav2lip") for _ in range(12)]
    rows += [row(0, 0.1, "real") for _ in range(12)]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="FakeAVCeleb", scope="development_validation"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["class_balance"]["ranking_metrics_reliable"] is True
    assert "note" not in report["class_balance"]


def test_detection_rate_is_recorded_even_when_both_classes_exist(
    checkpoint: Path,
) -> None:
    """It is the statistic that survives a lopsided or unseen-generator set."""
    rows = [row(1, 0.9, "g"), row(1, 0.1, "g"), row(0, 0.1, "real")]

    report = _branch_evaluation_report(
        arguments(checkpoint, dataset="MNW", scope="external_mnw"),
        rows,
        FakeState(),
        "prep-hash",
    )

    assert report["detection_rate"] == 0.5
