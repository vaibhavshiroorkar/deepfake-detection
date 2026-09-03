import json
from pathlib import Path

import pytest

from deepfake_detection.dashboard.evidence import load_validation_evidence


def _write_evidence(metrics_path: Path, history_path: Path) -> None:
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "FakeAVCeleb",
                "rows": 400,
                "fixed_threshold": 0.5,
                "evidence_scope": "development_validation",
                "checkpoint_run_id": "4243b35e64c743b89cc33000cc9d3d3e",
                "evaluation_run_id": "56182266f70a424581f763b2d3b41989",
                "checkpoint_sha256": (
                    "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0"
                ),
                "preprocessing_hash": (
                    "fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96"
                ),
                "split_hash": (
                    "3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c"
                ),
                "metrics": {
                    "roc_auc": 0.999175,
                    "pr_auc": 0.999292,
                    "balanced_accuracy": 0.9975,
                    "f1": 0.997494,
                },
                "confusion": {
                    "true_positive": 199,
                    "true_negative": 200,
                    "false_positive": 0,
                    "false_negative": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(
            {
                "best_epoch": 4,
                "epochs": [{"epoch": value} for value in range(1, 6)],
                "metadata": {
                    "run_id": "4243b35e64c743b89cc33000cc9d3d3e",
                    "preprocessing_hash": (
                        "fd372dbe6bb64f359db4d57b05c3b5cd"
                        "27ed6660f2bb8bdc50567224e0928c96"
                    ),
                    "split_hash": (
                        "3255ae334536336c73058941285925f3d"
                        "d5b094c02b1037e19f379c6f45db30c"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_validation_evidence_reads_the_tracked_metric_contract(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.json"
    _write_evidence(metrics_path, history_path)

    evidence = load_validation_evidence(metrics_path, history_path)

    assert evidence.dataset == "FakeAVCeleb"
    assert evidence.rows == 400
    assert evidence.best_epoch == 4
    assert len(evidence.epochs) == 5
    assert evidence.metrics["roc_auc"] == 0.999175
    assert evidence.confusion["false_negative"] == 1


def test_validation_evidence_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="metrics"):
        load_validation_evidence(tmp_path / "metrics.json", tmp_path / "history.json")


def test_validation_evidence_rejects_the_wrong_scope(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.json"
    _write_evidence(metrics_path, history_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["evidence_scope"] = "prototype_only"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evidence_scope"):
        load_validation_evidence(metrics_path, history_path)


def test_validation_evidence_rejects_a_missing_required_metric(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.json"
    _write_evidence(metrics_path, history_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    del payload["metrics"]["f1"]
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="f1"):
        load_validation_evidence(metrics_path, history_path)


def test_validation_evidence_rejects_a_different_training_run(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.json"
    _write_evidence(metrics_path, history_path)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload["metadata"]["run_id"] = "different-run"
    history_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training run"):
        load_validation_evidence(metrics_path, history_path)
