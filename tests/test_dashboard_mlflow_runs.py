from pathlib import Path

import pytest

from deepfake_detection.dashboard.lib import mlflow_runs


def summary(**overrides) -> mlflow_runs.RunSummary:
    values = {
        "run_id": "0123456789abcdef",
        "run_name": "visual-efficientnet-b0-initial-seed17",
        "experiment": "initial-baseline-20260902",
        "status": "FINISHED",
        "start_time": 1,
        "params": {"arguments.seed": "17", "arguments.learning-rate": "0.0001"},
        "metrics": {"validation.loss": 0.019},
        "tags": {"evidence_scope": "development_baseline"},
    }
    values.update(overrides)
    return mlflow_runs.RunSummary(**values)


def test_tracking_uri_points_at_the_local_store() -> None:
    uri = mlflow_runs.tracking_uri()
    assert uri.startswith("sqlite:///")
    assert uri.endswith("mlflow.db")


def test_tracking_uri_accepts_another_database(tmp_path: Path) -> None:
    database = tmp_path / "other.db"
    assert mlflow_runs.tracking_uri(database).endswith("other.db")


def test_comparison_rows_drop_the_arguments_prefix() -> None:
    row = mlflow_runs.comparison_rows([summary()])[0]
    assert row["seed"] == "17"
    assert row["learning-rate"] == "0.0001"
    assert "arguments.seed" not in row


def test_comparison_rows_leave_an_unrecorded_value_blank() -> None:
    row = mlflow_runs.comparison_rows([summary()])[0]
    assert row["evaluation.roc_auc"] is None
    assert row["epochs"] is None


def test_comparison_rows_carry_identity_and_metrics() -> None:
    row = mlflow_runs.comparison_rows([summary()])[0]
    assert row["run"] == "visual-efficientnet-b0-initial-seed17"
    assert row["status"] == "FINISHED"
    assert row["experiment"] == "initial-baseline-20260902"
    assert row["validation.loss"] == 0.019
    assert row["run_id"] == "0123456789abcdef"


def test_comparison_rows_name_an_unnamed_run_by_its_id() -> None:
    row = mlflow_runs.comparison_rows([summary(run_name="")])[0]
    assert row["run"] == "01234567"


def test_comparison_rows_share_one_set_of_columns() -> None:
    rows = mlflow_runs.comparison_rows(
        [summary(), summary(run_id="ff", params={}, metrics={})]
    )
    assert set(rows[0]) == set(rows[1])


def test_reading_the_store_reports_a_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("no mlflow here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        mlflow_runs.experiments()


def test_experiments_excludes_the_empty_default() -> None:
    pytest.importorskip("mlflow")
    assert "Default" not in mlflow_runs.experiments()


def test_comparison_rows_surface_metrics_nobody_hardcoded() -> None:
    """A cross-dataset metric must reach the table without a code change.

    The column list used to be a literal tuple, so any run recording something
    new showed up with that value silently missing.
    """
    summaries = [
        mlflow_runs.RunSummary(
            run_id="a" * 32,
            run_name="celebdf-zero-shot",
            experiment="generalization",
            status="FINISHED",
            start_time=0,
            params={"arguments.seed": "17", "arguments.fake-ratio": "3.0"},
            metrics={"evaluation.roc_auc": 0.71, "evaluation.detection_rate": 0.42},
            tags={"evidence_scope": "generalization_celebdf"},
        )
    ]

    rows = mlflow_runs.comparison_rows(summaries)

    assert rows[0]["evaluation.detection_rate"] == 0.42
    assert rows[0]["fake-ratio"] == "3.0"
    # The familiar columns keep their place at the front.
    assert rows[0]["evaluation.roc_auc"] == 0.71
    assert list(rows[0])[:3] == ["run", "status", "experiment"]
