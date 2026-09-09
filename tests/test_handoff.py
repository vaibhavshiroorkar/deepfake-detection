"""The handoff's generated blocks must describe what is on disk, not memory."""

import json
import sqlite3
from pathlib import Path

import pytest

from deepfake_detection.documentation.handoff import (
    collect_evidence,
    collect_mlflow,
    render_datasets,
    render_evidence,
    render_mlflow,
    replace_block,
    survey_datasets,
    update_handoff,
)

HANDOFF = """# Handoff

## Datasets

<!-- BEGIN GENERATED DATASETS -->
stale
<!-- END GENERATED DATASETS -->

Prose that must survive.

## Evidence

<!-- BEGIN GENERATED EVIDENCE -->
stale
<!-- END GENERATED EVIDENCE -->

## MLflow

<!-- BEGIN GENERATED MLFLOW -->
stale
<!-- END GENERATED MLFLOW -->
"""


def test_survey_counts_videos_that_are_actually_present(tmp_path: Path) -> None:
    data = tmp_path / "data"
    celeb = data / "Celeb-DF-v2" / "Celeb-real"
    celeb.mkdir(parents=True)
    (celeb / "id0_0000.mp4").write_bytes(b"x" * 10)
    (celeb / "id0_0001.mp4").write_bytes(b"x" * 20)
    (celeb / "notes.txt").write_text("ignored")

    states = {state.name: state for state in survey_datasets(data)}

    assert states["Celeb-DF-v2"].present
    assert states["Celeb-DF-v2"].videos == 2
    assert states["Celeb-DF-v2"].bytes_on_disk == 30
    assert not states["FakeAVCeleb_v1.2"].present


def test_render_datasets_says_absent_rather_than_omitting(tmp_path: Path) -> None:
    """A missing dataset has to appear as absent. Dropping the row is how the
    old hand-written table implied things existed that did not."""
    table = render_datasets(survey_datasets(tmp_path / "data"))

    assert "`data/MNW`" in table
    assert "absent" in table


def test_evidence_reports_a_single_class_set_as_undefined(tmp_path: Path) -> None:
    runs = tmp_path / "runs" / "external"
    runs.mkdir(parents=True)
    (runs / "mnw-metrics.json").write_text(
        json.dumps(
            {
                "dataset": "MNW",
                "evidence_scope": "external_mnw",
                "rows": 120,
                "metrics": None,
                "single_class": {"class": "fake", "detection_rate": 0.31},
                "confusion": {},
            }
        ),
        encoding="utf-8",
    )

    table = render_evidence(collect_evidence(tmp_path / "runs"))

    assert "undefined" in table
    assert "0.3100 (fake only)" in table
    assert "external_mnw" in table


def test_evidence_renders_ranking_metrics_when_both_classes_exist(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs" / "pilot"
    runs.mkdir(parents=True)
    (runs / "visual-metrics.json").write_text(
        json.dumps(
            {
                "dataset": "FakeAVCeleb",
                "evidence_scope": "development_validation",
                "rows": 600,
                "metrics": {"roc_auc": 0.9312, "balanced_accuracy": 0.8899},
                "confusion": {},
            }
        ),
        encoding="utf-8",
    )

    table = render_evidence(collect_evidence(tmp_path / "runs"))

    assert "0.9312" in table
    assert "0.8899" in table


def test_evidence_states_plainly_when_there_is_none(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    assert "No evaluation reports" in render_evidence(collect_evidence(tmp_path / "runs"))


def test_mlflow_summary_counts_runs_by_experiment_and_status(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE experiments (experiment_id INTEGER, name TEXT)")
    connection.execute("CREATE TABLE runs (run_uuid TEXT, experiment_id INTEGER, status TEXT)")
    connection.execute("INSERT INTO experiments VALUES (1, 'pilot')")
    connection.executemany(
        "INSERT INTO runs VALUES (?, 1, ?)",
        [("a", "FINISHED"), ("b", "FINISHED"), ("c", "FAILED")],
    )
    connection.commit()
    connection.close()

    table = render_mlflow(collect_mlflow(database))

    assert "`pilot`" in table
    assert "3 (1 failed, 2 finished)" in table


def test_mlflow_summary_handles_a_missing_database(tmp_path: Path) -> None:
    assert "No MLflow database" in render_mlflow(collect_mlflow(tmp_path / "absent.db"))


def test_update_preserves_prose_outside_the_markers(tmp_path: Path) -> None:
    path = tmp_path / "handoff.md"
    path.write_text(HANDOFF, encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()

    assert update_handoff(tmp_path, path) is True

    text = path.read_text(encoding="utf-8")
    assert "Prose that must survive." in text
    assert "stale" not in text


def test_update_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "handoff.md"
    path.write_text(HANDOFF, encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "runs").mkdir()

    update_handoff(tmp_path, path)

    assert update_handoff(tmp_path, path) is False


def test_replace_block_refuses_a_file_without_markers() -> None:
    with pytest.raises(ValueError, match="missing the datasets markers"):
        replace_block("# Handoff\n", "datasets", "body")
