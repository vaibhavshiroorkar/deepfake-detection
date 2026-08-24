import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("mlflow")

from mlflow import MlflowClient

from deepfake_detection.cli import main
from deepfake_detection.experiments.configuration import load_configuration


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_configured_smoke_run_logs_finished_mlflow_evidence(tmp_path: Path) -> None:
    local = tmp_path / "local.yaml"
    local.write_text(
        """
schema_version: 1
tracking:
  enabled: true
  tracking_uri: sqlite:///tracking.db
  artifact_root: artifacts
  experiment_name: smoke-test
  run_name: smoke-test-run
""".lstrip(),
        encoding="utf-8",
    )
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(
        """
schema_version: 1
command: [smoke]
arguments:
  output-dir: evidence
  seed: 17
  samples: 32
tracking:
  experiment_name: smoke-test
  run_name: smoke-test-run
""".lstrip(),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                "--root",
                str(tmp_path),
                "--config",
                str(local),
                "--config",
                str(smoke),
            ]
        )
        == 0
    )

    configuration = load_configuration((local, smoke))
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = client.get_experiment_by_name("smoke-test")
    assert experiment is not None
    runs = client.search_runs((experiment.experiment_id,))
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["configuration_sha256"] == configuration.sha256
    assert run.data.params["smoke.evidence_scope"] == "software_fixture_only"
    assert "smoke.report_sha256" in run.data.params
    artifacts = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert {
        "resolved-config.yaml",
        "runtime.json",
        "fusion.joblib",
        "smoke-report.json",
        "predictions.csv",
    } <= artifacts
    downloaded = {
        name: Path(
            client.download_artifacts(run.info.run_id, name, tmp_path / "download")
        )
        for name in ("fusion.joblib", "predictions.csv", "smoke-report.json")
    }
    report = json.loads(downloaded["smoke-report.json"].read_text(encoding="utf-8"))
    assert report["evidence_scope"] == "software_fixture_only"
    assert set(report["metric_evidence_scope"].values()) == {"software_fixture_only"}
    assert (
        _file_hash(downloaded["fusion.joblib"])
        == report["artifact_hashes"]["fusion.joblib"]
    )
    assert (
        _file_hash(downloaded["predictions.csv"])
        == report["artifact_hashes"]["predictions.csv"]
    )
    assert (
        _file_hash(downloaded["smoke-report.json"])
        == run.data.params["smoke.report_sha256"]
    )
    for artifact_name, artifact_hash in report["artifact_hashes"].items():
        assert run.data.params[f"smoke.payload.{artifact_name}.sha256"] == artifact_hash
    scoped_metric_prefix = "smoke.software_fixture_only.validation."
    assert set(run.data.metrics) == {
        f"{scoped_metric_prefix}{name}" for name in report["metrics"]
    }
    assert {
        name.removeprefix(scoped_metric_prefix): value
        for name, value in run.data.metrics.items()
    } == pytest.approx(report["metrics"])
