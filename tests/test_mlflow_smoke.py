import pytest

pytest.importorskip("mlflow")

from pathlib import Path

from mlflow import MlflowClient

from deepfake_detection.cli import main
from deepfake_detection.experiments.configuration import load_configuration


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
    assert "smoke.validation.roc_auc" in run.data.metrics
    assert "smoke.report_sha256" in run.data.params
    artifacts = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert {
        "resolved-config.yaml",
        "runtime.json",
        "fusion.joblib",
        "smoke-report.json",
        "predictions.csv",
    } <= artifacts
