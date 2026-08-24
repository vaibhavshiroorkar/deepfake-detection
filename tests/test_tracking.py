import hashlib
import logging
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from deepfake_detection.experiments import tracking as experiment_tracking
from deepfake_detection.experiments.configuration import ResolvedConfiguration
from deepfake_detection.experiments.runtime import RuntimeSnapshot
from deepfake_detection.views.tracking import Box, Detection, select_primary_track


@dataclass(frozen=True)
class _FakeRunInfo:
    run_id: str


@dataclass(frozen=True)
class _FakeRun:
    info: _FakeRunInfo


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.experiments: dict[str, str] = {}

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def set_tracking_uri(self, uri: str) -> None:
        self._record("set_tracking_uri", uri)

    def get_experiment_by_name(self, name: str) -> object | None:
        self._record("get_experiment_by_name", name)
        return self.experiments.get(name)

    def create_experiment(self, name: str, artifact_location: str) -> str:
        self._record("create_experiment", name, artifact_location)
        self.experiments[name] = "1"
        return "1"

    def set_experiment(self, name: str) -> None:
        self._record("set_experiment", name)

    def start_run(self, *, run_name: str) -> _FakeRun:
        self._record("start_run", run_name=run_name)
        return _FakeRun(info=_FakeRunInfo(run_id="fake-run-id"))

    def log_params(self, values: dict[str, Any]) -> None:
        self._record("log_params", values)

    def set_tags(self, values: dict[str, str]) -> None:
        self._record("set_tags", values)

    def log_dict(self, values: dict[str, Any], artifact_file: str) -> None:
        self._record("log_dict", values, artifact_file)

    def log_metrics(self, values: dict[str, float], *, step: int | None = None) -> None:
        self._record("log_metrics", values, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._record("log_artifact", path, artifact_path=artifact_path)

    def end_run(self, status: str) -> None:
        self._record("end_run", status=status)


def _configuration() -> ResolvedConfiguration:
    return ResolvedConfiguration(
        values={
            "schema_version": 1,
            "command": ["smoke"],
            "arguments": {"seed": 17, "samples": 32},
            "tracking": {
                "enabled": True,
                "tracking_uri": "sqlite:///mlflow.db",
                "artifact_root": "mlartifacts",
                "experiment_name": "tracking-tests",
                "run_name": "fake-run",
                "tags": {"project": "deepfake-generalization"},
            },
        },
        sources=(Path("configs/local.yaml"),),
        sha256="a" * 64,
    )


def _runtime() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        started_at_utc="2026-08-25T00:00:00+00:00",
        git_commit="abc123",
        git_dirty=False,
        python_version="3.11.0",
        platform="test-platform",
        packages={"numpy": "2.4.6"},
        cpu="test-cpu",
        gpu=None,
        gpu_memory_mib=None,
        available_memory_mib=1024,
        ffmpeg_version="ffmpeg test",
    )


def _calls(
    fake: _FakeMlflow, name: str
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    return [
        (args, kwargs) for call_name, args, kwargs in fake.calls if call_name == name
    ]


def _shortened(value: str, limit: int) -> str:
    suffix = f"...[sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}]"
    return f"{value[: limit - len(suffix)]}{suffix}"


def test_disabled_tracking_does_not_import_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = experiment_tracking.TrackingSettings.from_configuration(
        {"enabled": False}, root=tmp_path
    )

    def fail_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(experiment_tracking.importlib, "import_module", fail_import)

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=_configuration(),
        runtime=_runtime(),
    ) as logger:
        logger.log_metrics({"accuracy": 0.5})

    assert logger.run_id == ""


def test_tracked_run_logs_config_runtime_metrics_and_finishes(tmp_path: Path) -> None:
    configuration = _configuration()
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()
    artifact = tmp_path / "history.json"
    artifact.write_text("{}", encoding="utf-8")

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=_runtime(),
        mlflow_module=fake,
    ) as logger:
        assert logger.run_id == "fake-run-id"
        logger.log_metrics({"accuracy": 0.75}, step=3)
        logger.log_artifact(artifact, artifact_path="reports")

    parameters = _calls(fake, "log_params")[0][0][0]
    tags = _calls(fake, "set_tags")[0][0][0]
    logged_dicts = {
        artifact_file: values for (values, artifact_file), _ in _calls(fake, "log_dict")
    }
    assert parameters["arguments.seed"] == "17"
    assert parameters["configuration_sha256"] == "a" * 64
    assert tags["configuration_sha256"] == "a" * 64
    assert tags["git_commit"] == "abc123"
    assert logged_dicts["resolved-config.yaml"] == configuration.values
    assert logged_dicts["runtime.json"] == _runtime().as_dict()
    assert _calls(fake, "log_metrics") == [(({"accuracy": 0.75},), {"step": 3})]
    assert _calls(fake, "log_artifact") == [((artifact,), {"artifact_path": "reports"})]
    assert _calls(fake, "end_run") == [((), {"status": "FINISHED"})]


def test_tracking_normalizes_over_limit_parameter_and_tag_data(tmp_path: Path) -> None:
    configuration = _configuration()
    long_parameter_value = "p" * 6001
    long_tag_value = "t" * 8001
    long_runtime_value = "r" * 8001
    long_parameter_key = f"caller:{'k' * 300}"
    long_tag_key = f"tag:{'k' * 300}"
    unsafe_path_keys = (".", "..", "../tag", "..tag", "paths/.")
    values = dict(configuration.values)
    values["arguments"] = {
        **configuration.values["arguments"],
        "description": long_parameter_value,
    }
    tracking_values = dict(configuration.values["tracking"])
    tracking_values["run_name"] = "n" * 8001
    tracking_values["tags"] = {
        "project": "deepfake-generalization",
        "custom": long_tag_value,
        long_tag_key: "custom tag value",
    }
    values["tracking"] = tracking_values
    configuration = ResolvedConfiguration(
        values=values,
        sources=configuration.sources,
        sha256=configuration.sha256,
    )
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=replace(_runtime(), cpu=long_runtime_value),
        mlflow_module=fake,
    ) as logger:
        logger.log_params(
            {
                long_parameter_key: long_parameter_value,
                **{key: f"unsafe value {key}" for key in unsafe_path_keys},
            }
        )

    initial_parameters, caller_parameters = [
        values for (values,), _ in _calls(fake, "log_params")
    ]
    tags = _calls(fake, "set_tags")[0][0][0]
    run_name = _calls(fake, "start_run")[0][1]["run_name"]
    parameter_key_hash = hashlib.sha256(long_parameter_key.encode("utf-8")).hexdigest()[
        :16
    ]
    tag_key_hash = hashlib.sha256(long_tag_key.encode("utf-8")).hexdigest()[:16]

    assert initial_parameters["arguments.seed"] == "17"
    assert initial_parameters["arguments.description"] == _shortened(
        long_parameter_value, 6000
    )
    assert len(caller_parameters) == 1 + len(unsafe_path_keys)
    assert long_parameter_key not in caller_parameters
    normalized_parameter_key = next(
        key for key in caller_parameters if key.endswith(f"__{parameter_key_hash}")
    )
    assert len(normalized_parameter_key) <= 250
    assert caller_parameters[normalized_parameter_key] == _shortened(
        long_parameter_value, 6000
    )
    for unsafe_path_key in unsafe_path_keys:
        assert unsafe_path_key not in caller_parameters
        key_hash = hashlib.sha256(unsafe_path_key.encode("utf-8")).hexdigest()[:16]
        normalized_key = next(
            key for key in caller_parameters if key.endswith(f"__{key_hash}")
        )
        assert caller_parameters[normalized_key] == f"unsafe value {unsafe_path_key}"
    assert tags["project"] == "deepfake-generalization"
    assert tags["custom"] == _shortened(long_tag_value, 8000)
    assert tags["cpu"] == _shortened(long_runtime_value, 8000)
    assert long_tag_key not in tags
    assert any(key.endswith(f"__{tag_key_hash}") for key in tags)
    assert all(len(key) <= 250 and len(value) <= 8000 for key, value in tags.items())
    assert run_name == _shortened("n" * 8001, 8000)


def test_logger_keeps_values_when_normalized_parameter_keys_collide(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=_runtime(),
        mlflow_module=fake,
    ) as logger:
        logger.log_params({1: "integer key", "1": "string key"})

    caller_parameters = _calls(fake, "log_params")[1][0][0]
    assert len(caller_parameters) == 2
    assert set(caller_parameters.values()) == {"integer key", "string key"}
    assert "1" in caller_parameters
    assert any(key.startswith("1__") for key in caller_parameters)


def test_tracked_run_records_failure_and_marks_run_failed(tmp_path: Path) -> None:
    configuration = _configuration()
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()

    class SentinelError(Exception):
        pass

    with pytest.raises(SentinelError, match="sentinel failure"):
        with experiment_tracking.start_tracked_run(
            settings,
            configuration=configuration,
            runtime=_runtime(),
            mlflow_module=fake,
        ):
            raise SentinelError("sentinel failure")

    logged_dicts = {
        artifact_file: values for (values, artifact_file), _ in _calls(fake, "log_dict")
    }
    assert logged_dicts["failure.json"] == {
        "exception_type": "SentinelError",
        "message": "sentinel failure",
    }
    assert _calls(fake, "end_run") == [((), {"status": "FAILED"})]


def test_tracked_run_redacts_secret_configuration_values(tmp_path: Path) -> None:
    configuration = _configuration()
    values = dict(configuration.values)
    values["credentials"] = {"api_token": "top-secret"}
    tracking_values = dict(configuration.values["tracking"])
    tracking_values["tags"] = {"api_token": "top-secret"}
    values["tracking"] = tracking_values
    configuration = ResolvedConfiguration(
        values=values,
        sources=configuration.sources,
        sha256=configuration.sha256,
    )
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=_runtime(),
        mlflow_module=fake,
    ):
        pass

    parameters = _calls(fake, "log_params")[0][0][0]
    tags = _calls(fake, "set_tags")[0][0][0]
    logged_dicts = {
        artifact_file: values for (values, artifact_file), _ in _calls(fake, "log_dict")
    }
    assert "credentials.api_token" not in parameters
    assert "api_token" not in tags
    assert "top-secret" not in str(logged_dicts["resolved-config.yaml"])


def test_failed_run_redacts_secret_values_from_the_exception_message(
    tmp_path: Path,
) -> None:
    configuration = _configuration()
    values = dict(configuration.values)
    values["credentials"] = {"api_token": "top-secret"}
    configuration = ResolvedConfiguration(
        values=values,
        sources=configuration.sources,
        sha256=configuration.sha256,
    )
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )
    fake = _FakeMlflow()

    with pytest.raises(RuntimeError, match="top-secret"):
        with experiment_tracking.start_tracked_run(
            settings,
            configuration=configuration,
            runtime=_runtime(),
            mlflow_module=fake,
        ):
            raise RuntimeError("request used top-secret")

    logged_dicts = {
        artifact_file: values for (values, artifact_file), _ in _calls(fake, "log_dict")
    }
    assert logged_dicts["failure.json"]["message"] == "request used [REDACTED]"


def test_relative_sqlite_and_artifact_paths_resolve_from_project_root(
    tmp_path: Path,
) -> None:
    settings = experiment_tracking.TrackingSettings.from_configuration(
        {
            "enabled": True,
            "tracking_uri": "sqlite:///state/mlflow.db",
            "artifact_root": "artifacts/tracking",
            "experiment_name": "path-test",
            "run_name": "path-run",
        },
        root=tmp_path,
    )
    configuration = _configuration()
    fake = _FakeMlflow()

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=_runtime(),
        mlflow_module=fake,
    ):
        pass

    expected_uri = f"sqlite:///{(tmp_path / 'state' / 'mlflow.db').as_posix()}"
    assert settings.tracking_uri == expected_uri
    assert settings.artifact_root == tmp_path / "artifacts" / "tracking"
    assert settings.artifact_root.is_dir()
    assert _calls(fake, "set_tracking_uri") == [((expected_uri,), {})]
    assert _calls(fake, "create_experiment") == [
        (("path-test", settings.artifact_root.as_uri()), {})
    ]


def test_real_local_backend_finishes_a_run(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    configuration = _configuration()
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )

    with experiment_tracking.start_tracked_run(
        settings,
        configuration=configuration,
        runtime=_runtime(),
    ) as logger:
        logger.log_metrics({"accuracy": 0.75}, step=3)

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.params["arguments.seed"] == "17"
    assert runs[0].data.metrics["accuracy"] == 0.75


def test_real_local_backend_accepts_normalized_limit_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mlflow = pytest.importorskip("mlflow")
    long_parameter_value = "p" * 6001
    long_tag_value = "t" * 8001
    long_runtime_value = "r" * 8001
    configuration = _configuration()
    values = dict(configuration.values)
    values["arguments"] = {
        **configuration.values["arguments"],
        "description": long_parameter_value,
    }
    tracking_values = dict(configuration.values["tracking"])
    tracking_values["run_name"] = "n" * 8001
    tracking_values["tags"] = {"custom": long_tag_value}
    values["tracking"] = tracking_values
    configuration = ResolvedConfiguration(
        values=values,
        sources=configuration.sources,
        sha256=configuration.sha256,
    )
    settings = experiment_tracking.TrackingSettings.from_configuration(
        configuration.values["tracking"], root=tmp_path
    )

    with caplog.at_level(logging.WARNING, logger="mlflow.utils.validation"):
        with experiment_tracking.start_tracked_run(
            settings,
            configuration=configuration,
            runtime=replace(_runtime(), cpu=long_runtime_value),
        ) as logger:
            logger.log_params(
                {key: f"unsafe value {key}" for key in (".", "..", "../tag", "..tag")}
            )

    client = mlflow.tracking.MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["arguments.description"] == _shortened(
        long_parameter_value, 6000
    )
    assert run.data.tags["custom"] == _shortened(long_tag_value, 8000)
    assert run.data.tags["cpu"] == _shortened(long_runtime_value, 8000)
    assert run.data.tags["mlflow.runName"] == _shortened("n" * 8001, 8000)
    for unsafe_path_key in (".", "..", "../tag", "..tag"):
        assert unsafe_path_key not in run.data.params
        key_hash = hashlib.sha256(unsafe_path_key.encode("utf-8")).hexdigest()[:16]
        normalized_key = next(
            key for key in run.data.params if key.endswith(f"__{key_hash}")
        )
        assert run.data.params[normalized_key] == f"unsafe value {unsafe_path_key}"
    assert not [record for record in caplog.records if "truncated" in record.message]


class FaceTrackingTests(unittest.TestCase):
    def test_selects_the_longest_consistent_track(self) -> None:
        frames = (
            (Detection(Box(0, 0, 10, 10), 0.95),),
            (
                Detection(Box(1, 0, 11, 10), 0.94),
                Detection(Box(30, 30, 40, 40), 0.99),
            ),
            (Detection(Box(2, 0, 12, 10), 0.93),),
        )

        result = select_primary_track(frames, min_iou=0.3)

        self.assertEqual(result.frame_indices, (0, 1, 2))
        self.assertEqual(result.coverage, 1.0)
        self.assertTrue(result.stable)

    def test_marks_equal_length_faces_as_ambiguous(self) -> None:
        frames = (
            (
                Detection(Box(0, 0, 10, 10), 0.95),
                Detection(Box(30, 30, 40, 40), 0.95),
            ),
            (
                Detection(Box(1, 0, 11, 10), 0.95),
                Detection(Box(31, 30, 41, 40), 0.95),
            ),
        )

        result = select_primary_track(frames, min_iou=0.3)

        self.assertFalse(result.stable)


if __name__ == "__main__":
    unittest.main()
