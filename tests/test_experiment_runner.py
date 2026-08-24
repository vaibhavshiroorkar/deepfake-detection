from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import pytest

from deepfake_detection.experiments import runner
from deepfake_detection.experiments.configuration import ResolvedConfiguration
from deepfake_detection.experiments.runtime import RuntimeSnapshot
from deepfake_detection.experiments.tracking import NullRunLogger


def _configuration(
    *, command: list[str] | None = None, tracking_enabled: bool = True
) -> ResolvedConfiguration:
    return ResolvedConfiguration(
        values={
            "schema_version": 1,
            "command": command or ["target", "--value", "configured"],
            "arguments": {},
            "tracking": {"enabled": tracking_enabled},
        },
        sources=(Path("first.yaml"), Path("second.yaml")),
        sha256="configuration-hash",
    )


def _runtime() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        started_at_utc="2026-08-24T00:00:00+00:00",
        git_commit="commit",
        git_dirty=False,
        python_version="python",
        platform="platform",
        packages={},
        cpu="cpu",
        gpu=None,
        gpu_memory_mib=None,
        available_memory_mib=None,
        ffmpeg_version=None,
    )


def test_execute_configured_run_dispatches_from_root_and_attaches_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    observed: dict[str, object] = {}
    logger = NullRunLogger()

    def handler(arguments: argparse.Namespace) -> int:
        observed["cwd"] = Path.cwd()
        observed["value"] = arguments.value
        observed["logger"] = arguments._run_logger
        observed["configuration"] = arguments._resolved_configuration
        observed["hash"] = arguments._config_hash
        observed["run_id"] = arguments.run_id
        return 0

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--value")
        parser.add_argument("--run-id")
        parser.set_defaults(handler=handler)
        return parser

    loaded: list[tuple[Path, ...]] = []
    monkeypatch.setattr(
        runner,
        "load_configuration",
        lambda paths: loaded.append(tuple(paths)) or _configuration(),
    )
    monkeypatch.setattr(runner, "capture_runtime", lambda actual_root: _runtime())
    monkeypatch.setattr(
        runner,
        "start_tracked_run",
        lambda *args, **kwargs: _logger_context(logger),
    )
    original_cwd = Path.cwd()

    result = runner.execute_configured_run(
        (Path("first.yaml"), Path("second.yaml")),
        root=root,
        parser_factory=parser_factory,
    )

    assert result == 0
    assert loaded == [(original_cwd / "first.yaml", original_cwd / "second.yaml")]
    assert observed == {
        "cwd": root.resolve(),
        "value": "configured",
        "logger": logger,
        "configuration": _configuration(),
        "hash": "configuration-hash",
        "run_id": "",
    }
    assert Path.cwd() == original_cwd


def test_execute_configured_run_marks_nonzero_results_failed_and_restores_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    states: list[str] = []

    def handler(arguments: argparse.Namespace) -> int:
        del arguments
        assert Path.cwd() == root.resolve()
        return 2

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.set_defaults(handler=handler)
        return parser

    @contextmanager
    def tracked_run(*args: object, **kwargs: object):
        del args, kwargs
        try:
            yield NullRunLogger()
        except BaseException:
            states.append("FAILED")
            raise
        else:
            states.append("FINISHED")

    monkeypatch.setattr(
        runner, "load_configuration", lambda paths: _configuration(command=["target"])
    )
    monkeypatch.setattr(runner, "capture_runtime", lambda actual_root: _runtime())
    monkeypatch.setattr(runner, "start_tracked_run", tracked_run)
    original_cwd = Path.cwd()

    result = runner.execute_configured_run(
        (Path("config.yaml"),), root=root, parser_factory=parser_factory
    )

    assert result == 2
    assert states == ["FAILED"]
    assert Path.cwd() == original_cwd


def test_execute_configured_run_without_tracking_preserves_supplied_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    observed: dict[str, object] = {}
    configuration = _configuration(
        command=["target", "--run-id", "configured-run"], tracking_enabled=True
    )

    def handler(arguments: argparse.Namespace) -> int:
        observed["logger"] = arguments._run_logger
        observed["configuration"] = arguments._resolved_configuration
        observed["hash"] = arguments._config_hash
        observed["run_id"] = arguments.run_id
        return 0

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--run-id")
        parser.set_defaults(handler=handler)
        return parser

    monkeypatch.setattr(runner, "load_configuration", lambda paths: configuration)
    monkeypatch.setattr(runner, "capture_runtime", lambda actual_root: _runtime())

    result = runner.execute_configured_run(
        (Path("config.yaml"),),
        root=root,
        parser_factory=parser_factory,
        disable_tracking=True,
    )

    assert result == 0
    assert isinstance(observed["logger"], NullRunLogger)
    assert observed["configuration"] is configuration
    assert observed["hash"] == "configuration-hash"
    assert observed["run_id"] == "configured-run"


def test_execute_configured_run_rejects_a_parsed_run_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.set_defaults(handler=lambda arguments: 0)
        return parser

    monkeypatch.setattr(
        runner, "load_configuration", lambda paths: _configuration(command=["run"])
    )

    with pytest.raises(ValueError, match="cannot dispatch run"):
        runner.execute_configured_run(
            (Path("config.yaml"),), root=tmp_path, parser_factory=parser_factory
        )


def test_execute_configured_run_rejects_the_configured_run_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def handler(arguments: argparse.Namespace) -> int:
        nonlocal called
        del arguments
        called = True
        return 0

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.set_defaults(
            handler=handler,
            _configured_run_sentinel=runner._CONFIGURED_RUN_SENTINEL,
        )
        return parser

    monkeypatch.setattr(
        runner, "load_configuration", lambda paths: _configuration(command=["target"])
    )

    with pytest.raises(ValueError, match="cannot dispatch run"):
        runner.execute_configured_run(
            (Path("config.yaml"),), root=tmp_path, parser_factory=parser_factory
        )

    assert not called


def test_execute_configured_run_restores_cwd_after_a_handler_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def handler(arguments: argparse.Namespace) -> int:
        del arguments
        assert Path.cwd() == root.resolve()
        raise RuntimeError("handler failed")

    def parser_factory() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.set_defaults(handler=handler)
        return parser

    monkeypatch.setattr(
        runner, "load_configuration", lambda paths: _configuration(command=["target"])
    )
    monkeypatch.setattr(runner, "capture_runtime", lambda actual_root: _runtime())
    original_cwd = Path.cwd()

    with pytest.raises(RuntimeError, match="handler failed"):
        runner.execute_configured_run(
            (Path("config.yaml"),),
            root=root,
            parser_factory=parser_factory,
            disable_tracking=True,
        )

    assert Path.cwd() == original_cwd


def test_call_from_root_preserves_handler_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def change_directory(path: Path) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("cleanup failed")

    def handler(arguments: argparse.Namespace) -> int:
        del arguments
        raise RuntimeError("handler failed")

    monkeypatch.setattr(runner.os, "chdir", change_directory)

    with pytest.raises(RuntimeError, match="handler failed"):
        runner._call_from_root(handler, argparse.Namespace(), tmp_path)

    assert calls[0] == tmp_path
    assert len(calls) == 2


@contextmanager
def _logger_context(logger: NullRunLogger):
    yield logger
