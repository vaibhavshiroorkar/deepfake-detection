from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from deepfake_detection.experiments.configuration import (
    configuration_argv,
    load_configuration,
)
from deepfake_detection.experiments.runtime import capture_runtime
from deepfake_detection.experiments.tracking import TrackingSettings, start_tracked_run

_CONFIGURED_RUN_SENTINEL = object()


class _HandlerFailed(Exception):
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code


def execute_configured_run(
    configuration_paths: Sequence[Path],
    *,
    root: Path,
    parser_factory: Callable[[], argparse.ArgumentParser],
    disable_tracking: bool = False,
) -> int:
    project_root = root.resolve()
    resolved_paths = tuple(Path(path).resolve() for path in configuration_paths)
    configuration = load_configuration(resolved_paths)
    arguments = parser_factory().parse_args(configuration_argv(configuration))
    handler = arguments.handler
    if (
        getattr(arguments, "command", None) == "run"
        or getattr(arguments, "_configured_run_sentinel", None)
        is _CONFIGURED_RUN_SENTINEL
    ):
        raise ValueError("Configured runs cannot dispatch run")

    runtime = capture_runtime(project_root)
    settings = TrackingSettings.from_configuration(
        configuration.values, root=project_root
    )
    if disable_tracking:
        settings = replace(settings, enabled=False)

    try:
        with start_tracked_run(
            settings,
            configuration=configuration,
            runtime=runtime,
        ) as logger:
            arguments._run_logger = logger
            arguments._resolved_configuration = configuration
            arguments._config_hash = configuration.sha256
            if settings.enabled and hasattr(arguments, "run_id"):
                arguments.run_id = logger.run_id
            exit_code = _call_from_root(handler, arguments, project_root)
            if exit_code:
                raise _HandlerFailed(exit_code)
            return 0
    except _HandlerFailed as error:
        return error.exit_code


def _call_from_root(
    handler: Callable[[argparse.Namespace], int],
    arguments: argparse.Namespace,
    root: Path,
) -> int:
    original_directory = Path.cwd()
    primary_error: BaseException | None = None
    try:
        os.chdir(root)
        return int(handler(arguments))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            os.chdir(original_directory)
        except BaseException:
            if primary_error is None:
                raise
