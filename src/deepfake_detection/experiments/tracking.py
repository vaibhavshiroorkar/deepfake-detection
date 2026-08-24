from __future__ import annotations

import hashlib
import importlib
import posixpath
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from deepfake_detection.experiments.configuration import ResolvedConfiguration
from deepfake_detection.experiments.runtime import RuntimeSnapshot

_SENSITIVE_KEYS = frozenset(
    {
        "credential",
        "key",
        "password",
        "apikey",
        "accesskey",
        "privatekey",
        "secret",
        "token",
    }
)
_MAX_KEY_LENGTH = 250
_MAX_PARAMETER_VALUE_LENGTH = 6000
_MAX_TAG_VALUE_LENGTH = 8000
_HASH_LENGTH = 16
_KEY_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- /"
)


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    enabled: bool
    tracking_uri: str
    artifact_root: Path
    experiment_name: str
    run_name: str
    tags: dict[str, str]

    @classmethod
    def from_configuration(
        cls,
        values: Mapping[str, Any],
        *,
        root: Path,
    ) -> TrackingSettings:
        tracking = values.get("tracking", values)
        if not isinstance(tracking, Mapping):
            raise ValueError("Configuration tracking must be a mapping")

        project_root = root.resolve()
        enabled = _boolean_value(tracking.get("enabled", False), "enabled")
        tracking_uri = _tracking_uri(
            _string_value(
                tracking.get("tracking_uri", "sqlite:///mlflow.db"), "tracking_uri"
            ),
            root=project_root,
        )
        artifact_root = _absolute_path(
            _string_value(
                tracking.get("artifact_root", "mlartifacts"), "artifact_root"
            ),
            root=project_root,
        )
        experiment_name = _string_value(
            tracking.get("experiment_name", "default"), "experiment_name"
        )
        run_name = _string_value(tracking.get("run_name", "run"), "run_name")
        tags = _tags(tracking.get("tags", {}))

        return cls(
            enabled=enabled,
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
        )


class RunLogger(Protocol):
    @property
    def run_id(self) -> str: ...

    def log_params(self, values: Mapping[str, Any]) -> None: ...

    def log_metrics(
        self, values: Mapping[str, float], *, step: int | None = None
    ) -> None: ...

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None: ...

    def log_dict(self, values: Mapping[str, Any], artifact_file: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NullRunLogger:
    @property
    def run_id(self) -> str:
        return ""

    def log_params(self, values: Mapping[str, Any]) -> None:
        del values

    def log_metrics(
        self, values: Mapping[str, float], *, step: int | None = None
    ) -> None:
        del values, step

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        del path, artifact_path

    def log_dict(self, values: Mapping[str, Any], artifact_file: str) -> None:
        del values, artifact_file


@dataclass(frozen=True, slots=True)
class _MlflowRunLogger:
    _mlflow: Any
    _run_id: str

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_params(self, values: Mapping[str, Any]) -> None:
        self._mlflow.log_params(_normalize_params(values))

    def log_metrics(
        self, values: Mapping[str, float], *, step: int | None = None
    ) -> None:
        self._mlflow.log_metrics(dict(values), step=step)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_dict(self, values: Mapping[str, Any], artifact_file: str) -> None:
        self._mlflow.log_dict(dict(values), artifact_file)


@contextmanager
def start_tracked_run(
    settings: TrackingSettings,
    *,
    configuration: ResolvedConfiguration,
    runtime: RuntimeSnapshot,
    mlflow_module: Any | None = None,
) -> Iterator[RunLogger]:
    if not settings.enabled:
        yield NullRunLogger()
        return

    mlflow = mlflow_module if mlflow_module is not None else _import_mlflow()
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.tracking_uri)
    _select_experiment(mlflow, settings)

    run = mlflow.start_run(
        run_name=_bounded_value(settings.run_name, limit=_MAX_TAG_VALUE_LENGTH)
    )
    logger = _MlflowRunLogger(mlflow, _run_id(run))
    try:
        parameters = _scalar_values(configuration.values)
        parameters["configuration_sha256"] = configuration.sha256
        logger.log_params(parameters)
        mlflow.set_tags(_normalize_tags(_run_tags(settings, configuration, runtime)))
        logger.log_dict(_redacted_values(configuration.values), "resolved-config.yaml")
        logger.log_dict(runtime.as_dict(), "runtime.json")
        yield logger
    except BaseException as error:
        _record_failure(logger, mlflow, error, configuration.values)
        raise
    else:
        try:
            mlflow.end_run(status="FINISHED")
        except BaseException as error:
            _record_failure(logger, mlflow, error, configuration.values)
            raise


def _import_mlflow() -> Any:
    try:
        return importlib.import_module("mlflow")
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking needs the optional dependency. Run `uv sync --extra tracking`."
        ) from error


def _select_experiment(mlflow: Any, settings: TrackingSettings) -> None:
    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        try:
            mlflow.create_experiment(
                settings.experiment_name,
                artifact_location=settings.artifact_root.as_uri(),
            )
        except Exception:
            if mlflow.get_experiment_by_name(settings.experiment_name) is None:
                raise
    mlflow.set_experiment(settings.experiment_name)


def _run_id(run: Any) -> str:
    run_id = run.info.run_id
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("MLflow did not return a run ID")
    return run_id


def _record_failure(
    logger: RunLogger,
    mlflow: Any,
    error: BaseException,
    configuration_values: Mapping[str, Any],
) -> None:
    _log_failure(logger, error, _secret_values(configuration_values))
    _end_failed_run(mlflow)


def _log_failure(
    logger: RunLogger, error: BaseException, secret_values: set[str]
) -> None:
    try:
        logger.log_dict(
            {
                "exception_type": type(error).__name__,
                "message": _redacted_message(str(error), secret_values),
            },
            "failure.json",
        )
    except BaseException:
        return


def _end_failed_run(mlflow: Any) -> None:
    try:
        mlflow.end_run(status="FAILED")
    except BaseException:
        return


def _boolean_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Configuration tracking {name} must be a boolean")
    return value


def _string_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration tracking {name} must be a nonempty string")
    return value


def _tags(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("Configuration tracking tags must be a mapping")
    if any(
        not isinstance(name, str) or not isinstance(tag, str)
        for name, tag in value.items()
    ):
        raise ValueError("Configuration tracking tags must map strings to strings")
    return dict(value)


def _absolute_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _tracking_uri(value: str, *, root: Path) -> str:
    prefix = "sqlite:///"
    if not value.startswith(prefix) or value == "sqlite:///:memory:":
        return value
    database = Path(value.removeprefix(prefix))
    if not database.is_absolute():
        database = root / database
    return f"{prefix}{database.resolve().as_posix()}"


def _scalar_values(values: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    scalars: dict[str, str | int | float | bool] = {}
    _collect_scalars(values, prefix="", output=scalars)
    return scalars


def _collect_scalars(
    value: Any,
    *,
    prefix: str,
    output: dict[str, str | int | float | bool],
) -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if _is_sensitive_key(str(name)):
                continue
            _collect_scalars(child, prefix=_join_key(prefix, str(name)), output=output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_scalars(child, prefix=_join_key(prefix, str(index)), output=output)
    elif isinstance(value, (str, int, float, bool)) and not isinstance(value, Path):
        output[prefix] = value


def _join_key(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"


def _redacted_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(name): "[REDACTED]"
            if _is_sensitive_key(str(name))
            else _redacted_values(child)
            for name, child in value.items()
        }
    if isinstance(value, list):
        return [_redacted_values(item) for item in value]
    return value


def _is_sensitive_key(name: str) -> bool:
    return any(part in _SENSITIVE_KEYS for part in _key_tokens(name))


def _key_tokens(name: str) -> tuple[str, ...]:
    tokens = [name.casefold()]
    current: list[str] = []

    for index, character in enumerate(name):
        if not character.isalnum():
            if current:
                tokens.append("".join(current).casefold())
                current = []
            continue

        next_is_lower = index + 1 < len(name) and name[index + 1].islower()
        if (
            character.isupper()
            and current
            and (current[-1].islower() or (current[-1].isupper() and next_is_lower))
        ):
            tokens.append("".join(current).casefold())
            current = []
        current.append(character)

    if current:
        tokens.append("".join(current).casefold())
    return tuple(tokens)


def _secret_values(value: Any) -> set[str]:
    values: set[str] = set()
    _collect_secret_values(value, values)
    return values


def _collect_secret_values(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if _is_sensitive_key(str(name)):
                _collect_strings(child, output)
            else:
                _collect_secret_values(child, output)
    elif isinstance(value, list):
        for item in value:
            _collect_secret_values(item, output)


def _collect_strings(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _collect_strings(child, output)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, output)
    elif isinstance(value, str) and value:
        output.add(value)


def _redacted_message(message: str, secret_values: set[str]) -> str:
    for value in sorted(secret_values, key=len, reverse=True):
        message = message.replace(value, "[REDACTED]")
    return message


def _run_tags(
    settings: TrackingSettings,
    configuration: ResolvedConfiguration,
    runtime: RuntimeSnapshot,
) -> dict[str, str]:
    tags = {
        name: value
        for name, value in settings.tags.items()
        if not _is_sensitive_key(name)
    }
    tags.update(
        {
            "configuration_sha256": configuration.sha256,
            "started_at_utc": runtime.started_at_utc,
            "git_commit": runtime.git_commit,
            "git_dirty": str(runtime.git_dirty).lower(),
            "python_version": runtime.python_version,
            "platform": runtime.platform,
            "cpu": runtime.cpu,
        }
    )
    if runtime.gpu is not None:
        tags["gpu"] = runtime.gpu
    return tags


def _normalize_params(values: Mapping[str, Any]) -> dict[str, str]:
    return _normalize_values(values, value_limit=_MAX_PARAMETER_VALUE_LENGTH)


def _normalize_tags(values: Mapping[str, Any]) -> dict[str, str]:
    return _normalize_values(values, value_limit=_MAX_TAG_VALUE_LENGTH)


def _normalize_values(values: Mapping[str, Any], *, value_limit: int) -> dict[str, str]:
    normalized: dict[str, str] = {}
    used_keys: set[str] = set()
    for key, value in sorted(values.items(), key=_entry_sort_key):
        normalized_key = _normalized_key(str(key), used_keys)
        normalized[normalized_key] = _bounded_value(value, limit=value_limit)
    return normalized


def _entry_sort_key(item: tuple[Any, Any]) -> tuple[str, str, str]:
    key = item[0]
    return (str(key), type(key).__qualname__, repr(key))


def _normalized_key(value: str, used_keys: set[str]) -> str:
    candidate = value if _is_safe_key(value) else _encoded_key(value)
    unique_key = candidate
    index = 1
    while unique_key in used_keys:
        suffix = f"__{_short_hash(f'{value}:{index}')}"
        unique_key = f"{candidate[: _MAX_KEY_LENGTH - len(suffix)]}{suffix}"
        index += 1
    used_keys.add(unique_key)
    return unique_key


def _is_safe_key(value: str) -> bool:
    if not value or len(value) > _MAX_KEY_LENGTH:
        return False
    if any(character not in _KEY_CHARACTERS for character in value):
        return False
    normalized = posixpath.normpath(value)
    return (
        normalized == value
        and normalized != "."
        and not normalized.startswith("..")
        and not normalized.startswith("/")
    )


def _encoded_key(value: str) -> str:
    readable = "".join(
        character if character in _KEY_CHARACTERS and character != "/" else "_"
        for character in value
    ).strip(" .")
    if not readable:
        readable = "key"
    suffix = f"__{_short_hash(value)}"
    return f"{readable[: _MAX_KEY_LENGTH - len(suffix)]}{suffix}"


def _bounded_value(value: Any, *, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    suffix = f"...[sha256:{_short_hash(text)}]"
    return f"{text[: limit - len(suffix)]}{suffix}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_HASH_LENGTH]
