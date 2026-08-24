from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ResolvedConfiguration:
    values: dict[str, Any]
    sources: tuple[Path, ...]
    sha256: str

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.values, allow_unicode=False, sort_keys=True),
            encoding="utf-8",
        )


def load_configuration(paths: Sequence[Path]) -> ResolvedConfiguration:
    if not paths:
        raise ValueError("At least one configuration path is required")
    if isinstance(paths, (str, bytes)):
        raise ValueError("Configuration paths must be a sequence of paths")

    sources = tuple(Path(path) for path in paths)
    values: dict[str, Any] = {}
    for source in sources:
        try:
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML in configuration: {source}") from error
        if not isinstance(document, Mapping):
            raise ValueError(f"Configuration must be a mapping: {source}")
        values = _merge_mappings(values, document)

    _validate_configuration(values)
    canonical = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return ResolvedConfiguration(
        values=values,
        sources=sources,
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def configuration_argv(configuration: ResolvedConfiguration) -> tuple[str, ...]:
    command = configuration.values["command"]
    arguments = configuration.values["arguments"]
    argv = list(command)
    for name in sorted(arguments):
        value = arguments[name]
        if value is None or value is False:
            continue
        option = f"--{name}"
        if value is True:
            argv.append(option)
        elif isinstance(value, list):
            if value:
                argv.extend((option, *(str(item) for item in value)))
        else:
            argv.extend((option, str(value)))
    return tuple(argv)


def _merge_mappings(
    base: Mapping[object, Any], override: Mapping[object, Any]
) -> dict[str, Any]:
    merged = {key: _copy_value(value) for key, value in base.items()}
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(current, value)
        else:
            merged[key] = _copy_value(value)
    return merged


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _validate_configuration(values: dict[str, Any]) -> None:
    _validate_json_value(values, "configuration")
    if values.get("schema_version") != 1 or isinstance(
        values.get("schema_version"), bool
    ):
        raise ValueError("Configuration schema_version must be 1")

    command = values.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("Configuration command must be a nonempty list of strings")
    if command[0] == "run":
        raise ValueError("Configuration command cannot start with run")

    arguments = values.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Configuration arguments must be a mapping")
    for name, value in arguments.items():
        if not isinstance(name, str):
            raise ValueError("Configuration argument names must be strings")
        if not _is_argument_value(value):
            raise ValueError(
                f"Configuration argument {name!r} must be a scalar, list of scalars, or null"
            )

    if not isinstance(values.get("tracking"), Mapping):
        raise ValueError("Configuration tracking must be a mapping")


def _validate_json_value(value: Any, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Configuration mapping keys must be strings at {location}"
                )
            _validate_json_value(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Configuration float must be finite at {location}")
    elif not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"Configuration value is not JSON-compatible at {location}")


def _is_argument_value(value: Any) -> bool:
    if isinstance(value, list):
        return all(_is_list_scalar(item) for item in value)
    return _is_scalar(value, allow_none=True)


def _is_list_scalar(value: Any) -> bool:
    return _is_scalar(value, allow_none=False)


def _is_scalar(value: Any, *, allow_none: bool) -> bool:
    if value is None:
        return allow_none
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)
