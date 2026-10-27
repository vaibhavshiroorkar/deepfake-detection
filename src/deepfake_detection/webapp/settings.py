"""What an administrator can change about the user app, kept in one YAML file.

The file is the only state the admin page writes. The user page reads it on
every request, so a change shows up without a restart.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from deepfake_detection.paths import PROJECT_ROOT

from .engines import MODEL_IDS

SETTINGS_PATH = PROJECT_ROOT / "configs" / "webapp.yaml"
_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class Settings:
    default_model: str = "ensemble"
    enabled_models: tuple[str, ...] = field(default_factory=lambda: MODEL_IDS)
    default_threshold: float = 0.5
    # Off lets a user move the threshold; on pins every verdict to the default.
    lock_threshold: bool = False
    withhold_unfamiliar: bool = True
    # Whether a user may switch the familiarity check off for one analysis.
    allow_unfamiliar_override: bool = True
    show_research_tabs: bool = True
    max_upload_mb: int = 200

    def validated(self) -> Settings:
        enabled = tuple(m for m in MODEL_IDS if m in set(self.enabled_models))
        if not enabled:
            raise ValueError("At least one model must stay enabled")
        if self.default_model not in enabled:
            raise ValueError("The default model must be one of the enabled models")
        if not 0 < self.default_threshold < 1:
            raise ValueError("The default threshold must be between 0 and 1")
        if not 1 <= self.max_upload_mb <= 2048:
            raise ValueError("The upload limit must be between 1 and 2048 MB")
        return Settings(
            default_model=self.default_model,
            enabled_models=enabled,
            default_threshold=float(self.default_threshold),
            lock_threshold=bool(self.lock_threshold),
            withhold_unfamiliar=bool(self.withhold_unfamiliar),
            allow_unfamiliar_override=bool(self.allow_unfamiliar_override),
            show_research_tabs=bool(self.show_research_tabs),
            max_upload_mb=int(self.max_upload_mb),
        )

    def as_dict(self) -> dict:
        values = asdict(self)
        values["enabled_models"] = list(self.enabled_models)
        return values


def from_mapping(values: dict) -> Settings:
    known = {item.name for item in fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    if "enabled_models" in values:
        values = {**values, "enabled_models": tuple(values["enabled_models"])}
    return Settings(**values).validated()


def load(path: Path = SETTINGS_PATH) -> Settings:
    """The saved settings, or the defaults when no file has been written."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return Settings()
    return from_mapping(raw)


def save(settings: Settings, path: Path = SETTINGS_PATH) -> Settings:
    checked = settings.validated()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(checked.as_dict(), sort_keys=False), encoding="utf-8"
        )
        temporary.replace(path)
    return checked
