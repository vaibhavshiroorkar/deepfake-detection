from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from deepfake_detection.dashboard.configuration import dashboard_defaults

_REQUIRED_METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "f1")
_REQUIRED_CONFUSION = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    dataset: str
    rows: int
    threshold: float
    metrics: Mapping[str, float]
    confusion: Mapping[str, int]
    epochs: tuple[int, ...]
    best_epoch: int
    checkpoint_run_id: str
    evaluation_run_id: str
    checkpoint_sha256: str
    preprocessing_hash: str
    split_hash: str


def _read_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain an object")
    return value


def _required_string(record: Mapping[str, object], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing {field}")
    return value


def _required_int(record: Mapping[str, object], field: str, label: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} has an invalid {field}")
    return value


def _required_float(record: Mapping[str, object], field: str, label: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} has an invalid {field}")
    return float(value)


def _required_object(
    record: Mapping[str, object], field: str, label: str
) -> Mapping[str, object]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing {field}")
    return value


def load_validation_evidence(
    metrics_path: Path, history_path: Path
) -> ValidationEvidence:
    """Read the frozen visual development-validation record from local files."""
    metrics_record = _read_object(metrics_path, "metrics")
    history_record = _read_object(history_path, "history")
    defaults = dashboard_defaults(root=Path.cwd())

    if _required_string(metrics_record, "dataset", "metrics") != "FakeAVCeleb":
        raise ValueError("metrics dataset must be FakeAVCeleb")
    if (
        _required_string(metrics_record, "evidence_scope", "metrics")
        != "development_validation"
    ):
        raise ValueError("metrics evidence_scope must be development_validation")

    rows = _required_int(metrics_record, "rows", "metrics")
    if rows != 400:
        raise ValueError("metrics rows must be 400")
    threshold = _required_float(metrics_record, "fixed_threshold", "metrics")
    if threshold != 0.5:
        raise ValueError("metrics fixed_threshold must be 0.5")

    checkpoint_run_id = _required_string(metrics_record, "checkpoint_run_id", "metrics")
    if checkpoint_run_id != defaults.run_id:
        raise ValueError("metrics training run does not match dashboard defaults")
    checkpoint_sha256 = _required_string(metrics_record, "checkpoint_sha256", "metrics")
    if checkpoint_sha256 != defaults.checkpoint_sha256:
        raise ValueError("metrics checkpoint hash does not match dashboard defaults")
    preprocessing_hash = _required_string(
        metrics_record, "preprocessing_hash", "metrics"
    )
    if preprocessing_hash != defaults.preprocessing_hash:
        raise ValueError("metrics preprocessing hash does not match dashboard defaults")
    split_hash = _required_string(metrics_record, "split_hash", "metrics")
    if split_hash != defaults.split_hash:
        raise ValueError("metrics split hash does not match dashboard defaults")

    metrics = _required_object(metrics_record, "metrics", "metrics")
    parsed_metrics = {
        name: _required_float(metrics, name, "metrics") for name in _REQUIRED_METRICS
    }
    for name, value in parsed_metrics.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"metrics has an invalid {name}")
    confusion = _required_object(metrics_record, "confusion", "metrics")
    parsed_confusion = {
        name: _required_int(confusion, name, "metrics") for name in _REQUIRED_CONFUSION
    }
    if any(value < 0 for value in parsed_confusion.values()):
        raise ValueError("metrics confusion counts must be nonnegative")
    if sum(parsed_confusion.values()) != rows:
        raise ValueError("metrics confusion counts must total rows")

    evaluation_run_id = _required_string(metrics_record, "evaluation_run_id", "metrics")
    if evaluation_run_id != defaults.evaluation_run_id:
        raise ValueError("metrics evaluation run does not match dashboard defaults")

    metadata = _required_object(history_record, "metadata", "history")
    history_run_id = _required_string(metadata, "run_id", "history")
    if history_run_id != checkpoint_run_id:
        raise ValueError("history training run does not match metrics")
    for field, expected in (
        ("preprocessing_hash", preprocessing_hash),
        ("split_hash", split_hash),
    ):
        if _required_string(metadata, field, "history") != expected:
            raise ValueError(f"history {field} does not match metrics")
    history_checkpoint = history_record.get("checkpoint_hash")
    if history_checkpoint is not None and history_checkpoint != checkpoint_sha256:
        raise ValueError("history checkpoint hash does not match metrics")

    best_epoch = _required_int(history_record, "best_epoch", "history")
    epochs_record = history_record.get("epochs")
    if not isinstance(epochs_record, list):
        raise ValueError("history is missing epochs")
    epochs = tuple(
        _required_int(epoch, "epoch", "history epoch")
        for epoch in epochs_record
        if isinstance(epoch, dict)
    )
    if len(epochs) != len(epochs_record) or epochs != (1, 2, 3, 4, 5):
        raise ValueError("history must contain epochs 1 through 5")
    if best_epoch != 4:
        raise ValueError("history best_epoch must be 4")

    return ValidationEvidence(
        dataset="FakeAVCeleb",
        rows=rows,
        threshold=threshold,
        metrics=MappingProxyType(parsed_metrics),
        confusion=MappingProxyType(parsed_confusion),
        epochs=epochs,
        best_epoch=best_epoch,
        checkpoint_run_id=checkpoint_run_id,
        evaluation_run_id=evaluation_run_id,
        checkpoint_sha256=checkpoint_sha256,
        preprocessing_hash=preprocessing_hash,
        split_hash=split_hash,
    )
