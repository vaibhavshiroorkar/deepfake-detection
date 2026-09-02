from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deepfake_detection.views.contracts import PreparedClip

if TYPE_CHECKING:
    from deepfake_detection.inference.predictor import PredictionResult


@dataclass(frozen=True, slots=True)
class UploadedClip:
    name: str
    suffix: str
    content: bytes
    sha256: str


def uploaded_clip(values: Mapping[str, object]) -> UploadedClip | None:
    value = values.get("dashboard.upload")
    return value if isinstance(value, UploadedClip) else None


def store_upload(
    values: MutableMapping[str, object], *, name: str, content: bytes
) -> UploadedClip:
    suffix = Path(name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi"}:
        raise ValueError("Unsupported video format")
    clip = UploadedClip(
        name=name,
        suffix=suffix,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
    values["dashboard.upload"] = clip
    return clip


@contextmanager
def temporary_video(clip: UploadedClip) -> Iterator[Path]:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=clip.suffix) as handle:
            path = Path(handle.name)
            handle.write(clip.content)
        yield path
    finally:
        if path is not None and path.exists():
            path.unlink()


def store_prepared(
    values: MutableMapping[str, object],
    clip_sha256: str,
    prepared: PreparedClip,
) -> None:
    values["dashboard.prepared"] = (clip_sha256, prepared)


def prepared_for_upload(
    values: Mapping[str, object], clip_sha256: str
) -> PreparedClip | None:
    value = values.get("dashboard.prepared")
    if not isinstance(value, tuple) or len(value) != 2 or value[0] != clip_sha256:
        return None
    return value[1] if isinstance(value[1], PreparedClip) else None


def store_prediction(
    values: MutableMapping[str, object],
    clip_sha256: str,
    result: PredictionResult,
) -> None:
    values["dashboard.prediction"] = (clip_sha256, result)


def prediction_for_upload(
    values: Mapping[str, object], clip_sha256: str
) -> PredictionResult | None:
    value = values.get("dashboard.prediction")
    if not isinstance(value, tuple) or len(value) != 2 or value[0] != clip_sha256:
        return None
    from deepfake_detection.inference.predictor import PredictionResult

    return value[1] if isinstance(value[1], PredictionResult) else None
