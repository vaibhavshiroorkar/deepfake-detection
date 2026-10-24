"""The adapter interface between the inference library and a scorer (real or mock).

An adapter receives a ``Media`` (the validated input) and returns an ``AdapterResult``. It reports per-modality
scores before fusion, its own warnings (face detection lives here), the model block of D-027 (None for a mock) and
the time of its own stages. The library, not the adapter, builds the response, fuses the modalities and checks input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol


@dataclass(frozen=True)
class Media:
    path: Path
    filename: str
    media_type: str            # image, video or audio
    mime_type: str
    size_bytes: int
    sha256: str
    duration_s: Optional[float]
    has_audio: bool            # an audio stream exists (always True for audio files)
    width: Optional[int]
    height: Optional[int]


@dataclass
class ModalityScore:
    available: bool
    score: Optional[float] = None          # calibrated probability of fake, None when unavailable
    n_units: Optional[int] = None          # frames or audio windows actually used
    reason: Optional[str] = None           # why the modality is unavailable


@dataclass
class AdapterResult:
    per_modality: dict[str, ModalityScore]
    warnings: list[dict] = field(default_factory=list)
    model: Optional[dict] = None           # D-027 model block; None for a mock
    threshold: float = 0.5
    stage_ms: dict[str, float] = field(default_factory=dict)   # detect, score, calibrate


class Adapter(Protocol):
    inference_source: str                  # "real" or "mock"

    def analyse(self, media: Media, progress: Optional[Callable[[str], None]] = None) -> AdapterResult: ...
