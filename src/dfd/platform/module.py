"""The detector-module interface (WS2).

A module declares what it needs through ``applicability(probe)`` and does its work in ``run(ctx)``. The coordinator
(``coordinator.py``) calls both, so a module never decides whether it is shown: skipped, unsupported and not-applicable
modules are reported with their reason. A module may still return ``not_applicable`` from ``run`` when the answer
needs the decoded media (for example, no face found).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from .schema import ModuleResult

APPLICABLE, NOT_APPLICABLE, UNSUPPORTED = "applicable", "not_applicable", "unsupported"


@dataclass(frozen=True)
class Applicability:
    kind: str                       # APPLICABLE, NOT_APPLICABLE (input lacks what the module needs) or UNSUPPORTED (input kind unknown to the module)
    reason: Optional[str] = None

    @staticmethod
    def yes() -> "Applicability":
        return Applicability(APPLICABLE)

    @staticmethod
    def no(reason: str) -> "Applicability":
        return Applicability(NOT_APPLICABLE, reason)

    @staticmethod
    def unsupported(reason: str) -> "Applicability":
        return Applicability(UNSUPPORTED, reason)


@dataclass(frozen=True)
class Probe:
    """What is known about the media before any module runs. ``None`` means not determined, never "absent"."""
    path: str
    filename: str
    media_type: str                 # image, video or audio
    mime_type: str
    size_bytes: int
    sha256: str
    duration_s: Optional[float] = None
    has_video: Optional[bool] = None
    has_audio: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None
    container: Optional[str] = None
    n_faces: Optional[int] = None
    speech_detected: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "path"}


@dataclass
class Context:
    probe: Probe
    cache: dict = field(default_factory=dict)       # shared extraction cache (frames, face tracks, audio), filled by modules or the pipeline
    progress: Optional[Callable[[str], None]] = None


class DetectorModule(Protocol):
    module_id: str
    version: str
    section: str                    # authenticity, cross_modal or provenance
    timeout_s: float

    def applicability(self, probe: Probe) -> Applicability: ...
    def run(self, ctx: Context) -> ModuleResult: ...


def probe_from_media(media: Any, n_faces: Optional[int] = None, speech_detected: Optional[bool] = None) -> Probe:
    """Build a Probe from a ``dfd.infer.adapters.Media`` (duck-typed, so the v1 package is not imported)."""
    mt = media.media_type
    return Probe(path=str(media.path), filename=media.filename, media_type=mt, mime_type=media.mime_type,
                 size_bytes=media.size_bytes, sha256=media.sha256, duration_s=media.duration_s,
                 has_video=(mt in ("image", "video")), has_audio=bool(media.has_audio), width=media.width,
                 height=media.height, n_faces=n_faces, speech_detected=speech_detected)
