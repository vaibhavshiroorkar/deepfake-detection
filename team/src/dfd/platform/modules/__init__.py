"""Detector modules that wrap the frozen v1 detectors without changing them (WS2)."""
from __future__ import annotations

from typing import Optional

from .base import AdapterProvider, shared_provider
from .face_video import FaceVideoModule
from .image_detector import ImageDetectorModule
from .speech import SpeechModule

__all__ = ["AdapterProvider", "FaceVideoModule", "ImageDetectorModule", "SpeechModule", "build_default_modules", "shared_provider"]


def build_default_modules(provider: Optional[AdapterProvider] = None, timeout_s: float = 120.0) -> list:
    """The three modules, sharing one provider (hence one adapter built on first use and one analysis per file)."""
    p = provider or shared_provider()
    return [FaceVideoModule(p, timeout_s), ImageDetectorModule(p, timeout_s), SpeechModule(p, timeout_s)]
