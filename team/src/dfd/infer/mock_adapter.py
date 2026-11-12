"""MOCK adapter (D-029). It is a development stand-in, never a measurement.

Rules (D-029): this module's name contains `mock`; it always reports ``inference_source = "mock"`` and no model block;
every interface shows a MOCK banner for it (tested); its scores are deterministic functions of the file hash, so tests
are repeatable, and they carry no meaning whatsoever. Nothing in the evaluation harness or in the paper builds may
consume a result whose source is mock.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from dfd.infer.adapters import AdapterResult, Media, ModalityScore

INFERENCE_SOURCE = "mock"


def _unit(sha256: str, salt: str) -> float:
    """A number in [0, 1) derived from the hash and a salt; deterministic and meaningless."""
    import hashlib
    return int(hashlib.sha256((sha256 + salt).encode()).hexdigest()[:8], 16) / 0x100000000


class MockAdapter:
    inference_source = INFERENCE_SOURCE

    def analyse(self, media: Media, progress: Optional[Callable[[str], None]] = None) -> AdapterResult:
        t0 = time.perf_counter()
        tick = progress or (lambda stage: None)
        per: dict[str, ModalityScore] = {}
        warnings: list[dict] = []
        tick("detect")
        if media.media_type == "image":
            per["image"] = ModalityScore(True, round(_unit(media.sha256, "image"), 4), 1)
        elif media.media_type == "audio":
            n = max(1, int((media.duration_s or 0) // 4) + 1)
            per["audio"] = ModalityScore(True, round(_unit(media.sha256, "audio"), 4), n)
        else:
            per["visual"] = ModalityScore(True, round(_unit(media.sha256, "visual"), 4), 16)
            if media.has_audio:
                per["audio"] = ModalityScore(True, round(_unit(media.sha256, "audio"), 4), max(1, int((media.duration_s or 0) // 4) + 1))
            else:
                per["audio"] = ModalityScore(False, None, None, "video has no audio track")
        # a simulated warning on some files so that the interface can be tested; it says nothing about the content
        if media.media_type in ("image", "video") and _unit(media.sha256, "faces") < 0.25:
            warnings.append({"code": "MULTIPLE_FACES", "message": "MOCK: simulated warning, the largest face would be used.",
                             "modality": "visual" if media.media_type == "video" else "image"})
        tick("score")
        tick("calibrate")
        dt = (time.perf_counter() - t0) * 1000
        return AdapterResult(per, warnings, None, 0.5, {"detect": dt / 3, "score": dt / 3, "calibrate": dt / 3})
