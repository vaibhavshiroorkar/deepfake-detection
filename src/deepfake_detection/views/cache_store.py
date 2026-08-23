from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import PreparedClip, QualityReport


def _safe_component(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"
    digest = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{readable[:80]}-{digest}"


class CacheStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, prepared: PreparedClip, *, dataset: str) -> Path:
        return (
            self.root
            / _safe_component(dataset)
            / _safe_component(prepared.clip_id)
            / f"{prepared.preprocessing_fingerprint}.npz"
        )

    def save(self, prepared: PreparedClip, *, dataset: str) -> Path:
        path = self.path_for(prepared, dataset=dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "clip_id": prepared.clip_id,
            "preprocessing_fingerprint": prepared.preprocessing_fingerprint,
            "preprocessing_config_hash": prepared.preprocessing_config_hash,
            "quality": asdict(prepared.quality),
        }
        arrays = {
            name: value
            for name, value in {
                "visual_view": prepared.visual_view,
                "audio_view": prepared.audio_view,
                "sync_video_view": prepared.sync_video_view,
                "sync_audio_view": prepared.sync_audio_view,
                "sync_audio_context": prepared.sync_audio_context,
            }.items()
            if value is not None
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".npz",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    **arrays,
                )
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

    def load(self, path: Path) -> PreparedClip:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            quality = QualityReport(**metadata["quality"])
            return PreparedClip(
                clip_id=metadata["clip_id"],
                visual_view=archive["visual_view"].copy()
                if "visual_view" in archive
                else None,
                audio_view=archive["audio_view"].copy()
                if "audio_view" in archive
                else None,
                sync_video_view=archive["sync_video_view"].copy()
                if "sync_video_view" in archive
                else None,
                sync_audio_view=archive["sync_audio_view"].copy()
                if "sync_audio_view" in archive
                else None,
                quality=quality,
                preprocessing_fingerprint=metadata["preprocessing_fingerprint"],
                sync_audio_context=archive["sync_audio_context"].copy()
                if "sync_audio_context" in archive
                else None,
                preprocessing_config_hash=metadata.get("preprocessing_config_hash", ""),
            )
