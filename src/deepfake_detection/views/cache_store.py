from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Collection
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

    def path_for_parts(self, *, clip_id: str, fingerprint: str, dataset: str) -> Path:
        """Where a clip would land, without having to prepare it first.

        A resumable cache build needs the path before it does the work, and the
        fingerprint is derivable from the media bytes and the view config alone
        (see views/cache.py). Preparing a clip just to learn where it would be
        written would defeat the point of skipping it.
        """
        return (
            self.root
            / _safe_component(dataset)
            / _safe_component(clip_id)
            / f"{fingerprint}.npz"
        )

    def path_for(self, prepared: PreparedClip, *, dataset: str) -> Path:
        return self.path_for_parts(
            clip_id=prepared.clip_id,
            fingerprint=prepared.preprocessing_fingerprint,
            dataset=dataset,
        )

    def available_views(self, path: Path) -> frozenset[str]:
        """Which views a cache entry holds, without decompressing any of them.

        A clip whose primary face track was unstable has no `visual_view` at
        all: the pipeline abstains rather than substituting a full-frame crop.
        Callers need to know that before they build a loader, because the
        alternative is discovering it as an exception halfway through an epoch.
        """
        with np.load(path, allow_pickle=False) as archive:
            return frozenset(name for name in archive.files if name != "metadata")

    def load_metadata(self, path: Path) -> dict:
        """The metadata header only, leaving the view arrays on disk.

        Resuming a build re-reads the quality report of every already-cached clip
        so the audit stays exact. Loading the arrays too would make a resume as
        slow as the work it is avoiding.
        """
        with np.load(path, allow_pickle=False) as archive:
            return json.loads(str(archive["metadata"].item()))

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

    def load(self, path: Path, views: Collection[str] | None = None) -> PreparedClip:
        """Read one cached clip. `views` limits which arrays are materialised.

        A clip holds four views and they are not small: the visual view is 9.19
        MiB and the mouth view another 7.18 MiB, so decompressing all of them
        costs about 20 MiB per clip when a visual branch reads exactly one.

        That is not only waste. Visual stream training died on it, twice with a
        DataLoader worker and once single-process after two and a half hours,
        each time inside the `sync_video_view` allocation of a run that never
        touches the mouth crops.

        A view that is not requested comes back None, which is the same thing a
        view that was never cached does, so a caller that asks for less has to
        mean it. Default stays every view, so existing callers are unchanged.
        """
        wanted = None if views is None else set(views)

        def read(archive, name: str):
            if wanted is not None and name not in wanted:
                return None
            return archive[name].copy() if name in archive else None

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            quality = QualityReport(**metadata["quality"])
            return PreparedClip(
                clip_id=metadata["clip_id"],
                visual_view=read(archive, "visual_view"),
                audio_view=read(archive, "audio_view"),
                sync_video_view=read(archive, "sync_video_view"),
                sync_audio_view=read(archive, "sync_audio_view"),
                quality=quality,
                preprocessing_fingerprint=metadata["preprocessing_fingerprint"],
                sync_audio_context=read(archive, "sync_audio_context"),
                preprocessing_config_hash=metadata.get("preprocessing_config_hash", ""),
            )
