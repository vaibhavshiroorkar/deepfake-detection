from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepfake_detection.views.cache_store import CacheStore

from .manifest import ClipRecord


@dataclass(frozen=True, slots=True)
class CacheBuildReport:
    succeeded: int
    failed: int
    full_fusion_ready: int
    blocker_counts: dict[str, int]
    preprocessing_hash: str | None
    cache_index: dict[str, Path]
    failures: dict[str, str]


def build_cache(
    *,
    records: Sequence[ClipRecord],
    dataset_root: Path,
    preprocessor: Any,
    cache_store: CacheStore,
) -> CacheBuildReport:
    root = dataset_root.resolve()
    index: dict[str, Path] = {}
    failures: dict[str, str] = {}
    blocker_counts: Counter[str] = Counter()
    full_fusion_ready = 0
    preprocessing_hashes: set[str] = set()
    for record in records:
        try:
            media_path = (
                record.video_path.resolve()
                if record.video_path.is_absolute()
                else (root / record.video_path).resolve()
            )
            if not media_path.is_relative_to(root):
                raise ValueError("Media path escapes the dataset root")
            if not media_path.is_file():
                raise FileNotFoundError(f"Media file does not exist: {media_path}")
            prepared = preprocessor.prepare(record, media_path)
            blockers = prepared.quality.full_fusion_blockers()
            if prepared.preprocessing_config_hash:
                preprocessing_hashes.add(prepared.preprocessing_config_hash)
            blocker_counts.update(blockers)
            if not blockers:
                full_fusion_ready += 1
            index[record.clip_id] = cache_store.save(prepared, dataset=record.dataset)
        except (OSError, RuntimeError, ValueError) as error:
            failures[record.clip_id] = str(error)
    if len(preprocessing_hashes) > 1:
        raise ValueError("Cache build produced mixed preprocessing hashes")
    return CacheBuildReport(
        succeeded=len(index),
        failed=len(failures),
        full_fusion_ready=full_fusion_ready,
        blocker_counts=dict(sorted(blocker_counts.items())),
        preprocessing_hash=(
            next(iter(preprocessing_hashes)) if preprocessing_hashes else None
        ),
        cache_index=index,
        failures=failures,
    )
