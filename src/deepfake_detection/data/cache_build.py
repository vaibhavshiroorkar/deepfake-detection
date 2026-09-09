from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import QualityReport

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
    skipped: int = 0


def select_shard(
    records: Sequence[ClipRecord], shard: tuple[int, int] | None
) -> Sequence[ClipRecord]:
    """The slice of records this worker owns, as (index, count).

    Round-robin rather than contiguous blocks, so every shard sees the same mix
    of manipulation types and none of them ends up with the one family whose
    clips are slowest to detect faces in.
    """
    if shard is None:
        return records
    index, count = shard
    if count <= 0:
        raise ValueError("Shard count must be positive")
    if not 0 <= index < count:
        raise ValueError("Shard index must be in [0, count)")
    return [record for position, record in enumerate(records) if position % count == index]


def build_cache(
    *,
    records: Sequence[ClipRecord],
    dataset_root: Path,
    preprocessor: Any,
    cache_store: CacheStore,
    skip_cached: bool = False,
    shard: tuple[int, int] | None = None,
) -> CacheBuildReport:
    root = dataset_root.resolve()
    index: dict[str, Path] = {}
    failures: dict[str, str] = {}
    blocker_counts: Counter[str] = Counter()
    full_fusion_ready = 0
    skipped = 0
    preprocessing_hashes: set[str] = set()
    for record in select_shard(records, shard):
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
            cached = (
                _cached_path(record, media_path, preprocessor, cache_store)
                if skip_cached
                else None
            )
            if cached is not None:
                # A skipped clip still has to contribute its index row and its
                # quality counters, or a resumed build would report a short index
                # and an audit that disagrees with the cache on disk.
                metadata = cache_store.load_metadata(cached)
                quality = QualityReport(**metadata["quality"])
                blockers = quality.full_fusion_blockers()
                config_hash = metadata.get("preprocessing_config_hash", "")
                if config_hash:
                    preprocessing_hashes.add(config_hash)
                blocker_counts.update(blockers)
                if not blockers:
                    full_fusion_ready += 1
                index[record.clip_id] = cached
                skipped += 1
                continue
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
        skipped=skipped,
    )


def _cached_path(
    record: ClipRecord,
    media_path: Path,
    preprocessor: Any,
    cache_store: CacheStore,
) -> Path | None:
    """An existing cache entry for this clip, or None if it has to be built.

    The fingerprint covers the media bytes as well as the view config, so a clip
    whose source file changed does not match its old entry and is rebuilt.
    """
    fingerprint = preprocessor.fingerprint_for(record, media_path)
    path = cache_store.path_for_parts(
        clip_id=record.clip_id,
        fingerprint=fingerprint,
        dataset=record.dataset,
    )
    return path if path.is_file() else None
