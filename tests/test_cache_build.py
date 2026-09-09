from pathlib import Path

import numpy as np
import pytest

from deepfake_detection.data.cache_build import build_cache
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def record(clip_id: str, path: str) -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": clip_id,
            "dataset": "fixture",
            "video_path": path,
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": clip_id,
        }
    )


class FixturePreprocessor:
    def prepare(self, clip: ClipRecord, path: Path) -> PreparedClip:
        quality = (
            QualityReport(0.5, False, False, False, 0.4, False)
            if clip.clip_id == "blocked"
            else QualityReport(1.0, True, True, False, 0.0)
        )
        return PreparedClip(
            clip_id=clip.clip_id,
            visual_view=np.zeros((1, 3, 2, 2), dtype=np.float32),
            audio_view=np.zeros((4,), dtype=np.float32),
            sync_video_view=np.zeros((1, 3, 2, 2), dtype=np.float32),
            sync_audio_view=np.zeros((2,), dtype=np.float32),
            quality=quality,
            preprocessing_fingerprint=f"hash-{clip.clip_id}",
            preprocessing_config_hash="pipeline-hash",
        )


def test_cache_build_reports_missing_media_without_hiding_it(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "exists.mp4").write_bytes(b"fixture")

    report = build_cache(
        records=(
            record("exists", "exists.mp4"),
            record("missing", "missing.mp4"),
        ),
        dataset_root=dataset_root,
        preprocessor=FixturePreprocessor(),
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert report.succeeded == 1
    assert report.failed == 1
    assert set(report.cache_index) == {"exists"}
    assert "missing" in report.failures
    assert report.full_fusion_ready == 1
    assert report.blocker_counts == {}
    assert report.preprocessing_hash == "pipeline-hash"


def test_cache_build_counts_quality_blockers(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "blocked.mp4").write_bytes(b"fixture")

    report = build_cache(
        records=(record("blocked", "blocked.mp4"),),
        dataset_root=dataset_root,
        preprocessor=FixturePreprocessor(),
        cache_store=CacheStore(tmp_path / "cache"),
    )

    assert report.full_fusion_ready == 0
    assert report.blocker_counts == {
        "av_duration_mismatch": 1,
        "insufficient_sync_duration": 1,
        "low_face_coverage": 1,
        "missing_audio": 1,
        "unstable_face_track": 1,
    }


class CountingPreprocessor(FixturePreprocessor):
    """Records how many clips were actually prepared, so a skip is observable."""

    def __init__(self) -> None:
        self.prepared: list[str] = []

    def prepare(self, clip: ClipRecord, path: Path) -> PreparedClip:
        self.prepared.append(clip.clip_id)
        return super().prepare(clip, path)

    def fingerprint_for(self, clip: ClipRecord, path: Path) -> str:
        return f"hash-{clip.clip_id}"


def dataset_with(tmp_path: Path, names: tuple[str, ...]) -> Path:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(exist_ok=True)
    for name in names:
        (dataset_root / f"{name}.mp4").write_bytes(b"fixture")
    return dataset_root


def test_skip_cached_reuses_existing_entries_without_preparing_them(
    tmp_path: Path,
) -> None:
    names = ("one", "two", "three")
    dataset_root = dataset_with(tmp_path, names)
    records = tuple(record(name, f"{name}.mp4") for name in names)
    store = CacheStore(tmp_path / "cache")
    first = CountingPreprocessor()
    build_cache(
        records=records,
        dataset_root=dataset_root,
        preprocessor=first,
        cache_store=store,
    )
    assert first.prepared == list(names)

    second = CountingPreprocessor()
    report = build_cache(
        records=records,
        dataset_root=dataset_root,
        preprocessor=second,
        cache_store=store,
        skip_cached=True,
    )

    assert second.prepared == []
    assert report.skipped == 3
    # A resumed build still has to report a full index and an intact audit.
    assert set(report.cache_index) == set(names)
    assert report.succeeded == 3
    assert report.full_fusion_ready == 3
    assert report.preprocessing_hash == "pipeline-hash"


def test_skip_cached_still_builds_clips_that_are_absent(tmp_path: Path) -> None:
    names = ("one", "two")
    dataset_root = dataset_with(tmp_path, names)
    store = CacheStore(tmp_path / "cache")
    build_cache(
        records=(record("one", "one.mp4"),),
        dataset_root=dataset_root,
        preprocessor=CountingPreprocessor(),
        cache_store=store,
    )

    preprocessor = CountingPreprocessor()
    report = build_cache(
        records=tuple(record(name, f"{name}.mp4") for name in names),
        dataset_root=dataset_root,
        preprocessor=preprocessor,
        cache_store=store,
        skip_cached=True,
    )

    assert preprocessor.prepared == ["two"]
    assert report.skipped == 1
    assert set(report.cache_index) == {"one", "two"}


def test_shards_partition_the_records_and_reassemble(tmp_path: Path) -> None:
    names = tuple(f"clip{index}" for index in range(7))
    dataset_root = dataset_with(tmp_path, names)
    records = tuple(record(name, f"{name}.mp4") for name in names)

    merged: dict[str, object] = {}
    for index in range(3):
        preprocessor = CountingPreprocessor()
        report = build_cache(
            records=records,
            dataset_root=dataset_root,
            preprocessor=preprocessor,
            cache_store=CacheStore(tmp_path / "cache"),
            shard=(index, 3),
        )
        # Each shard prepares only what it owns, and nothing twice.
        assert set(report.cache_index).isdisjoint(merged)
        merged.update(report.cache_index)

    assert set(merged) == set(names)


def test_shard_rejects_an_index_outside_the_count(tmp_path: Path) -> None:
    dataset_root = dataset_with(tmp_path, ("one",))
    with pytest.raises(ValueError, match="Shard index"):
        build_cache(
            records=(record("one", "one.mp4"),),
            dataset_root=dataset_root,
            preprocessor=CountingPreprocessor(),
            cache_store=CacheStore(tmp_path / "cache"),
            shard=(3, 3),
        )
