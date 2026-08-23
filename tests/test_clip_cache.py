from pathlib import Path

import numpy as np

from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def fixture_prepared() -> PreparedClip:
    return PreparedClip(
        clip_id="clip-1",
        visual_view=np.ones((2, 3, 4, 4), dtype=np.float32),
        audio_view=np.ones((8,), dtype=np.float32),
        sync_video_view=np.ones((3, 3, 4, 4), dtype=np.float32),
        sync_audio_view=np.ones((6,), dtype=np.float32),
        quality=QualityReport(1.0, True, True, False, 0.01),
        preprocessing_fingerprint="abc123",
        preprocessing_config_hash="pipeline123",
        sync_audio_context=np.ones((10,), dtype=np.float32),
    )


def test_cache_round_trip_preserves_views_quality_and_fingerprint(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path)
    prepared = fixture_prepared()

    path = store.save(prepared, dataset="fixture")
    restored = store.load(path)

    assert np.array_equal(restored.visual_view, prepared.visual_view)
    assert np.array_equal(restored.audio_view, prepared.audio_view)
    assert np.array_equal(restored.sync_audio_context, prepared.sync_audio_context)
    assert restored.quality == prepared.quality
    assert restored.preprocessing_fingerprint == "abc123"
    assert restored.preprocessing_config_hash == "pipeline123"


def test_cache_namespace_prevents_cross_dataset_clip_collisions(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    prepared = fixture_prepared()

    first = store.save(prepared, dataset="dataset-a")
    second = store.save(prepared, dataset="dataset-b")

    assert first != second
