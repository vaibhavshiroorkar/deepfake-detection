from pathlib import Path

import numpy as np
import torch

from deepfake_detection.branches.sync_objective import MISMATCH_CLASS_INDEX
from deepfake_detection.data.datasets import CachedGlobalSyncDataset, CachedSyncDataset
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport


def real_record(clip_id: str, source: str) -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": clip_id,
            "dataset": "fixture",
            "video_path": f"{clip_id}.mp4",
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": source,
        }
    )


def cached_clip(clip_id: str, audio_value: float) -> PreparedClip:
    return PreparedClip(
        clip_id=clip_id,
        visual_view=None,
        audio_view=None,
        sync_video_view=np.ones((4, 3, 4, 4), dtype=np.float32),
        sync_audio_view=np.full((16,), audio_value, dtype=np.float32),
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint=f"hash-{clip_id}",
        sync_audio_context=np.full((656,), audio_value, dtype=np.float32),
    )


def test_sync_dataset_uses_authentic_offsets_and_cross_clip_mismatches(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path)
    first = store.save(cached_clip("real-1", 1.0), dataset="fixture")
    second = store.save(cached_clip("real-2", 2.0), dataset="fixture")
    records = (real_record("real-1", "id1"), real_record("real-2", "id2"))
    dataset = CachedSyncDataset(
        records=records,
        cache_index={"real-1": first, "real-2": second},
        cache_store=store,
        sample_rate=1_000,
    )

    aligned = dataset[3]
    mismatch = dataset[MISMATCH_CLASS_INDEX]

    assert len(dataset) == 16
    assert aligned.offset_class.item() == 3
    assert torch.equal(aligned.waveform, torch.zeros(16))
    assert mismatch.offset_class.item() == MISMATCH_CLASS_INDEX
    assert torch.equal(mismatch.waveform, torch.zeros(16))


def test_global_label_sync_ablation_maps_fake_clips_to_mismatch(
    tmp_path: Path,
) -> None:
    store = CacheStore(tmp_path)
    real_path = store.save(cached_clip("real-1", 1.0), dataset="fixture")
    fake_path = store.save(cached_clip("fake-1", 2.0), dataset="fixture")
    fake = ClipRecord.from_mapping(
        {
            "clip_id": "fake-1",
            "dataset": "fixture",
            "video_path": "fake-1.mp4",
            "manipulation_type": "FakeVideo-RealAudio",
            "method": "faceswap",
            "source": "id2",
        }
    )
    dataset = CachedGlobalSyncDataset(
        records=(real_record("real-1", "id1"), fake),
        cache_index={"real-1": real_path, "fake-1": fake_path},
        cache_store=store,
    )

    assert dataset[0].offset_class.item() == 3
    assert dataset[1].offset_class.item() == MISMATCH_CLASS_INDEX
