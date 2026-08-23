from pathlib import Path

import numpy as np
import pytest
import torch

from deepfake_detection.data.datasets import CachedBranchDataset
from deepfake_detection.data.manifest import ClipRecord
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
        preprocessing_config_hash="pipeline-1",
    )


def fake_video_record() -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": "clip-1",
            "dataset": "fixture",
            "video_path": "clip.mp4",
            "manipulation_type": "FakeVideo-RealAudio",
            "method": "faceswap",
            "source": "id1",
        }
    )


def test_branch_datasets_use_cue_specific_labels(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    cache_path = store.save(fixture_prepared(), dataset="fixture")
    index = {"clip-1": cache_path}

    visual = CachedBranchDataset(
        records=(fake_video_record(),),
        cache_index=index,
        cache_store=store,
        branch="visual",
        preprocessing_hash="pipeline-1",
    )
    audio = CachedBranchDataset(
        records=(fake_video_record(),),
        cache_index=index,
        cache_store=store,
        branch="audio",
        preprocessing_hash="pipeline-1",
    )

    assert torch.equal(visual[0].values, torch.ones(2, 3, 4, 4))
    assert visual[0].label.item() == 1.0
    assert audio[0].label.item() == 0.0


def test_branch_dataset_fails_when_a_required_cache_entry_is_missing() -> None:
    with pytest.raises(ValueError, match="Missing cache entries: clip-1"):
        CachedBranchDataset(
            records=(fake_video_record(),),
            cache_index={},
            cache_store=CacheStore(Path("unused")),
            branch="visual",
        )


def test_branch_dataset_rejects_wrong_preprocessing_hash(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    cache_path = store.save(fixture_prepared(), dataset="fixture")
    dataset = CachedBranchDataset(
        records=(fake_video_record(),),
        cache_index={"clip-1": cache_path},
        cache_store=store,
        branch="visual",
        preprocessing_hash="pipeline-2",
    )

    with pytest.raises(ValueError, match="preprocessing hash"):
        dataset[0]
