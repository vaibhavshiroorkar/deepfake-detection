from pathlib import Path

import numpy as np
import torch
from torch import nn

from deepfake_detection.branches.contracts import BranchOutput
from deepfake_detection.branches.sync import SyncOutput
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.fusion.export import export_features
from deepfake_detection.fusion.store import FeatureStore
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport


class FixtureBranch(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, values: torch.Tensor) -> BranchOutput:
        batch = values.shape[0]
        return BranchOutput(
            logits=torch.full((batch,), self.logit),
            embedding=torch.full((batch, 2), self.logit),
            token_count=1,
        )


class FixtureSync(nn.Module):
    def forward(
        self, *, mouth_video: torch.Tensor, waveform: torch.Tensor
    ) -> SyncOutput:
        logits = torch.zeros(mouth_video.shape[0], 8)
        logits[:, 7] = 2.0
        tokens = torch.zeros(mouth_video.shape[0], 2, 2)
        return SyncOutput(tokens, tokens, logits, torch.zeros(mouth_video.shape[0], 2))


def test_feature_export_is_a_real_producer_with_clip_level_fusion_labels(
    tmp_path: Path,
) -> None:
    record = ClipRecord.from_mapping(
        {
            "clip_id": "clip-1",
            "dataset": "fixture",
            "video_path": "clip.mp4",
            "manipulation_type": "FakeVideo-RealAudio",
            "method": "faceswap",
            "source": "id1",
        }
    )
    prepared = PreparedClip(
        clip_id="clip-1",
        visual_view=np.zeros((2, 3, 4, 4), dtype=np.float32),
        audio_view=np.zeros((16,), dtype=np.float32),
        sync_video_view=np.zeros((4, 3, 4, 4), dtype=np.float32),
        sync_audio_view=np.zeros((16,), dtype=np.float32),
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint="prep-1",
        preprocessing_config_hash="pipeline-1",
    )
    cache_store = CacheStore(tmp_path / "cache")
    cache_path = cache_store.save(prepared, dataset="fixture")
    feature_store = FeatureStore(tmp_path / "features.parquet")

    report = export_features(
        records=(record,),
        cache_index={"clip-1": cache_path},
        cache_store=cache_store,
        feature_store=feature_store,
        visual_model=FixtureBranch(1.0),
        audio_model=FixtureBranch(-1.0),
        sync_model=FixtureSync(),
        checkpoint_hashes={"visual": "v", "audio": "a", "sync": "s"},
        split_hash="split",
        preprocessing_hash="pipeline-1",
        partition_role="oof",
        run_id="run",
        device="cpu",
    )

    rows = feature_store.read()
    assert report.exported_rows == 3
    assert {row.branch for row in rows} == {"visual", "audio", "sync"}
    assert {row.label for row in rows} == {1}
    assert {row.preprocessing_hash for row in rows} == {"pipeline-1"}
    assert {row.cache_fingerprint for row in rows} == {"prep-1"}
    assert {row.source_identity for row in rows} == {"id1"}
    assert {row.method for row in rows} == {"faceswap"}


def test_feature_export_keeps_missing_cache_entries_as_unavailable_rows(
    tmp_path: Path,
) -> None:
    record = ClipRecord.from_mapping(
        {
            "clip_id": "missing-1",
            "dataset": "fixture",
            "video_path": "missing.mp4",
            "manipulation_type": "RealVideo-RealAudio",
            "method": "real",
            "source": "id-missing",
        }
    )
    feature_store = FeatureStore(tmp_path / "features.parquet")

    report = export_features(
        records=(record,),
        cache_index={},
        cache_store=CacheStore(tmp_path / "cache"),
        feature_store=feature_store,
        visual_model=FixtureBranch(1.0),
        audio_model=FixtureBranch(-1.0),
        sync_model=FixtureSync(),
        checkpoint_hashes={"visual": "v", "audio": "a", "sync": "s"},
        split_hash="split",
        preprocessing_hash="pipeline-1",
        partition_role="oof",
        run_id="run",
        device="cpu",
    )

    rows = feature_store.read()
    assert report.unavailable_rows == 3
    assert report.failures == {"missing-1": "missing_cache_entry"}
    assert len(rows) == 3
    assert not any(row.available for row in rows)
    assert {row.quality_flags for row in rows} == {("missing_cache",)}
