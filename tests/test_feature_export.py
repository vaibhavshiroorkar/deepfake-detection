from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from deepfake_detection.branches.contracts import BranchOutput
from deepfake_detection.branches.sync import SyncOutput
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.fusion.export import (
    export_features,
    verify_out_of_fold,
)
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


def _fixture_record(clip_id: str, source: str) -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": clip_id,
            "dataset": "fixture",
            "video_path": f"{clip_id}.mp4",
            "manipulation_type": "FakeVideo-RealAudio",
            "method": "faceswap",
            "source": source,
        }
    )


def test_out_of_fold_export_is_refused_when_a_branch_trained_on_the_clips() -> None:
    """The defect this guards: `--partition-role oof` was an unverified string.

    Every fold shares one split hash, so no hash check can see fold
    contamination. Pointing one fold's checkpoint at another fold's manifest
    produced a fusion head fitted on branch scores for clips the branch had
    memorised, with provenance that looked clean.
    """
    records = [_fixture_record("clip-1", "id1"), _fixture_record("clip-2", "id2")]

    with pytest.raises(ValueError, match="contaminated"):
        verify_out_of_fold(
            records=records,
            partition_role="oof",
            trained_source_identities={"visual": ("id2", "id9")},
        )


def test_out_of_fold_export_is_allowed_when_no_source_overlaps() -> None:
    records = [_fixture_record("clip-1", "id1")]

    verify_out_of_fold(
        records=records,
        partition_role="oof",
        trained_source_identities={"visual": ("id2",), "audio": ("id3",)},
    )


def test_only_the_out_of_fold_role_is_checked() -> None:
    """A test or external export is meant to score unseen clips through a
    trained checkpoint, so an overlap there is the point, not a fault."""
    records = [_fixture_record("clip-1", "id1")]

    for role in ("test", "external", "validation", "stress"):
        verify_out_of_fold(
            records=records,
            partition_role=role,
            trained_source_identities={"visual": ("id1",)},
        )


def test_a_checkpoint_without_recorded_sources_skips_rather_than_passes() -> None:
    """Empty means "not recorded", not "trained on nothing".

    Checkpoints written before the field existed report an empty tuple, so the
    check must not read that as a clean bill of health.
    """
    records = [_fixture_record("clip-1", "id1")]

    verify_out_of_fold(
        records=records,
        partition_role="oof",
        trained_source_identities={"visual": ()},
    )
    verify_out_of_fold(
        records=records, partition_role="oof", trained_source_identities=None
    )
