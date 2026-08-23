from dataclasses import replace
from pathlib import Path

import pytest

from deepfake_detection.fusion.store import FeatureRecord, FeatureStore


def record(clip_id: str, branch: str, *, logit: float = 0.0) -> FeatureRecord:
    return FeatureRecord(
        dataset="fixture",
        clip_id=clip_id,
        segment_id="segment-0",
        branch=branch,
        logit=logit,
        embedding=(1.0, 2.0),
        available=True,
        checkpoint_hash=f"{branch}-checkpoint",
        preprocessing_hash="preprocess-1",
        split_hash="split-1",
        run_id="run-1",
        label=0,
    )


def test_feature_store_round_trip_preserves_provenance(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    expected = record("clip-1", "visual", logit=1.25)

    store.write([expected])

    assert store.read() == (expected,)


def test_feature_store_refuses_incomplete_fusion_coverage(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    store.write(
        [
            record("clip-1", "visual"),
            record("clip-1", "audio"),
            record("clip-2", "visual"),
        ]
    )

    with pytest.raises(ValueError, match="clip-2.*audio"):
        store.assemble(required_branches=("visual", "audio"))


def test_feature_store_rejects_duplicate_provenance_keys(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    duplicate = record("clip-1", "visual")
    store.write([duplicate])

    with pytest.raises(ValueError, match="Duplicate feature key"):
        store.write([duplicate])


def test_feature_store_treats_unavailable_rows_as_missing_coverage(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    visual = record("clip-1", "visual")
    unavailable_audio = FeatureRecord(
        dataset="fixture",
        clip_id="clip-1",
        segment_id="segment-0",
        branch="audio",
        logit=0.0,
        embedding=(),
        available=False,
        checkpoint_hash="audio-checkpoint",
        preprocessing_hash="preprocess-1",
        split_hash="split-1",
        run_id="run-1",
        label=0,
        quality_flags=("missing_audio",),
    )
    store.write([visual, unavailable_audio])

    with pytest.raises(ValueError, match="clip-1.*audio"):
        store.assemble(required_branches=("visual", "audio"))

    rows = store.assemble(
        required_branches=("visual", "audio"),
        strict=False,
    )
    assert len(rows) == 1
    assert not rows[0].available
    assert rows[0].missing_branches == ("audio",)
    assert rows[0].branch_logits == {"visual": 0.0}


def test_feature_store_rejects_mixed_preprocessing_provenance(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features.parquet")
    store.write(
        [
            record("clip-1", "visual"),
            replace(
                record("clip-1", "audio"),
                preprocessing_hash="preprocess-2",
            ),
        ]
    )

    with pytest.raises(ValueError, match="clip-1.*preprocessing"):
        store.assemble(required_branches=("visual", "audio"))
