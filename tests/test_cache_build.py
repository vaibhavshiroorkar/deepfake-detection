from pathlib import Path

import numpy as np

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
