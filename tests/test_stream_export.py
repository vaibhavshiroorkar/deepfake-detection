"""Exporting stream embeddings into the feature store.

The test that earns its place is the one pinning export to the same views and
the same audio normalisation the dataset uses. A stream trained on the mouth
crops and scored on the face crops raises nothing: the shapes are compatible and
the only symptom is a disappointing number. A first draft of `stream_export.py`
did exactly that with the waveform, normalising by peak where the dataset
centres and divides by standard deviation.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from torch import nn

from deepfake_detection.data.datasets import (
    STREAM_VIEWS,
    CachedAVPairDataset,
    CachedBranchDataset,
)
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.fusion.store import FeatureStore
from deepfake_detection.fusion.stream_export import (
    StreamSpec,
    export_stream_features,
)
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport

FRAMES = 4
SIZE = 8
SYNC_FRAMES = 6
AUDIO = 800
HASH = "pipeline-hash"


class RecordingVisualStream(nn.Module):
    """Returns the tuple a VisualStream returns, and keeps what it was given."""

    def __init__(self, width: int = 5) -> None:
        super().__init__()
        self.width = width
        self.seen: list[torch.Tensor] = []

    def forward(self, frames: torch.Tensor):
        self.seen.append(frames.clone())
        batch = frames.shape[0]
        return torch.zeros(batch), torch.arange(
            self.width, dtype=torch.float32
        ).repeat(batch, 1)


class RecordingCrossModalStream(nn.Module):
    """Returns a StreamOutput-shaped result, and keeps both inputs."""

    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.width = width
        self.video: list[torch.Tensor] = []
        self.audio: list[torch.Tensor] = []

    def forward(self, *, video: torch.Tensor, audio: torch.Tensor):
        self.video.append(video.clone())
        self.audio.append(audio.clone())
        batch = video.shape[0]

        class Output:
            logit = torch.full((batch,), 0.25)
            embedding = torch.ones(batch, self.width)

        return Output()


def record(clip_id: str, *, fake: bool) -> ClipRecord:
    return ClipRecord.from_mapping(
        {
            "clip_id": clip_id,
            "dataset": "fixture",
            "video_path": f"{clip_id}.mp4",
            "manipulation_type": (
                "FakeVideo-RealAudio" if fake else "RealVideo-RealAudio"
            ),
            "method": "wav2lip" if fake else "real",
            "source": f"identity-{clip_id}",
        }
    )


def store_clip(store: CacheStore, clip_id: str, *, full: bool = True) -> Path:
    generator = np.random.default_rng(abs(hash(clip_id)) % 2**32)
    prepared = PreparedClip(
        clip_id=clip_id,
        visual_view=generator.normal(size=(FRAMES, 3, SIZE, SIZE)).astype(np.float32),
        audio_view=generator.normal(size=(AUDIO,)).astype(np.float32),
        sync_video_view=(
            generator.normal(size=(SYNC_FRAMES, 3, SIZE, SIZE)).astype(np.float32)
            if full
            else None
        ),
        sync_audio_view=(
            generator.normal(size=(AUDIO,)).astype(np.float32) if full else None
        ),
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint=f"fingerprint-{clip_id}",
        preprocessing_config_hash=HASH,
    )
    return store.save(prepared, dataset="fixture")


@pytest.fixture
def cached(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    records = []
    index = {}
    for number in range(4):
        clip_id = f"clip-{number}"
        index[clip_id] = store_clip(store, clip_id)
        records.append(record(clip_id, fake=number % 2 == 0))
    return records, index, store


def run_export(tmp_path, records, index, store, streams):
    feature_store = FeatureStore(tmp_path / "features.parquet")
    report = export_stream_features(
        records=records,
        cache_index=index,
        cache_store=store,
        feature_store=feature_store,
        streams=streams,
        split_hash="split",
        preprocessing_hash=HASH,
        partition_role="oof",
        run_id="run",
        device="cpu",
    )
    return report, feature_store


def test_every_stream_gets_a_row_per_clip(tmp_path: Path, cached) -> None:
    records, index, store = cached
    streams = [
        StreamSpec("dinov3", RecordingVisualStream(), "hash-v", "visual"),
        StreamSpec("lipsync", RecordingCrossModalStream(), "hash-l", "lipsync"),
    ]

    report, feature_store = run_export(tmp_path, records, index, store, streams)

    assert report.clips == 4
    assert report.exported_rows == 8
    assert report.unavailable_rows == 0
    assert {row.branch for row in feature_store.read()} == {"dinov3", "lipsync"}


def test_embeddings_reach_the_store_not_just_the_logit(tmp_path: Path, cached) -> None:
    """Late fusion kept only the scalar. Feature-level fusion is the whole point
    of this exporter, so the embedding has to survive the round trip."""
    records, index, store = cached
    streams = [StreamSpec("dinov3", RecordingVisualStream(width=5), "h", "visual")]

    _report, feature_store = run_export(tmp_path, records, index, store, streams)

    rows = list(feature_store.read())
    assert all(len(row.embedding) == 5 for row in rows)
    assert rows[0].embedding == (0.0, 1.0, 2.0, 3.0, 4.0)


def test_a_visual_stream_is_given_exactly_what_training_gives_it(
    tmp_path: Path, cached
) -> None:
    records, index, store = cached
    model = RecordingVisualStream()
    streams = [StreamSpec("dinov3", model, "h", "visual")]
    dataset = CachedBranchDataset(
        records=records, cache_index=index, cache_store=store, branch="visual"
    )

    run_export(tmp_path, records, index, store, streams)

    assert torch.allclose(model.seen[0][0], dataset[0].values)


def test_a_crossmodal_stream_is_given_exactly_what_training_gives_it(
    tmp_path: Path, cached
) -> None:
    """Same views, same frame trim, same waveform normalisation."""
    records, index, store = cached
    model = RecordingCrossModalStream()
    streams = [StreamSpec("lipsync", model, "h", "lipsync")]
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )

    run_export(tmp_path, records, index, store, streams)

    assert torch.allclose(model.video[0][0], dataset[0].video)
    assert torch.allclose(model.audio[0][0], dataset[0].audio)


def test_emotion_reads_the_face_views_not_the_mouth_views(
    tmp_path: Path, cached
) -> None:
    """The two audiovisual streams differ only in their inputs, so a swap here
    would be invisible in the shapes."""
    records, index, store = cached
    model = RecordingCrossModalStream()
    streams = [StreamSpec("emotion", model, "h", "emotion")]
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="emotion"
    )

    run_export(tmp_path, records, index, store, streams)

    assert STREAM_VIEWS["emotion"] == ("visual_view", "audio_view")
    assert torch.allclose(model.video[0][0], dataset[0].video)


def test_a_missing_view_abstains_rather_than_dropping_the_clip(
    tmp_path: Path
) -> None:
    """The clip is still counted, which is what keeps the abstention rate
    honest. Dropping it would shrink the denominator instead."""
    store = CacheStore(tmp_path / "cache")
    index = {"clip-0": store_clip(store, "clip-0", full=False)}
    records = [record("clip-0", fake=True)]
    streams = [
        StreamSpec("dinov3", RecordingVisualStream(), "h", "visual"),
        StreamSpec("lipsync", RecordingCrossModalStream(), "h", "lipsync"),
    ]

    report, feature_store = run_export(tmp_path, records, index, store, streams)

    rows = {row.branch: row for row in feature_store.read()}
    assert rows["dinov3"].available is True
    assert rows["lipsync"].available is False
    assert report.unavailable_rows == 1
    assert "clip-0:lipsync" in report.failures


def test_an_unreadable_clip_abstains_on_every_stream(tmp_path: Path) -> None:
    records = [record("absent", fake=True)]
    store = CacheStore(tmp_path / "cache")
    streams = [StreamSpec("dinov3", RecordingVisualStream(), "h", "visual")]

    report, feature_store = run_export(tmp_path, records, {}, store, streams)

    assert report.unavailable_rows == 1
    assert report.failures["absent"] == "missing_cache_entry"
    assert [row.available for row in feature_store.read()] == [False]


def test_a_different_preprocessing_hash_is_an_error(tmp_path: Path, cached) -> None:
    """Silently scoring against a cache built by different code is the failure
    this pipeline hashes everything to prevent."""
    records, index, store = cached
    feature_store = FeatureStore(tmp_path / "features.parquet")

    with pytest.raises(ValueError, match="different preprocessing hash"):
        export_stream_features(
            records=records,
            cache_index=index,
            cache_store=store,
            feature_store=feature_store,
            streams=[StreamSpec("dinov3", RecordingVisualStream(), "h", "visual")],
            split_hash="split",
            preprocessing_hash="a-different-hash",
            partition_role="oof",
            run_id="run",
            device="cpu",
        )


def test_rejects_an_unknown_stream_kind() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        StreamSpec("thing", RecordingVisualStream(), "h", "nonsense")


def test_rejects_duplicate_stream_names(tmp_path: Path, cached) -> None:
    records, index, store = cached
    streams = [
        StreamSpec("visual", RecordingVisualStream(), "a", "visual"),
        StreamSpec("visual", RecordingVisualStream(), "b", "visual"),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        run_export(tmp_path, records, index, store, streams)


def test_rejects_an_empty_stream_list(tmp_path: Path, cached) -> None:
    records, index, store = cached

    with pytest.raises(ValueError, match="At least one stream"):
        run_export(tmp_path, records, index, store, [])
