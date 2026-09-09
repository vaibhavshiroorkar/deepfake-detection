"""The stream trains, and the dataset feeds it the right pairs.

The overfit test is the important one. A model that cannot drive the loss down
on a handful of examples it sees every epoch has a wiring or gradient fault, and
finding that out here costs seconds instead of finding it out after a GPU run.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.data.datasets import (
    CachedAVPairDataset,
    collate_av_pair_items,
)
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.streams.config import StreamConfig
from deepfake_detection.streams.cross_modal_stream import CrossModalStream
from deepfake_detection.training.streams import StreamTrainingConfig, fit_stream
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport

VIDEO_STEPS = 8
AUDIO_SAMPLES = 4_000


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


def store_clip(store: CacheStore, clip_id: str, *, fake: bool) -> Path:
    """A cached clip whose views separate the two classes.

    Fake clips get a constant offset the model can find. The point is to test
    that the machinery learns, not that it solves deepfake detection.
    """
    generator = np.random.default_rng(abs(hash(clip_id)) % 2**32)
    bias = 1.5 if fake else -1.5
    prepared = PreparedClip(
        clip_id=clip_id,
        visual_view=None,
        audio_view=None,
        sync_video_view=(
            generator.normal(size=(VIDEO_STEPS, 3, 16, 16)).astype(np.float32) + bias
        ),
        sync_audio_view=(
            generator.normal(size=(AUDIO_SAMPLES,)).astype(np.float32) + bias
        ),
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint=f"fingerprint-{clip_id}",
        preprocessing_config_hash="pipeline-hash",
    )
    return store.save(prepared, dataset="fixture")


@pytest.fixture
def cached(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    records = []
    index = {}
    for number in range(10):
        for fake in (False, True):
            clip_id = f"clip-{number}-{'fake' if fake else 'real'}"
            index[clip_id] = store_clip(store, clip_id, fake=fake)
            records.append(record(clip_id, fake=fake))
    return records, index, store


def tiny_stream() -> CrossModalStream:
    config = StreamConfig(
        stream_name="lipsync",
        backbone_name="resnet18",
        pretrained=False,
        common_dim=16,
        image_size=16,
        frame_chunk_size=0,
    )
    return CrossModalStream(
        config=config, pretrained=False, attention_heads=2, query_steps=VIDEO_STEPS
    )


def batches(dataset, size: int = 5):
    items = [dataset[index] for index in range(len(dataset))]
    return [
        collate_av_pair_items(items[start : start + size])
        for start in range(0, len(items), size)
    ]


def test_dataset_yields_paired_views_and_a_binary_label(cached) -> None:
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )

    item = dataset[0]

    assert item.video.shape == (VIDEO_STEPS, 3, 16, 16)
    assert item.audio.shape == (AUDIO_SAMPLES,)
    assert item.label.item() in {0.0, 1.0}


def test_dataset_label_is_clip_fake_not_the_cue_label(cached) -> None:
    """Either modality being manipulated produces a mismatch, so the
    cue-specific labels do not describe what this stream can see."""
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )

    by_id = {dataset[i].clip_id: dataset[i].label.item() for i in range(len(dataset))}

    assert by_id["clip-0-fake"] == 1.0
    assert by_id["clip-0-real"] == 0.0


def test_dataset_rejects_an_unknown_stream(cached) -> None:
    records, index, store = cached
    with pytest.raises(ValueError, match="Unsupported stream"):
        CachedAVPairDataset(
            records=records, cache_index=index, cache_store=store, stream="nonsense"
        )


def test_dataset_reports_a_missing_view_rather_than_guessing(cached) -> None:
    """These clips have no face view at all, which is the abstention case."""
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="emotion"
    )

    with pytest.raises(ValueError, match="has no emotion view"):
        dataset[0]


def test_collate_refuses_an_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty audiovisual batch"):
        collate_av_pair_items([])


def test_stream_overfits_a_tiny_set(cached) -> None:
    """If this cannot memorise 20 clips, the gradient path is broken."""
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )
    torch.manual_seed(17)
    model = tiny_stream()
    data = batches(dataset)

    history = fit_stream(
        model=model,
        train_batches=data,
        validation_batches=data,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=StreamTrainingConfig(
            epochs=8, accumulation_steps=1, freeze_epochs=0, early_stopping_patience=8
        ),
        device="cpu",
    )

    first = history.epochs[0].train_loss
    last = history.epochs[-1].train_loss
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_history_records_diagonal_mass_per_epoch(cached) -> None:
    """Tracked so a falling loss with flat diagonal mass is visible as the
    shortcut it would be."""
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )
    torch.manual_seed(17)
    model = tiny_stream()
    data = batches(dataset)

    history = fit_stream(
        model=model,
        train_batches=data,
        validation_batches=data,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=StreamTrainingConfig(
            epochs=2, accumulation_steps=1, freeze_epochs=0, early_stopping_patience=2
        ),
        device="cpu",
    )

    for epoch in history.epochs:
        assert 0.0 <= epoch.validation_diagonal_mass <= 1.0


def test_freeze_schedule_is_reported_per_epoch(cached) -> None:
    records, index, store = cached
    dataset = CachedAVPairDataset(
        records=records, cache_index=index, cache_store=store, stream="lipsync"
    )
    torch.manual_seed(17)
    model = tiny_stream()
    data = batches(dataset)

    history = fit_stream(
        model=model,
        train_batches=data,
        validation_batches=data,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=StreamTrainingConfig(
            epochs=3, accumulation_steps=1, freeze_epochs=2, early_stopping_patience=3
        ),
        device="cpu",
    )

    assert [epoch.encoders_trainable for epoch in history.epochs] == [
        False,
        False,
        True,
    ]


def test_config_rejects_a_freeze_longer_than_the_run() -> None:
    with pytest.raises(ValueError, match="within the training run"):
        StreamTrainingConfig(epochs=2, freeze_epochs=5)


def test_encoders_get_a_much_lower_learning_rate() -> None:
    """The bug that wrecked the first real run: one learning rate for a
    randomly initialised head and 100M pretrained encoder parameters drove
    training loss to 0.039 while validation climbed to 2.56."""
    from deepfake_detection.training.streams import parameter_groups

    model = tiny_stream()
    groups = parameter_groups(model, head_lr=1e-3, encoder_lr=5e-6)

    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-3
    assert groups[1]["lr"] == 5e-6
    # Every parameter appears exactly once, or some would never be updated.
    counted = sum(len(group["params"]) for group in groups)
    assert counted == len(list(model.parameters()))


def test_encoder_group_holds_the_bulk_of_the_parameters() -> None:
    from deepfake_detection.training.streams import parameter_groups

    model = tiny_stream()
    head, encoders = parameter_groups(model, head_lr=1e-3, encoder_lr=5e-6)

    head_size = sum(p.numel() for p in head["params"])
    encoder_size = sum(p.numel() for p in encoders["params"])
    assert encoder_size > head_size


def test_emotion_stream_trims_video_to_the_audio_aligned_prefix(tmp_path: Path) -> None:
    """`visual_view` spans the whole clip while `audio_view` is a fixed 4 second
    window, so the later face frames have no audio to be compared against.
    Pairing them anyway would measure nothing, which is fatal to an
    affect-consistency claim."""
    from deepfake_detection.data.datasets import STREAM_FRAME_LIMITS

    store = CacheStore(tmp_path / "cache")
    prepared = PreparedClip(
        clip_id="clip-a",
        visual_view=np.zeros((16, 3, 16, 16), dtype=np.float32),
        audio_view=np.zeros((AUDIO_SAMPLES,), dtype=np.float32),
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint="fingerprint-a",
        preprocessing_config_hash="pipeline-hash",
    )
    index = {"clip-a": store.save(prepared, dataset="fixture")}
    dataset = CachedAVPairDataset(
        records=[record("clip-a", fake=False)],
        cache_index=index,
        cache_store=store,
        stream="emotion",
    )

    item = dataset[0]

    assert item.video.shape[0] == STREAM_FRAME_LIMITS["emotion"] == 8
    # Lip-sync must not be trimmed: its two views already share a window.
    assert STREAM_FRAME_LIMITS["lipsync"] is None
