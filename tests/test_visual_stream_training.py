"""The configurable visual stream trains, and a frozen backbone stays frozen.

`fit_stream` cannot be reused here: a visual batch carries one tensor rather
than a video and an audio tensor, and `VisualStream` returns a plain tuple
rather than a `StreamOutput`. So the same overfit check is repeated against the
separate code path, because that is where a wiring fault would live.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.data.datasets import (
    CachedBranchDataset,
    collate_branch_items,
)
from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.streams.config import StreamConfig
from deepfake_detection.streams.visual_stream import build_visual_stream
from deepfake_detection.training.streams import (
    StreamTrainingConfig,
    fit_visual_stream,
    parameter_groups,
)
from deepfake_detection.views.cache_store import CacheStore
from deepfake_detection.views.contracts import PreparedClip, QualityReport

FRAMES = 4
SIZE = 32


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
    generator = np.random.default_rng(abs(hash(clip_id)) % 2**32)
    bias = 1.5 if fake else -1.5
    prepared = PreparedClip(
        clip_id=clip_id,
        visual_view=(
            generator.normal(size=(FRAMES, 3, SIZE, SIZE)).astype(np.float32) + bias
        ),
        audio_view=None,
        sync_video_view=None,
        sync_audio_view=None,
        quality=QualityReport(1.0, True, True, False, 0.0),
        preprocessing_fingerprint=f"fingerprint-{clip_id}",
        preprocessing_config_hash="pipeline-hash",
    )
    return store.save(prepared, dataset="fixture")


@pytest.fixture
def batches(tmp_path: Path):
    store = CacheStore(tmp_path / "cache")
    records = []
    index = {}
    for number in range(10):
        for fake in (False, True):
            clip_id = f"clip-{number}-{'fake' if fake else 'real'}"
            index[clip_id] = store_clip(store, clip_id, fake=fake)
            records.append(record(clip_id, fake=fake))
    dataset = CachedBranchDataset(
        records=records, cache_index=index, cache_store=store, branch="visual"
    )
    items = [dataset[i] for i in range(len(dataset))]
    return [collate_branch_items(items[start : start + 5]) for start in range(0, 20, 5)]


def tiny_stream(*, freeze: bool = False):
    config = StreamConfig(
        stream_name="fixture",
        backbone_name="resnet18",
        pretrained=False,
        common_dim=16,
        num_frames=FRAMES,
        image_size=SIZE,
        temporal_hidden=16,
        frame_chunk_size=0,
        freeze_backbone=freeze,
    )
    return build_visual_stream(config)


def train(model, batches, *, epochs: int = 8, freeze_epochs: int = 0):
    return fit_visual_stream(
        model=model,
        train_batches=batches,
        validation_batches=batches,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        config=StreamTrainingConfig(
            epochs=epochs,
            accumulation_steps=1,
            freeze_epochs=freeze_epochs,
            early_stopping_patience=epochs,
        ),
        device="cpu",
    )


def test_visual_stream_overfits_a_tiny_set(batches) -> None:
    torch.manual_seed(17)
    history = train(tiny_stream(), batches)

    first = history.epochs[0].train_loss
    last = history.epochs[-1].train_loss
    assert last < first * 0.6, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_freeze_schedule_is_reported_per_epoch(batches) -> None:
    torch.manual_seed(17)
    history = train(tiny_stream(), batches, epochs=4, freeze_epochs=2)

    trainable = [epoch.encoders_trainable for epoch in history.epochs]
    assert trainable == [False, False, True, True]


def test_diagonal_mass_is_zero_because_there_is_no_attention_map(batches) -> None:
    """A visual stream makes no correspondence claim, so the field carries no
    meaning here and must not be read as a measurement."""
    torch.manual_seed(17)
    history = train(tiny_stream(), batches, epochs=2)

    assert all(epoch.validation_diagonal_mass == 0.0 for epoch in history.epochs)


def test_a_frozen_backbone_yields_only_a_head_parameter_group() -> None:
    """An optimizer group with an empty parameter list raises, so the frozen
    case has to drop the group rather than pass it empty."""
    model = tiny_stream(freeze=True)

    groups = parameter_groups(
        model, head_lr=1e-3, encoder_lr=5e-6, encoder_names=("backbone",)
    )

    assert len(groups) == 1
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_a_trainable_backbone_gets_its_own_learning_rate() -> None:
    model = tiny_stream(freeze=False)

    groups = parameter_groups(
        model, head_lr=1e-3, encoder_lr=5e-6, encoder_names=("backbone",)
    )

    assert [group["lr"] for group in groups] == [1e-3, 5e-6]
