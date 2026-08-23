import pytest
import torch
from torch import nn

from deepfake_detection.training.engine import run_accumulated_epoch
from deepfake_detection.training.stages import apply_sync_training_stage


def test_partial_accumulation_group_performs_a_correctly_scaled_step() -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [(torch.ones(1, 1), torch.ones(1, 1)) for _ in range(3)]

    stats = run_accumulated_epoch(
        model=model,
        batches=batches,
        optimizer=optimizer,
        accumulation_steps=8,
        loss_for_batch=lambda current, batch: nn.functional.mse_loss(
            current(batch[0]), batch[1]
        ),
    )

    assert stats.optimizer_steps == 1
    assert model.weight.item() == pytest.approx(0.2)


class DummyVideoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(1, 1)
        self.layer4 = nn.Linear(1, 1)


class DummyAudioStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(4)])


class DummyAudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Linear(1, 1)
        self.encoder = DummyAudioStack()


class DummySyncModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_encoder = DummyVideoEncoder()
        self.audio_encoder = DummyAudioEncoder()
        self.head = nn.Linear(1, 1)


def test_sync_upper_stage_unfreezes_only_approved_backbone_blocks() -> None:
    model = DummySyncModel()

    apply_sync_training_stage(model, "upper", audio_layers=2)

    assert not any(
        parameter.requires_grad for parameter in model.video_encoder.stem.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.video_encoder.layer4.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.audio_encoder.feature_extractor.parameters()
    )
    assert not any(
        parameter.requires_grad
        for layer in model.audio_encoder.encoder.layers[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.audio_encoder.encoder.layers[-2:]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.head.parameters())
