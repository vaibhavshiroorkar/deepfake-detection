from types import SimpleNamespace

import torch
from torch import nn

from deepfake_detection.branches.audio import AudioSpoofBranch
from deepfake_detection.branches.sync import SynchronizationBranch
from deepfake_detection.branches.visual import VisualArtifactBranch


class FrameBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projection(frames.mean(dim=(-1, -2)))


class AudioBackbone(nn.Module):
    def __init__(self, output_dim: int, token_count: int) -> None:
        super().__init__()
        self.projection = nn.Linear(1, output_dim)
        self.token_count = token_count

    def forward(self, waveform: torch.Tensor) -> SimpleNamespace:
        chunks = waveform.unfold(
            1,
            waveform.shape[1] // self.token_count,
            waveform.shape[1] // self.token_count,
        )
        tokens = chunks.mean(dim=-1, keepdim=True)
        return SimpleNamespace(last_hidden_state=self.projection(tokens))


def test_visual_branch_preserves_batch_and_returns_one_logit_per_clip() -> None:
    model = VisualArtifactBranch(
        backbone=FrameBackbone(12),
        backbone_dim=12,
        hidden_dim=8,
    )

    output = model(torch.randn(2, 4, 3, 8, 8))

    assert output.logits.shape == (2,)
    assert output.embedding.shape == (2, 8)


def test_audio_branch_uses_temporal_tokens_before_pooling() -> None:
    model = AudioSpoofBranch(
        encoder=AudioBackbone(output_dim=10, token_count=5),
        encoder_dim=10,
        projection_dim=8,
    )

    output = model(torch.randn(3, 100))

    assert output.logits.shape == (3,)
    assert output.embedding.shape == (3, 8)
    assert output.token_count == 5


def test_sync_branch_keeps_time_axis_and_scores_offsets() -> None:
    model = SynchronizationBranch(
        video_encoder=FrameBackbone(12),
        video_dim=12,
        audio_encoder=AudioBackbone(output_dim=10, token_count=10),
        audio_dim=10,
        projection_dim=8,
        transformer_layers=1,
        attention_heads=2,
        offset_classes=8,
    )

    output = model(
        mouth_video=torch.randn(2, 5, 3, 8, 8),
        waveform=torch.randn(2, 100),
    )

    assert output.video_tokens.shape == (2, 5, 8)
    assert output.audio_tokens.shape == (2, 5, 8)
    assert output.offset_logits.shape == (2, 8)
    assert output.aligned_similarity.shape == (2, 5)
