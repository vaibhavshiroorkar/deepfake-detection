from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .audio import _audio_tokens
from .visual import _flatten_features


def _nearest_temporal_tokens(tokens: Tensor, *, size: int) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError("Temporal tokens must have shape [batch, time, features]")
    if size <= 0 or tokens.shape[1] < size:
        raise ValueError("Temporal token count must be at least the requested size")
    indices = (
        torch.linspace(
            0,
            tokens.shape[1] - 1,
            steps=size,
            device=tokens.device,
        )
        .round()
        .long()
    )
    return tokens.index_select(1, indices)


@dataclass(frozen=True, slots=True)
class SyncOutput:
    video_tokens: Tensor
    audio_tokens: Tensor
    offset_logits: Tensor
    aligned_similarity: Tensor


class SynchronizationBranch(nn.Module):
    def __init__(
        self,
        *,
        video_encoder: nn.Module,
        video_dim: int,
        audio_encoder: nn.Module,
        audio_dim: int,
        projection_dim: int = 256,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        offset_classes: int = 8,
    ) -> None:
        super().__init__()
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.video_projection = nn.Linear(video_dim, projection_dim)
        self.audio_projection = nn.Linear(audio_dim, projection_dim)
        layer_options = {
            "d_model": projection_dim,
            "nhead": attention_heads,
            "dim_feedforward": projection_dim * 4,
            "batch_first": True,
            "norm_first": True,
        }
        self.video_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options),
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.audio_temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**layer_options),
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.offset_head = nn.Sequential(
            nn.Linear(projection_dim * 2 + 2, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, offset_classes),
        )

    def forward(self, *, mouth_video: Tensor, waveform: Tensor) -> SyncOutput:
        if mouth_video.ndim != 5:
            raise ValueError(
                "Mouth video must have shape [batch, time, channels, height, width]"
            )
        if waveform.ndim != 2:
            raise ValueError("Waveform must have shape [batch, samples]")
        batch, time, channels, height, width = mouth_video.shape
        frames = mouth_video.reshape(batch * time, channels, height, width)
        video = _flatten_features(self.video_encoder(frames)).reshape(batch, time, -1)
        video = self.video_temporal(self.video_projection(video))

        audio = self.audio_projection(_audio_tokens(self.audio_encoder(waveform)))
        audio = _nearest_temporal_tokens(audio, size=time)
        audio = self.audio_temporal(audio)

        normalized_video = functional.normalize(video, dim=-1)
        normalized_audio = functional.normalize(audio, dim=-1)
        similarity = (normalized_video * normalized_audio).sum(dim=-1)
        pooled = torch.cat(
            (
                video.mean(dim=1),
                audio.mean(dim=1),
                similarity.mean(dim=1, keepdim=True),
                similarity.amax(dim=1, keepdim=True),
            ),
            dim=-1,
        )
        return SyncOutput(
            video_tokens=video,
            audio_tokens=audio,
            offset_logits=self.offset_head(pooled),
            aligned_similarity=similarity,
        )

    def freeze_backbones(self) -> None:
        for encoder in (self.video_encoder, self.audio_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False


def build_sync_branch(
    *,
    audio_model_name: str = "facebook/wav2vec2-base",
    projection_dim: int = 256,
    pretrained: bool = True,
) -> SynchronizationBranch:
    from torchvision.models import ResNet18_Weights, resnet18
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    video_encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    video_dim = video_encoder.fc.in_features
    video_encoder.fc = nn.Identity()
    audio_encoder = (
        Wav2Vec2Model.from_pretrained(audio_model_name)
        if pretrained
        else Wav2Vec2Model(Wav2Vec2Config())
    )
    return SynchronizationBranch(
        video_encoder=video_encoder,
        video_dim=video_dim,
        audio_encoder=audio_encoder,
        audio_dim=audio_encoder.config.hidden_size,
        projection_dim=projection_dim,
    )
