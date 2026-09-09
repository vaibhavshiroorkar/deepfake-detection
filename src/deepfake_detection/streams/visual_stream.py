"""The reusable visual-stream template.

Every visual stream is an instance of this one class, differing only by the
StreamConfig passed in.

Data flow for one clip, batched as [B, T, 3, H, W] where T is num_frames:

    frames [B, T, 3, 224, 224]
      -> backbone per frame, in VRAM-sized chunks   [B, T, F]
      -> temporal model (LSTM, GRU or mean-pool)    [B, temporal_out]
      -> projection (Linear + LayerNorm)            [B, common_dim]
      -> temporary head, development only           logit [B]

The projection output is the real product. The head only lets a stream's
standalone power be measured before fusion exists.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import StreamConfig


def _create_backbone(config: StreamConfig) -> nn.Module:
    """The timm backbone, built at the config's input resolution.

    A ViT needs `img_size` to be told: DINOv3 ships a 256-pixel pretrained
    config and this pipeline feeds 224-pixel face crops, so without it the
    positional embedding is sized for an input that never arrives. A CNN has no
    such parameter and raises TypeError on the keyword, which is the signal to
    build it plainly rather than a list of which backbones are transformers.

    Pooling comes from the config for the same reason it cannot be hardcoded to
    "avg": that choice makes timm build `fc_norm` in place of `norm`, and
    DINOv3's released weights carry `norm`, so the pretrained load fails.
    """
    import timm

    kwargs = {
        "pretrained": config.pretrained,
        "num_classes": 0,
        "global_pool": config.global_pool,
    }
    try:
        return timm.create_model(
            config.backbone_name, img_size=config.image_size, **kwargs
        )
    except TypeError:
        pass
    except Exception as error:
        raise RuntimeError(
            f"Failed to build backbone {config.backbone_name!r} from timm: {error}"
        ) from error
    try:
        return timm.create_model(config.backbone_name, **kwargs)
    except Exception as error:
        raise RuntimeError(
            f"Failed to build backbone {config.backbone_name!r} from timm: {error}"
        ) from error


class VisualStream(nn.Module):
    def __init__(self, config: StreamConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = _create_backbone(config)
        self.feature_dim = self.backbone.num_features

        # Gradient checkpointing is a VRAM trade, not a correctness requirement,
        # and legacy_xception does not implement it: timm's method is present
        # but asserts on enable. Losing it costs memory, so it is recorded
        # rather than swallowed, and it is never fatal.
        self.grad_checkpointing = False
        if config.grad_checkpointing and hasattr(
            self.backbone, "set_grad_checkpointing"
        ):
            try:
                self.backbone.set_grad_checkpointing(True)
                self.grad_checkpointing = True
            except (AssertionError, NotImplementedError, TypeError):
                self.grad_checkpointing = False

        temporal_type = config.temporal_type.lower()
        if temporal_type == "mean":
            self.temporal = None
            temporal_out = self.feature_dim
        elif temporal_type in ("lstm", "gru"):
            rnn_cls = nn.LSTM if temporal_type == "lstm" else nn.GRU
            self.temporal = rnn_cls(
                input_size=self.feature_dim,
                hidden_size=config.temporal_hidden,
                num_layers=config.temporal_layers,
                batch_first=True,
                bidirectional=config.temporal_bidirectional,
            )
            directions = 2 if config.temporal_bidirectional else 1
            temporal_out = config.temporal_hidden * directions
        else:
            raise ValueError(
                f"temporal_type must be lstm, gru or mean, got {config.temporal_type!r}"
            )
        self._temporal_out = temporal_out

        self.projection = nn.Sequential(
            nn.Linear(temporal_out, config.common_dim),
            nn.LayerNorm(config.common_dim),
        )
        self.temp_head = nn.Linear(config.common_dim, 1)

    def _run_backbone_chunked(self, frames_flat: torch.Tensor) -> torch.Tensor:
        chunk = self.config.frame_chunk_size
        if not chunk or chunk >= frames_flat.shape[0]:
            return self.backbone(frames_flat)
        return torch.cat(
            [
                self.backbone(frames_flat[index : index + chunk])
                for index in range(0, frames_flat.shape[0], chunk)
            ],
            dim=0,
        )

    def forward(self, frames: torch.Tensor):
        """frames [B, T, 3, H, W] -> (logit [B], embedding [B, common_dim])."""
        batch, steps, channels, height, width = frames.shape
        features = self._run_backbone_chunked(
            frames.reshape(batch * steps, channels, height, width)
        )
        features = features.reshape(batch, steps, self.feature_dim)

        if self.temporal is None:
            clip_vector = features.mean(dim=1)
        else:
            self.temporal.flatten_parameters()
            if self.config.temporal_type.lower() == "lstm":
                _, (hidden, _) = self.temporal(features)
            else:
                _, hidden = self.temporal(features)
            directions = 2 if self.config.temporal_bidirectional else 1
            last = hidden.view(
                self.config.temporal_layers,
                directions,
                batch,
                self.config.temporal_hidden,
            )[-1]
            clip_vector = torch.cat(
                [last[index] for index in range(directions)], dim=-1
            )

        embedding = self.projection(clip_vector)
        logit = self.temp_head(embedding).squeeze(-1)
        return logit, embedding

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "feature_dim": self.feature_dim,
            "embedding_dim": self.config.common_dim,
        }


def build_visual_stream(config: StreamConfig) -> VisualStream:
    model = VisualStream(config)
    if config.freeze_backbone:
        model.set_backbone_trainable(False)
    return model
