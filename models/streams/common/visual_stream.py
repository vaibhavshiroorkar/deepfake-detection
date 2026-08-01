"""The reusable visual-stream template — every visual stream is an instance of
this one class, differing only by the StreamConfig passed in (Xception vs
EfficientNet vs DINOv2 later).

Data flow for one clip, batched as [B, T, 3, H, W] (T = num_frames):

    frames [B, T, 3, 224, 224]
      -> backbone per frame (in VRAM-sized chunks)     [B, T, F]
      -> temporal model (LSTM / GRU / mean-pool)        [B, temporal_out]
      -> projection (Linear + LayerNorm)                [B, common_dim]   <- feature store
      -> temporary head (dev only)                      logit [B]

The projection output is the real product; the head only lets a stream's
standalone power be measured before fusion exists.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import timm

from models.streams.common.config import StreamConfig


def _create_backbone(config: StreamConfig) -> nn.Module:
    """The timm backbone, built at the config's input resolution.

    A ViT needs `img_size` to be told: DINOv2 ships a 518-pixel pretrained config,
    and this pipeline feeds 224-pixel face crops, so without it the positional
    embedding is sized for an input that never arrives. A CNN has no such
    parameter and raises TypeError on the keyword, which is the signal to build
    it plainly rather than a list of which backbones are transformers.

    Pooling comes from the config for the same reason it cannot be hardcoded to
    "avg": that choice makes timm build `fc_norm` in place of `norm`, and
    DINOv2's released weights carry `norm`, so the pretrained load fails.
    """
    kwargs = dict(pretrained=config.pretrained, num_classes=0,
                  global_pool=config.global_pool)
    try:
        return timm.create_model(config.backbone_name, img_size=config.image_size, **kwargs)
    except TypeError:
        pass
    except Exception as e:
        raise RuntimeError(
            f"Failed to build backbone '{config.backbone_name}' from timm: {e}") from e
    try:
        return timm.create_model(config.backbone_name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build backbone '{config.backbone_name}' from timm: {e}") from e


class VisualStream(nn.Module):
    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config

        self.backbone = _create_backbone(config)
        self.feature_dim = self.backbone.num_features

        # Gradient checkpointing is a VRAM trade, not a correctness requirement,
        # and legacy_xception does not implement it: timm's method is present but
        # asserts on enable. Losing it costs memory, so it is recorded rather
        # than swallowed, and never fatal.
        self.grad_checkpointing = False
        if config.grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            try:
                self.backbone.set_grad_checkpointing(True)
                self.grad_checkpointing = True
            except (AssertionError, NotImplementedError, TypeError):
                self.grad_checkpointing = False

        ttype = config.temporal_type.lower()
        if ttype == "mean":
            self.temporal = None
            temporal_out = self.feature_dim
        elif ttype in ("lstm", "gru"):
            rnn_cls = nn.LSTM if ttype == "lstm" else nn.GRU
            self.temporal = rnn_cls(
                input_size=self.feature_dim, hidden_size=config.temporal_hidden,
                num_layers=config.temporal_layers, batch_first=True,
                bidirectional=config.temporal_bidirectional,
            )
            temporal_out = config.temporal_hidden * (2 if config.temporal_bidirectional else 1)
        else:
            raise ValueError(f"temporal_type must be lstm/gru/mean, got '{config.temporal_type}'")
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
        return torch.cat([self.backbone(frames_flat[i:i + chunk])
                          for i in range(0, frames_flat.shape[0], chunk)], dim=0)

    def forward(self, frames: torch.Tensor):
        """frames [B, T, 3, H, W] -> (logit [B], embedding [B, common_dim])."""
        B, T, C, H, W = frames.shape
        feats = self._run_backbone_chunked(frames.reshape(B * T, C, H, W))
        feats = feats.reshape(B, T, self.feature_dim)

        if self.temporal is None:
            clip_vec = feats.mean(dim=1)                          # [B, F]
        else:
            self.temporal.flatten_parameters()
            if self.config.temporal_type.lower() == "lstm":
                _, (h_n, _) = self.temporal(feats)
            else:
                _, h_n = self.temporal(feats)
            num_dir = 2 if self.config.temporal_bidirectional else 1
            h_last = h_n.view(self.config.temporal_layers, num_dir, B,
                              self.config.temporal_hidden)[-1]
            clip_vec = torch.cat([h_last[d] for d in range(num_dir)], dim=-1)

        embedding = self.projection(clip_vec)                    # [B, common_dim]
        logit = self.temp_head(embedding).squeeze(-1)            # [B]
        return logit, embedding

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "feature_dim": self.feature_dim,
                "embedding_dim": self.config.common_dim}


def build_visual_stream(config: StreamConfig) -> VisualStream:
    model = VisualStream(config)
    if config.freeze_backbone:
        model.set_backbone_trainable(False)
    return model
