"""Temporal Transformer for video deepfake detection.

Architecture:
  Sequence of DINOv2 [CLS] embeddings (one per sampled frame)
  → Learned positional encoding
  → TransformerEncoder (4 layers, 8 heads, dim=1024)
  → Mean pooling
  → Linear(1024, 2)   [real, fake]

This model captures temporal inconsistencies that per-frame analysis misses:
face generation flicker, lighting drift, boundary jitter across time.
Deepfakes generated frame-by-frame produce embedding sequences with
tell-tale repetitive or jittery patterns.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EMBED_DIM = 1024     # DINOv2-large CLS dimension
MAX_FRAMES = 64      # maximum supported sequence length


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = MAX_FRAMES):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """Transformer encoder over a sequence of frame embeddings."""

    def __init__(
        self,
        d_model: int = EMBED_DIM,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """frame_embeddings: (B, T, D) → logits (B, 2)."""
        x = self.norm_in(frame_embeddings)
        x = self.pos_enc(x)
        x = self.transformer(x)           # (B, T, D)
        x = x.mean(dim=1)                 # (B, D) — mean pooling
        return self.classifier(x)          # (B, 2)

    def per_frame_scores(self, frame_embeddings: torch.Tensor) -> np.ndarray:
        """Get per-frame suspicion scores via leave-one-out attention analysis.

        Returns an (T,) array of 0..1 scores — higher means that frame
        contributes more to the "fake" prediction.
        """
        x = self.norm_in(frame_embeddings)
        x = self.pos_enc(x)
        encoded = self.transformer(x)  # (1, T, D)

        # Score each frame's encoded representation individually
        T = encoded.size(1)
        logits_per_frame = []
        for t in range(T):
            frame_logit = self.classifier(encoded[:, t])  # (1, 2)
            p_fake = torch.softmax(frame_logit, dim=-1)[0, 1].item()
            logits_per_frame.append(p_fake)

        return np.array(logits_per_frame, dtype=np.float32)


def load_temporal_transformer(
    weights_path: str | Path | None,
    device: torch.device,
) -> TemporalTransformer:
    """Load the temporal transformer model."""
    model = TemporalTransformer()

    if weights_path:
        wp = Path(weights_path)
        if wp.exists():
            state = torch.load(wp, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            logger.info("Loaded temporal transformer weights from %s", wp)
        else:
            logger.warning(
                "Temporal transformer weights not found at %s — running untrained.",
                wp,
            )
    else:
        logger.warning(
            "No VERITAS_TEMPORAL_WEIGHTS path set — temporal transformer is untrained."
        )

    return model.to(device).eval()


def predict_video(
    frame_embeddings: np.ndarray,
    model: TemporalTransformer,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Classify a video from its frame embeddings.

    Args:
        frame_embeddings: (N, 1024) numpy array of DINOv2 CLS embeddings.
        model: trained TemporalTransformer.
        device: torch device.

    Returns:
        (p_fake, p_real, per_frame_scores)
    """
    embs = torch.from_numpy(frame_embeddings).unsqueeze(0).float().to(device)  # (1, N, 1024)

    with torch.no_grad():
        logits = model(embs)[0]  # (2,)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p_real = float(probs[0])
    p_fake = float(probs[1])

    with torch.no_grad():
        per_frame = model.per_frame_scores(embs)

    return p_fake, p_real, per_frame
