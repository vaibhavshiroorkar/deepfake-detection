"""Whisper-base encoder with trained classifier head for audio deepfake detection.

Architecture:
  openai/whisper-base encoder (frozen) → hidden states [T, 512]
  → Attention-weighted temporal pooling → (512,)
  → Linear(512, 256) → GELU → Dropout(0.1)
  → Linear(256, 2)    [real, fake]

The Whisper encoder is frozen — only the pooling + MLP head are trained.
Whisper was trained on ~680k hours of real speech, making its encoder a
powerful feature extractor for detecting synthetic audio artifacts that
differ from real speech spectrogram patterns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WHISPER_DIM = 512  # whisper-base hidden size


class AttentionPool(nn.Module):
    """Learned attention-weighted mean pooling over the time axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, D)"""
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * weights).sum(dim=1)  # (B, D)


class WhisperClassifierHead(nn.Module):
    """MLP classifier on pooled Whisper encoder output."""

    def __init__(self, in_dim: int = WHISPER_DIM, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentionPool(in_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """encoder_output: (B, T, D) → logits: (B, 2) [real, fake]"""
        pooled = self.pool(encoder_output)
        return self.classifier(pooled)


def load_whisper_classifier(
    weights_path: str | Path | None,
    device: torch.device,
) -> WhisperClassifierHead:
    """Load the classifier head.  Whisper encoder is loaded separately
    via models.get_whisper_encoder()."""
    head = WhisperClassifierHead(in_dim=WHISPER_DIM)

    if weights_path:
        wp = Path(weights_path)
        if wp.exists():
            state = torch.load(wp, map_location="cpu", weights_only=True)
            head.load_state_dict(state)
            logger.info("Loaded Whisper classifier head from %s", wp)
        else:
            logger.warning(
                "Whisper classifier weights not found at %s — head is untrained.",
                wp,
            )
    else:
        logger.warning(
            "No VERITAS_WHISPER_WEIGHTS path set — Whisper classifier head is untrained."
        )

    return head.to(device).eval()


def predict_audio(
    encoder_hidden: np.ndarray,
    head: WhisperClassifierHead,
    device: torch.device,
) -> tuple[float, float, dict]:
    """Classify audio from pre-computed Whisper encoder output.

    Args:
        encoder_hidden: (T, 512) numpy array from Whisper encoder.
        head: trained classifier head.
        device: torch device.

    Returns:
        (p_fake, p_real, stats_dict)
    """
    h = torch.from_numpy(encoder_hidden).unsqueeze(0).to(device)  # (1, T, 512)

    with torch.no_grad():
        logits = head(h)[0]  # (2,)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    p_real = float(probs[0])
    p_fake = float(probs[1])

    # Also compute the frame-level attention pattern for interpretability
    with torch.no_grad():
        attn_weights = torch.softmax(head.pool.attn(h), dim=1)[0, :, 0].cpu().numpy()

    stats = {
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
        "attention_entropy": round(float(-np.sum(attn_weights * np.log(attn_weights + 1e-9))), 3),
        "n_frames": int(encoder_hidden.shape[0]),
    }
    return p_fake, p_real, stats
