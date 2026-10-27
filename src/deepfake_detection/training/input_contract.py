"""Checkpoint-controlled audio masking shared by training and inference."""

from __future__ import annotations

import torch

MASKING_MODES = ("legacy_none", "valid_samples")


def audio_forward(model, waveform, valid_samples=None):
    mode = getattr(model, "masking_mode", "legacy_none")
    if mode not in MASKING_MODES:
        raise ValueError(f"Unknown checkpoint masking mode: {mode}")
    if mode == "legacy_none":
        return model(waveform)
    if valid_samples is None:
        raise ValueError(
            "Checkpoint requires valid audio sample counts; rebuild the cache"
        )
    counts = torch.as_tensor(valid_samples, device=waveform.device).reshape(-1)
    if (
        counts.shape != (waveform.shape[0],)
        or torch.any(counts <= 0)
        or torch.any(counts > waveform.shape[1])
    ):
        raise ValueError("Invalid audio valid sample counts")
    return model(waveform, valid_samples=counts)
