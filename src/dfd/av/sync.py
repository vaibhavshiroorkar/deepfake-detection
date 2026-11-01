"""Zero-shot lip-audio agreement with AV-HuBERT: nothing is trained.

AV-HuBERT was pretrained with modality dropout, so a lips-only pass and an
audio-only pass land in one feature space. On genuine speech the two agree
frame by frame. `agreement` returns their per-frame cosine similarity; a fake
whose voice does not match the lips should score lower.

A small audio-video offset is common in genuine uploads, so `best_offset`
takes the highest mean agreement over shifts of up to `max_shift` frames
(40 ms each) instead of trusting zero.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from dfd.av.avhubert import AVHubert, audio_features

FRAME_SAMPLES = 640  # 16 kHz / 25 fps


def model_inputs(lips: np.ndarray, audio: np.ndarray, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Lips [T, 88, 88] and a waveform to model tensors with the same T.

    The audio is cut or zero-padded to exactly T * 640 samples first, so both
    streams run at 25 steps per second over the same span.
    """
    frames = len(lips)
    wave = np.zeros(frames * FRAME_SAMPLES, np.float32)
    wave[: min(len(audio), len(wave))] = audio[: len(wave)]
    video = torch.from_numpy(lips).to(device)[None, None]
    sound = audio_features(torch.from_numpy(wave).to(device)[None])
    return video, sound[..., :frames]


@torch.no_grad()
def agreement(model: AVHubert, video: torch.Tensor, audio: torch.Tensor, *, layer: int | None = None) -> torch.Tensor:
    """Per-frame cosine of the lips-only and audio-only passes, [B, T]."""
    lips = model(video=video, output_layer=layer)
    sound = model(audio=audio, output_layer=layer)
    return F.cosine_similarity(lips, sound, dim=-1)


@torch.no_grad()
def best_offset(model: AVHubert, video: torch.Tensor, audio: torch.Tensor, *, max_shift: int = 3,
                layer: int | None = None) -> tuple[float, int]:
    """Highest mean agreement over audio shifts in [-max_shift, max_shift] frames, and that shift."""
    lips = model(video=video, output_layer=layer)[0]
    sound = model(audio=audio, output_layer=layer)[0]
    best = (-2.0, 0)
    frames = lips.shape[0]
    for shift in range(-max_shift, max_shift + 1):
        a, b = (lips[shift:], sound[: frames - shift]) if shift >= 0 else (lips[:shift], sound[-shift:])
        value = F.cosine_similarity(a, b, dim=-1).mean().item()
        best = max(best, (value, shift))
    return best
