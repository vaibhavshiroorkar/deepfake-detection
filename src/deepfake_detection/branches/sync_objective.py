from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

OFFSET_MILLISECONDS = (-320, -160, -80, 0, 80, 160, 320)
MISMATCH_CLASS_INDEX = len(OFFSET_MILLISECONDS)


def sync_anomaly_logit(offset_logits: Tensor) -> Tensor:
    if offset_logits.ndim != 2 or offset_logits.shape[1] != MISMATCH_CLASS_INDEX + 1:
        raise ValueError("Offset logits must have one column per offset and mismatch")
    aligned_index = OFFSET_MILLISECONDS.index(0)
    anomaly_indices = [
        index for index in range(offset_logits.shape[1]) if index != aligned_index
    ]
    return (
        torch.logsumexp(offset_logits[:, anomaly_indices], dim=1)
        - offset_logits[:, aligned_index]
    )


def crop_audio_context(
    context: Tensor,
    *,
    output_samples: int,
    offset_ms: int,
    sample_rate: int,
) -> Tensor:
    if context.ndim != 2:
        raise ValueError("Audio context must have shape [batch, samples]")
    if output_samples <= 0 or output_samples > context.shape[1]:
        raise ValueError("Output length must fit inside the audio context")
    sample_offset = round(offset_ms * sample_rate / 1_000)
    center_start = (context.shape[1] - output_samples) // 2
    start = center_start + sample_offset
    end = start + output_samples
    if start < 0 or end > context.shape[1]:
        raise ValueError("Audio context is too short for the requested offset")
    return context[:, start:end]


def contrastive_alignment_loss(
    video_tokens: Tensor,
    audio_tokens: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if video_tokens.ndim != 3 or audio_tokens.ndim != 3:
        raise ValueError("Contrastive inputs must have shape [batch, time, features]")
    if video_tokens.shape[0] != audio_tokens.shape[0]:
        raise ValueError("Audio and video batches must have equal size")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    video = functional.normalize(video_tokens.mean(dim=1), dim=-1)
    audio = functional.normalize(audio_tokens.mean(dim=1), dim=-1)
    similarities = video @ audio.transpose(0, 1) / temperature
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    return 0.5 * (
        functional.cross_entropy(similarities, targets)
        + functional.cross_entropy(similarities.transpose(0, 1), targets)
    )
