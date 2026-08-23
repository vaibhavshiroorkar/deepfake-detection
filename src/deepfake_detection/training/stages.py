from __future__ import annotations

from torch import nn


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def apply_sync_training_stage(
    model: nn.Module,
    stage: str,
    *,
    audio_layers: int = 2,
) -> None:
    if stage not in {"heads", "upper", "full"}:
        raise ValueError(f"Unknown sync training stage: {stage}")
    _set_trainable(model, True)
    if stage == "full":
        return

    video_encoder = model.video_encoder
    audio_encoder = model.audio_encoder
    _set_trainable(video_encoder, False)
    _set_trainable(audio_encoder, False)
    if stage == "heads":
        return

    video_upper = getattr(video_encoder, "layer4", None)
    if video_upper is None:
        raise ValueError("Video encoder has no layer4 block for staged tuning")
    audio_stack = getattr(getattr(audio_encoder, "encoder", None), "layers", None)
    if audio_stack is None:
        raise ValueError("Audio encoder has no encoder.layers stack for staged tuning")
    if not 0 < audio_layers <= len(audio_stack):
        raise ValueError("Requested audio layer count is outside the encoder stack")
    _set_trainable(video_upper, True)
    for layer in audio_stack[-audio_layers:]:
        _set_trainable(layer, True)
