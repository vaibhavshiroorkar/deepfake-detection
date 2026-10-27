"""Rebuild a Design A branch from its checkpoint, whatever it was trained as.

Each loader used to build the model it expected and then load weights into it.
That fails loudly for a ConvNeXt checkpoint loaded into EfficientNet, and
quietly for a WavLM checkpoint loaded into a Wav2Vec2 body (F28). Checkpoints
now record their architecture, so the model is built from the file rather than
from the caller's guess.

Checkpoints written before the field existed record nothing. For those the
file name decides, which is the convention the evaluation reports already use.
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from deepfake_detection.training.checkpoints import (
    CheckpointState,
    load_checkpoint,
    read_metadata,
)

DEFAULT_AUDIO_MODEL = "facebook/wav2vec2-base"
WAVLM_AUDIO_MODEL = "microsoft/wavlm-base"


def architecture_of(path: Path, branch: str, recorded: str = "") -> str:
    """The architecture a checkpoint was trained as."""
    if recorded:
        return recorded
    stem = Path(path).stem
    if branch == "visual":
        return "convnext-tiny" if "convnext" in stem else "efficientnet-b0"
    return WAVLM_AUDIO_MODEL if "wavlm" in stem else DEFAULT_AUDIO_MODEL


def build_branch(branch: str, architecture: str, *, pretrained: bool = False):
    """A fresh branch model. `architecture` is a backbone or an audio model name."""
    if branch == "visual":
        from deepfake_detection.branches.visual import build_visual_branch

        return build_visual_branch(architecture=architecture, pretrained=pretrained)
    if branch == "audio":
        from deepfake_detection.branches.audio import build_audio_branch

        return build_audio_branch(model_name=architecture, pretrained=pretrained)
    if branch == "sync":
        from deepfake_detection.branches.sync import build_sync_branch

        return build_sync_branch(audio_model_name=architecture, pretrained=pretrained)
    raise ValueError(f"Unknown branch {branch!r}")


def load_branch(
    path: Path, branch: str, *, fallback: str = ""
) -> tuple[nn.Module, CheckpointState]:
    """Build the model a checkpoint was trained as, load it, check its branch.

    `fallback` is the caller's architecture for a checkpoint that records none,
    such as a `--architecture` flag. A recorded architecture always wins.
    """
    metadata = read_metadata(path)
    if metadata.branch != branch:
        raise ValueError(f"{path} is a {metadata.branch} checkpoint, not {branch}")
    architecture = architecture_of(path, branch, metadata.architecture or fallback)
    model = build_branch(branch, architecture)
    return model, load_checkpoint(path, model=model)


def load_design_a_branches(
    *,
    visual: Path,
    audio: Path,
    sync: Path,
    audio_fallback: str = DEFAULT_AUDIO_MODEL,
) -> tuple[dict[str, nn.Module], dict[str, CheckpointState]]:
    """The three Design A branches, each rebuilt from its own checkpoint."""
    models: dict[str, nn.Module] = {}
    states: dict[str, CheckpointState] = {}
    for branch, path, fallback in (
        ("visual", visual, ""),
        ("audio", audio, audio_fallback),
        ("sync", sync, audio_fallback),
    ):
        models[branch], states[branch] = load_branch(path, branch, fallback=fallback)
    return models, states
