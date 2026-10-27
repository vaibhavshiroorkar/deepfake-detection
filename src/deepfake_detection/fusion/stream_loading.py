"""Rebuild the trained Design B streams from a run directory.

This lived in `scripts/score_streams.py`, which made it unreachable from the
dashboard: serving a clip through the same five streams that were scored meant
either importing a script or writing the loader a second time. A second copy is
the failure this module exists to prevent, because a stream rebuilt with the
wrong temporal model still loads most of its tensors and then computes a
different vector, silently.

The architecture is read back from each checkpoint's own history file rather
than passed in as flags, so the shape always comes from the run that produced
the weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from deepfake_detection.digest import file_sha256

# Which cached views each stream reads, and so which trainer built it. Keyed by
# the checkpoint stem the training script writes.
STREAM_KINDS: dict[str, tuple[str, str | None]] = {
    "visual-dinov3": ("visual", "dinov3"),
    "visual-efficientnet": ("visual", "efficientnet"),
    "stream-lipsync": ("lipsync", None),
    "stream-emotion": ("emotion", None),
    # The same two streams with the encoders their design named: AV-HuBERT for
    # lip-sync, HSEmotion for the face side of emotion. The encoder itself is
    # read from each history file, so the second field is informational.
    "stream-lipsync-avhubert": ("lipsync", "avhubert"),
    "stream-emotion-hsemotion": ("emotion", "hsemotion"),
    # The Design A audio branch, reused rather than retrained: it already emits
    # a 256-dim embedding, the same width the streams project to, so it drops
    # into the fusion head unchanged. It is also the only model an audio-only
    # input can drive.
    "final-audio-seed17": ("audio", None),
}

# A stream whose checkpoint lives outside the run directory. The audio branch
# was trained by the Design A program run and there is no reason to train it
# again.
EXTERNAL_CHECKPOINTS = {
    "final-audio-seed17": Path("runs/program-20260906/checkpoints"),
}


def build_model(name: str, history: dict):
    """Rebuild a stream exactly as it was trained, from its own history file.

    Returns the module and its export kind. `pretrained=False` throughout: the
    checkpoint carries every weight, and downloading a backbone only to
    overwrite it costs a network round trip the dashboard cannot assume.
    """
    kind, backbone = STREAM_KINDS[name]
    model_config = history["config"]["model"]

    if kind == "visual":
        from deepfake_detection.streams.config import (
            dinov3_config,
            efficientnet_config,
        )
        from deepfake_detection.streams.visual_stream import build_visual_stream

        presets = {"dinov3": dinov3_config, "efficientnet": efficientnet_config}
        config = presets[backbone](
            pretrained=False,
            common_dim=model_config["common_dim"],
            temporal_type=model_config["temporal"],
            temporal_hidden=model_config["temporal_hidden"],
            freeze_backbone=model_config["frozen_backbone"],
            frame_chunk_size=8,
        )
        return build_visual_stream(config), kind

    if kind == "audio":
        from deepfake_detection.branches.audio import build_wav2vec2_audio_branch

        return (
            build_wav2vec2_audio_branch(
                model_name=model_config.get("audio_model", "facebook/wav2vec2-base"),
                pretrained=False,
            ),
            kind,
        )

    from deepfake_detection.streams.cross_modal_stream import (
        build_emotion_stream,
        build_lipsync_stream,
    )

    builder = build_lipsync_stream if kind == "lipsync" else build_emotion_stream
    return builder(
        pretrained=False,
        common_dim=model_config.get("common_dim", 256),
        encoder=model_config.get("encoder", "default"),
        video_backbone=model_config.get(
            "video_backbone", "tf_efficientnet_b0.ns_jft_in1k"
        ),
        audio_model=model_config.get("audio_model", "facebook/wav2vec2-base"),
        attention_heads=model_config.get("attention_heads", 4),
        temporal_position_mode=model_config.get("temporal_position_mode", "none"),
        query_steps=model_config.get("query_steps", 50 if kind == "lipsync" else 8),
    ), kind


def load_streams(
    run_dir: Path,
    device: str,
    *,
    root: Path | None = None,
    only: tuple[str, ...] | None = None,
    report: Callable[[str], None] | None = None,
):
    """Every trained checkpoint in the run, rebuilt and loaded.

    `only` restricts the set, which is what an audio-only upload wants: loading
    four unused models to score a wav file costs about a minute of GPU and all
    of their memory.
    """
    from deepfake_detection.fusion.stream_export import StreamSpec
    from deepfake_detection.training.checkpoints import load_checkpoint

    say = report or (lambda _message: None)
    base = root or Path.cwd()
    specs = []
    for name in STREAM_KINDS:
        if only is not None and name not in only:
            continue
        external = EXTERNAL_CHECKPOINTS.get(name)
        directory = run_dir / "checkpoints" if external is None else base / external
        checkpoint = directory / f"{name}.pt"
        history_path = directory / f"{name}-history.json"
        if not checkpoint.is_file() or not history_path.is_file():
            say(f"  {name}: not trained yet, skipping")
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        model, kind = build_model(name, history)
        load_checkpoint(checkpoint, model=model)
        specs.append(
            StreamSpec(
                name=name,
                model=model.to(device).eval(),
                checkpoint_hash=file_sha256(checkpoint),
                kind=kind,
            )
        )
        best = history["epochs"][history["best_epoch"] - 1]
        say(
            f"  {name}: epoch {history['best_epoch']} of {len(history['epochs'])}, "
            f"val loss {best['validation_loss']:.4f}"
        )
    return specs
