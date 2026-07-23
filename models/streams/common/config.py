"""Config for a visual stream — everything that differs between streams lives here.

Which backbone, temporal model, dims, freezing, VRAM knobs are all fields so the
model code never hardcodes a choice. Xception and EfficientNet are just two
StreamConfigs with a different `backbone_name` (see the preset helpers below).
"""
from dataclasses import dataclass

# timm model ids for the two visual backbones the dashboard exposes.
EFFICIENTNET_B0 = "tf_efficientnet_b0.ns_jft_in1k"   # ~5M params, light, 1280-dim
XCEPTION = "legacy_xception"                          # ~22M params, 2048-dim


@dataclass
class StreamConfig:
    # --- identity ---
    stream_name: str = "efficientnet"
    backbone_name: str = EFFICIENTNET_B0
    pretrained: bool = True

    # --- temporal model: turns num_frames per-frame embeddings into one clip vector ---
    temporal_type: str = "lstm"                 # "lstm", "gru" or "mean"
    temporal_hidden: int = 256
    temporal_layers: int = 1
    temporal_bidirectional: bool = True

    # --- projection to the shared space every stream writes into ---
    common_dim: int = 256                       # PROJECT_OVERVIEW.md fixes this at 256

    # --- data ---
    num_frames: int = 16
    image_size: int = 224

    # --- training (used by the background trainer, not by the dashboard) ---
    freeze_backbone: bool = True
    lr_head: float = 1e-3
    lr_backbone: float = 5e-6
    grad_clip_norm: float = 1.0
    epochs: int = 8
    batch_size: int = 2
    grad_accum_steps: int = 8
    weight_decay: float = 1e-4

    # --- VRAM knobs ---
    grad_checkpointing: bool = True
    frame_chunk_size: int = 8

    seed: int = 42


def efficientnet_config(**overrides) -> StreamConfig:
    base = dict(stream_name="efficientnet", backbone_name=EFFICIENTNET_B0)
    base.update(overrides)
    return StreamConfig(**base)


def xception_config(**overrides) -> StreamConfig:
    base = dict(stream_name="xception", backbone_name=XCEPTION)
    base.update(overrides)
    return StreamConfig(**base)
