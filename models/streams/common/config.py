"""Config for a visual stream — everything that differs between streams lives here.

Which backbone, temporal model, dims, freezing, VRAM knobs are all fields so the
model code never hardcodes a choice. Xception and EfficientNet are just two
StreamConfigs with a different `backbone_name` (see the preset helpers below).
"""
from dataclasses import dataclass

# timm model ids for the visual backbones the dashboard exposes.
EFFICIENTNET_B0 = "tf_efficientnet_b0.ns_jft_in1k"   # ~5M params, light, 1280-dim
XCEPTION = "legacy_xception"                          # ~22M params, 2048-dim
# A ViT, unlike the two above, so it is built with an explicit img_size: its
# pretrained config is 256 pixels and this pipeline feeds 224 (see
# visual_stream._create_backbone). 224/16 = 14, so a face crop becomes a 14x14
# patch grid, and the prefix is 5 rows (CLS + 4 registers) rather than 1.
DINOV3 = "vit_small_patch16_dinov3.lvd1689m"         # ~22M params, 384-dim


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

    # How the backbone reduces its spatial/token output to one vector per frame.
    # "avg" suits CNNs and is timm's usual choice. DINOv3 must use "token": its
    # released weights carry `norm`, but timm swaps that for `fc_norm` whenever
    # global_pool="avg", so a strict pretrained load fails outright.
    global_pool: str = "avg"

    # --- data ---
    num_frames: int = 16
    image_size: int = 224
    num_workers: int = 0                        # DataLoader workers; 0 = load in the main process

    # --- training (used by the background trainer, not by the dashboard) ---
    freeze_backbone: bool = True
    # Two-phase freeze schedule (see train_visual_stream.py): the backbone stays
    # frozen for the first `freeze_backbone_epochs`, then unfreezes to fine-tune.
    # Set == epochs to keep the backbone frozen for the WHOLE run (the "keep
    # frozen default" choice), or 0 to fine-tune from the start.
    freeze_backbone_epochs: int = 8
    # Even once the backbone is trainable, keep its BatchNorm running stats fixed
    # (eval mode) so tiny fine-tune batches don't corrupt ImageNet stats.
    freeze_batchnorm_on_finetune: bool = True
    use_amp: bool = True                        # mixed-precision (AMP) to fit in ~6GB VRAM
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


def dinov3_config(**overrides) -> StreamConfig:
    # global_pool="token" is not a tuning choice: with "avg" the pretrained
    # weights refuse to load at all (see StreamConfig.global_pool).
    base = dict(stream_name="dinov3", backbone_name=DINOV3, global_pool="token")
    base.update(overrides)
    return StreamConfig(**base)
