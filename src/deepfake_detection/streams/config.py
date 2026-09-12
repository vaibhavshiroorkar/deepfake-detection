"""Config for a visual stream. Everything that differs between streams is here.

Backbone, temporal model, dims, freezing and the VRAM knobs are all fields, so
the model code never hardcodes a choice. EfficientNet and DINOv3 are two
StreamConfigs with a different `backbone_name`.
"""

from __future__ import annotations

from dataclasses import dataclass

# timm model ids for the visual backbones the dashboard exposes.
EFFICIENTNET_B0 = "tf_efficientnet_b0.ns_jft_in1k"  # ~5M params, 1280-dim
# A ViT, unlike EfficientNet, so it is built with an explicit img_size: its
# pretrained config is 256 pixels and this pipeline feeds 224. 224/16 = 14, so a
# face crop becomes a 14x14 patch grid, and the prefix is 5 rows (CLS plus 4
# registers) rather than 1.
DINOV3 = "vit_small_patch16_dinov3.lvd1689m"  # ~22M params, 384-dim


@dataclass
class StreamConfig:
    # identity
    stream_name: str = "efficientnet"
    backbone_name: str = EFFICIENTNET_B0
    pretrained: bool = True

    # temporal model: num_frames per-frame embeddings into one clip vector
    temporal_type: str = "lstm"  # "lstm", "gru" or "mean"
    temporal_hidden: int = 256
    temporal_layers: int = 1
    temporal_bidirectional: bool = True

    # projection to the shared space every stream writes into
    common_dim: int = 256

    # How the backbone reduces its spatial or token output to one vector per
    # frame. "avg" suits CNNs and is timm's usual choice. DINOv3 must use
    # "token": its released weights carry `norm`, but timm swaps that for
    # `fc_norm` whenever global_pool="avg", so a strict pretrained load fails.
    global_pool: str = "avg"

    # data
    num_frames: int = 16
    image_size: int = 224
    num_workers: int = 0

    # training, used by the trainer rather than the dashboard
    freeze_backbone: bool = True
    # Two-phase freeze schedule: the backbone stays frozen for the first
    # `freeze_backbone_epochs`, then unfreezes to fine-tune. Set it equal to
    # epochs to keep the backbone frozen for the whole run, or 0 to fine-tune
    # from the start.
    freeze_backbone_epochs: int = 8
    # Even once the backbone is trainable, keep its BatchNorm running stats fixed
    # so tiny fine-tune batches do not corrupt the ImageNet statistics.
    freeze_batchnorm_on_finetune: bool = True
    use_amp: bool = True
    lr_head: float = 1e-3
    lr_backbone: float = 5e-6
    grad_clip_norm: float = 1.0
    epochs: int = 8
    batch_size: int = 2
    grad_accum_steps: int = 8
    weight_decay: float = 1e-4

    # VRAM knobs
    grad_checkpointing: bool = True
    frame_chunk_size: int = 8

    seed: int = 42


def efficientnet_config(**overrides) -> StreamConfig:
    base = {"stream_name": "efficientnet", "backbone_name": EFFICIENTNET_B0}
    base.update(overrides)
    return StreamConfig(**base)


def dinov3_config(**overrides) -> StreamConfig:
    # global_pool="token" is not a tuning choice: with "avg" the pretrained
    # weights refuse to load at all. See StreamConfig.global_pool.
    base = {
        "stream_name": "dinov3",
        "backbone_name": DINOV3,
        "global_pool": "token",
    }
    base.update(overrides)
    return StreamConfig(**base)
