"""
Config for a visual stream.

Everything that differs between streams (which backbone, LSTM vs GRU, dims,
freezing, VRAM knobs) lives here so the model/training code never hardcodes a
choice. Stage 3 builds the Xception and DINOv2 streams by writing a new
StreamConfig with a different `backbone_name` -- no model-code changes.

The requirement "config-driven, not hardcoded" (see the project brief) is why
this is a dataclass passed everywhere rather than constants scattered in code.
"""
from dataclasses import dataclass, field


@dataclass
class StreamConfig:
    # --- identity ---
    stream_name: str = "efficientnet"          # feature-store key; also the checkpoint folder name
    # timm model id. tf_efficientnet_b0 is light (1280-dim, ~5M params) -- a
    # deliberate choice for the 6GB laptop GPU. Swap this string for Xception
    # ("legacy_xception") or DINOv2 ("vit_small_patch14_dinov2.lvd142m") in Stage 3.
    backbone_name: str = "tf_efficientnet_b0.ns_jft_in1k"
    pretrained: bool = True

    # --- temporal model (turns 16 per-frame embeddings into 1 clip embedding) ---
    temporal_type: str = "lstm"                # "lstm" or "gru"
    temporal_hidden: int = 256
    temporal_layers: int = 1
    temporal_bidirectional: bool = True         # read the frame sequence both directions

    # --- projection to the shared space every stream writes into ---
    common_dim: int = 256                       # PROJECT_OVERVIEW.md fixes this at 256

    # --- data ---
    num_frames: int = 16                        # must match preprocessing/extract_clip.py NUM_FRAMES
    image_size: int = 224

    # --- freezing / fine-tuning (two-phase schedule) ---
    # Phase 1: freeze backbone, train temporal+projection+head (fast, stable).
    # Phase 2: unfreeze backbone, fine-tune end-to-end at a lower LR.
    freeze_backbone_epochs: int = 2             # how many epochs to keep the backbone frozen first
    lr_head: float = 1e-3                       # LR for temporal/projection/head
    lr_backbone: float = 5e-6                   # LR for the backbone once unfrozen (much smaller)

    # Stability for the unfreeze step (see training/train_visual_stream.py):
    grad_clip_norm: float = 1.0                 # clip global grad norm -- kills the unfreeze loss spike
    freeze_batchnorm_on_finetune: bool = True   # keep backbone BatchNorm in eval() while fine-tuning:
                                                # with tiny batches, updating BN running stats shifts the
                                                # feature distribution and destabilizes the LSTM/head.

    # --- training ---
    epochs: int = 8
    batch_size: int = 2                         # small: 6GB VRAM with 16 frames/clip
    grad_accum_steps: int = 8                   # effective batch = batch_size * grad_accum_steps = 16
    weight_decay: float = 1e-4
    num_workers: int = 0                        # Windows + heavy __getitem__: keep 0 unless proven safe

    # --- 6GB-VRAM knobs ---
    use_amp: bool = True                        # mixed precision: ~halves activation memory
    grad_checkpointing: bool = True             # recompute backbone activations in backward to save VRAM
    frame_chunk_size: int = 8                   # run the backbone on <=8 frames at a time

    # --- label convention for a VISUAL-only stream (see docs/stage-2-plan.md) ---
    # A visual stream sees only frames, so its label is the authenticity of the
    # VIDEO track: FakeVideo-* -> 1 (fake), RealVideo-* -> 0 (real, even
    # RealVideo-FakeAudio, whose fakeness is audio-only). This is intentional.
    visual_fake_types: tuple = ("FakeVideo-RealAudio", "FakeVideo-FakeAudio")

    seed: int = 42
