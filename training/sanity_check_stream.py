"""
Forward-pass sanity check for a visual stream. NO TRAINING -- this only proves
the model instantiates, runs one real batch on the GPU, produces the right
shapes (embedding [B, common_dim], logit [B]), and fits in VRAM. Run this
before committing to a full training run.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from models.streams.efficientnet.config import efficientnet_config
from models.streams.common.visual_stream import VisualStream
from preprocessing.dataset import make_dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = efficientnet_config(batch_size=2)
    print(f"Device: {device}")

    model = VisualStream(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built: {config.backbone_name}, feature_dim={model.feature_dim}, "
          f"params={n_params/1e6:.1f}M")

    loader = make_dataloader("train", batch_size=config.batch_size, shuffle=False,
                             label_mode="visual")
    frames, audio, labels, clip_ids = next(iter(loader))
    frames = frames.to(device)
    print(f"Input frames: {tuple(frames.shape)} on {frames.device}")

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=config.use_amp):
        logit, embedding = model(frames)

    print(f"logit shape:     {tuple(logit.shape)}      (expected [{config.batch_size}])")
    print(f"embedding shape: {tuple(embedding.shape)}  (expected [{config.batch_size}, {config.common_dim}])")
    print(f"labels (visual): {labels.tolist()}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Peak VRAM (forward only): {peak:.2f} GB / {total:.1f} GB")

    assert tuple(embedding.shape) == (config.batch_size, config.common_dim)
    assert tuple(logit.shape) == (config.batch_size,)
    print("\nPASS: forward pass produces correct shapes. No training was performed.")


if __name__ == "__main__":
    main()
