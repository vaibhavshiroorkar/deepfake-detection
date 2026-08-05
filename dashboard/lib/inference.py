"""Turning a clip on disk into the tensor a visual stream takes.

Decodes a clip to a face-crop sequence (reusing dashboard.lib.media, so it
mirrors the preprocessing pipeline) and normalizes it the way the pretrained
backbones expect. Both functions are free of Streamlit so they stay testable
without a running app.

What happens to the tensor afterwards lives elsewhere: the Streams pages build a
model, load a checkpoint if one exists, and trace the forward pass through
models/streams/common/introspect.py.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ImageNet statistics. The default timm backbones (EfficientNet-NS, Xception)
# were pretrained with these. Single source of truth in preprocessing.ops.
from preprocessing.ops.constants import IMAGENET_MEAN, IMAGENET_STD


def frames_to_tensor(faces: list[np.ndarray]):
    """List of T face crops (H, W, 3) uint8 RGB -> tensor [1, T, 3, H, W] float."""
    import torch

    arr = np.stack(faces).astype(np.float32) / 255.0        # [T, H, W, 3]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)           # [T, 3, H, W]
    return t.unsqueeze(0).contiguous()                      # [1, T, 3, H, W]


def decode_face_clip(video_path: str, num_frames: int, detector,
                     conf_thresh: float = 0.9, margin: float = 0.3) -> list[np.ndarray]:
    """Sample num_frames timestamps, decode, detect-and-crop each to a 224 face crop."""
    from dashboard.lib import media

    duration, _ = media.frame_meta(video_path)
    timestamps = media.sample_timestamps(duration, num_frames, 0.35)
    frames = media.decode_frames(video_path, timestamps)
    return [media.detect_and_crop(f, detector, conf_thresh, margin)[0] for f in frames]
