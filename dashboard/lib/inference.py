"""Real-clip inference for the visual streams — the dashboard's Run action.

Decodes a clip to a face-crop sequence (reusing dashboard.lib.media, so it
mirrors the preprocessing pipeline) and forward-passes the config-driven
VisualStream. There is no training and no checkpoint loading here: no trained
weights exist on disk yet, so the returned probability is a plumbing check on an
untrained head, not a real detection.

The tensor/normalization logic is factored out of the Streamlit page (frames_to_
tensor) so it stays unit-testable without a running app or a video on disk.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ImageNet statistics — the default timm backbones (EfficientNet-NS, Xception)
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
    """Sample num_frames timestamps, decode, MTCNN-crop each to a 224 face crop."""
    from dashboard.lib import media

    duration, _ = media.frame_meta(video_path)
    timestamps = media.sample_timestamps(duration, num_frames, 0.35)
    frames = media.decode_frames(video_path, timestamps)
    return [media.detect_and_crop(f, detector, conf_thresh, margin)[0] for f in frames]


def run_visual_stream(config, video_path: str, detector,
                      conf_thresh: float = 0.9, margin: float = 0.3) -> dict:
    """Build the model, run the real clip through it, and report what came out."""
    import torch

    from models.streams.common.visual_stream import build_visual_stream

    model = build_visual_stream(config).eval()
    faces = decode_face_clip(video_path, config.num_frames, detector, conf_thresh, margin)
    x = frames_to_tensor(faces)
    with torch.no_grad():
        logit, embedding = model(x)
    return {
        "counts": model.param_counts(),
        "logit": float(logit.item()),
        "prob": float(torch.sigmoid(logit).item()),
        "embed_shape": tuple(embedding.shape),
        "num_frames": len(faces),
    }
