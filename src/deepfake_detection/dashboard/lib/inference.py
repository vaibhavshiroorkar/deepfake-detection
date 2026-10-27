"""Turning a clip on disk into the tensor a visual stream takes.

Decodes a clip to a face-crop sequence (reusing dashboard.lib.media, so it
mirrors the preprocessing pipeline) and normalizes it the way the pretrained
backbones expect. Both functions are free of Streamlit so they stay testable
without a running app.

What happens to the tensor afterwards lives elsewhere: the Streams pages build a
model, load a checkpoint if one exists, and trace the forward pass through
deepfake_detection/streams/introspect.py.
"""

from pathlib import Path

import numpy as np

# ImageNet statistics. The default timm backbone (EfficientNet-NS)
# were pretrained with these. Single source of truth in preprocessing.ops.
from deepfake_detection.preprocessing.ops.constants import IMAGENET_MEAN, IMAGENET_STD


def frames_to_tensor(faces: list[np.ndarray]):
    """List of T face crops (H, W, 3) uint8 RGB -> tensor [1, T, 3, H, W] float."""
    import torch

    arr = np.stack(faces).astype(np.float32) / 255.0  # [T, H, W, 3]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T, 3, H, W]
    return t.unsqueeze(0).contiguous()  # [1, T, 3, H, W]


def decode_face_clip(
    video_path: str,
    num_frames: int,
    detector,
    conf_thresh: float = 0.9,
    margin: float = 0.3,
) -> list[np.ndarray]:
    """Sample num_frames timestamps, decode, detect-and-crop each to a 224 face crop."""
    from deepfake_detection.dashboard.lib import media

    duration, _ = media.frame_meta(video_path)
    timestamps = media.sample_timestamps(duration, num_frames, 0.35)
    frames = media.decode_frames(video_path, timestamps)
    return [media.detect_and_crop(f, detector, conf_thresh, margin)[0] for f in frames]


def pipeline_face_clip(video_path: str):
    """The pipeline's own visual view: (crops as uint8 RGB, tensor) or (None, None).

    Built through `views.preprocessor`, as the trained checkpoints were, with the
    streams' code version so the preprocessing hash matches theirs.
    """
    import torch

    from deepfake_detection.dashboard.configuration import dashboard_defaults
    from deepfake_detection.inference.loading import build_preprocessor
    from deepfake_detection.inference.predictor import _InferenceRecord
    from deepfake_detection.paths import PROJECT_ROOT
    from deepfake_detection.training.augment import to_pixels

    code_version = dashboard_defaults(root=PROJECT_ROOT).stream_code_version
    preprocessor = build_preprocessor(code_version=code_version, device="cuda")
    prepared = preprocessor.prepare_visual(
        _InferenceRecord(clip_id="dashboard"), Path(video_path)
    )
    if prepared.visual_view is None:
        return None, None
    view = np.asarray(prepared.visual_view, dtype=np.float32)
    crops = list(to_pixels(view))
    return crops, torch.from_numpy(view).unsqueeze(0).contiguous()
