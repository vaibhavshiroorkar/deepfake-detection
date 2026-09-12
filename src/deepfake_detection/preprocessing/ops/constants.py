"""The single source of truth for preprocessing shapes and constants."""

from __future__ import annotations

import numpy as np

NUM_FRAMES = 16
FRAME_SIZE = 224
MOUTH_SIZE = 96
AUDIO_SR = 16000
AUDIO_WINDOW_SEC = 0.35

# Bump whenever the cached crop or audio pixels change, so stale caches are
# detected and re-extracted. v1 = plain MTCNN crop; v2 = 5-point alignment plus
# leading-silence-aware sampling; v3 = aligned crops pad with black; v4 =
# alignment removed, back to a margin-padded bbox crop.
#
# The detector is not a version. It varies per run rather than moving forward,
# so it is stamped beside this number ("4:mtcnn") instead of bumping it.
PIPELINE_VERSION = 4

# Both visual backbones (EfficientNet, DINOv3) are
# ImageNet-pretrained in timm, so face crops are normalized with these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
