"""The single source of truth for preprocessing shapes and constants.

These were previously copy-pasted across extract_clip.py, dataset.py,
dashboard/lib/media.py, inference.py and visual_ops.py. Import them from here.
"""
import numpy as np

NUM_FRAMES = 16
FRAME_SIZE = 224
MOUTH_SIZE = 96
AUDIO_SR = 16000            # target sample rate every audio window is resampled to
AUDIO_WINDOW_SEC = 0.35     # duration of the audio window centered on each frame

# Bump whenever the CACHED crop/audio pixels change so stale caches are detected
# and re-extracted (see extract_clip._cache_valid). v1 = plain MTCNN crop; v2 =
# 5-point face alignment + leading-silence-aware frame/audio sampling; v3 =
# aligned crops pad with black instead of reflecting (v2 mirrored a second face
# into every crop) and drop the template inset; v4 = alignment removed, back to a
# margin-padded bbox crop (parked in docs/ideas.md).
PIPELINE_VERSION = 4

# ImageNet statistics — all three visual backbones (Xception/EfficientNet/DINOv2)
# are ImageNet-pretrained in timm, so face crops are normalized with these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
