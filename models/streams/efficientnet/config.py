"""
EfficientNet stream = the shared VisualStream template with an EfficientNet-B0
backbone. This file only overrides config; there is no EfficientNet-specific
model code. Stage 3 adds xception/config.py and dinov2/config.py the same way.
"""
from models.streams.common.config import StreamConfig


def efficientnet_config(**overrides) -> StreamConfig:
    base = dict(
        stream_name="efficientnet",
        backbone_name="tf_efficientnet_b0.ns_jft_in1k",  # light: fits 6GB, 1280-dim features
        temporal_type="lstm",
        common_dim=256,
    )
    base.update(overrides)
    return StreamConfig(**base)
