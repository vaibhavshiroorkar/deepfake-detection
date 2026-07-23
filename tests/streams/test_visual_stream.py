"""Build the real config-driven visual stream and forward-pass it.

pretrained=False keeps the test offline and fast; grad_checkpointing off avoids
a backward-only feature warning. We check shapes, the two backbones, all three
temporal modes, and that freezing actually drops trainable params.
"""
import pytest
import torch

from models.streams.common.config import (
    StreamConfig, efficientnet_config, xception_config,
    EFFICIENTNET_B0, XCEPTION,
)
from models.streams.common.visual_stream import VisualStream, build_visual_stream


def _cfg(**kw):
    base = dict(pretrained=False, grad_checkpointing=False, frame_chunk_size=0)
    base.update(kw)
    return StreamConfig(**base)


@pytest.mark.parametrize("backbone", [EFFICIENTNET_B0, XCEPTION])
def test_forward_shapes_for_both_backbones(backbone):
    model = VisualStream(_cfg(backbone_name=backbone, temporal_type="lstm")).eval()
    x = torch.randn(2, 3, 3, 224, 224)          # B=2, T=3 frames
    with torch.no_grad():
        logit, emb = model(x)
    assert logit.shape == (2,)
    assert emb.shape == (2, 256)                 # common_dim


@pytest.mark.parametrize("temporal", ["lstm", "gru", "mean"])
def test_all_temporal_modes_work(temporal):
    model = VisualStream(_cfg(temporal_type=temporal)).eval()
    with torch.no_grad():
        logit, emb = model(torch.randn(1, 2, 3, 224, 224))
    assert logit.shape == (1,) and emb.shape == (1, 256)


def test_freeze_backbone_reduces_trainable_params():
    model = build_visual_stream(_cfg(freeze_backbone=True))
    counts = model.param_counts()
    assert counts["trainable"] < counts["total"]
    assert counts["feature_dim"] == 1280         # efficientnet default


def test_preset_helpers_set_backbone():
    assert efficientnet_config().backbone_name == EFFICIENTNET_B0
    assert xception_config().backbone_name == XCEPTION
