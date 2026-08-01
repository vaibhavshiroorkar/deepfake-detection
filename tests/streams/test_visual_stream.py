"""Build the real config-driven visual stream and forward-pass it.

pretrained=False keeps the test offline and fast; grad_checkpointing off avoids
a backward-only feature warning. We check shapes, the two backbones, all three
temporal modes, and that freezing actually drops trainable params.
"""
import pytest
import torch

from models.streams.common.config import (
    StreamConfig, efficientnet_config, xception_config, dinov2_config,
    EFFICIENTNET_B0, XCEPTION, DINOV2,
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
    assert dinov2_config().backbone_name == DINOV2


def test_dinov2_preset_pools_the_class_token():
    """Offline guard for the bug that made dinov2_config() unusable.

    global_pool="avg" makes timm build `fc_norm` where DINOv2's released weights
    carry `norm`, so the pretrained load fails with a state_dict key mismatch.
    The CNNs must keep "avg".
    """
    assert dinov2_config().global_pool == "token"
    assert efficientnet_config().global_pool == "avg"
    assert xception_config().global_pool == "avg"


def _skip_if_weights_unreachable(exc: Exception):
    """Fail on a genuine load error; skip only when the weights can't be fetched.

    The distinction is the whole point of the test below: a state_dict key
    mismatch is the regression being guarded and must fail loudly, while a
    machine with no network should not report a false failure.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    offline_signs = ("connection", "timeout", "timed out", "network", "dns",
                     "temporarily unavailable", "max retries", "404", "403",
                     "rate limit", "offline", "resolve")
    if any(sign in text for sign in offline_signs):
        pytest.skip(f"pretrained weights unreachable: {exc}")
    raise exc


@pytest.mark.parametrize("preset", [efficientnet_config, xception_config, dinov2_config])
def test_pretrained_weights_actually_load(preset):
    """Build each backbone WITH pretrained weights.

    Every other test here passes pretrained=False to stay offline and fast --
    which is exactly why the DINOv2 pooling bug shipped undetected: the
    architecture was fine, and only the pretrained load path was broken. This
    test is slower and needs the weights (cached after first run), but it is the
    only one that exercises the path the trainer actually uses.
    """
    config = preset(grad_checkpointing=False, frame_chunk_size=0)
    assert config.pretrained is True, "the preset must default to pretrained weights"
    try:
        model = VisualStream(config).eval()
    except Exception as e:                      # noqa: BLE001 - re-raised unless offline
        _skip_if_weights_unreachable(e)
        return
    with torch.no_grad():
        logit, emb = model(torch.randn(1, 2, 3, config.image_size, config.image_size))
    assert logit.shape == (1,)
    assert emb.shape == (1, 256)


def test_dinov2_feature_dim_matches_the_documented_384():
    """The cached preprocessing and the fusion input size both assume 384-d.

    Pooling choice changes what the backbone emits, so this pins the dimension
    the rest of the pipeline is written against.
    """
    try:
        model = VisualStream(dinov2_config(grad_checkpointing=False, frame_chunk_size=0))
    except Exception as e:                      # noqa: BLE001 - re-raised unless offline
        _skip_if_weights_unreachable(e)
        return
    assert model.feature_dim == 384
