import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.streams import introspect
from deepfake_detection.streams.config import (
    DINOV3,
    EFFICIENTNET_B0,
    XCEPTION,
    StreamConfig,
    dinov3_config,
    efficientnet_config,
    xception_config,
)
from deepfake_detection.streams.visual_stream import build_visual_stream


def tiny_config(**overrides) -> StreamConfig:
    base = {
        "pretrained": False,
        "num_frames": 4,
        "frame_chunk_size": 2,
        "grad_checkpointing": False,
    }
    base.update(overrides)
    return efficientnet_config(**base)


def clip(frames: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(1, frames, 3, 224, 224, generator=generator)


def test_presets_carry_their_own_backbone() -> None:
    assert efficientnet_config().backbone_name == EFFICIENTNET_B0
    assert xception_config().backbone_name == XCEPTION
    assert dinov3_config().backbone_name == DINOV3


def test_dinov3_preset_pools_over_the_class_token() -> None:
    # With "avg" timm builds fc_norm in place of norm and the released DINOv3
    # weights refuse to load, so this is a correctness constraint, not a taste.
    assert dinov3_config().global_pool == "token"
    assert efficientnet_config().global_pool == "avg"


def test_forward_returns_a_logit_and_an_embedding() -> None:
    model = build_visual_stream(tiny_config(common_dim=128)).eval()
    with torch.no_grad():
        logit, embedding = model(clip())
    assert logit.shape == (1,)
    assert embedding.shape == (1, 128)


def test_mean_pooling_skips_the_temporal_model() -> None:
    model = build_visual_stream(tiny_config(temporal_type="mean")).eval()
    assert model.temporal is None
    with torch.no_grad():
        logit, embedding = model(clip())
    assert logit.shape == (1,)


def test_an_unknown_temporal_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="temporal_type"):
        build_visual_stream(tiny_config(temporal_type="transformer"))


def test_freezing_the_backbone_shrinks_the_trainable_count() -> None:
    frozen = build_visual_stream(tiny_config(freeze_backbone=True)).param_counts()
    trainable = build_visual_stream(tiny_config(freeze_backbone=False)).param_counts()
    assert frozen["trainable"] < trainable["trainable"]
    assert frozen["total"] == trainable["total"]


def test_chunking_does_not_change_the_output() -> None:
    whole = build_visual_stream(tiny_config(frame_chunk_size=0)).eval()
    chunked = build_visual_stream(tiny_config(frame_chunk_size=2)).eval()
    chunked.load_state_dict(whole.state_dict())
    frames = clip()
    with torch.no_grad():
        assert torch.allclose(whole(frames)[1], chunked(frames)[1], atol=1e-5)


def test_token_grid_prefers_the_smallest_prefix() -> None:
    assert introspect.token_grid(201) == (5, 14)
    assert introspect.token_grid(197) == (1, 14)


def test_token_grid_rejects_a_count_that_is_not_a_grid() -> None:
    with pytest.raises(ValueError, match="do not decompose"):
        introspect.token_grid(115)


def test_trace_records_every_backbone_stage() -> None:
    model = build_visual_stream(tiny_config()).eval()
    trace = introspect.trace_visual_stream(model, clip(), detail_frame=1)
    assert trace.stages
    assert all(stage.kind == introspect.SPATIAL for stage in trace.stages)
    assert trace.frame_features.shape[0] == 4
    assert trace.embedding.shape == (256,)
    assert 0.0 <= trace.prob <= 1.0
    assert trace.detail_frame == 1


def test_trace_summary_covers_every_frame() -> None:
    model = build_visual_stream(tiny_config()).eval()
    trace = introspect.trace_visual_stream(model, clip())
    for stage in trace.stages:
        assert stage.summary.shape[0] == 4


def test_trace_needs_exactly_one_clip() -> None:
    model = build_visual_stream(tiny_config()).eval()
    with pytest.raises(ValueError, match="one clip"):
        introspect.trace_visual_stream(model, torch.randn(2, 4, 3, 224, 224))


def test_trace_rejects_a_detail_frame_outside_the_clip() -> None:
    model = build_visual_stream(tiny_config()).eval()
    with pytest.raises(ValueError, match="outside"):
        introspect.trace_visual_stream(model, clip(), detail_frame=9)


def test_trace_clip_vector_matches_the_model_embedding() -> None:
    # The tracer rebuilds the clip vector from the RNN's final state rather than
    # hooking the projection, so this is the check that the two paths agree.
    model = build_visual_stream(tiny_config()).eval()
    trace = introspect.trace_visual_stream(model, clip())
    with torch.no_grad():
        rebuilt = model.projection(torch.from_numpy(trace.clip_vector).unsqueeze(0))
    assert np.allclose(rebuilt.squeeze(0).numpy(), trace.embedding, atol=1e-4)


def test_normalize01_maps_a_constant_map_to_zero() -> None:
    assert np.all(introspect.normalize01(np.full((4, 4), 3.0)) == 0.0)


def test_top_channels_ranks_by_mean_response() -> None:
    detail = np.zeros((3, 2, 2), dtype=np.float32)
    detail[1] = 5.0
    detail[2] = 1.0
    assert introspect.top_channels(detail, 2) == [1, 2]


def test_top_channels_needs_a_spatial_activation() -> None:
    with pytest.raises(ValueError, match=r"\[C, H, W\]"):
        introspect.top_channels(np.zeros((4, 4)), 1)


def test_token_maps_come_back_as_a_square_grid() -> None:
    detail = np.random.default_rng(3).random((201, 8)).astype(np.float32)
    assert introspect.patch_token_map(detail).shape == (14, 14)
    assert introspect.cls_similarity_map(detail).shape == (14, 14)


def test_cls_similarity_needs_a_prefix_token() -> None:
    detail = np.random.default_rng(3).random((196, 8)).astype(np.float32)
    with pytest.raises(ValueError, match="no prefix token"):
        introspect.cls_similarity_map(detail)


def test_batchnorm_stays_in_eval_mode_while_the_module_trains() -> None:
    """`freeze_batchnorm_on_finetune` was documented in StreamConfig and
    implemented nowhere. Without it, fine-tuning rewrites all 49 running means
    and variances from batches of eight drawn by an inverse-frequency sampler,
    which matches neither the ImageNet statistics the weights came from nor the
    distribution validation is drawn from. Measured cost: 0.5154 validation AUC
    against 0.9058 with it applied."""
    from torch.nn.modules.batchnorm import _BatchNorm

    model = build_visual_stream(
        efficientnet_config(pretrained=False, common_dim=32)
    ).train()

    frozen = [
        module
        for module in model.backbone.modules()
        if isinstance(module, _BatchNorm)
    ]
    assert frozen, "fixture backbone has no BatchNorm to freeze"
    assert not any(module.training for module in frozen)
    # Only the backbone: the head still has to train.
    assert model.temporal.training
    assert model.projection.training


def test_batchnorm_freezing_can_be_turned_off() -> None:
    from torch.nn.modules.batchnorm import _BatchNorm

    model = build_visual_stream(
        efficientnet_config(
            pretrained=False, common_dim=32, freeze_batchnorm_on_finetune=False
        )
    ).train()

    assert all(
        module.training
        for module in model.backbone.modules()
        if isinstance(module, _BatchNorm)
    )
