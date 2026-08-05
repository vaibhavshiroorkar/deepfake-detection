"""Trace a real visual stream and check what comes back describes the forward pass.

pretrained=False keeps this offline; two frames and a chunk size that splits them
keep it quick while still exercising the accumulate-across-chunks path, which is
where a hook-based trace is most likely to be wrong.
"""
import numpy as np
import pytest
import torch

from models.streams.common.config import (
    StreamConfig, DINOV3, EFFICIENTNET_B0, XCEPTION, dinov3_config,
)
from models.streams.common.visual_stream import build_visual_stream
from models.streams.common import introspect

CNN_BACKBONES = [EFFICIENTNET_B0, XCEPTION]


def _model(**kw):
    base = dict(pretrained=False, num_frames=2, frame_chunk_size=0, temporal_type="lstm")
    base.update(kw)
    return build_visual_stream(StreamConfig(**base)).eval()


def _clip(t=2):
    return torch.randn(1, t, 3, 224, 224)


@pytest.mark.parametrize("backbone", CNN_BACKBONES)
def test_stages_come_from_timm_feature_info(backbone):
    model = _model(backbone_name=backbone)
    stages = introspect.backbone_stages(model.backbone)
    assert stages, "every backbone here publishes feature_info"
    assert all(s.reduction > 0 and s.channels > 0 for s in stages)
    # increasing reduction is what makes them a depth order rather than a set
    assert [s.reduction for s in stages] == sorted(s.reduction for s in stages)
    for stage in stages:
        model.backbone.get_submodule(stage.name)     # raises if the name is stale


@pytest.mark.parametrize("backbone", CNN_BACKBONES)
def test_trace_reports_every_stage_for_every_frame(backbone):
    trace = introspect.trace_visual_stream(_model(backbone_name=backbone), _clip(2), detail_frame=1)
    assert trace.stages
    for stage in trace.stages:
        assert stage.kind == introspect.SPATIAL
        assert stage.summary.shape[0] == 2, "one summary map per frame"
        assert stage.detail.ndim == 3, "the detail frame keeps its channels"
        assert stage.detail.shape == stage.shape
        assert stage.summary.shape[1:] == stage.shape[1:], "summary is the mean over channels"


def test_trace_shapes_follow_the_documented_data_flow():
    model = _model(temporal_hidden=64, common_dim=128)
    trace = introspect.trace_visual_stream(model, _clip(2))
    assert trace.input_shape == (1, 2, 3, 224, 224)
    assert trace.folded_shape == (2, 3, 224, 224)
    assert trace.frame_features.shape == (2, model.feature_dim)
    assert trace.temporal_sequence.shape == (2, 128)      # 64 hidden, bidirectional
    assert trace.clip_vector.shape == (128,)
    assert trace.embedding.shape == (128,)                # common_dim
    assert 0.0 <= trace.prob <= 1.0


def test_traced_clip_vector_reproduces_the_models_own_embedding():
    """The trace rebuilds the clip vector rather than reading it off the model.

    Projecting it has to land back on the embedding the model returned, or the
    trace is describing a forward pass that did not happen.
    """
    model = _model()
    trace = introspect.trace_visual_stream(model, _clip(2))
    with torch.no_grad():
        rebuilt = model.projection(torch.from_numpy(trace.clip_vector).unsqueeze(0))
    assert np.allclose(rebuilt.squeeze(0).numpy(), trace.embedding, atol=1e-5)


def test_mean_pooling_has_no_temporal_sequence():
    trace = introspect.trace_visual_stream(_model(temporal_type="mean"), _clip(2))
    assert trace.temporal_sequence is None
    assert trace.clip_vector.shape == (1280,)      # the backbone width, unpooled


def test_chunked_backbone_still_sees_every_frame():
    """frame_chunk_size splits the forward pass, so each hook fires several times."""
    trace = introspect.trace_visual_stream(_model(num_frames=3, frame_chunk_size=2), _clip(3),
                                           detail_frame=2)
    assert trace.frame_features.shape[0] == 3
    for stage in trace.stages:
        assert stage.summary.shape[0] == 3


def test_grad_checkpointing_does_not_silence_the_hooks():
    """timm runs a checkpointed backbone as one flattened segment, so the stage
    modules are never called and every hook would come back empty."""
    model = _model(backbone_name=EFFICIENTNET_B0, grad_checkpointing=True)
    assert model.grad_checkpointing, "this backbone supports it, so the test is meaningful"
    trace = introspect.trace_visual_stream(model, _clip(2))
    assert all(stage.summary.shape[0] == 2 for stage in trace.stages)
    assert model.grad_checkpointing, "and it is restored afterwards"


def test_xception_does_not_fail_on_unsupported_grad_checkpointing():
    """legacy_xception has the method but asserts on enable; it must not be fatal."""
    model = _model(backbone_name=XCEPTION, grad_checkpointing=True)
    assert model.grad_checkpointing is False


def test_detail_frame_must_be_inside_the_clip():
    with pytest.raises(ValueError, match="detail_frame"):
        introspect.trace_visual_stream(_model(), _clip(2), detail_frame=5)


def test_trace_refuses_a_batch():
    """A stage map averaged over several clips explains none of them."""
    with pytest.raises(ValueError, match=r"\[1, T, 3, H, W\]"):
        introspect.trace_visual_stream(_model(), torch.randn(2, 2, 3, 224, 224))


# ------------------------------------------------------------------- ViT branch

def test_dinov3_builds_at_the_pipeline_resolution_and_traces_as_tokens():
    """The slow one: ~22M params. DINOv3 is a ViT, so its stages are token
    matrices and the spatial helpers do not apply to them."""
    model = build_visual_stream(dinov3_config(pretrained=False, num_frames=2,
                                              frame_chunk_size=0)).eval()
    assert model.feature_dim == 384
    trace = introspect.trace_visual_stream(model, _clip(2), detail_frame=1)
    assert len(trace.stages) == 12
    stage = trace.stages[-1]
    assert stage.kind == introspect.TOKENS
    # patch16 at 224 gives a 14x14 grid, behind CLS plus four register tokens
    assert stage.shape == (201, 384)
    assert stage.summary.shape == (2, 14, 14)
    assert introspect.cls_similarity_map(stage.detail).shape == (14, 14)
    assert introspect.patch_token_map(stage.detail).shape == (14, 14)


def test_dinov3_preset_names_the_vit():
    assert dinov3_config().backbone_name == DINOV3
    assert dinov3_config().stream_name == "dinov3"


# ------------------------------------------------------------- numpy view helpers

def test_token_grid_picks_the_smallest_prefix():
    assert introspect.token_grid(257) == (1, 16)      # CLS + 16x16, not 32 + 15x15
    assert introspect.token_grid(196) == (0, 14)
    assert introspect.token_grid(261) == (5, 16)      # CLS + 4 registers


def test_token_grid_rejects_a_shape_that_is_not_a_grid():
    with pytest.raises(ValueError, match="square grid"):
        introspect.token_grid(110)      # no prefix under 9 leaves a square


def test_normalize01_spans_the_unit_interval():
    out = introspect.normalize01(np.array([[-4.0, 0.0, 4.0]]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_normalize01_maps_a_constant_array_to_zeros():
    """A dead channel is common in an untrained model and must not divide by zero."""
    out = introspect.normalize01(np.full((4, 4), 2.5))
    assert np.array_equal(out, np.zeros((4, 4)))


def test_top_channels_ranks_by_mean_response():
    act = np.zeros((4, 2, 2), dtype=np.float32)
    act[2] = 5.0
    act[0] = 1.0
    act[3, 0, 0] = 10.0         # peaks highest, but its mean is 2.5: must not win
    assert introspect.top_channels(act, 2) == [2, 3]


def test_cls_similarity_map_is_one_where_a_patch_equals_the_cls_token():
    tokens = np.zeros((5, 3), dtype=np.float32)     # CLS + 2x2 patches
    tokens[0] = [1.0, 0.0, 0.0]
    tokens[1] = [2.0, 0.0, 0.0]                     # same direction, different length
    tokens[2] = [0.0, 1.0, 0.0]                     # orthogonal
    sims = introspect.cls_similarity_map(tokens)
    assert sims.shape == (2, 2)
    assert sims[0, 0] == pytest.approx(1.0)
    assert sims[0, 1] == pytest.approx(0.0)
