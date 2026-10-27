import pytest
import torch
from torch import nn

from deepfake_detection.streams.cross_modal_stream import (
    build_emotion_stream,
    build_lipsync_stream,
)
from deepfake_detection.streams.hsemotion import hsemotion_state_dict


def test_hsemotion_release_loads_through_the_allowlist(tmp_path):
    import timm

    # The release is a whole pickled timm model with its classifier wrapped in
    # a Sequential.
    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=8)
    model.classifier = nn.Sequential(nn.Linear(1280, 8))
    path = tmp_path / "release.pt"
    torch.save(model, path)
    state = hsemotion_state_dict(path)
    assert "classifier.weight" in state and "classifier.0.weight" not in state
    rebuilt = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=8)
    rebuilt.load_state_dict(state, strict=True)


def test_avhubert_lipsync_stream_trains_only_its_head():
    stream = build_lipsync_stream(pretrained=False, encoder="avhubert")
    stream.set_backbone_trainable(True)
    stream.train()
    # Frozen and in eval mode even inside a training-mode stream, so the
    # ResNet's BatchNorm statistics never move.
    assert not stream.pair_encoder.model.training
    assert not any(p.requires_grad for p in stream.pair_encoder.parameters())
    video = torch.randn(1, 10, 3, 112, 112)
    audio = torch.randn(1, 6_400)
    output = stream(video=video, audio=audio)
    assert output.embedding.shape == (1, 256)
    output.logit.sum().backward()
    assert stream.audio_projection.weight.grad is not None


def test_hsemotion_emotion_stream_is_the_same_architecture():
    hsemotion = build_emotion_stream(pretrained=False, encoder="hsemotion")
    default = build_emotion_stream(pretrained=False)
    shapes = [p.shape for p in hsemotion.video_encoder.parameters()]
    assert shapes == [p.shape for p in default.video_encoder.parameters()]


def test_streams_reject_an_encoder_meant_for_the_other():
    with pytest.raises(ValueError):
        build_lipsync_stream(pretrained=False, encoder="hsemotion")
    with pytest.raises(ValueError):
        build_emotion_stream(pretrained=False, encoder="avhubert")
