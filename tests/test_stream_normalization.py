"""A visual stream converts the cache's ImageNet normalization to its backbone's own (amendment A11).

The older study scored CLIP and PE on ImageNet-normalized inputs, which those
models were never trained on. The conversion now comes from each backbone's
own pretrained config, so it cannot be forgotten for a new backbone.
"""

import torch

from deepfake_detection.streams.config import CLIP_VIT_L14, StreamConfig, clip_config
from deepfake_detection.streams.visual_stream import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VisualStream,
    renormalize,
)

OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)


def test_renormalize_maps_pixels_from_one_normalization_to_another():
    pixel = torch.tensor([0.2, 0.5, 0.9]).view(1, 3, 1, 1)
    imagenet = (pixel - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    clip = renormalize(imagenet, IMAGENET_MEAN, IMAGENET_STD, OPENAI_MEAN, OPENAI_STD)
    expected = (pixel - torch.tensor(OPENAI_MEAN).view(1, 3, 1, 1)) / torch.tensor(OPENAI_STD).view(1, 3, 1, 1)
    assert torch.allclose(clip, expected, atol=1e-6)


def test_same_normalization_is_left_alone():
    x = torch.randn(2, 3, 4, 4)
    assert torch.equal(renormalize(x, IMAGENET_MEAN, IMAGENET_STD, IMAGENET_MEAN, IMAGENET_STD), x)


def test_clip_preset_names_the_openai_vit_l14():
    config = clip_config()
    assert config.backbone_name == CLIP_VIT_L14 == "vit_large_patch14_clip_224.openai"
    assert config.global_pool == "token"


def test_a_clip_stream_reads_its_normalization_from_the_backbone():
    # A small CLIP ViT keeps the test light; the mechanism is the same for ViT-L.
    config = StreamConfig(stream_name="clip-small", backbone_name="vit_base_patch32_clip_224.openai",
                          pretrained=False, global_pool="token", temporal_type="mean", grad_checkpointing=False)
    stream = VisualStream(config)
    assert torch.allclose(stream.backbone_mean.flatten(), torch.tensor(OPENAI_MEAN))
    assert stream.renormalizes
    assert "backbone_mean" not in stream.state_dict()      # old checkpoints still load


def test_an_imagenet_backbone_is_not_converted():
    config = StreamConfig(stream_name="tiny", backbone_name="resnet18", pretrained=False,
                          temporal_type="mean", grad_checkpointing=False)
    assert not VisualStream(config).renormalizes
