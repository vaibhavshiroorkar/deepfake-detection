"""HSEmotion's facial-expression EfficientNet-B0 as a timm backbone.

HSEmotion (now EmotiEffLib, Apache-2.0) fine-tuned `tf_efficientnet_b0` on
AffectNet's eight expressions. Its release is the whole pickled timm model,
saved under timm 0.9, and those block objects lack attributes timm 1.0 reads
at forward time, so the pickle cannot be run here. Its weights still fit timm
1.0's `tf_efficientnet_b0` exactly, apart from the classifier, which the
release wraps in a Sequential. So the weights are lifted out and loaded into a
freshly built model.

The pickle is opened with `weights_only=True` and an allowlist of the timm and
torch classes it names, so loading it cannot run arbitrary code.
`scripts/check_encoder_ports.py` compares this backbone's logits with the
original model's on the same faces.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

ARCHITECTURE = "tf_efficientnet_b0"
# AffectNet's eight classes, in the order of HSEmotion's classifier.
EXPRESSIONS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)


def _allowed_classes() -> list[type]:
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
    from timm.layers.conv2d_same import Conv2dSame
    from timm.layers.norm_act import BatchNormAct2d
    from timm.models._efficientnet_blocks import (
        DepthwiseSeparableConv,
        InvertedResidual,
        SqueezeExcite,
    )
    from timm.models.efficientnet import EfficientNet

    return [
        SelectAdaptivePool2d,
        Conv2dSame,
        BatchNormAct2d,
        DepthwiseSeparableConv,
        InvertedResidual,
        SqueezeExcite,
        EfficientNet,
        nn.SiLU,
        nn.Sigmoid,
        nn.Sequential,
        nn.Conv2d,
        nn.Flatten,
        nn.Identity,
        nn.Linear,
        nn.AdaptiveAvgPool2d,
        set,
    ]


def hsemotion_state_dict(path: Path) -> dict[str, Tensor]:
    """The released model's weights, keyed for timm 1.0's `tf_efficientnet_b0`."""
    with torch.serialization.safe_globals(_allowed_classes()):
        released = torch.load(path, map_location="cpu", weights_only=True)
    state = released.state_dict() if isinstance(released, nn.Module) else released
    return {key.replace("classifier.0.", "classifier."): v for key, v in state.items()}


def build_hsemotion(path: Path, *, with_classifier: bool = False) -> nn.Module:
    """The expression model. Without the classifier it pools to 1280 features
    per image, which is what a stream backbone returns."""
    import timm

    state = hsemotion_state_dict(path)
    if with_classifier:
        model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=8)
    else:
        model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=0)
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
    model.load_state_dict(state, strict=True)
    return model.eval()
