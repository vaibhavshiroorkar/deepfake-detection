from __future__ import annotations

from torch import Tensor, nn

from .contracts import BranchOutput


def _flatten_features(features: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    if isinstance(features, (tuple, list)):
        features = features[-1]
    if features.ndim == 4:
        return features.mean(dim=(-1, -2))
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim != 2:
        raise ValueError(f"Backbone returned unsupported shape {tuple(features.shape)}")
    return features


class VisualArtifactBranch(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        backbone_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.temporal = nn.GRU(backbone_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, frames: Tensor) -> BranchOutput:
        if frames.ndim != 5:
            raise ValueError(
                "Visual input must have shape [batch, time, channels, height, width]"
            )
        batch, time, channels, height, width = frames.shape
        flattened = frames.reshape(batch * time, channels, height, width)
        features = _flatten_features(self.backbone(flattened))
        sequence = features.reshape(batch, time, -1)
        encoded, _ = self.temporal(sequence)
        embedding = encoded[:, -1]
        logits = self.classifier(embedding).squeeze(-1)
        return BranchOutput(logits=logits, embedding=embedding, token_count=time)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


def build_efficientnet_b0(
    *, pretrained: bool = True, hidden_dim: int = 256
) -> VisualArtifactBranch:
    import timm

    backbone = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    return VisualArtifactBranch(
        backbone=backbone,
        backbone_dim=backbone.num_features,
        hidden_dim=hidden_dim,
    )
