"""Small heads on frozen AV-HuBERT features (src/dfd/av/features.py).

AlignmentHead (I-8b) never sees a fake. It learns, on genuine speech only,
whether a lips frame and an audio frame belong together; negatives are the
same lips paired with audio from another time or another clip. A window's
fake score is its mean misfit, -log sigmoid(g), over aligned frame pairs.

WindowHead (I-8c) is supervised on LAV-DF's forged and genuine windows. Each
frame carries the fused feature, both single-modality features, their cosine
and their absolute difference; attention pooling over the 50 frames lets a
short edit dominate the window, then a linear layer gives the logit.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def pair_features(lips: Tensor, audio: Tensor) -> Tensor:
    """[..., width] twice to [..., 4 width]: both, their product and their absolute difference."""
    return torch.cat([lips, audio, lips * audio, (lips - audio).abs()], dim=-1)


class AlignmentHead(nn.Module):
    def __init__(self, width: int = 768, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4 * width), nn.Linear(4 * width, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, lips: Tensor, audio: Tensor) -> Tensor:
        """Logit that the frames belong together, shape [...]."""
        return self.net(pair_features(lips, audio)).squeeze(-1)

    def window_score(self, lips: Tensor, audio: Tensor) -> Tensor:
        """[B, T, width] aligned pairs to a fake score [B]: the mean misfit."""
        return F.softplus(-self(lips, audio)).mean(dim=1)


class WindowHead(nn.Module):
    def __init__(self, width: int = 768, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.frame = nn.Sequential(
            nn.LayerNorm(3 * width + width + 1),
            nn.Linear(3 * width + width + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attend = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, fused: Tensor, lips: Tensor, audio: Tensor) -> Tensor:
        """[B, T, width] three times to logits [B]."""
        cosine = F.cosine_similarity(lips, audio, dim=-1).unsqueeze(-1)
        frames = self.frame(torch.cat([fused, lips, audio, (lips - audio).abs(), cosine], dim=-1))
        weights = torch.softmax(self.attend(frames), dim=1)
        return self.out((weights * frames).sum(dim=1)).squeeze(-1)
