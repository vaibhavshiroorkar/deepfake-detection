"""Feature-level fusion: concatenate stream embeddings, then a small MLP.

The late-fusion path in `late.py` collapses each branch to one calibrated
scalar before combining. That threw away 255 of every 256 dimensions, and it
produced the clearest failure in this project's record: with three scalars it
learned a -3.019 coefficient on the sync branch, whose raw ROC-AUC was 0.4364,
below chance. Platt calibration inverted that signal into a +0.0247 in-domain
gain which vanished cross-corpus, where the same branch scored 0.4959.

So the risk here is not underfitting. It is that a model with more capacity
finds that kind of artifact more easily, and 1,280 input dimensions against
roughly 7,600 out-of-fold clips is where an MLP starts memorising.

`stream_dropout` is the direct answer. During training an entire stream's block
is zeroed at random, so the head cannot become dependent on any single stream
being present and usable. A stream whose only contribution is an invertible
artifact stops paying for itself once the head has to work without it.

Missing streams are zero-filled with a presence flag rather than dropped,
matching the pipeline's abstention policy: a clip that one stream cannot read
is still a clip, and deleting it would quietly shrink the denominator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class StreamFusionOutput:
    logit: Tensor
    presence: Tensor


class StreamFusion(nn.Module):
    """Concatenated per-stream embeddings to one fake probability.

        [visual 256 | lipsync 256 | emotion 256 | ...] -> MLP -> logit

    Each stream is projected before concatenation, so streams whose encoders
    emit different widths still line up, and a stream can be swapped for another
    encoder without changing the head.
    """

    def __init__(
        self,
        *,
        stream_dims: Mapping[str, int],
        common_dim: int = 256,
        hidden_sizes: Sequence[int] = (128,),
        dropout: float = 0.2,
        stream_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not stream_dims:
            raise ValueError("At least one stream is required")
        if not 0.0 <= stream_dropout < 1.0:
            raise ValueError("Stream dropout must be in [0, 1)")
        self.stream_names = tuple(sorted(stream_dims))
        self.common_dim = common_dim
        self.stream_dropout = stream_dropout
        self.projections = nn.ModuleDict(
            {name: nn.Linear(stream_dims[name], common_dim) for name in self.stream_names}
        )
        # One extra input per stream carrying whether it was present, so the
        # head can tell "this stream said nothing" from "this stream said zero".
        width = common_dim * len(self.stream_names) + len(self.stream_names)
        layers: list[nn.Module] = []
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.GELU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        presence: Mapping[str, Tensor] | None = None,
    ) -> StreamFusionOutput:
        missing = [name for name in self.stream_names if name not in embeddings]
        if missing:
            raise ValueError(f"Missing stream embeddings: {', '.join(missing)}")

        blocks: list[Tensor] = []
        flags: list[Tensor] = []
        for name in self.stream_names:
            values = embeddings[name]
            if values.ndim != 2:
                raise ValueError(f"Stream {name} must be [batch, features]")
            present = (
                presence[name].to(values.dtype)
                if presence is not None and name in presence
                else torch.ones(values.shape[0], device=values.device, dtype=values.dtype)
            )
            projected = self.projections[name](values) * present.unsqueeze(1)
            if self.training and self.stream_dropout:
                # Drop the whole stream for a random subset of the batch, not
                # individual features. Feature dropout would leave the head able
                # to lean on the stream's average; this makes it survive the
                # stream being absent entirely.
                keep = (
                    torch.rand(values.shape[0], device=values.device)
                    >= self.stream_dropout
                ).to(values.dtype)
                projected = projected * keep.unsqueeze(1)
                present = present * keep
            blocks.append(projected)
            flags.append(present.unsqueeze(1))

        combined = torch.cat(blocks + flags, dim=1)
        return StreamFusionOutput(
            logit=self.head(combined).squeeze(-1),
            presence=torch.cat(flags, dim=1),
        )
