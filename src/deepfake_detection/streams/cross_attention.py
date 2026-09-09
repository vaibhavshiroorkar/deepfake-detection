"""Cross-modal attention: one modality asks, the other answers.

This is the trainable version of the mechanism the dashboard has demonstrated on
random projections since the streams pages were written. `dashboard/lib/
cross_modal.py` is the specification, and `tests/test_cross_attention.py` asserts
this module matches its numpy result to floating-point tolerance, so the two
cannot drift apart.

Why the project wants this at all: a visual-only classifier learns what authentic
video looks like, which is a dataset-specific question, and the measured result
was a model at 1.0000 ROC-AUC in-domain that called 130 of 155 genuine Celeb-DF
videos fake. A cross-modal mismatch is asked and answered *inside* one clip.
There is no dataset-level appearance prior for it to latch onto.

`diagonal_mass` is the reason this module returns its attention weights rather
than just the attended vector. In a genuine recording sound arrives at a
near-fixed offset from the articulation that produced it, so a stream that has
learned anything real concentrates its attention near the diagonal. That makes
the stream checkable on its own, without reference to whether it improves
classification, which is a stronger position than most of this pipeline is in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CrossAttentionOutput:
    weights: Tensor
    attended: Tensor


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor
) -> CrossAttentionOutput:
    """softmax(QK^T / sqrt(d_k)) V for batched sequences.

    Shapes are `[batch, queries, dim]` and `[batch, keys, dim]`, giving weights
    `[batch, queries, keys]` and an attended vector per query step.

    The divisor is not cosmetic: the dot product of two d-dimensional vectors
    with unit-variance components has variance d, so without it the softmax
    saturates towards one-hot as the width grows. Softmax runs over the key
    axis, so each query step's attention over the evidence sums to one.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Query, key and value must be [batch, time, dim]")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must share a feature width")
    if key.shape[1] != value.shape[1]:
        raise ValueError("Key and value must share a time length")

    scores = query @ key.transpose(1, 2) / query.shape[-1] ** 0.5
    weights = scores.softmax(dim=-1)
    return CrossAttentionOutput(weights=weights, attended=weights @ value)


def diagonal_mass(weights: Tensor, band: int = 1) -> Tensor:
    """Fraction of attention mass within `band` steps of the diagonal, per item.

    Ported from `dashboard/lib/cross_modal.py` so the eventual stream and the
    teaching page measure the same thing. The `cols / rows` scaling matters
    whenever the two modalities are sampled at different rates, which is the
    normal case here: a 2 second window is 50 video frames but a different
    number of audio tokens.

    Returns `[batch]`. Not a loss term. A model rewarded for producing diagonal
    attention could learn to produce it without learning synchronisation, which
    would destroy the only independent check the stream has.
    """
    if weights.ndim != 3:
        raise ValueError("Weights must be [batch, queries, keys]")
    rows, cols = weights.shape[1], weights.shape[2]
    r = torch.arange(rows, device=weights.device).unsqueeze(1)
    c = torch.arange(cols, device=weights.device).unsqueeze(0)
    mask = ((r * (cols / rows) - c).abs() <= band).to(weights.dtype)
    return (weights * mask).sum(dim=(1, 2)) / rows


class CrossModalAttention(nn.Module):
    """Learned projections around the attention above.

    One head reproduces the specification exactly, which is what the equivalence
    test pins. More heads give the stream capacity to attend to several kinds of
    correspondence at once; the returned weights are averaged across heads so
    `diagonal_mass` keeps its meaning either way.
    """

    def __init__(self, *, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must divide evenly into {num_heads} heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, values: Tensor) -> Tensor:
        batch, time, _ = values.shape
        return values.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, *, query: Tensor, key: Tensor, value: Tensor) -> CrossAttentionOutput:
        batch, queries, _ = query.shape
        heads_query = self._split(self.query_projection(query))
        heads_key = self._split(self.key_projection(key))
        heads_value = self._split(self.value_projection(value))

        scores = heads_query @ heads_key.transpose(-2, -1) / self.head_dim**0.5
        weights = scores.softmax(dim=-1)
        attended = self.dropout(weights) @ heads_value
        attended = attended.transpose(1, 2).reshape(batch, queries, self.dim)
        return CrossAttentionOutput(
            # Averaged across heads so a multi-head stream still reports one
            # interpretable attention map for diagonal_mass to read.
            weights=weights.mean(dim=1),
            attended=self.output_projection(attended),
        )
