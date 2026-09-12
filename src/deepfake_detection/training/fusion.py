"""Train `StreamFusion` over assembled out-of-fold stream embeddings.

The sklearn path in `fusion/late.py` needed no validation split: logistic
regression on three scalars has a closed-form fit and nothing to stop early. An
MLP over roughly 1,280 input dimensions and 7,600 clips does, so this carves a
validation slice out of the out-of-fold rows.

That slice is grouped by source identity, not sampled at random. Splitting
FakeAVCeleb at random puts the same speaker on both sides, and a head that has
seen a speaker's other clips is scored on recognising the speaker. The whole
point of cross-fitting the branch checkpoints was to keep that out, and a random
split here would put it straight back in one layer higher.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from deepfake_detection.fusion.deep import StreamFusion
from deepfake_detection.fusion.store import AssembledFeature


@dataclass(frozen=True, slots=True)
class FusionEpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_auc: float


@dataclass(frozen=True, slots=True)
class FusionTrainingHistory:
    epochs: tuple[FusionEpochRecord, ...]
    best_epoch: int
    train_clips: int
    validation_clips: int
    stream_dims: dict[str, int]


def stream_dimensions(rows: Sequence[AssembledFeature]) -> dict[str, int]:
    """The embedding width of every stream present, checked for agreement.

    A stream whose width changes between rows means two different checkpoints
    were exported into one store, which no amount of training recovers from.
    """
    widths: dict[str, set[int]] = {}
    for row in rows:
        for name, embedding in row.branch_embeddings.items():
            widths.setdefault(name, set()).add(len(embedding))
    disagree = {name: sorted(sizes) for name, sizes in widths.items() if len(sizes) > 1}
    if disagree:
        raise ValueError(f"Streams have inconsistent embedding widths: {disagree}")
    if not widths:
        raise ValueError("No stream embeddings in these rows")
    return {name: sizes.pop() for name, sizes in sorted(widths.items())}


def group_split(
    rows: Sequence[AssembledFeature], *, fraction: float, seed: int
) -> tuple[list[AssembledFeature], list[AssembledFeature]]:
    """Hold out whole source identities, never individual clips."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("Validation fraction must be in (0, 1)")
    identities = sorted({row.source_identity for row in rows})
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(identities), generator=generator).tolist()
    held = {identities[i] for i in order[: max(1, round(len(identities) * fraction))]}
    train = [row for row in rows if row.source_identity not in held]
    validation = [row for row in rows if row.source_identity in held]
    if not train or not validation:
        raise ValueError(
            f"Grouped split left one side empty: {len(identities)} identities, "
            f"{len(train)} train and {len(validation)} validation clips"
        )
    return train, validation


def as_tensors(
    rows: Sequence[AssembledFeature], dims: dict[str, int], device: str
) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
    """Rows to (embeddings, presence, labels).

    A stream a clip lacks becomes a zero vector with presence 0, rather than the
    clip being dropped. Dropping it would shrink the denominator, which is what
    the abstention policy exists to prevent, and it is also the case the head
    most needs to have trained on.
    """
    values: dict[str, Tensor] = {}
    presence: dict[str, Tensor] = {}
    for name, width in dims.items():
        block = torch.zeros(len(rows), width)
        flags = torch.zeros(len(rows))
        for index, row in enumerate(rows):
            embedding = row.branch_embeddings.get(name)
            if embedding:
                block[index] = torch.tensor(embedding, dtype=torch.float32)
                flags[index] = 1.0
        values[name] = block.to(device)
        presence[name] = flags.to(device)
    labels = torch.tensor([float(row.label) for row in rows], device=device)
    return values, presence, labels


def _auc(scores: Tensor, labels: Tensor) -> float:
    """ROC-AUC by rank, so the training loop needs no sklearn import."""
    positive = labels > 0.5
    count_positive = int(positive.sum())
    count_negative = int(labels.numel() - count_positive)
    if count_positive == 0 or count_negative == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - count_positive * (count_positive + 1) / 2) / (
        count_positive * count_negative
    )



# What an input can drive, and how often each case is expected at inference.
# A video reaches every stream; a still image reaches only the visual ones; a
# sound file only the audio branch. `dashboard/lib/media_kind.py` is the same
# table on the serving side.
#
# This exists because i.i.d. stream dropout does not produce these patterns
# often enough to train on. With four streams at p=0.2 the chance a row happens
# to look like "visual only", which is every image upload, is 0.8 * 0.2^3, about
# 49 rows of 7,621. The head is regularised against any one stream vanishing and
# is effectively untrained on the regimes it will actually be asked to run in.
VIDEO = "video"
IMAGE = "image"
AUDIO = "audio"

# Which stream names belong to which modality. Matched by substring so a stream
# called "visual-dinov3" or "final-audio-seed17" lands in the right group
# without the trainer carrying a registry of checkpoint names.
MODALITY_TOKENS = {
    IMAGE: ("visual",),
    AUDIO: ("audio",),
}

DEPLOYMENT_PATTERNS = {VIDEO: 0.7, IMAGE: 0.15, AUDIO: 0.15}


def pattern_masks(
    dims: dict[str, int], patterns: Mapping[str, float]
) -> tuple[list[str], Tensor, Tensor]:
    """Per-pattern stream masks and their sampling weights.

    Returns the pattern names, a `[patterns, streams]` mask of 1 where a stream
    is reachable, and the normalised weights to sample patterns with.

    A pattern that reaches no stream at all is dropped: it would train the head
    to produce a confident number from an all-zero input, which is the one case
    where it should abstain instead.
    """
    names = sorted(dims)
    kept: list[str] = []
    rows: list[list[float]] = []
    weights: list[float] = []
    for pattern, weight in patterns.items():
        if weight <= 0:
            continue
        tokens = MODALITY_TOKENS.get(pattern)
        mask = [
            1.0 if tokens is None or any(t in name for t in tokens) else 0.0
            for name in names
        ]
        if not any(mask):
            continue
        kept.append(pattern)
        rows.append(mask)
        weights.append(float(weight))
    if not kept:
        raise ValueError("No presence pattern reaches any stream")
    total = sum(weights)
    return (
        kept,
        torch.tensor(rows, dtype=torch.float32),
        torch.tensor([w / total for w in weights], dtype=torch.float32),
    )


def fit_stream_fusion(
    *,
    rows: Sequence[AssembledFeature],
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    stream_dropout: float = 0.0,
    dropout: float = 0.2,
    presence_patterns: Mapping[str, float] | None = None,
    hidden_sizes: Sequence[int] = (128,),
    common_dim: int = 256,
    validation_fraction: float = 0.2,
    patience: int = 20,
    seed: int = 17,
    device: str = "cpu",
) -> tuple[StreamFusion, FusionTrainingHistory]:
    torch.manual_seed(seed)
    dims = stream_dimensions(rows)
    train_rows, validation_rows = group_split(
        rows, fraction=validation_fraction, seed=seed
    )
    train = as_tensors(train_rows, dims, device)
    validation = as_tensors(validation_rows, dims, device)

    model = StreamFusion(
        stream_dims=dims,
        common_dim=common_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        stream_dropout=stream_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # FakeAVCeleb runs about ten to one fake, so an unweighted loss is minimised
    # by leaning on the majority class.
    labels = train[2]
    positive = float(labels.sum())
    negative = float(labels.numel() - positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / max(positive, 1.0), device=device)
    )

    # Sampled per row per epoch, so over a run the head sees every regime at
    # roughly its configured rate rather than relying on dropout to stumble into
    # them. Multiplied into the real presence flags, never replacing them: a
    # stream a clip genuinely lacks stays absent whatever pattern is drawn.
    patterns = DEPLOYMENT_PATTERNS if presence_patterns is None else presence_patterns
    pattern_names, masks, pattern_weights = pattern_masks(dims, patterns)
    stream_order = sorted(dims)

    records: list[FusionEpochRecord] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(labels.numel(), generator=generator)
        losses: list[float] = []
        for start in range(0, order.numel(), batch_size):
            index = order[start : start + batch_size].to(device)
            drawn = torch.multinomial(
                pattern_weights, index.numel(), replacement=True, generator=generator
            )
            chosen = masks[drawn].to(device)
            output = model(
                {name: block[index] for name, block in train[0].items()},
                {
                    name: flags[index] * chosen[:, position]
                    for position, name in enumerate(stream_order)
                    for flags in (train[1][name],)
                },
            )
            loss = criterion(output.logit, labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        with torch.inference_mode():
            output = model(validation[0], validation[1])
            validation_loss = float(criterion(output.logit, validation[2]))
            auc = _auc(output.logit, validation[2])
        records.append(
            FusionEpochRecord(
                epoch=epoch + 1,
                train_loss=sum(losses) / len(losses),
                validation_loss=validation_loss,
                validation_auc=auc,
            )
        )
        if validation_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Fusion training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    return model, FusionTrainingHistory(
        epochs=tuple(records),
        best_epoch=best_epoch,
        train_clips=len(train_rows),
        validation_clips=len(validation_rows),
        stream_dims=dims,
    )
