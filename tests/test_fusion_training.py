"""Training the feature-level fusion head.

The grouped split is the test worth having. A random split puts the same speaker
on both sides, and then early stopping is tuned against a validation set the
head has effectively seen. Cross-fitting the branch checkpoints exists to keep
that out; a random split here would reintroduce it one layer higher.
"""

import dataclasses

import pytest

pytest.importorskip("torch")
import torch

from deepfake_detection.fusion.store import AssembledFeature
from deepfake_detection.training.fusion import (
    as_tensors,
    fit_stream_fusion,
    group_split,
    stream_dimensions,
)

DIMS = {"lipsync": 4, "visual": 8}


def row(
    clip: str,
    *,
    label: int,
    identity: str,
    streams: dict[str, int] | None = None,
) -> AssembledFeature:
    """One assembled clip whose embeddings separate the classes linearly."""
    present = DIMS if streams is None else streams
    generator = torch.Generator().manual_seed(abs(hash(clip)) % 2**31)
    bias = 1.0 if label else -1.0
    return AssembledFeature(
        dataset="fixture",
        clip_id=clip,
        segment_id="0",
        label=label,
        branch_logits={name: float(bias) for name in present},
        branch_embeddings={
            name: tuple(
                (torch.randn(width, generator=generator) * 0.1 + bias).tolist()
            )
            for name, width in present.items()
        },
        face_coverage=1.0,
        audio_clipped=False,
        av_duration_delta_sec=0.0,
        checkpoint_hashes={name: "hash" for name in present},
        preprocessing_hash="prep",
        split_hash="split",
        run_id="run",
        source_identity=identity,
        method="wav2lip" if label else "real",
        race="unknown",
        gender="unknown",
        available=True,
        missing_branches=(),
        partition_role="oof",
    )


def population(identities: int = 20, per_identity: int = 4):
    return [
        row(
            f"clip-{i}-{j}",
            label=j % 2,
            identity=f"identity-{i}",
        )
        for i in range(identities)
        for j in range(per_identity)
    ]


def test_stream_dimensions_reads_every_stream_width() -> None:
    assert stream_dimensions(population()) == DIMS


def test_inconsistent_widths_are_an_error_not_a_silent_truncation() -> None:
    """Two checkpoints exported into one store, which training cannot recover
    from."""
    rows = population(identities=2)
    wide = dict(rows[0].branch_embeddings)
    wide["visual"] = tuple([0.0] * 16)
    rows[0] = dataclasses.replace(rows[0], branch_embeddings=wide)

    with pytest.raises(ValueError, match="inconsistent embedding widths"):
        stream_dimensions(rows)


def test_rows_without_embeddings_are_an_error() -> None:
    rows = [
        dataclasses.replace(item, branch_embeddings={})
        for item in population(identities=2)
    ]

    with pytest.raises(ValueError, match="No stream embeddings"):
        stream_dimensions(rows)


def test_split_never_puts_one_identity_on_both_sides() -> None:
    train, validation = group_split(population(), fraction=0.25, seed=17)

    left = {item.source_identity for item in train}
    right = {item.source_identity for item in validation}
    assert left & right == set()
    assert left and right


def test_split_refuses_a_fraction_that_would_empty_a_side() -> None:
    with pytest.raises(ValueError, match="Validation fraction"):
        group_split(population(), fraction=1.0, seed=17)


def test_split_reports_when_there_are_too_few_identities() -> None:
    """One identity cannot be split, and the message says so rather than
    returning an empty training set."""
    rows = [row(f"clip-{i}", label=i % 2, identity="only") for i in range(8)]

    with pytest.raises(ValueError, match="left one side empty"):
        group_split(rows, fraction=0.5, seed=17)


def test_a_missing_stream_becomes_a_zero_vector_with_presence_zero() -> None:
    """The abstention policy: a clip one stream cannot read is still a clip."""
    rows = [
        row("with", label=1, identity="a"),
        row("without", label=0, identity="b", streams={"visual": 8}),
    ]

    values, presence, labels = as_tensors(rows, DIMS, "cpu")

    assert presence["lipsync"].tolist() == [1.0, 0.0]
    assert values["lipsync"][1].abs().sum().item() == 0.0
    assert labels.tolist() == [1.0, 0.0]


def test_fusion_learns_a_separable_signal() -> None:
    model, history = fit_stream_fusion(
        rows=population(),
        epochs=60,
        batch_size=32,
        patience=60,
        stream_dropout=0.0,
        seed=17,
    )

    assert history.stream_dims == DIMS
    assert history.train_clips + history.validation_clips == 80
    best = history.epochs[history.best_epoch - 1]
    assert best.validation_auc > 0.9, f"did not learn: AUC {best.validation_auc:.3f}"
    assert isinstance(model.stream_names, tuple)


def test_history_records_every_epoch_it_ran() -> None:
    _model, history = fit_stream_fusion(
        rows=population(), epochs=5, patience=5, seed=17
    )

    assert [record.epoch for record in history.epochs] == [1, 2, 3, 4, 5]
    assert 1 <= history.best_epoch <= 5


def test_early_stopping_ends_the_run_before_the_epoch_budget() -> None:
    _model, history = fit_stream_fusion(
        rows=population(), epochs=200, patience=3, seed=17
    )

    assert len(history.epochs) < 200
