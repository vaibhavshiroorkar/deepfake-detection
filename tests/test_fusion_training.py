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


def store_with_role(tmp_path, role: str):
    """A small feature store whose rows all carry one partition role."""
    from deepfake_detection.fusion.store import FeatureRecord, FeatureStore

    path = tmp_path / f"{role}.parquet"
    records = [
        FeatureRecord(
            dataset="fixture",
            clip_id=f"clip-{index}",
            segment_id="segment-0",
            branch=branch,
            logit=float(value),
            embedding=(float(value),),
            available=True,
            checkpoint_hash=f"{branch}-checkpoint",
            preprocessing_hash="prep",
            split_hash="split",
            run_id="run",
            label=int(value > 0),
            source_identity=f"id-{index}",
            method="real" if value < 0 else "fixture-fake",
            race="fixture",
            gender="fixture",
            partition_role=role,
        )
        for index, value in enumerate((-4, -3, -2, -1, 1, 2, 3, 4))
        for branch in ("visual", "audio", "sync")
    ]
    FeatureStore(path).write(records)
    return path


def train_fusion_cli(tmp_path, store_path):
    from deepfake_detection.cli import main

    return main(
        [
            "train",
            "fusion",
            "--feature-store",
            str(store_path),
            "--output",
            str(tmp_path / "fusion.joblib"),
            "--metadata",
            str(tmp_path / "fusion.json"),
        ]
    )


def test_holdout_rows_are_accepted_for_fusion(tmp_path) -> None:
    """Streams trained on the training partition have never seen validation, so
    validation rows are as leakage-free as cross-fitted ones and cost one
    training run per stream instead of one per fold."""
    assert train_fusion_cli(tmp_path, store_with_role(tmp_path, "holdout")) == 0


def test_test_partition_rows_are_refused(tmp_path) -> None:
    """Fitting fusion on the test partition is the leak the roles exist to
    prevent."""
    with pytest.raises(ValueError, match="never trained on"):
        train_fusion_cli(tmp_path, store_with_role(tmp_path, "test"))


def test_mixed_roles_are_refused(tmp_path) -> None:
    """Two provenance rules in one store means a clip's row cannot be trusted to
    follow either."""
    from deepfake_detection.fusion.store import FeatureStore

    mixed = tmp_path / "mixed.parquet"
    rows = list(FeatureStore(store_with_role(tmp_path, "oof")).read())
    swapped = [
        dataclasses.replace(row, partition_role="holdout")
        if row.clip_id == "clip-0"
        else row
        for row in rows
    ]
    FeatureStore(mixed).write(swapped)

    with pytest.raises(ValueError, match="mix out-of-fold and holdout"):
        train_fusion_cli(tmp_path, mixed)


def test_pattern_masks_route_each_modality_to_its_streams() -> None:
    """An image reaches only the visual streams, a sound file only the audio
    branch, a video everything. Matched by substring so a checkpoint called
    visual-dinov3 or final-audio-seed17 lands correctly without a registry."""
    from deepfake_detection.training.fusion import DEPLOYMENT_PATTERNS, pattern_masks

    dims = {"visual-dinov3": 8, "stream-lipsync": 4, "final-audio-seed17": 6}
    names, masks, weights = pattern_masks(dims, DEPLOYMENT_PATTERNS)
    order = sorted(dims)

    reach = {
        name: {order[i] for i, on in enumerate(row) if on}
        for name, row in zip(names, masks.tolist(), strict=False)
    }
    assert reach["video"] == set(order)
    assert reach["image"] == {"visual-dinov3"}
    assert reach["audio"] == {"final-audio-seed17"}
    assert pytest.approx(float(weights.sum())) == 1.0


def test_a_pattern_reaching_no_stream_is_dropped() -> None:
    """Training the head to produce a confident number from an all-zero input
    would teach it the one case where it should abstain."""
    from deepfake_detection.training.fusion import pattern_masks

    names, _masks, _weights = pattern_masks(
        {"visual-dinov3": 8}, {"video": 0.5, "audio": 0.5}
    )

    assert names == ["video"]


def test_every_pattern_reaching_nothing_is_an_error() -> None:
    from deepfake_detection.training.fusion import pattern_masks

    with pytest.raises(ValueError, match="reaches any stream"):
        pattern_masks({"stream-lipsync": 4}, {"image": 1.0})


def test_training_with_patterns_still_learns() -> None:
    """The masking must not break the gradient path it rides on."""
    _model, history = fit_stream_fusion(
        rows=population(),
        epochs=60,
        batch_size=32,
        patience=60,
        presence_patterns={"video": 1.0},
        seed=17,
    )

    best = history.epochs[history.best_epoch - 1]
    assert best.validation_auc > 0.9, f"did not learn: {best.validation_auc:.3f}"


def test_a_genuinely_absent_stream_stays_absent_whatever_pattern_is_drawn() -> None:
    """Patterns multiply into the real presence flags rather than replacing
    them. A clip that lacks a stream must not be handed one by a video pattern."""
    rows = [
        row(f"clip-{i}", label=i % 2, identity=f"id-{i}", streams={"visual": 8})
        for i in range(12)
    ]
    values, presence, _labels = as_tensors(rows, DIMS, "cpu")

    assert presence["lipsync"].sum().item() == 0.0
    assert values["lipsync"].abs().sum().item() == 0.0
