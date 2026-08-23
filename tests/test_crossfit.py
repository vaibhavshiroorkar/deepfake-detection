from dataclasses import replace
from pathlib import Path

from deepfake_detection.data.manifest import ClipRecord
from deepfake_detection.training.crossfit import build_group_folds


def record(clip_id: str, source: str) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        dataset="fixture",
        video_path=Path(f"{clip_id}.mp4"),
        manipulation_type="RealVideo-RealAudio",
        method="real",
        source=source,
        targets=(),
        clip_fake=False,
        video_fake=False,
        audio_fake=False,
    )


def test_group_crossfit_holds_out_each_source_exactly_once() -> None:
    records = tuple(
        record(f"{source}-{index}", source)
        for source in (f"id{number}" for number in range(12))
        for index in range(2)
    )

    folds = build_group_folds(records, folds=3, seed=17)

    held_out = [source for fold in folds for source in fold.holdout_sources]
    assert sorted(held_out) == sorted({record.source for record in records})
    for fold in folds:
        assert not set(fold.train_sources) & set(fold.holdout_sources)
        assert len(fold.holdout_indices) == 8
    assert folds == build_group_folds(records, folds=3, seed=17)


def test_group_crossfit_keeps_folds_nonempty_with_sparse_strata() -> None:
    records = tuple(
        replace(record(f"clip-{index}", f"id-{index}"), race=f"race-{index}")
        for index in range(4)
    )

    folds = build_group_folds(records, folds=4, seed=17)

    assert [len(fold.holdout_sources) for fold in folds] == [1, 1, 1, 1]
