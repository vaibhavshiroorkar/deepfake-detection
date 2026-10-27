import argparse
import csv

import pytest

from deepfake_detection.cli.data import _split_subsample
from deepfake_detection.data.manifest import ClipRecord, load_manifest, write_manifest
from deepfake_detection.training.crossfit import build_group_folds


def _record(index, *, dataset="fixture", fake=False):
    return ClipRecord.from_mapping(
        {
            "clip_id": f"clip-{index}",
            "dataset": dataset,
            "video_path": f"clip-{index}.mp4",
            "source": f"source-{index}",
            "manipulation_type": "FakeVideo-RealAudio"
            if fake
            else "RealVideo-RealAudio",
            "method": "fake" if fake else "real",
        }
    )


def test_stored_dataset_cannot_be_relabelled_by_a_cli_argument(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest([_record(0, dataset="MNW")], path)
    with pytest.raises(ValueError, match="dataset"):
        load_manifest(path, dataset="FakeAVCeleb")


def test_mixed_manifest_keeps_each_dataset_without_hiding_locked_inputs(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest([_record(0, dataset="FakeAVCeleb"), _record(1, dataset="MNW")], path)
    with pytest.raises(ValueError, match="dataset"):
        load_manifest(path, dataset="FakeAVCeleb+FF++")


def test_manifest_round_trip_distinguishes_no_window_from_zero(tmp_path):
    from dataclasses import replace

    path = tmp_path / "manifest.csv"
    write_manifest([_record(0), replace(_record(1), sync_start_sec=0.0)], path)
    loaded = load_manifest(path, dataset="fixture").records
    assert [r.sync_start_sec for r in loaded] == [None, 0.0]


def test_pilot_changes_only_training_rows(tmp_path):
    source = tmp_path / "full"
    target = tmp_path / "pilot"
    for name in ("train", "val", "test"):
        rows = [_record(0)] + [_record(i, fake=True) for i in range(1, 10)]
        write_manifest(rows, source / f"{name}.csv")
    assert (
        _split_subsample(
            argparse.Namespace(
                split_dir=source,
                output_dir=target,
                dataset="fixture",
                seed=17,
                fake_ratio=2,
            )
        )
        == 0
    )
    assert len(load_manifest(target / "train.csv", dataset="fixture").records) == 3
    for name in ("val", "test"):
        with (source / f"{name}.csv").open(newline="") as a:
            with (target / f"{name}.csv").open(newline="") as b:
                assert list(csv.DictReader(a)) == list(csv.DictReader(b))


def test_nested_folds_keep_selection_and_outer_holdout_disjoint():
    records = tuple(_record(i) for i in range(12))
    folds = build_group_folds(records, folds=3, seed=17, validation_fraction=0.25)
    assert folds == build_group_folds(
        records, folds=3, seed=17, validation_fraction=0.25
    )
    for fold in folds:
        training, validation, holdout = map(
            set,
            (
                fold.train_sources,
                fold.validation_sources,
                fold.holdout_sources,
            ),
        )
        assert training and validation and holdout
        assert not training & validation
        assert not training & holdout
        assert not validation & holdout
        assert training | validation | holdout == {r.source for r in records}
        assert {records[i].source for i in fold.validation_indices} == validation
