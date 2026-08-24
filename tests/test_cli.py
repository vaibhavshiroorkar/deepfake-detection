import csv
import json
from pathlib import Path

import joblib
import pytest

from deepfake_detection.cli import build_parser, main
from deepfake_detection.experiments import runtime
from deepfake_detection.fusion.late import FusionArtifact
from deepfake_detection.fusion.store import FeatureRecord, FeatureStore


def test_public_parser_exposes_the_documented_command_tree() -> None:
    parser = build_parser()

    assert parser.prog == "ddf"


def test_run_command_accepts_a_root_and_multiple_configuration_layers() -> None:
    arguments = build_parser().parse_args(
        [
            "run",
            "--root",
            ".",
            "--config",
            "configs/local.yaml",
            "--config",
            "configs/smoke.yaml",
        ]
    )

    assert arguments.root == Path(".")
    assert arguments.config == [Path("configs/local.yaml"), Path("configs/smoke.yaml")]
    assert not arguments.no_tracking


@pytest.mark.parametrize("branch", ["visual", "sync"])
def test_branch_training_uses_shared_runtime_seed_function(
    branch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, bool]] = []

    class StopTraining(Exception):
        pass

    def stop_after_seed(seed: int, *, deterministic: bool) -> None:
        calls.append((seed, deterministic))
        raise StopTraining

    monkeypatch.setattr(runtime, "seed_everything", stop_after_seed)

    with pytest.raises(StopTraining):
        main(
            [
                "train",
                branch,
                "--train-manifest",
                "train.csv",
                "--validation-manifest",
                "validation.csv",
                "--cache-index",
                "cache.csv",
                "--cache-root",
                "cache",
                "--dataset",
                "fixture",
                "--checkpoint",
                "checkpoint.pt",
                "--history",
                "history.json",
                "--run-id",
                "run",
                "--split-hash",
                "split",
                "--preprocessing-hash",
                "preprocessing",
                "--seed",
                "23",
            ]
        )

    assert calls == [(23, True)]


def write_fixture_manifest(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "video_path",
                "manipulation_type",
                "method",
                "source",
                "race",
                "gender",
            ),
        )
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "clip_id": f"clip-{index}",
                    "video_path": f"clip-{index}.mp4",
                    "manipulation_type": "RealVideo-RealAudio",
                    "method": "real",
                    "source": f"id-{index}",
                    "race": "A" if index % 2 else "B",
                    "gender": "fixture",
                }
            )


def test_manifest_and_split_commands_emit_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.csv"
    normalized = tmp_path / "normalized.csv"
    manifest_audit = tmp_path / "manifest-audit.json"
    split_directory = tmp_path / "splits"
    write_fixture_manifest(raw)

    assert (
        main(
            [
                "manifest",
                "build",
                "--input",
                str(raw),
                "--output",
                str(normalized),
                "--audit",
                str(manifest_audit),
                "--dataset",
                "fixture",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "split",
                "build",
                "--manifest",
                str(normalized),
                "--output-dir",
                str(split_directory),
                "--dataset",
                "fixture",
                "--seed",
                "17",
            ]
        )
        == 0
    )

    assert json.loads(manifest_audit.read_text(encoding="utf-8"))["records"] == 10
    split_audit = json.loads(
        (split_directory / "audit.json").read_text(encoding="utf-8")
    )
    assert split_audit["source_overlaps"] == {}
    assert len(split_audit["split_hash"]) == 64
    assert (
        sum(
            1
            for name in ("train.csv", "val.csv", "test.csv")
            for _ in csv.DictReader((split_directory / name).open(encoding="utf-8"))
        )
        == 10
    )


def test_evaluate_command_keeps_blank_predictions_as_abstentions(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "label,probability,visual_probability,source,method,race,gender\n"
        "0,0.1,0.2,id1,real,A,women\n"
        "1,0.9,0.8,id2,wav2lip,B,men\n"
        "1,,,id3,rtvc,A,men\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.json"

    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(predictions),
                "--output",
                str(output),
                "--threshold",
                "0.5",
                "--bootstrap-samples",
                "20",
            ]
        )
        == 0
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["overall"]["abstained"] == 1
    assert report["overall"]["coverage"] == 2 / 3
    assert report["fusion_vs_visual_auc"]["estimate"] == 0.0


def test_threshold_command_writes_validation_selection(tmp_path: Path) -> None:
    predictions = tmp_path / "validation-predictions.csv"
    predictions.write_text(
        "label,probability\n0,0.1\n0,0.4\n1,0.6\n1,0.9\n",
        encoding="utf-8",
    )
    output = tmp_path / "threshold.json"

    assert (
        main(
            [
                "threshold",
                "--predictions",
                str(predictions),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    selection = json.loads(output.read_text(encoding="utf-8"))
    assert selection["objective"] == "balanced_accuracy"
    assert selection["threshold"] == 0.5
    assert selection["balanced_accuracy"] == 1.0


def test_train_fusion_command_fits_a_serialized_model(tmp_path: Path) -> None:
    store_path = tmp_path / "features.parquet"
    records = []
    for index, value in enumerate((-4, -3, -2, -1, 1, 2, 3, 4)):
        label = int(value > 0)
        for branch, scale in (("visual", 1.0), ("audio", 0.8), ("sync", 1.2)):
            records.append(
                FeatureRecord(
                    dataset="fixture",
                    clip_id=f"clip-{index}",
                    segment_id="segment-0",
                    branch=branch,
                    logit=value * scale,
                    embedding=(value,),
                    available=True,
                    checkpoint_hash=f"{branch}-checkpoint",
                    preprocessing_hash="prep",
                    split_hash="split",
                    run_id="run",
                    label=label,
                    source_identity=f"id-{index}",
                    method="real" if label == 0 else "fixture-fake",
                    race="fixture",
                    gender="fixture",
                    partition_role="oof",
                )
            )
    FeatureStore(store_path).write(records)
    model = tmp_path / "fusion.joblib"
    metadata = tmp_path / "fusion.json"

    assert (
        main(
            [
                "train",
                "fusion",
                "--feature-store",
                str(store_path),
                "--output",
                str(model),
                "--metadata",
                str(metadata),
            ]
        )
        == 0
    )

    assert model.is_file()
    artifact = joblib.load(model)
    assert isinstance(artifact, FusionArtifact)
    assert artifact.split_hash == "split"
    assert artifact.preprocessing_hash == "prep"
    report = json.loads(metadata.read_text(encoding="utf-8"))
    assert report["samples"] == 8
    assert report["split_hash"] == "split"

    predictions = tmp_path / "fusion-predictions.csv"
    assert (
        main(
            [
                "features",
                "score",
                "--feature-store",
                str(store_path),
                "--fusion-model",
                str(model),
                "--output",
                str(predictions),
            ]
        )
        == 0
    )
    rows = tuple(csv.DictReader(predictions.open(encoding="utf-8")))
    assert len(rows) == 8
    assert float(rows[0]["probability"]) < float(rows[-1]["probability"])
    assert all(row["visual_probability"] for row in rows)
    assert {row["source"] for row in rows} == {f"id-{index}" for index in range(8)}

    incomplete = []
    for branch in ("visual", "audio", "sync"):
        incomplete.append(
            FeatureRecord(
                dataset="fixture",
                clip_id="clip-8",
                segment_id="segment-0",
                branch=branch,
                logit=0.0,
                embedding=(),
                available=branch == "visual",
                checkpoint_hash=f"{branch}-checkpoint",
                preprocessing_hash="prep",
                split_hash="split",
                run_id="run",
                label=1,
                source_identity="id-8",
                method="fixture-fake",
                race="fixture",
                gender="fixture",
                partition_role="oof",
            )
        )
    FeatureStore(store_path).write(incomplete)
    assert (
        main(
            [
                "features",
                "score",
                "--feature-store",
                str(store_path),
                "--fusion-model",
                str(model),
                "--output",
                str(predictions),
            ]
        )
        == 0
    )
    rows = tuple(csv.DictReader(predictions.open(encoding="utf-8")))
    assert rows[-1]["probability"] == ""
    assert rows[-1]["visual_probability"] != ""
