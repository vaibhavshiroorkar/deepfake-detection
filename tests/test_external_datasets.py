"""Adapters for the two datasets the model is never trained on."""

from pathlib import Path

import pytest

from deepfake_detection.data.celebdf import manifest_from_celebdf, read_test_list
from deepfake_detection.data.guards import (
    EvaluationOnlyDatasetError,
    reject_evaluation_only,
    reject_evaluation_only_datasets,
)
from deepfake_detection.data.manifest import load_manifest
from deepfake_detection.data.mnw import manifest_from_mnw


def celebdf_tree(root: Path) -> Path:
    drop = root / "Celeb-DF-v2"
    for directory, names in (
        ("Celeb-real", ("id0_0000.mp4", "id1_0001.mp4")),
        ("Celeb-synthesis", ("id0_id1_0000.mp4", "id2_id3_0005.mp4")),
        ("YouTube-real", ("00063.mp4",)),
    ):
        folder = drop / directory
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(b"fixture")
    (drop / "List_of_testing_videos.txt").write_text(
        "1 YouTube-real/00063.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n",
        encoding="utf-8",
    )
    return drop


def test_celebdf_inverts_the_published_label_convention(tmp_path: Path) -> None:
    """Celeb-DF writes 1 for real; this project writes 1 for fake."""
    drop = celebdf_tree(tmp_path)

    frame = manifest_from_celebdf(drop, tmp_path)

    by_id = {row.clip_id: row for row in frame.itertuples()}
    assert (
        by_id["celebdf__Celeb-synthesis__id0_id1_0000"].manipulation_type
        == "FakeVideo-RealAudio"
    )
    assert (
        by_id["celebdf__Celeb-real__id0_0000"].manipulation_type
        == "RealVideo-RealAudio"
    )
    assert (
        by_id["celebdf__YouTube-real__00063"].manipulation_type
        == "RealVideo-RealAudio"
    )


def test_celebdf_reads_the_swapped_identity_pair(tmp_path: Path) -> None:
    drop = celebdf_tree(tmp_path)

    frame = manifest_from_celebdf(drop, tmp_path)

    row = next(
        item
        for item in frame.itertuples()
        if item.clip_id == "celebdf__Celeb-synthesis__id0_id1_0000"
    )
    assert row.source == "celeb-id0"
    assert row.target1 == "celeb-id1"


def test_celebdf_gives_each_youtube_clip_its_own_source(tmp_path: Path) -> None:
    """They carry no identity, so sharing one source would strand them all in
    a single split partition."""
    drop = celebdf_tree(tmp_path)

    frame = manifest_from_celebdf(drop, tmp_path)

    row = next(
        item for item in frame.itertuples() if item.clip_id == "celebdf__YouTube-real__00063"
    )
    assert row.source == "youtube-00063"


def test_celebdf_test_only_honours_the_official_list(tmp_path: Path) -> None:
    drop = celebdf_tree(tmp_path)
    assert read_test_list(drop) == {
        "YouTube-real/00063.mp4",
        "Celeb-synthesis/id0_id1_0000.mp4",
    }

    frame = manifest_from_celebdf(drop, tmp_path, test_only=True)

    assert set(frame["clip_id"]) == {
        "celebdf__YouTube-real__00063",
        "celebdf__Celeb-synthesis__id0_id1_0000",
    }


def test_celebdf_rows_survive_the_manifest_contract(tmp_path: Path) -> None:
    drop = celebdf_tree(tmp_path)
    frame = manifest_from_celebdf(drop, tmp_path)
    path = tmp_path / "celebdf.csv"
    frame.to_csv(path, index=False)

    result = load_manifest(path, dataset="Celeb-DF-v2")

    assert len(result.records) == len(frame)
    assert not result.quarantined_paths
    fake = next(item for item in result.records if item.video_fake)
    assert fake.clip_fake
    # Celeb-DF leaves the audio track alone, so the audio branch sees a real one.
    assert not fake.audio_fake


def mnw_tree(root: Path) -> Path:
    drop = root / "MNW"
    for generator, count in (("Wav2lip", 2), ("Vasa_1", 1)):
        folder = drop / "Deepfake_Video" / generator
        folder.mkdir(parents=True)
        for index in range(count):
            (folder / f"{generator}_no_audio_0{index}.mp4").write_bytes(b"fixture")
    wild = drop / "AI_media_in_the_wild" / "Video"
    for label, names in (
        ("likely_manipulated", ("M056.mp4",)),
        ("likely_authentic", ("A022.mp4",)),
        ("inconsistent", ("I043.mp4",)),
    ):
        folder = wild / label
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(b"fixture")
    return drop


def test_mnw_labels_every_lab_clip_fake_and_names_its_generator(tmp_path: Path) -> None:
    drop = mnw_tree(tmp_path)

    frame = manifest_from_mnw(drop, tmp_path, include_wild=False)

    assert len(frame) == 3
    assert set(frame["manipulation_type"]) == {"FakeVideo-RealAudio"}
    assert set(frame["method"]) == {"mnw-wav2lip", "mnw-vasa_1"}


def test_mnw_skips_the_inconsistent_verdict(tmp_path: Path) -> None:
    """'inconsistent' means the experts could not decide, which is not a label."""
    drop = mnw_tree(tmp_path)

    frame = manifest_from_mnw(drop, tmp_path)

    assert not any("inconsistent" in clip_id for clip_id in frame["clip_id"])
    assert any("likely_authentic" in clip_id for clip_id in frame["clip_id"])
    assert any("likely_manipulated" in clip_id for clip_id in frame["clip_id"])


def test_mnw_gives_every_clip_its_own_source(tmp_path: Path) -> None:
    drop = mnw_tree(tmp_path)

    frame = manifest_from_mnw(drop, tmp_path)

    assert frame["source"].is_unique


def test_mnw_rows_survive_the_manifest_contract(tmp_path: Path) -> None:
    drop = mnw_tree(tmp_path)
    frame = manifest_from_mnw(drop, tmp_path)
    path = tmp_path / "mnw.csv"
    frame.to_csv(path, index=False)

    result = load_manifest(path, dataset="MNW")

    assert len(result.records) == len(frame)
    assert not result.quarantined_paths


def test_mnw_raises_when_no_videos_are_present(tmp_path: Path) -> None:
    empty = tmp_path / "MNW"
    empty.mkdir()
    with pytest.raises(ValueError, match="No MNW videos"):
        manifest_from_mnw(empty, tmp_path)


@pytest.mark.parametrize(
    "operation",
    ["training", "split building", "fusion training", "an oof-role feature export"],
)
def test_guard_refuses_mnw_for_every_model_influencing_operation(
    operation: str,
) -> None:
    with pytest.raises(EvaluationOnlyDatasetError, match="evaluation-only"):
        reject_evaluation_only("MNW", operation=operation)


def test_guard_allows_datasets_that_are_not_locked() -> None:
    reject_evaluation_only("FakeAVCeleb", operation="training")
    reject_evaluation_only("Celeb-DF-v2", operation="training")


def test_guard_catches_a_locked_dataset_blended_with_others() -> None:
    with pytest.raises(EvaluationOnlyDatasetError):
        reject_evaluation_only_datasets(
            ["FakeAVCeleb", "MNW", "Celeb-DF-v2"], operation="fusion training"
        )
