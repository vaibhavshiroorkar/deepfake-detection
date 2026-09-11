import os
import time
from pathlib import Path

import pandas as pd
import pytest

from deepfake_detection.dashboard.lib import checkpoints, datasets, selectors
from deepfake_detection.data.meta import (
    clip_label,
    manifest_from_meta,
    resolve_video_path,
)

META_ROW = [
    "id00018",
    "id00018",
    "id00018",
    "real",
    "RealVideo-RealAudio",
    "RealVideo-RealAudio",
    "Asian",
    "men",
    "00109.mp4",
    "FakeAVCeleb/RealVideo-RealAudio/Asian/men/id00018",
]


def write_drop(root: Path) -> Path:
    drop = root / "FakeAVCeleb"
    clip_dir = drop / "RealVideo-RealAudio" / "Asian" / "men" / "id00018"
    clip_dir.mkdir(parents=True)
    (clip_dir / "00109.mp4").write_bytes(b"not really a video")
    pd.DataFrame([META_ROW]).to_csv(drop / "meta_data.csv", index=False)
    return drop


def manifest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clip_id": ["a1", "b2", "c3"],
            "video_path": ["a.mp4", "b.mp4", "c.mp4"],
            "label": [0, 1, 1],
            "manipulation_type": [
                "RealVideo-RealAudio",
                "FakeVideo-RealAudio",
                "FakeVideo-FakeAudio",
            ],
            "method": ["real", "fsgan", "wav2lip"],
            "source": ["id1", "id1", "id2"],
        }
    )


def test_clip_label_counts_a_fake_track_as_fake() -> None:
    assert clip_label("RealVideo-RealAudio") == 0
    assert clip_label("RealVideo-FakeAudio") == 1
    assert clip_label("FakeVideo-RealAudio") == 1


def test_clip_label_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unrecognised"):
        clip_label("RealVideo-SomethingElse")


def test_resolve_video_path_drops_the_leading_dataset_segment() -> None:
    resolved = resolve_video_path(
        Path("/data/FakeAVCeleb"), "FakeAVCeleb/RealVideo-RealAudio/x", "a.mp4"
    )
    assert resolved.as_posix().endswith("FakeAVCeleb/RealVideo-RealAudio/x/a.mp4")


def test_manifest_from_meta_drops_rows_with_no_file(tmp_path: Path) -> None:
    drop = write_drop(tmp_path)
    meta = pd.DataFrame([META_ROW, [*META_ROW[:8], "absent.mp4", META_ROW[9]]])
    frame = manifest_from_meta(meta, drop, tmp_path)
    assert len(frame) == 1
    assert frame.loc[0, "label"] == 0
    assert frame.loc[0, "clip_id"].endswith("00109")


def test_manifest_from_meta_dedupes_a_repeated_file(tmp_path: Path) -> None:
    drop = write_drop(tmp_path)
    meta = pd.DataFrame([META_ROW, META_ROW])
    assert len(manifest_from_meta(meta, drop, tmp_path)) == 1


def test_manifest_from_meta_rejects_a_reshaped_meta_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected"):
        manifest_from_meta(pd.DataFrame([[1, 2, 3]]), tmp_path, tmp_path)


def test_discover_finds_a_raw_drop(tmp_path: Path) -> None:
    write_drop(tmp_path)
    found = datasets.discover(tmp_path)
    assert list(found) == ["FakeAVCeleb"]
    assert found["FakeAVCeleb"].splits == [datasets.RAW_SPLIT]


def test_discover_attaches_a_manifest_to_the_drop_it_points_into(
    tmp_path: Path,
) -> None:
    write_drop(tmp_path)
    video_path = "FakeAVCeleb/RealVideo-RealAudio/Asian/men/id00018/00109.mp4"
    pd.DataFrame(
        {"clip_id": ["a"], "video_path": [video_path], "label": [0]}
    ).to_csv(tmp_path / "train.csv", index=False)
    found = datasets.discover(tmp_path)
    assert found["FakeAVCeleb"].splits == ["train", datasets.RAW_SPLIT]


def test_discover_ignores_a_csv_that_is_not_a_manifest(tmp_path: Path) -> None:
    write_drop(tmp_path)
    pd.DataFrame({"note": ["hello"]}).to_csv(tmp_path / "notes.csv", index=False)
    assert datasets.discover(tmp_path)["FakeAVCeleb"].manifests == {}


def test_discover_returns_nothing_for_a_missing_directory(tmp_path: Path) -> None:
    assert datasets.discover(tmp_path / "absent") == {}


def test_load_split_reads_the_raw_manifest(tmp_path: Path) -> None:
    write_drop(tmp_path)
    dataset = datasets.discover(tmp_path)["FakeAVCeleb"]
    assert len(datasets.load_split(dataset, datasets.RAW_SPLIT, tmp_path)) == 1


def test_load_split_rejects_an_unknown_split(tmp_path: Path) -> None:
    write_drop(tmp_path)
    dataset = datasets.discover(tmp_path)["FakeAVCeleb"]
    with pytest.raises(KeyError):
        datasets.load_split(dataset, "test", tmp_path)


def test_filter_manifest_narrows_by_type_method_and_label() -> None:
    frame = manifest_frame()
    assert len(selectors.filter_manifest(frame, ["FakeVideo-RealAudio"], [], "all")) == 1
    assert len(selectors.filter_manifest(frame, [], ["wav2lip"], "all")) == 1
    assert len(selectors.filter_manifest(frame, [], [], "real")) == 1
    assert len(selectors.filter_manifest(frame, [], [], "fake")) == 2


def test_search_manifest_matches_clip_id_and_identity() -> None:
    frame = manifest_frame()
    assert len(selectors.search_manifest(frame, "id1")) == 2
    assert len(selectors.search_manifest(frame, "c3")) == 1
    assert len(selectors.search_manifest(frame, "  ")) == 3


def test_group_by_identity_keeps_variants_together() -> None:
    grouped = selectors.group_by_identity(manifest_frame())
    assert list(grouped["source"]) == ["id1", "id1", "id2"]


def test_label_text_names_an_upload_unknown() -> None:
    assert selectors.label_text({"label": 0}) == "real"
    assert selectors.label_text({"label": 1}) == "fake"
    assert selectors.label_text({"label": selectors.LABEL_UNKNOWN}) == "unknown"


def test_clip_path_leaves_an_upload_absolute(tmp_path: Path) -> None:
    absolute = tmp_path / "clip.mp4"
    assert selectors.clip_path({"video_path": str(absolute)}) == absolute
    assert selectors.clip_path({"video_path": "drop/clip.mp4"}).is_absolute()


def test_upload_row_writes_outside_the_data_directory() -> None:
    row = selectors.upload_row("clip.mp4", b"bytes", "file-1")
    written = Path(row["video_path"])
    try:
        assert written.exists()
        assert selectors.DATA_DIR not in written.parents
        assert row["label"] == selectors.LABEL_UNKNOWN
    finally:
        written.unlink()


def test_discover_checkpoints_returns_nothing_without_a_directory(
    tmp_path: Path,
) -> None:
    assert (
        checkpoints.discover("visual", root=tmp_path, runs_root=tmp_path / "runs") == []
    )


def test_discover_checkpoints_lists_newest_first(tmp_path: Path) -> None:
    directory = tmp_path / "visual"
    directory.mkdir()
    (directory / "old.pt").write_bytes(b"a")
    (directory / "new.pt").write_bytes(b"b")
    (directory / "notes.txt").write_bytes(b"c")
    earlier = time.time() - 60
    os.utime(directory / "old.pt", (earlier, earlier))
    found = checkpoints.discover("visual", root=tmp_path, runs_root=tmp_path / "runs")
    assert [path.name for path in found] == ["new.pt", "old.pt"]


def run_checkpoint(runs: Path, run: str, name: str) -> Path:
    """A checkpoint where `ddf run` actually leaves one."""
    directory = runs / run / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"weights")
    return path


def test_discover_finds_checkpoints_inside_run_directories(tmp_path: Path) -> None:
    """The bug this covers: the picker offered only untrained weights while
    twenty trained checkpoints sat in `runs/`, because nothing copies them to
    the top-level `checkpoints/` directory the dashboard was looking in."""
    runs = tmp_path / "runs"
    run_checkpoint(runs, "program", "final-visual-seed17.pt")

    found = checkpoints.discover("efficientnet", root=tmp_path / "none", runs_root=runs)

    assert [path.name for path in found] == ["final-visual-seed17.pt"]


def test_discover_keeps_streams_apart(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_checkpoint(runs, "program", "final-visual-seed17.pt")
    run_checkpoint(runs, "program", "final-audio-seed17.pt")
    run_checkpoint(runs, "program", "pilot-lipsync-stream.pt")

    visual = checkpoints.discover("efficientnet", root=tmp_path, runs_root=runs)
    lipsync = checkpoints.discover("lipsync", root=tmp_path, runs_root=runs)

    assert [path.name for path in visual] == ["final-visual-seed17.pt"]
    assert [path.name for path in lipsync] == ["pilot-lipsync-stream.pt"]


def test_discover_keeps_identically_named_files_from_different_runs(
    tmp_path: Path,
) -> None:
    """`ddf run` writes the same `fold0-visual.pt` into every run it is given,
    so the filename is not an identity."""
    runs = tmp_path / "runs"
    run_checkpoint(runs, "first", "fold0-visual.pt")
    run_checkpoint(runs, "second", "fold0-visual.pt")

    found = checkpoints.discover("efficientnet", root=tmp_path, runs_root=runs)

    assert len(found) == 2
    assert {path.parent.parent.name for path in found} == {"first", "second"}


def test_architecture_reads_a_gru_from_its_gate_count() -> None:
    """A GRU holds three gate matrices per layer and an LSTM four, which is the
    only record of which one a bare state dict was trained with."""
    torch = pytest.importorskip("torch")
    state = {"temporal.weight_hh_l0": torch.zeros(768, 256)}

    assert checkpoints.architecture(state) == {
        "temporal": "gru",
        "hidden": 256,
        "bidirectional": False,
        # None because a bare temporal state dict carries no projection layer,
        # which is also what a Design A branch checkpoint looks like: it ends in
        # a classifier and has no shared fusion width to report.
        "common_dim": None,
    }


def test_architecture_reads_a_bidirectional_lstm() -> None:
    torch = pytest.importorskip("torch")
    state = {
        "temporal.weight_hh_l0": torch.zeros(512, 128),
        "temporal.weight_hh_l0_reverse": torch.zeros(512, 128),
    }

    assert checkpoints.architecture(state) == {
        "temporal": "lstm",
        "hidden": 128,
        "bidirectional": True,
        "common_dim": None,
    }


def test_architecture_reports_mean_pooling_when_there_is_no_temporal_model() -> None:
    assert checkpoints.architecture({})["temporal"] == "mean"


def test_describe_reports_an_unreadable_checkpoint(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    assert checkpoints.describe(broken)["error"]


def test_load_into_reports_what_did_not_fit(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = nn.Linear(4, 2)
    path = tmp_path / "other.pt"
    torch.save({"model_state": {"weight": torch.zeros(3, 4)}}, path)
    report = checkpoints.load_into(model, path)
    assert report["clean"] is False
    assert report["mismatched"][0][0] == "weight"
    assert "bias" in report["missing"]


def test_load_into_reports_a_clean_load(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    model = nn.Linear(4, 2)
    path = tmp_path / "same.pt"
    torch.save({"model_state": model.state_dict()}, path)
    assert checkpoints.load_into(model, path)["clean"] is True


def test_discover_does_not_walk_a_run_cache(tmp_path: Path) -> None:
    """A run directory holds its preprocessed cache beside its weights. Walking
    the tree visits about 62,000 entries and takes ten seconds, once per
    Streamlit rerun, which reads on screen as a page that stops rendering at the
    checkpoint picker."""
    runs = tmp_path / "runs"
    run_checkpoint(runs, "program", "final-visual-seed17.pt")
    buried = runs / "program" / "cache" / "a" / "b" / "c"
    buried.mkdir(parents=True)
    (buried / "stray-visual.pt").write_bytes(b"not a checkpoint")

    found = checkpoints.discover("efficientnet", root=tmp_path, runs_root=runs)

    assert [path.name for path in found] == ["final-visual-seed17.pt"]


def test_a_generic_token_does_not_claim_another_backbone(tmp_path: Path) -> None:
    """`ddf train visual` names its output "visual" with no backbone in it, so
    "visual" has to reach efficientnet. But `ddf train visual-stream` writes
    visual-dinov3.pt, which must not also land under efficientnet."""
    runs = tmp_path / "runs"
    run_checkpoint(runs, "design-b", "visual-dinov3.pt")
    run_checkpoint(runs, "design-b", "visual-efficientnet.pt")
    run_checkpoint(runs, "design-b", "visual-xception.pt")
    run_checkpoint(runs, "program", "final-visual-seed17.pt")

    def names(stream: str) -> set[str]:
        return {
            path.name
            for path in checkpoints.discover(stream, root=tmp_path, runs_root=runs)
        }

    assert names("dinov3") == {"visual-dinov3.pt"}
    assert names("xception") == {"visual-xception.pt"}
    # Its own, plus the Design A checkpoint that is EfficientNet-B0 unnamed.
    assert names("efficientnet") == {"visual-efficientnet.pt", "final-visual-seed17.pt"}


def manifest_csv(path: Path, rows: list[dict]) -> Path:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_clip_fake_manifest_gets_a_label(tmp_path: Path) -> None:
    """`ddf manifest build` writes clip_fake, not label. Widening discovery to
    accept it without deriving a label crashed the picker with KeyError."""
    frame = pd.read_csv(
        manifest_csv(
            tmp_path / "m.csv",
            [
                {"clip_id": "a", "video_path": "d/a.mp4", "clip_fake": "True"},
                {"clip_id": "b", "video_path": "d/b.mp4", "clip_fake": "False"},
            ],
        )
    )

    labelled = datasets.ensure_label(frame)

    assert labelled["label"].tolist() == [1, 0]


def test_manipulation_type_manifest_gets_a_label(tmp_path: Path) -> None:
    """The external adapters emit manipulation_type, where anything other than
    RealVideo-RealAudio involves a manipulation."""
    frame = pd.read_csv(
        manifest_csv(
            tmp_path / "m.csv",
            [
                {
                    "clip_id": "a",
                    "video_path": "d/a.mp4",
                    "manipulation_type": "RealVideo-RealAudio",
                },
                {
                    "clip_id": "b",
                    "video_path": "d/b.mp4",
                    "manipulation_type": "FakeVideo-RealAudio",
                },
                {
                    "clip_id": "c",
                    "video_path": "d/c.mp4",
                    "manipulation_type": "RealVideo-FakeAudio",
                },
            ],
        )
    )

    assert datasets.ensure_label(frame)["label"].tolist() == [0, 1, 1]


def test_an_existing_label_is_left_alone(tmp_path: Path) -> None:
    frame = pd.read_csv(
        manifest_csv(
            tmp_path / "m.csv",
            [{"clip_id": "a", "video_path": "d/a.mp4", "label": 1, "clip_fake": "False"}],
        )
    )

    assert datasets.ensure_label(frame)["label"].tolist() == [1]


def test_a_manifest_with_no_label_information_is_unknown(tmp_path: Path) -> None:
    """Unknown, not 0: a clip nothing is known about must not read as
    confirmed real."""
    frame = pd.read_csv(
        manifest_csv(tmp_path / "m.csv", [{"clip_id": "a", "video_path": "d/a.mp4"}])
    )

    labelled = datasets.ensure_label(frame)

    assert labelled["label"].tolist() == [datasets.UNKNOWN_LABEL]
    assert selectors.label_text(labelled.iloc[0]) == "unknown"


def test_label_text_survives_a_row_with_no_label_column() -> None:
    """A frame read by something other than load_split still has to render."""
    assert selectors.label_text(pd.Series({"clip_id": "a"})) == "unknown"
