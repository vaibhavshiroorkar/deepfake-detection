"""LAV-DF's matched pairs, and DFDC's deliberately cautious label mapping."""

import json
from pathlib import Path

import pytest

from deepfake_detection.data.dfdc import manifest_from_dfdc
from deepfake_detection.data.lavdf import (
    SYNC_SECONDS,
    forged_window,
    genuine_window,
    manifest_from_lavdf,
    manipulation_type,
)
from deepfake_detection.data.manifest import load_manifest


def lavdf_tree(root: Path, entries: list[dict]) -> Path:
    drop = root / "LAV-DF"
    for entry in entries:
        path = drop / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (drop / "metadata.min.json").write_text(json.dumps(entries), encoding="utf-8")
    return drop


def entry(name, *, periods, duration, split="train", video=False, audio=False, original=None):
    return {
        "file": f"{split}/{name}.mp4",
        "n_fakes": len(periods),
        "fake_periods": periods,
        "duration": duration,
        "original": original,
        "modify_video": video,
        "modify_audio": audio,
        "split": split,
    }


@pytest.mark.parametrize(
    ("video", "audio", "expected"),
    [
        (False, False, "RealVideo-RealAudio"),
        (True, False, "FakeVideo-RealAudio"),
        (False, True, "RealVideo-FakeAudio"),
        (True, True, "FakeVideo-FakeAudio"),
    ],
)
def test_both_flags_map_onto_the_enum(video, audio, expected) -> None:
    """LAV-DF is the only corpus here that populates all four honestly."""
    assert manipulation_type(modify_video=video, modify_audio=audio) == expected


def test_forged_window_centres_the_longest_span() -> None:
    start = forged_window([[1.0, 1.4], [5.0, 6.0]], duration=10.0)
    assert start == pytest.approx(4.5)


def test_forged_window_is_clamped_inside_the_clip() -> None:
    start = forged_window([[4.5, 4.9]], duration=5.0)
    assert 0.0 <= start <= 5.0 - SYNC_SECONDS


def test_genuine_window_avoids_every_manipulated_span() -> None:
    start = genuine_window([[3.0, 3.6]], duration=8.0)
    assert start is not None
    assert start + SYNC_SECONDS <= 3.0 or start >= 3.6


def test_genuine_window_is_none_when_no_room_remains() -> None:
    """Better to emit only the fake window than to put a manipulated span under
    a real label."""
    assert genuine_window([[0.0, 2.5], [2.6, 5.0]], duration=5.0) is None


def test_a_forgery_yields_a_matched_pair_from_one_file(tmp_path: Path) -> None:
    drop = lavdf_tree(
        tmp_path,
        [entry("000001", periods=[[3.0, 3.7]], duration=8.0, video=True, audio=True,
               original="train/000009.mp4")],
    )

    frame = manifest_from_lavdf(drop, tmp_path)

    assert len(frame) == 2
    kinds = dict(
        zip(frame["clip_id"], frame["manipulation_type"], strict=True)
    )
    assert kinds["lavdf__000001__forged"] == "FakeVideo-FakeAudio"
    assert kinds["lavdf__000001__genuine"] == "RealVideo-RealAudio"
    # One file, two windows: the codec and the speaker are held constant.
    assert frame["video_path"].nunique() == 1
    assert frame["sync_start_sec"].nunique() == 2


def test_matched_pair_shares_an_identity_group(tmp_path: Path) -> None:
    """Both windows must move together through a source-disjoint split, and a
    forgery must group with the clip it was made from."""
    drop = lavdf_tree(
        tmp_path,
        [entry("000001", periods=[[3.0, 3.7]], duration=8.0, video=True,
               original="train/000009.mp4")],
    )

    frame = manifest_from_lavdf(drop, tmp_path)

    assert set(frame["source"]) == {"lavdf-000009"}


def test_pairs_can_be_disabled(tmp_path: Path) -> None:
    drop = lavdf_tree(
        tmp_path, [entry("000001", periods=[[3.0, 3.7]], duration=8.0, video=True)]
    )

    frame = manifest_from_lavdf(drop, tmp_path, matched_pairs=False)

    assert len(frame) == 1
    assert frame.iloc[0]["clip_id"].endswith("__forged")


def test_real_clips_yield_one_row(tmp_path: Path) -> None:
    drop = lavdf_tree(tmp_path, [entry("000002", periods=[], duration=6.0)])

    frame = manifest_from_lavdf(drop, tmp_path)

    assert len(frame) == 1
    assert frame.iloc[0]["manipulation_type"] == "RealVideo-RealAudio"
    assert frame.iloc[0]["sync_start_sec"] == 0.0


def test_split_filter_honours_the_published_protocol(tmp_path: Path) -> None:
    drop = lavdf_tree(
        tmp_path,
        [
            entry("000001", periods=[], duration=6.0, split="train"),
            entry("000002", periods=[], duration=6.0, split="test"),
        ],
    )

    assert len(manifest_from_lavdf(drop, tmp_path, split="train")) == 1
    assert len(manifest_from_lavdf(drop, tmp_path, split="test")) == 1


def test_clips_shorter_than_the_window_are_dropped(tmp_path: Path) -> None:
    drop = lavdf_tree(tmp_path, [entry("000003", periods=[], duration=1.0)])

    with pytest.raises(ValueError, match="No LAV-DF clips"):
        manifest_from_lavdf(drop, tmp_path)


def test_lavdf_rows_survive_the_manifest_contract(tmp_path: Path) -> None:
    drop = lavdf_tree(
        tmp_path,
        [
            entry("000001", periods=[[3.0, 3.7]], duration=8.0, video=True),
            entry("000002", periods=[], duration=6.0),
        ],
    )
    frame = manifest_from_lavdf(drop, tmp_path)
    path = tmp_path / "lavdf.csv"
    frame.to_csv(path, index=False)

    result = load_manifest(path, dataset="LAV-DF")

    assert len(result.records) == len(frame)
    assert not result.quarantined_paths
    forged = next(r for r in result.records if r.clip_id.endswith("__forged"))
    genuine = next(r for r in result.records if r.clip_id.endswith("__genuine"))
    assert forged.sync_start_sec != genuine.sync_start_sec
    assert forged.clip_fake and not genuine.clip_fake


def dfdc_tree(root: Path) -> Path:
    drop = root / "DFDC" / "part_00"
    drop.mkdir(parents=True)
    for name in ("aaa.mp4", "bbb.mp4"):
        (drop / name).write_bytes(b"fixture")
    (drop / "metadata.json").write_text(
        json.dumps(
            {
                "aaa.mp4": {"label": "FAKE", "split": "train", "original": "bbb.mp4"},
                "bbb.mp4": {"label": "REAL", "split": "train", "original": None},
            }
        ),
        encoding="utf-8",
    )
    return root / "DFDC"


def test_dfdc_maps_labels_and_groups_a_fake_with_its_source(tmp_path: Path) -> None:
    frame = manifest_from_dfdc(dfdc_tree(tmp_path), tmp_path)

    by_id = {row.clip_id: row for row in frame.itertuples()}
    assert by_id["dfdc__aaa"].manipulation_type == "FakeVideo-RealAudio"
    assert by_id["dfdc__bbb"].manipulation_type == "RealVideo-RealAudio"
    # The forgery and the clip it came from share a group.
    assert by_id["dfdc__aaa"].source == by_id["dfdc__bbb"].source == "dfdc-bbb"


def test_dfdc_rows_survive_the_manifest_contract(tmp_path: Path) -> None:
    frame = manifest_from_dfdc(dfdc_tree(tmp_path), tmp_path)
    path = tmp_path / "dfdc.csv"
    frame.to_csv(path, index=False)

    result = load_manifest(path, dataset="DFDC")

    assert len(result.records) == 2
    assert not result.quarantined_paths


def test_dfdc_raises_when_nothing_is_present(tmp_path: Path) -> None:
    empty = tmp_path / "DFDC"
    empty.mkdir()
    with pytest.raises(ValueError, match="No DFDC videos"):
        manifest_from_dfdc(empty, tmp_path)
