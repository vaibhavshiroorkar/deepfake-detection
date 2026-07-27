"""Discovery of datasets under data/, the dashboard has no hardcoded registry,
so these pin down what counts as a dataset and which manifest belongs to which.
"""
import pandas as pd
import pytest

from dashboard.lib import datasets


META_HEADER = "source,target1,target2,method,category,type,race,gender,path,\n"


def _meta_row(source, method, category, mtype, filename, dirpath):
    return f"{source},-,-,{method},{category},{mtype},African,men,{filename},{dirpath}\n"


def _make_drop(data_dir, name="FakeAVCeleb_v1.2"):
    """A raw drop: meta_data.csv plus the two videos it lists that exist."""
    root = data_dir / name
    clips = root / "RealVideo-RealAudio" / "African" / "men" / "id00076"
    clips.mkdir(parents=True)
    (clips / "00109.mp4").write_bytes(b"")
    (clips / "00110.mp4").write_bytes(b"")
    rel = "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076"
    (root / "meta_data.csv").write_text(
        META_HEADER
        + _meta_row("id00076", "real", "A", "RealVideo-RealAudio", "00109.mp4", rel)
        + _meta_row("id00076", "real", "A", "RealVideo-RealAudio", "00110.mp4", rel)
        # listed in meta but never downloaded, must not reach the manifest
        + _meta_row("id00099", "wav2lip", "D", "FakeVideo-FakeAudio", "00001.mp4", rel)
    )
    return root


def _write_manifest(path, video_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "clip_id": [f"c{i}" for i in range(len(video_paths))],
        "video_path": video_paths,
        "label": [0] * len(video_paths),
    }).to_csv(path, index=False)


def test_raw_drop_is_found_without_any_manifest(tmp_path):
    _make_drop(tmp_path)
    found = datasets.discover(tmp_path)
    assert list(found) == ["FakeAVCeleb_v1.2"]
    assert found["FakeAVCeleb_v1.2"].splits == [datasets.RAW_SPLIT]


def test_root_manifest_attaches_to_the_drop_its_clips_live_in(tmp_path):
    _make_drop(tmp_path)
    # The pipeline writes splits flat in data/ while the clips sit in the drop.
    _write_manifest(tmp_path / "train.csv",
                    ["FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4"])
    found = datasets.discover(tmp_path)
    assert list(found) == ["FakeAVCeleb_v1.2"]           # one dataset, not two
    ds = found["FakeAVCeleb_v1.2"]
    assert ds.splits == ["train", datasets.RAW_SPLIT]    # pipeline splits first


def test_manifest_pointing_outside_any_drop_becomes_its_own_dataset(tmp_path):
    _make_drop(tmp_path)
    _write_manifest(tmp_path / "deepfake_eval" / "test.csv", ["deepfake_eval/clips/x.mp4"])
    found = datasets.discover(tmp_path)
    assert set(found) == {"FakeAVCeleb_v1.2", "deepfake_eval"}
    assert found["deepfake_eval"].splits == ["test"]


def test_csv_without_manifest_columns_is_ignored(tmp_path):
    _make_drop(tmp_path)
    pd.DataFrame({"epoch": [1], "loss": [0.5]}).to_csv(tmp_path / "history.csv", index=False)
    found = datasets.discover(tmp_path)
    assert found["FakeAVCeleb_v1.2"].manifests == {}


def test_rescan_picks_up_a_dataset_added_later(tmp_path):
    assert datasets.discover(tmp_path) == {}     # empty data/ is not an error
    _make_drop(tmp_path, name="Celeb-DF")
    assert list(datasets.discover(tmp_path)) == ["Celeb-DF"]


def test_load_split_builds_a_manifest_from_meta_data(tmp_path):
    _make_drop(tmp_path)
    ds = datasets.discover(tmp_path)["FakeAVCeleb_v1.2"]
    df = datasets.load_split(ds, datasets.RAW_SPLIT, tmp_path)
    assert len(df) == 2                          # the missing third clip is dropped
    assert set(df["clip_id"]) == {"A__id00076__00109", "A__id00076__00110"}
    assert list(df["label"]) == [0, 0]
    # video_path stays relative to data/, the way every consumer resolves clips
    assert (tmp_path / df["video_path"].iloc[0]).exists()


def test_load_split_reads_a_manifest_csv_verbatim(tmp_path):
    _make_drop(tmp_path)
    _write_manifest(tmp_path / "val.csv", ["FakeAVCeleb_v1.2/x.mp4", "FakeAVCeleb_v1.2/y.mp4"])
    ds = datasets.discover(tmp_path)["FakeAVCeleb_v1.2"]
    assert list(datasets.load_split(ds, "val", tmp_path)["clip_id"]) == ["c0", "c1"]


def test_unknown_split_raises(tmp_path):
    _make_drop(tmp_path)
    ds = datasets.discover(tmp_path)["FakeAVCeleb_v1.2"]
    with pytest.raises(KeyError):
        datasets.load_split(ds, "train", tmp_path)
