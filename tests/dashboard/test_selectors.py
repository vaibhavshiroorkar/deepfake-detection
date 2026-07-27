from pathlib import Path

import pandas as pd
from dashboard.lib import selectors
from dashboard.lib.selectors import (
    filter_manifest, search_manifest, group_by_identity, clip_path, label_text,
    upload_row, LABEL_UNKNOWN,
)


def _df():
    return pd.DataFrame({
        "clip_id": ["a", "b", "c", "d"],
        "label": [0, 1, 1, 1],
        "manipulation_type": ["RealVideo-RealAudio", "FakeVideo-FakeAudio",
                              "FakeVideo-RealAudio", "RealVideo-FakeAudio"],
        "method": ["real", "wav2lip", "faceswap", "real"],
    })


def test_empty_filters_return_everything():
    out = filter_manifest(_df(), [], [], "all")
    assert list(out["clip_id"]) == ["a", "b", "c", "d"]


def test_label_filter_real_only():
    out = filter_manifest(_df(), [], [], "real")
    assert list(out["clip_id"]) == ["a"]


def test_type_and_method_and_together():
    out = filter_manifest(_df(), ["FakeVideo-FakeAudio", "FakeVideo-RealAudio"],
                          ["wav2lip"], "fake")
    assert list(out["clip_id"]) == ["b"]


def test_empty_result_is_empty_frame_not_error():
    out = filter_manifest(_df(), ["FakeVideo-FakeAudio"], ["faceswap"], "all")
    assert len(out) == 0


def test_filter_skips_missing_columns_gracefully():
    # A dataset without manipulation_type/method must not KeyError when those
    # filters are empty (the render layer only shows them when present).
    minimal = pd.DataFrame({"clip_id": ["x", "y"], "label": [0, 1]})
    out = filter_manifest(minimal, [], [], "fake")
    assert list(out["clip_id"]) == ["y"]


def _srcdf():
    return pd.DataFrame({
        "clip_id": ["A__id2__a", "C__id1__b", "A__id1__c", "D__id1__d"],
        "source": ["id2", "id1", "id1", "id1"],
        "manipulation_type": ["RealVideo-RealAudio", "FakeVideo-RealAudio",
                              "RealVideo-RealAudio", "FakeVideo-FakeAudio"],
        "label": [0, 1, 0, 1],
    })


def test_search_matches_clip_id_and_source_case_insensitive():
    out = search_manifest(_srcdf(), "ID1")
    assert set(out["clip_id"]) == {"C__id1__b", "A__id1__c", "D__id1__d"}


def test_search_empty_query_returns_all():
    assert len(search_manifest(_srcdf(), "  ")) == 4


def test_group_by_identity_keeps_same_source_adjacent():
    grouped = group_by_identity(_srcdf())
    sources = list(grouped["source"])
    # every source's rows form one contiguous block
    assert sources == sorted(sources)
    assert sources.count("id1") == 3 and sources[0] == "id1"


# ------------------------------------------------- clip paths and uploaded rows
#
# A dataset row's video_path is relative to data/; an uploaded clip's is absolute
# and outside it. Both flow through the same pages, so clip_path is the only
# place that difference is allowed to matter.

def test_clip_path_resolves_a_dataset_row_under_data_dir():
    row = {"video_path": "FakeAVCeleb_v1.2/RealVideo-RealAudio/x/00109.mp4"}
    assert clip_path(row) == selectors.DATA_DIR / row["video_path"]


def test_clip_path_leaves_an_absolute_upload_path_alone(tmp_path):
    absolute = tmp_path / "my_clip.mp4"
    assert clip_path({"video_path": str(absolute)}) == absolute
    # and it must NOT have been reparented under data/
    assert selectors.DATA_DIR not in clip_path({"video_path": str(absolute)}).parents


def test_label_text_covers_real_fake_and_unknown():
    assert label_text({"label": 0}) == "real"
    assert label_text({"label": 1}) == "fake"
    assert label_text({"label": LABEL_UNKNOWN}) == "unknown"


def test_upload_row_writes_the_file_and_reads_back_as_a_clip(monkeypatch, tmp_path):
    monkeypatch.setattr(selectors, "UPLOAD_DIR", tmp_path / "uploads")
    row = upload_row("Holiday Clip.mp4", b"not-really-video", "fid123")

    assert row["clip_id"] == "Holiday Clip"
    assert row["label"] == LABEL_UNKNOWN
    assert label_text(row) == "unknown"
    written = clip_path(row)
    assert written.is_file() and written.read_bytes() == b"not-really-video"
    # never into data/ — the dashboard is read-only with respect to the dataset
    assert selectors.DATA_DIR not in written.parents


def test_upload_row_is_idempotent_and_keyed_by_file_id(monkeypatch, tmp_path):
    monkeypatch.setattr(selectors, "UPLOAD_DIR", tmp_path / "uploads")
    first = upload_row("a.mp4", b"one", "fid1")
    again = upload_row("a.mp4", b"IGNORED", "fid1")
    assert clip_path(again) == clip_path(first)
    assert clip_path(again).read_bytes() == b"one"      # not rewritten on rerun

    # same filename, different upload: must not collide
    other = upload_row("a.mp4", b"two", "fid2")
    assert clip_path(other) != clip_path(first)
    assert clip_path(other).read_bytes() == b"two"
