import pandas as pd
from dashboard.lib.selectors import (
    filter_manifest, available_splits, load_manifest, DATASETS,
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


def test_all_four_datasets_are_registered():
    assert set(DATASETS) == {
        "FakeAVCeleb", "Deepfake-Eval-2024", "FaceForensics++", "Celeb-DF",
    }


def test_fakeavceleb_splits_available_others_not():
    # FakeAVCeleb manifests exist (built by the pipeline); the rest have none yet.
    assert set(available_splits("FakeAVCeleb")) >= {"train", "val", "test"}
    assert available_splits("Celeb-DF") == []
    assert available_splits("FaceForensics++") == []


def test_load_manifest_reads_fakeavceleb_train():
    df = load_manifest("FakeAVCeleb", "train")
    assert len(df) > 0
    assert "clip_id" in df.columns and "label" in df.columns


def test_filter_skips_missing_columns_gracefully():
    # A dataset without manipulation_type/method must not KeyError when those
    # filters are empty (the render layer only shows them when present).
    minimal = pd.DataFrame({"clip_id": ["x", "y"], "label": [0, 1]})
    out = filter_manifest(minimal, [], [], "fake")
    assert list(out["clip_id"]) == ["y"]
