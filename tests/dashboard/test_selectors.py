import pandas as pd
from dashboard.lib.selectors import filter_manifest


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
