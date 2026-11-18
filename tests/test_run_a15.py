import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "run_a15", Path(__file__).parents[1] / "scripts" / "run_a15.py"
)
a15 = importlib.util.module_from_spec(spec)
sys.modules["run_a15"] = a15  # dataclasses look their module up here
spec.loader.exec_module(a15)


def _clips(video_pred, audio_pred):
    # (video_true, audio_true): real, video only, audio only, both; one identity per clip.
    truth = [(0, 0), (1, 0), (0, 1), (1, 1)] * 8
    return [
        a15.Clip(
            f"c{i}",
            f"id{i}",
            v,
            a,
            int(v or a),
            (video_pred(v, a),) * 3,
            (audio_pred(v, a),) * 3,
        )
        for i, (v, a) in enumerate(truth)
    ]


def test_heads_that_name_the_right_stream_beat_the_clip_label_baseline() -> None:
    clips = _clips(lambda v, a: v, lambda v, a: a)
    assert a15.macro_f1(clips, 0) == 1.0
    assert a15.macro_f1(clips, None) == pytest.approx(
        0.8
    )  # baseline: precision 2/3, recall 1 per stream
    assert a15.difference(clips) == pytest.approx(0.2)


def test_heads_that_only_copy_the_clip_label_gain_nothing() -> None:
    clips = _clips(lambda v, a: int(v or a), lambda v, a: int(v or a))
    assert a15.difference(clips) == pytest.approx(0.0)


def test_bootstrap_reports_the_share_at_or_below_zero() -> None:
    result = a15.bootstrap(_clips(lambda v, a: v, lambda v, a: a), samples=200, seed=17)
    assert result["estimate"] == pytest.approx(0.2)
    assert result["p_one_sided"] < 0.05
    assert result["lower"] <= result["estimate"] <= result["upper"]
