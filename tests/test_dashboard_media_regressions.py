import json
import os
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from deepfake_detection.dashboard.lib import media, results, selectors  # noqa: E402


@pytest.mark.parametrize("operation", ["playback", "frames", "crops"])
def test_replacing_media_at_same_path_invalidates_cache(
    tmp_path, monkeypatch, operation
):
    st.cache_data.clear()
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"a")
    monkeypatch.setattr(
        media, "playable_video_bytes", lambda p: (Path(p).read_bytes(), None)
    )
    monkeypatch.setattr(media, "decode_frames", lambda p, ts: [Path(p).read_bytes()])
    monkeypatch.setattr(media, "get_detector", lambda name: (None, "cpu"))
    monkeypatch.setattr(media, "detect_face_and_mouth", lambda f, *args: (f, f, True))
    calls = {
        "playback": lambda: media.cached_playable_video(path)[0],
        "frames": lambda: media.cached_decode_frames(path, [0.0])[0],
        "crops": lambda: media.cached_face_mouth(path, [0.0], 0.9, 0.2)[0][0],
    }
    assert calls[operation]() == b"a"
    path.write_bytes(b"new")
    os.utime(
        path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2_000_000_000)
    )
    assert calls[operation]() == b"new"


def test_explainability_picker_preserves_reports_for_same_branch(tmp_path, monkeypatch):
    directory = tmp_path / results.PROGRAM_RUN / "evaluation"
    directory.mkdir(parents=True)
    for seed in (17, 29):
        (directory / f"visual-seed{seed}-metrics.json").write_text(
            json.dumps(
                {
                    "branch": "visual",
                    "dataset": "DFDC",
                    "seed": seed,
                    "architecture": "efficientnet_b0",
                    "checkpoint_sha256": str(seed),
                    "per_method": {"fake": {"accuracy": 0.8}},
                }
            )
        )
    reports = results.load_branch_breakdowns(root=tmp_path)
    monkeypatch.setattr(results, "load_branch_breakdowns", lambda: reports)
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/explainability.py"
    ).run()
    assert not page.exception
    assert len(page.selectbox[0].options) == 2
    assert {report.seed for report in reports} == {17, 29}
    assert all(report.report_path.is_file() for report in reports)


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_preprocess_routes_nonvideo_media(tmp_path, monkeypatch, kind):
    path = tmp_path / ("clip.wav" if kind == "audio" else "clip.png")
    if kind == "audio":
        with wave.open(str(path), "wb") as out:
            out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            out.writeframes(
                (np.sin(np.arange(32000) / 20) * 1000).astype("int16").tobytes()
            )
    else:
        import cv2

        cv2.imwrite(str(path), np.zeros((224, 224, 3), np.uint8))
    row = pd.Series({"clip_id": "fixture", "video_path": str(path), "label": 0})
    monkeypatch.setattr(selectors, "render_selection", lambda **kwargs: row)
    monkeypatch.setattr(
        media,
        "cached_face_mouth",
        lambda *args, **kwargs: ([np.zeros((224, 224, 3), np.uint8)], [], 0, 0.0),
    )

    def refuse_video(*args):
        raise AssertionError("Nonvideo media reached video playback")

    monkeypatch.setattr(media, "cached_playable_video", refuse_video)
    page = AppTest.from_file(
        "src/deepfake_detection/dashboard/pages/preprocess.py", default_timeout=60
    ).run()
    assert not page.exception, [e.message for e in page.exception]
    messages = [item.value for item in page.info]
    assert (
        any("no picture" in message.lower() for message in messages)
        if kind == "audio"
        else any("no audio" in message.lower() for message in messages)
    )
