from types import SimpleNamespace

import numpy as np
import pytest

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.runtime import (
    display_face_frames,
    prepare_uploaded_visual,
)
from deepfake_detection.dashboard.state import UploadedClip


def test_display_face_frames_reverses_imagenet_normalization() -> None:
    normalized = np.zeros((1, 3, 2, 2), dtype=np.float32)

    frames = display_face_frames(normalized)

    assert len(frames) == 1
    assert frames[0].shape == (2, 2, 3)
    assert frames[0].dtype == np.uint8
    assert frames[0][0, 0].tolist() == [123, 116, 103]


def test_prepare_uploaded_visual_uses_the_frozen_preprocessing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakePreprocessor:
        def prepare_visual(self, record, path):
            calls["record"] = record
            calls["prepared_path"] = path
            calls["content"] = path.read_bytes()
            return SimpleNamespace(
                preprocessing_config_hash=(
                    "fd372dbe6bb64f359db4d57b05c3b5cd"
                    "27ed6660f2bb8bdc50567224e0928c96"
                )
            )

    def fake_factory(**values):
        calls.update(values)
        return FakePreprocessor()

    monkeypatch.setattr(runtime, "build_preprocessor", fake_factory)
    clip = UploadedClip("sample.mp4", ".mp4", b"video", "a" * 64)

    prepared = prepare_uploaded_visual(clip, device="cpu")

    assert calls["code_version"] == "2689577"
    assert calls["device"] == "cpu"
    assert calls["detector"] == "mtcnn"
    assert calls["tracker"] == "greedy_iou"
    assert calls["crop_mode"] == "box"
    assert calls["content"] == b"video"
    assert calls["record"].clip_id == clip.sha256
    assert not calls["prepared_path"].exists()
    assert prepared.preprocessing_config_hash.endswith("e0928c96")


def test_prepare_uploaded_visual_rejects_mismatched_preprocessing_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreprocessor:
        def prepare_visual(self, record, path):
            return SimpleNamespace(preprocessing_config_hash="wrong")

    monkeypatch.setattr(
        runtime,
        "build_preprocessor",
        lambda **values: FakePreprocessor(),
    )
    clip = UploadedClip("sample.mp4", ".mp4", b"video", "a" * 64)

    with pytest.raises(ValueError, match="preprocessing.*checkpoint"):
        prepare_uploaded_visual(clip, device="cpu")
