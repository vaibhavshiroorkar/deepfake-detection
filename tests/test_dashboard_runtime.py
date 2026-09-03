from types import SimpleNamespace

import numpy as np
import pytest

from deepfake_detection.dashboard import runtime
from deepfake_detection.dashboard.configuration import DashboardDefaults
from deepfake_detection.dashboard.runtime import (
    display_face_frames,
    prepare_uploaded_visual,
)
from deepfake_detection.dashboard.state import UploadedClip
from deepfake_detection.inference.predictor import PredictionResult


def test_load_frozen_visual_engine_uses_all_frozen_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "visual.pt"
    defaults = DashboardDefaults(
        visual_checkpoint=checkpoint,
        code_version="code-version",
        preprocessing_hash="p" * 64,
        checkpoint_sha256="c" * 64,
        run_id="run-id",
        split_hash="s" * 64,
        git_commit="git-commit",
        seed=23,
    )
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(runtime, "dashboard_defaults", lambda *, root: defaults)

    def fake_loader(config):
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(
        runtime, "load_visual_prediction_engine", fake_loader, raising=False
    )
    runtime.load_frozen_visual_engine.clear()

    engine = runtime.load_frozen_visual_engine()

    config = captured["config"]
    assert engine is sentinel
    assert config.visual_checkpoint == checkpoint
    assert config.code_version == "code-version"
    assert config.expected_checkpoint_sha256 == "c" * 64
    assert config.expected_run_id == "run-id"
    assert config.expected_split_hash == "s" * 64
    assert config.expected_git_commit == "git-commit"
    assert config.expected_seed == 23
    assert config.threshold == 0.5
    assert config.device == "cuda"


def test_predict_upload_runs_the_frozen_engine_on_temporary_upload_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    expected = PredictionResult(
        clip_id="clip",
        verdict="real",
        probability=0.25,
        branch_logits={"visual": -1.1},
        blockers=(),
        preprocessing_fingerprint="fixture",
    )

    class FakeEngine:
        def predict(self, path):
            calls["path"] = path
            calls["content"] = path.read_bytes()
            return expected

    monkeypatch.setattr(runtime, "load_frozen_visual_engine", FakeEngine)
    clip = UploadedClip("sample.mp4", ".mp4", b"video bytes", "a" * 64)

    result = runtime.predict_upload(clip)

    assert result == expected
    assert calls["content"] == b"video bytes"
    assert not calls["path"].exists()


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
                    "fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96"
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
