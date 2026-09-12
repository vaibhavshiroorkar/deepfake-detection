import sys
from dataclasses import replace
from pathlib import Path
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

# Bound before the autouse fixture stubs the module attribute, so the preflight
# tests below can still reach the real function.
_REQUIRE_CUDA = runtime.require_cuda


@pytest.fixture(autouse=True)
def _allow_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the CUDA preflight so these stay host-independent.

    The preflight itself is covered below against a stubbed torch.
    """
    monkeypatch.setattr(runtime, "require_cuda", lambda: None)


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
        evaluation_run_id="evaluation-run-id",
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


def test_predict_upload_runs_the_multimodal_engine_on_temporary_upload_bytes(
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

    monkeypatch.setattr(runtime, "load_multimodal_gate_engine", FakeEngine)
    clip = UploadedClip("sample.mp4", ".mp4", b"video bytes", "a" * 64)

    result = runtime.predict_upload(clip)

    assert result == expected
    assert calls["content"] == b"video bytes"
    assert not calls["path"].exists()


def test_predict_upload_falls_back_to_the_visual_baseline_with_no_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A checkout with no trained fusion head still answers, through one model.

    The fallback has to be visible rather than silent, and it is: the visual
    engine reports no media kind, which is what makes the page label the verdict
    as the single-model baseline instead of a fusion.
    """
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
            return expected

    defaults = runtime.dashboard_defaults(root=Path.cwd())
    monkeypatch.setattr(
        runtime,
        "dashboard_defaults",
        lambda *, root: replace(
            defaults, gate_fusion_checkpoint=tmp_path / "absent.pt"
        ),
    )
    monkeypatch.setattr(runtime, "load_frozen_visual_engine", FakeEngine)

    result = runtime.predict_upload(UploadedClip("s.mp4", ".mp4", b"v", "a" * 64))

    assert result is expected
    assert result.media_kind == ""


def test_display_face_frames_reverses_imagenet_normalization() -> None:
    normalized = np.zeros((1, 3, 2, 2), dtype=np.float32)

    frames = display_face_frames(normalized)

    assert len(frames) == 1
    assert frames[0].shape == (2, 2, 3)
    assert frames[0].dtype == np.uint8
    assert frames[0][0, 0].tolist() == [123, 116, 103]


def test_prepare_uploaded_visual_uses_the_frozen_cuda_preprocessing_version(
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

    prepared = prepare_uploaded_visual(clip)

    assert calls["code_version"] == "2689577"
    assert calls["device"] == "cuda"
    assert calls["detector"] == "mtcnn"
    assert calls["tracker"] == "greedy_iou"
    assert calls["crop_mode"] == "box"
    assert calls["content"] == b"video"
    assert calls["record"].clip_id == clip.sha256
    assert not calls["prepared_path"].exists()
    assert prepared.preprocessing_config_hash.endswith("e0928c96")


def test_prepare_uploaded_visual_rejects_a_caller_device_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "build_preprocessor",
        lambda **values: pytest.fail("caller-controlled device reached the factory"),
    )
    clip = UploadedClip("sample.mp4", ".mp4", b"video", "a" * 64)

    with pytest.raises(TypeError, match="unexpected keyword argument 'device'"):
        prepare_uploaded_visual(clip, device="cpu")


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
        prepare_uploaded_visual(clip)


def _stub_torch(monkeypatch: pytest.MonkeyPatch, *, available: bool, cuda: str | None):
    """Install a fake torch so the preflight can be tested off a real GPU."""
    module = SimpleNamespace(
        __version__="2.12.1+cu130" if cuda else "2.12.1+cpu",
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: available),
    )
    monkeypatch.setitem(sys.modules, "torch", module)


def test_require_cuda_accepts_a_working_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_torch(monkeypatch, available=True, cuda="13.0")
    assert _REQUIRE_CUDA() is None


def test_require_cuda_names_a_cpu_only_build_and_the_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_torch(monkeypatch, available=False, cuda=None)
    with pytest.raises(RuntimeError) as failure:
        _REQUIRE_CUDA()
    message = str(failure.value)
    assert message.startswith("CUDA is unavailable:")
    assert "CPU-only build" in message
    assert "restart the server" in message
    assert "cu130" in message


def test_require_cuda_separates_a_missing_driver_from_a_cpu_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_torch(monkeypatch, available=False, cuda="13.0")
    with pytest.raises(RuntimeError) as failure:
        _REQUIRE_CUDA()
    message = str(failure.value)
    assert message.startswith("CUDA is unavailable:")
    assert "nvidia-smi" in message
    assert "CPU-only" not in message


def test_require_cuda_reports_a_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        _REQUIRE_CUDA()
