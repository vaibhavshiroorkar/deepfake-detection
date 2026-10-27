from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deepfake_detection.cli import build_parser
from deepfake_detection.experiments.configuration import load_configuration
from deepfake_detection.experiments.runtime import RuntimeSnapshot
from deepfake_detection.inference import loading
from deepfake_detection.views.model_assets import YUNET_SHA256


def _runtime() -> dict[str, object]:
    return RuntimeSnapshot(
        started_at_utc="2026-08-25T00:00:00+00:00",
        git_commit="fixture",
        git_dirty=False,
        python_version="3.11",
        platform="fixture-platform",
        packages={"opencv-python": "fixture"},
        cpu="fixture-cpu",
        gpu=None,
        gpu_memory_mib=None,
        available_memory_mib=1024,
        ffmpeg_version="fixture-ffmpeg",
    ).as_dict()


def test_shared_preprocessor_factory_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundFixtureMTCNN:
        def __init__(self, **values: object) -> None:
            self.values = values

        def model_sha256(self) -> str:
            return "a" * 64

    monkeypatch.setattr(loading, "MTCNNFaceDetector", BoundFixtureMTCNN)

    preprocessor = loading.build_preprocessor(code_version="revision", device="cpu")

    assert isinstance(preprocessor.detector, BoundFixtureMTCNN)
    assert preprocessor.detector.values == {"confidence": 0.8, "device": "cpu"}
    assert preprocessor.config.detector == "mtcnn"
    assert preprocessor.config.track_association == "greedy_iou"
    assert preprocessor.config.mouth_crop_mode == "box"
    assert preprocessor.config.detector_model_sha256 == "a" * 64


def test_shared_preprocessor_factory_checks_yunet_model_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"pinned-yunet")
    sentinel = object()
    monkeypatch.setattr(
        loading,
        "YuNetFaceDetector",
        lambda **values: (sentinel, values),
    )

    with pytest.raises(ValueError, match="hash"):
        loading.build_preprocessor(
            code_version="revision",
            detector="yunet",
            model_path=model,
            expected_model_hash="0" * 64,
        )


def test_shared_factory_rejects_claimed_mtcnn_hash_and_uses_observed_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundMTCNN:
        def __init__(self, **values: object) -> None:
            self.values = values

        def model_sha256(self) -> str:
            return "b" * 64

    monkeypatch.setattr(loading, "MTCNNFaceDetector", BoundMTCNN)

    with pytest.raises(ValueError, match="MTCNN.*expected model hash"):
        loading.build_preprocessor(
            code_version="revision",
            detector="mtcnn",
            expected_model_hash="a" * 64,
        )

    preprocessor = loading.build_preprocessor(
        code_version="revision",
        detector="mtcnn",
    )
    assert preprocessor.config.detector_model_sha256 == "b" * 64


def test_cache_and_predict_parsers_preserve_and_accept_preprocessing_choices() -> None:
    cache_defaults = build_parser().parse_args(
        [
            "cache",
            "build",
            "--manifest",
            "manifest.csv",
            "--dataset-root",
            "data",
            "--cache-root",
            "cache",
            "--index",
            "index.csv",
            "--audit",
            "audit.json",
            "--dataset",
            "fixture",
            "--code-version",
            "revision",
        ]
    )
    predict = build_parser().parse_args(
        [
            "predict",
            "clip.mp4",
            "--fusion-model",
            "fusion.joblib",
            "--visual-checkpoint",
            "visual.pt",
            "--audio-checkpoint",
            "audio.pt",
            "--sync-checkpoint",
            "sync.pt",
            "--output",
            "prediction.json",
            "--threshold",
            "0.5",
            "--code-version",
            "revision",
            "--detector",
            "yunet",
            "--tracker",
            "constant_velocity",
            "--crop-mode",
            "landmark",
            "--model-path",
            "models/yunet.onnx",
            "--expected-model-hash",
            "a" * 64,
        ]
    )

    assert cache_defaults.detector == "mtcnn"
    assert cache_defaults.tracker == "greedy_iou"
    assert cache_defaults.crop_mode == "box"
    assert cache_defaults.model_path is None
    assert cache_defaults.expected_model_hash is None
    assert predict.detector == "yunet"
    assert predict.tracker == "constant_velocity"
    assert predict.crop_mode == "landmark"
    assert predict.model_path == Path("models/yunet.onnx")
    assert predict.expected_model_hash == "a" * 64


def test_detector_configs_define_explicit_landmark_preprocessing_variants() -> None:
    mtcnn = load_configuration((Path("configs/detectors/mtcnn-landmark.yaml"),))
    yunet = load_configuration((Path("configs/detectors/yunet-landmark.yaml"),))

    assert mtcnn.values["arguments"] == {
        "audit": "runs/cache/mtcnn-landmark-audit.json",
        "cache-root": "cache/mtcnn-landmark",
        "code-version": "phase-2a",
        "crop-mode": "landmark",
        "dataset": "training",
        "dataset-root": "data/private",
        "detector": "mtcnn",
        "expected-model-hash": None,
        "index": "runs/cache/mtcnn-landmark-index.csv",
        "manifest": "data/private/train-manifest.csv",
        "model-path": None,
        "tracker": "greedy_iou",
    }
    assert yunet.values["arguments"]["detector"] == "yunet"
    assert yunet.values["arguments"]["expected-model-hash"] == (
        "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
    )
    assert yunet.values["arguments"]["model-path"] == (
        "models/face_detection_yunet_2026may.onnx"
    )


def test_inference_config_preserves_preprocessing_defaults() -> None:
    config = loading.InferenceConfig(
        visual_checkpoint=Path("visual.pt"),
        audio_checkpoint=Path("audio.pt"),
        sync_checkpoint=Path("sync.pt"),
        fusion_model=Path("fusion.joblib"),
        code_version="revision",
    )

    assert config.detector == "mtcnn"
    assert config.tracker == "greedy_iou"
    assert config.crop_mode == "box"
    assert config.model_path is None
    assert config.expected_model_hash is None


def test_real_yunet_load_is_opt_in_when_the_pinned_model_is_present() -> None:
    model = Path("models/face_detection_yunet_2026may.onnx")
    if not model.is_file():
        pytest.skip("Pinned YuNet model is not present")
    with model.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == YUNET_SHA256

    preprocessor = loading.build_preprocessor(
        code_version="real-yunet-load",
        detector="yunet",
        model_path=model,
        expected_model_hash=YUNET_SHA256,
    )

    assert preprocessor.config.detector == "yunet"


def test_detector_command_group_keeps_only_the_asset_fetch() -> None:
    detector = next(
        action for action in build_parser()._actions if action.dest == "command"
    ).choices["detector"]
    commands = next(
        action for action in detector._actions if action.dest == "detector_command"
    ).choices

    assert set(commands) == {"fetch-yunet"}
