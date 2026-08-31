from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from deepfake_detection.benchmarks.detector_metrics import (
    BOOTSTRAP_SEED,
    FROZEN_DETECTOR_RULE_REVISION,
    DetectorBenchmarkReport,
    DetectorLatency,
    DetectorMetrics,
    TrackerMetrics,
)
from deepfake_detection.cli import build_parser, main
from deepfake_detection.evaluation.bootstrap import BootstrapInterval
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


def _fixture_report(name: str) -> DetectorBenchmarkReport:
    metrics = DetectorMetrics(
        target_recall=1.0,
        false_detections_per_frame=0.0,
        non_target_detections_per_frame=0.0,
        non_target_candidate_count=0,
        landmark_nme=0.0,
        landmark_coverage=1.0,
        aligned_mouth_jitter=0.0,
    )
    trackers = tuple(
        TrackerMetrics(
            association=association,
            stable_track_coverage=1.0,
            abstention_rate=0.0,
            target_track_errors=0,
            tracked_frames=1,
            target_track_errors_per_1000=0.0,
        )
        for association in ("greedy_iou", "constant_velocity")
    )
    latency = DetectorLatency(
        timed_frames=1,
        median_ms=1.0,
        p95_ms=1.0,
        throughput_fps=1000.0,
        device="cpu",
        thread_count=1,
    )
    point_estimates = {
        "target_recall": 1.0,
        "false_detections_per_frame": 0.0,
        "non_target_detections_per_frame": 0.0,
        "non_target_candidate_count": 0.0,
        "landmark_nme": 0.0,
        "landmark_coverage": 1.0,
        "aligned_mouth_jitter": 0.0,
        "latency.median_ms": 1.0,
        "latency.p95_ms": 1.0,
        "latency.throughput_fps": 1000.0,
        "greedy_iou.stable_track_coverage": 1.0,
        "greedy_iou.abstention_rate": 0.0,
        "greedy_iou.target_track_errors_per_1000": 0.0,
        "constant_velocity.stable_track_coverage": 1.0,
        "constant_velocity.abstention_rate": 0.0,
        "constant_velocity.target_track_errors_per_1000": 0.0,
    }
    intervals = {
        key: BootstrapInterval(value, value, value, 1000)
        for key, value in point_estimates.items()
    }
    return DetectorBenchmarkReport(
        detector_name=name,
        detector_revision=f"{name}-fixture",
        model_sha256="a" * 64,
        threshold=0.5,
        collection_threshold=0.0,
        evidence_scope="software_fixture_only",
        rule_revision=FROZEN_DETECTOR_RULE_REVISION,
        frame_count=1,
        comparison_clip_count=1,
        source_count=1,
        metrics=metrics,
        trackers=trackers,
        latency=latency,
        intervals=intervals,
        runtime_snapshot=_runtime(),
        raw_results_sha256="b" * 64,
        evaluation_set_sha256="c" * 64,
        split_hash="d" * 64,
        identity_strict_split_hash="a" * 64,
        reviewed_sample_sha256="e" * 64,
        annotation_audit_sha256="f" * 64,
        annotation_audit_validated=True,
        bootstrap_seed=BOOTSTRAP_SEED,
    )


def test_detector_command_tree_exposes_all_operational_commands() -> None:
    detector = next(
        action for action in build_parser()._actions if action.dest == "command"
    ).choices["detector"]
    commands = next(
        action for action in detector._actions if action.dest == "detector_command"
    ).choices

    assert set(commands) == {
        "fetch-yunet",
        "sample",
        "validate-annotations",
        "run",
        "compare",
    }


def test_detector_compare_cli_rejects_the_unbound_downstream_score_option() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "detector",
                "compare",
                "--reports",
                "mtcnn.json",
                "yunet.json",
                "--downstream-validation",
                "scores.json",
                "--output",
                "decision.json",
            ]
        )


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


def test_detector_compare_fixture_smoke_cannot_select_a_real_detector(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    output = tmp_path / "decision.json"
    left.write_text(json.dumps(asdict(_fixture_report("left"))), encoding="utf-8")
    right.write_text(json.dumps(asdict(_fixture_report("right"))), encoding="utf-8")

    assert (
        main(
            [
                "detector",
                "compare",
                "--reports",
                str(left),
                str(right),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["selected_detector"] is None
    assert decision["selected_association"] is None
    assert decision["reason"] == "software_fixture_only"
    assert decision["input_report_sha256"] == {
        "left": hashlib.sha256(left.read_bytes()).hexdigest(),
        "right": hashlib.sha256(right.read_bytes()).hexdigest(),
    }


def test_detector_compare_rejects_fields_outside_the_aggregate_contract(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    payload = asdict(_fixture_report("fixture"))
    payload["private_annotation_path"] = "C:/private/annotations.jsonl"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown or missing fields"):
        main(
            [
                "detector",
                "compare",
                "--reports",
                str(report),
                "--output",
                str(tmp_path / "decision.json"),
            ]
        )


def test_detector_compare_rejects_invalid_nested_report_schemas(
    tmp_path: Path,
) -> None:
    base = asdict(_fixture_report("fixture"))
    mutations: list[tuple[str, dict[str, object]]] = []

    extra_metric = copy.deepcopy(base)
    extra_metric["metrics"]["private_path"] = "private/metric.json"
    mutations.append(("extra-metric", extra_metric))

    missing_latency = copy.deepcopy(base)
    del missing_latency["latency"]["thread_count"]
    mutations.append(("missing-latency", missing_latency))

    extra_runtime = copy.deepcopy(base)
    extra_runtime["runtime_snapshot"]["workspace"] = "private/workspace"
    mutations.append(("extra-runtime", extra_runtime))

    wrong_packages = copy.deepcopy(base)
    wrong_packages["runtime_snapshot"]["packages"] = ["opencv-python"]
    mutations.append(("wrong-packages", wrong_packages))

    for runtime_path in (
        "/private/runtime.json",
        "private/runtime.json",
        "C:\\private\\runtime.json",
        "private\\runtime.json",
    ):
        private_runtime = copy.deepcopy(base)
        private_runtime["runtime_snapshot"]["platform"] = runtime_path
        mutations.append((f"runtime-path-{len(mutations)}", private_runtime))

    for name, payload in mutations:
        report = tmp_path / f"{name}.json"
        report.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="schema|path-free"):
            main(
                [
                    "detector",
                    "compare",
                    "--reports",
                    str(report),
                    "--output",
                    str(tmp_path / f"{name}-decision.json"),
                ]
            )


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


def test_detector_compare_runs_through_configured_run(tmp_path: Path) -> None:
    report = tmp_path / "fixture.json"
    output = tmp_path / "decision.json"
    report.write_text(json.dumps(asdict(_fixture_report("fixture"))), encoding="utf-8")
    config = tmp_path / "compare.yaml"
    config.write_text(
        "schema_version: 1\n"
        "command: [detector, compare]\n"
        "arguments:\n"
        f"  reports: [{report.as_posix()}]\n"
        f"  output: {output.as_posix()}\n"
        "tracking:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )

    assert main(["run", "--root", str(tmp_path), "--config", str(config)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["reason"] == (
        "software_fixture_only"
    )


def test_validate_annotations_writes_aggregate_audit_without_copying_raw_rows(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.jsonl"
    annotations = tmp_path / "annotations.jsonl"
    report = tmp_path / "audit.json"
    sample.write_text("", encoding="utf-8")
    annotations.write_text("", encoding="utf-8")

    assert (
        main(
            [
                "detector",
                "validate-annotations",
                "--sample",
                str(sample),
                "--annotations",
                str(annotations),
                "--report",
                str(report),
            ]
        )
        == 2
    )
    audit = json.loads(report.read_text(encoding="utf-8"))
    assert audit["valid"] is False
    assert "annotations" not in audit


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
