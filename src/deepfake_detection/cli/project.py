"""Project-level commands: `ddf run`, `ddf handoff`, `ddf smoke`, `ddf detector`, `ddf predict`."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from deepfake_detection.cli.common import (
    _run_logger,
    _write_json,
    add_checkpoint_arguments,
    add_preprocessing_arguments,
)
from deepfake_detection.digest import file_sha256
from deepfake_detection.experiments import (
    NullRunLogger,
    execute_configured_run,
    run_fusion_smoke,
)
from deepfake_detection.experiments.runner import _CONFIGURED_RUN_SENTINEL
from deepfake_detection.views.model_assets import fetch_yunet_model


def _handoff_update(arguments: argparse.Namespace) -> int:
    """Regenerate the handoff's generated blocks from the working tree.

    Called after every stage of a long run, so the handoff describes the state
    that actually exists rather than the state someone last remembered to
    write down.
    """
    from deepfake_detection.documentation.handoff import update_handoff

    changed = update_handoff(arguments.root, arguments.path)
    print(f"{arguments.path}: {'updated' if changed else 'already current'}")
    return 0


def _smoke(arguments: argparse.Namespace) -> int:
    run_fusion_smoke(
        arguments.output_dir,
        seed=arguments.seed,
        samples=arguments.samples,
        logger=getattr(arguments, "_run_logger", NullRunLogger()),
    )
    return 0


def _run_configured(arguments: argparse.Namespace) -> int:
    # Imported here: the package's `__init__` imports this module.
    from deepfake_detection.cli import build_parser

    return execute_configured_run(
        arguments.config,
        root=arguments.root,
        parser_factory=build_parser,
        disable_tracking=arguments.no_tracking,
    )


def _detector_fetch_yunet(arguments: argparse.Namespace) -> int:
    model = fetch_yunet_model(arguments.destination, force=arguments.force)
    digest = file_sha256(model)
    payload = {
        "asset": "opencv-zoo-yunet-2026may",
        "sha256": digest,
    }
    if arguments.report is not None:
        _write_json(arguments.report, payload)
        _run_logger(arguments).log_artifact(
            arguments.report,
            artifact_path="detector/aggregate",
        )
    _run_logger(arguments).log_params(
        {
            "detector.asset": payload["asset"],
            "detector.model_sha256": digest,
        }
    )
    return 0


def _predict(arguments: argparse.Namespace) -> int:
    from deepfake_detection.inference.loading import (
        InferenceConfig,
        load_prediction_engine,
    )

    engine = load_prediction_engine(
        InferenceConfig(
            visual_checkpoint=arguments.visual_checkpoint,
            audio_checkpoint=arguments.audio_checkpoint,
            sync_checkpoint=arguments.sync_checkpoint,
            fusion_model=arguments.fusion_model,
            code_version=arguments.code_version,
            threshold=arguments.threshold,
            audio_model=arguments.audio_model,
            device=arguments.device,
            detector=arguments.detector,
            tracker=arguments.tracker,
            crop_mode=arguments.crop_mode,
            model_path=arguments.model_path,
            expected_model_hash=arguments.expected_model_hash,
        )
    )
    result = engine.predict(arguments.video)
    _write_json(arguments.output, asdict(result))
    return 0


def register_run(commands) -> None:
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--config", type=Path, action="append", required=True)
    run.add_argument("--no-tracking", action="store_true")
    run.set_defaults(
        handler=_run_configured,
        _configured_run_sentinel=_CONFIGURED_RUN_SENTINEL,
    )


def register_handoff(commands) -> None:
    handoff = commands.add_parser("handoff")
    handoff_commands = handoff.add_subparsers(dest="handoff_command", required=True)
    handoff_update = handoff_commands.add_parser("update")
    handoff_update.add_argument("--root", type=Path, default=Path("."))
    handoff_update.add_argument("--path", type=Path, default=Path("docs/handoff.md"))
    handoff_update.set_defaults(handler=_handoff_update)


def register_smoke(commands) -> None:
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--samples", type=int, default=32)
    smoke.set_defaults(handler=_smoke)


def register_detector(commands) -> None:
    detector = commands.add_parser("detector")
    detector_commands = detector.add_subparsers(
        dest="detector_command",
        required=True,
    )
    detector_fetch = detector_commands.add_parser("fetch-yunet")
    detector_fetch.add_argument(
        "--destination",
        type=Path,
        default=Path("models/face_detection_yunet_2026may.onnx"),
    )
    detector_fetch.add_argument("--force", action="store_true")
    detector_fetch.add_argument("--report", type=Path)
    detector_fetch.set_defaults(handler=_detector_fetch_yunet)


def register_predict(commands) -> None:
    predict = commands.add_parser("predict")
    predict.add_argument("video", type=Path)
    predict.add_argument("--fusion-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--threshold", type=float, required=True)
    predict.add_argument("--code-version", required=True)
    add_checkpoint_arguments(predict)
    add_preprocessing_arguments(predict)
    predict.set_defaults(handler=_predict)
