"""Helpers shared by every command group: JSON output, run logging, hashes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from deepfake_detection.experiments import (
    NullRunLogger,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "uncommitted"
    # The command has fixed arguments and never invokes a shell.
    process = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "uncommitted"


def _training_cost_metrics(
    *,
    train_samples: int,
    completed_epochs: int,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
) -> dict[str, int | float]:
    training_examples = train_samples * completed_epochs
    return {
        "training_examples": training_examples,
        "samples_per_second": training_examples / elapsed_seconds,
        "peak_gpu_memory_mib": peak_gpu_memory_bytes / (1024**2),
    }


def _configuration_hash(
    arguments: argparse.Namespace,
    run_config: object,
    fallback: Callable[[object], str],
) -> str:
    resolved = getattr(arguments, "_config_hash", None)
    return resolved if resolved is not None else fallback(run_config)


def _run_logger(arguments: argparse.Namespace):
    return getattr(arguments, "_run_logger", NullRunLogger())


def add_preprocessing_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--detector",
        choices=("mtcnn", "yunet"),
        default="mtcnn",
    )
    command.add_argument(
        "--tracker",
        choices=("greedy_iou", "constant_velocity"),
        default="greedy_iou",
    )
    command.add_argument(
        "--crop-mode",
        choices=("box", "landmark"),
        default="box",
    )
    command.add_argument("--model-path", type=Path)
    command.add_argument("--expected-model-hash")


def add_branch_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--train-manifest", type=Path, required=True)
    command.add_argument("--validation-manifest", type=Path, required=True)
    command.add_argument("--cache-index", type=Path, required=True)
    command.add_argument("--cache-root", type=Path, required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--history", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--split-hash", required=True)
    command.add_argument("--preprocessing-hash", required=True)
    command.add_argument("--device", default="cuda")
    command.add_argument("--epochs", type=int, default=12)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--accumulation-steps", type=int, default=4)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--weight-decay", type=float, default=1e-4)
    command.add_argument("--patience", type=int, default=3)
    command.add_argument("--workers", type=int, default=0)
    command.add_argument("--seed", type=int, default=17)


def add_checkpoint_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--visual-checkpoint", type=Path, required=True)
    command.add_argument("--audio-checkpoint", type=Path, required=True)
    command.add_argument("--sync-checkpoint", type=Path, required=True)
    command.add_argument(
        "--audio-model",
        default="facebook/wav2vec2-base",
    )
    command.add_argument("--device", default="cuda")
