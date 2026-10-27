"""The `ddf` command line.

Each command group lives in its own module with its handlers and its part
of the parser. This module only assembles the parser in a fixed order, so
`ddf --help` and the generated CLI reference stay stable.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from deepfake_detection.cli.data import (
    register_cache,
    register_manifest,
    register_split,
)
from deepfake_detection.cli.evaluate import register_evaluate, register_threshold
from deepfake_detection.cli.fusion import register_features
from deepfake_detection.cli.project import (
    register_detector,
    register_handoff,
    register_predict,
    register_run,
    register_smoke,
)
from deepfake_detection.cli.train import register_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddf")
    commands = parser.add_subparsers(dest="command", required=True)
    register_run(commands)
    register_handoff(commands)
    register_smoke(commands)
    register_manifest(commands)
    register_split(commands)
    register_cache(commands)
    register_detector(commands)
    register_evaluate(commands)
    register_threshold(commands)
    register_train(commands)
    register_features(commands)
    register_predict(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return int(arguments.handler(arguments))
