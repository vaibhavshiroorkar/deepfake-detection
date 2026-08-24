from pathlib import Path

import pytest

from deepfake_detection.experiments.configuration import (
    configuration_argv,
    load_configuration,
)


def test_configuration_layers_merge_and_hash_deterministically(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text(
        """
schema_version: 1
command: [smoke]
arguments:
  output-dir: runs/base
  seed: 17
tracking:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  artifact_root: mlartifacts
  experiment_name: smoke-base
  run_name: base-seed17
""".lstrip(),
        encoding="utf-8",
    )
    override.write_text(
        """
schema_version: 1
arguments:
  output-dir: runs/override
tracking:
  run_name: override-seed17
""".lstrip(),
        encoding="utf-8",
    )

    first = load_configuration((base, override))
    second = load_configuration((base, override))

    assert first.values["arguments"]["seed"] == 17
    assert first.values["arguments"]["output-dir"] == "runs/override"
    assert first.values["tracking"]["experiment_name"] == "smoke-base"
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_configuration_argv_handles_flags_lists_and_false_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        """
schema_version: 1
command: [cache, build]
arguments:
  manifest: data/manifest.csv
  methods: [faceswap, wav2lip]
  keep-leading-silence: true
  external: false
tracking:
  enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    resolved = load_configuration((path,))

    assert configuration_argv(resolved) == (
        "cache",
        "build",
        "--keep-leading-silence",
        "--manifest",
        "data/manifest.csv",
        "--methods",
        "faceswap",
        "wav2lip",
    )


@pytest.mark.parametrize(
    "body, message",
    [
        ("- not-a-mapping\n", "mapping"),
        ("schema_version: 2\ncommand: [smoke]\narguments: {}\n", "schema"),
        ("schema_version: 1\ncommand: run\narguments: {}\n", "command"),
    ],
)
def test_configuration_rejects_invalid_contracts(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_configuration((path,))
