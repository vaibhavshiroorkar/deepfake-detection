import json
from pathlib import Path

import pytest

from deepfake_detection.dashboard import configuration
from deepfake_detection.dashboard.configuration import dashboard_defaults


def test_dashboard_defaults_find_the_local_visual_baseline() -> None:
    root = Path("project")

    defaults = dashboard_defaults(root=root)

    assert defaults.visual_checkpoint == (
        root / "runs" / "initial-20260902" / "visual-initial.pt"
    )
    assert defaults.code_version == "2689577"
    assert defaults.checkpoint_sha256 == (
        "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0"
    )
    assert defaults.run_id == "4243b35e64c743b89cc33000cc9d3d3e"
    assert defaults.evaluation_run_id == "56182266f70a424581f763b2d3b41989"
    assert defaults.seed == 17


def test_dashboard_defaults_bind_the_frozen_training_protocol() -> None:
    defaults = dashboard_defaults(root=Path("project"))

    assert defaults.preprocessing_hash == (
        "fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96"
    )
    assert defaults.split_hash == (
        "3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c"
    )
    assert defaults.git_commit == ("268957796d366a81b5ab897dd1a4f523f1dc4b11")


def test_defaults_load_a_promoted_baseline_from_configs(tmp_path: Path) -> None:
    """Promoting a baseline must be a data change, not a code edit."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "frozen-baseline.json").write_text(
        json.dumps(
            {
                "run_id": "new-run",
                "validation_rows": 3288,
                "dataset": "FakeAVCeleb",
                "visual_checkpoint": "runs/full-20260904/checkpoints/full.pt",
            }
        ),
        encoding="utf-8",
    )

    defaults = configuration.dashboard_defaults(root=tmp_path)

    assert defaults.run_id == "new-run"
    assert defaults.validation_rows == 3288
    assert defaults.visual_checkpoint == tmp_path / "runs/full-20260904/checkpoints/full.pt"
    # Unspecified keys keep the current baseline's values.
    assert defaults.seed == 17


def test_defaults_reject_an_unknown_baseline_key(tmp_path: Path) -> None:
    """A typo must fail loudly instead of silently keeping the old baseline."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "frozen-baseline.json").write_text(
        json.dumps({"run-id": "typo"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown keys"):
        configuration.dashboard_defaults(root=tmp_path)


def test_defaults_fall_back_when_no_baseline_file_exists(tmp_path: Path) -> None:
    defaults = configuration.dashboard_defaults(root=tmp_path)

    assert defaults.run_id == "4243b35e64c743b89cc33000cc9d3d3e"
    assert defaults.validation_rows == 400
