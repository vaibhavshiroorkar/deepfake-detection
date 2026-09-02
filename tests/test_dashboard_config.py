from pathlib import Path

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
    assert defaults.seed == 17


def test_dashboard_defaults_bind_the_frozen_training_protocol() -> None:
    defaults = dashboard_defaults(root=Path("project"))

    assert defaults.split_hash == (
        "3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c"
    )
    assert defaults.git_commit == ("268957796d366a81b5ab897dd1a4f523f1dc4b11")
