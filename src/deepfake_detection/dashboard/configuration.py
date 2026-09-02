from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DashboardDefaults:
    visual_checkpoint: Path
    code_version: str
    preprocessing_hash: str
    checkpoint_sha256: str
    run_id: str
    split_hash: str
    git_commit: str
    seed: int


def dashboard_defaults(
    *,
    root: Path,
) -> DashboardDefaults:
    default_checkpoint = root / "runs" / "initial-20260902" / "visual-initial.pt"
    return DashboardDefaults(
        visual_checkpoint=default_checkpoint,
        code_version="2689577",
        preprocessing_hash=(
            "fd372dbe6bb64f359db4d57b05c3b5cd"
            "27ed6660f2bb8bdc50567224e0928c96"
        ),
        checkpoint_sha256=(
            "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0"
        ),
        run_id="4243b35e64c743b89cc33000cc9d3d3e",
        split_hash=("3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c"),
        git_commit="268957796d366a81b5ab897dd1a4f523f1dc4b11",
        seed=17,
    )
