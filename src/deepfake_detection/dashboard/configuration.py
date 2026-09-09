"""Which trained model the Evidence gate serves, and what its record must say.

These values bind the Evidence gate to one frozen checkpoint. They used to be
Python literals, which meant promoting a new baseline was a code edit in three
files, and the shape of the old baseline was welded into the validator: a
hardcoded row count of 400 and a hardcoded dataset name would both reject a new
record that was perfectly valid.

They now come from `configs/frozen-baseline.json` when it exists, so promoting a
baseline is a data change. The literals remain as the fallback, because the
current baseline predates the file and the dashboard has to keep working with
nothing added.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

BASELINE_CONFIG = Path("configs") / "frozen-baseline.json"


@dataclass(frozen=True, slots=True)
class DashboardDefaults:
    visual_checkpoint: Path
    code_version: str
    preprocessing_hash: str
    checkpoint_sha256: str
    run_id: str
    evaluation_run_id: str
    split_hash: str
    git_commit: str
    seed: int
    # What the evaluation record is required to say. Checked rather than
    # assumed, so a record from a different dataset or threshold cannot be
    # served as this baseline's evidence.
    dataset: str = "FakeAVCeleb"
    validation_rows: int = 400
    threshold: float = 0.5
    evidence_scope: str = "development_validation"


_FALLBACK = {
    "code_version": "2689577",
    "preprocessing_hash": (
        "fd372dbe6bb64f359db4d57b05c3b5cd27ed6660f2bb8bdc50567224e0928c96"
    ),
    "checkpoint_sha256": (
        "ac9a085e1017cf2743a7f78f3b632051c18acda695496d2f434c7d968fd627b0"
    ),
    "run_id": "4243b35e64c743b89cc33000cc9d3d3e",
    "evaluation_run_id": "56182266f70a424581f763b2d3b41989",
    "split_hash": "3255ae334536336c73058941285925f3dd5b094c02b1037e19f379c6f45db30c",
    "git_commit": "268957796d366a81b5ab897dd1a4f523f1dc4b11",
    "seed": 17,
    "dataset": "FakeAVCeleb",
    "validation_rows": 400,
    "threshold": 0.5,
    "evidence_scope": "development_validation",
    "visual_checkpoint": "runs/initial-20260902/visual-initial.pt",
}


def dashboard_defaults(
    *,
    root: Path,
) -> DashboardDefaults:
    values = dict(_FALLBACK)
    config = root / BASELINE_CONFIG
    if config.is_file():
        known = {field.name for field in fields(DashboardDefaults)}
        loaded = json.loads(config.read_text(encoding="utf-8"))
        unknown = sorted(set(loaded) - known)
        if unknown:
            raise ValueError(
                f"{config} has unknown keys: {', '.join(unknown)}. "
                f"Expected any of: {', '.join(sorted(known))}."
            )
        values.update(loaded)

    checkpoint = Path(values["visual_checkpoint"])
    return DashboardDefaults(
        visual_checkpoint=(
            checkpoint if checkpoint.is_absolute() else root / checkpoint
        ),
        code_version=str(values["code_version"]),
        preprocessing_hash=str(values["preprocessing_hash"]),
        checkpoint_sha256=str(values["checkpoint_sha256"]),
        run_id=str(values["run_id"]),
        evaluation_run_id=str(values["evaluation_run_id"]),
        split_hash=str(values["split_hash"]),
        git_commit=str(values["git_commit"]),
        seed=int(values["seed"]),
        dataset=str(values["dataset"]),
        validation_rows=int(values["validation_rows"]),
        threshold=float(values["threshold"]),
        evidence_scope=str(values["evidence_scope"]),
    )
