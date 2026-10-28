"""Shared constants and small helpers of the Evaluation Lab (no model, no training, no ledger write)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "_evallab"
DEFAULT_DIRS = ("data/_final/scores", "data/_evallab/imports")
ENV_DIRS = "DFD_EVALLAB_DIRS"

ROLES = {
    "train": "model development",
    "val": "validation (threshold and temperature are fitted here)",
    "dev_x": "development slice (repeatable, not final)",
    "test_in": "final held-out (sealed)",
    "test_x": "final held-out (sealed)",
}

METRICS = ("auc", "eer", "precision", "recall", "specificity", "fpr", "fnr", "f1", "accuracy",
           "balanced_accuracy")
PURPOSES = ("development", "validation", "final_reanalysis")
THRESHOLD_SOURCES = ("val_fit", "fixed")
BREAKDOWN_COLUMNS = ("generator", "source")
HASH_COLUMNS = ("sha256", "file_hash")
N_BOOT_DEFAULT = 2000
N_BOOT_MAX = 5000
N_BOOT_MIN = 100
SEED_DEFAULT = 20261027

LIMITATIONS = (
    "The Evaluation Lab re-analyses stored score files with the project's own metrics and group bootstrap. "
    "It does not run models, train, or approve anything. It cannot verify that the training data excluded the "
    "evaluated items beyond what the score file records (split labels, group ids and, when present, file hashes). "
    "A final re-analysis of a logged evaluation is not a new held-out look: the held-out split was already used "
    "once, and repeated re-analysis must not feed back into model or threshold choices."
)


class LabError(Exception):
    """An error with a stable code, shown to the client as {error: {code, message}}."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def repo_rel(path, root: Path = REPO_ROOT) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def file_id(rel_path: str) -> str:
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


def configured_dirs(override=None) -> list[str]:
    """Directories to scan: an explicit override, else env DFD_EVALLAB_DIRS (';' separated), else the defaults."""
    if override is not None:
        return [str(d) for d in override]
    env = os.environ.get(ENV_DIRS, "").strip()
    if env:
        return [d.strip() for d in env.split(";") if d.strip()]
    return list(DEFAULT_DIRS)


def jsonable(o):
    """Plain JSON types: numpy scalars converted, NaN and inf become null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def canonical_json(o) -> str:
    return json.dumps(jsonable(o), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
