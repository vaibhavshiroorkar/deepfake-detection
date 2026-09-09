"""Where the dashboard looks for things on disk.

One home for the three roots the pages resolve against, so a page never rebuilds
a path from `__file__` and lands somewhere different from its neighbour.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
