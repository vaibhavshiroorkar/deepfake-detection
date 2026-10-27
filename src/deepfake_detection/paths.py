"""Where the project looks for things on disk.

One home for the roots every package resolves against, so nothing rebuilds a
path from `__file__` and lands somewhere different from its neighbour.

The root is the checkout this file sits in. `DDF_PROJECT_ROOT` overrides it,
which an installed wheel needs because it does not live inside a checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("DDF_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
)
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
# Released third-party weights that no package index serves. Not committed.
MODELS_DIR = PROJECT_ROOT / "models"
AVHUBERT_CHECKPOINT = MODELS_DIR / "avhubert" / "base_vox_iter5.pt"
HSEMOTION_WEIGHTS = MODELS_DIR / "hsemotion" / "enet_b0_8_best_afew.pt"
