"""The Evaluation Lab object: catalogue, run store, table cache and runner wired to one data directory."""

from __future__ import annotations

from pathlib import Path

from dfd.eval.ledger import DEFAULT_LEDGER

from .catalogue import Catalogue
from .common import DATA_DIR, REPO_ROOT
from .runner import Runner
from .store import RunStore
from .tables import TableCache


class EvalLab:
    def __init__(self, root: Path = REPO_ROOT, dirs=None, data_dir: Path | None = None, ledger_path=None,
                 synchronous: bool = False):
        self.root = Path(root)
        self.data_dir = Path(data_dir) if data_dir else (self.root / "data" / "_evallab" if root != REPO_ROOT else DATA_DIR)
        self.ledger_path = Path(ledger_path) if ledger_path else DEFAULT_LEDGER
        self.catalogue = Catalogue(self.root, dirs, self.data_dir / "cache")
        self.store = RunStore(self.data_dir)
        self.tables = TableCache()
        self.runner = Runner(self.catalogue, self.store, self.tables, self.ledger_path, self.root, synchronous)
