"""Run records on disk: data/_evallab/runs/<run_id>/run.json (git-ignored). Atomic writes, in-memory mirror."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from pathlib import Path

from .common import DATA_DIR, jsonable

ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
ACTIVE = ("queued", "running")


class RunStore:
    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.dir = self.data_dir / "runs"
        self._mem: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._swept = False

    def _sweep(self) -> None:
        """Runs left queued or running by a process that no longer exists are marked failed."""
        if self._swept:
            return
        self._swept = True
        if not self.dir.is_dir():
            return
        for d in self.dir.iterdir():
            f = d / "run.json"
            if not f.is_file() or d.name in self._mem:
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if rec.get("state") in ACTIVE:
                rec["state"] = "failed"
                rec["error"] = {"code": "INTERRUPTED", "message": "the server stopped before this run finished"}
                self._write(rec)
            self._mem[rec["id"]] = rec

    def _write(self, rec: dict) -> None:
        d = self.dir / rec["id"]
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "run.json.tmp"
        tmp.write_text(json.dumps(jsonable(rec), indent=1), encoding="utf-8")
        os.replace(tmp, d / "run.json")

    def save(self, rec: dict) -> None:
        with self._lock:
            self._sweep()
            self._write(rec)
            self._mem[rec["id"]] = copy.deepcopy(jsonable(rec))

    def get(self, rid: str) -> dict | None:
        if not ID_RE.match(rid or ""):
            return None
        with self._lock:
            self._sweep()
            rec = self._mem.get(rid)
            if rec is None:
                f = self.dir / rid / "run.json"
                if f.is_file():
                    try:
                        rec = json.loads(f.read_text(encoding="utf-8"))
                        self._mem[rid] = rec
                    except (OSError, ValueError):
                        return None
            return copy.deepcopy(rec) if rec is not None else None

    def list(self) -> list[dict]:
        with self._lock:
            self._sweep()
            return sorted((copy.deepcopy(r) for r in self._mem.values()),
                          key=lambda r: (r.get("created", ""), r["id"]), reverse=True)
