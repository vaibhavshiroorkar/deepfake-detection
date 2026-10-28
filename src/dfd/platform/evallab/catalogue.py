"""Catalogue of score files: scans the configured directories, summarises every CSV that passes the
dfd.eval score-file reader, and caches the summary keyed by (path, mtime, size) so that a rescan is a stat call.

Nothing is uploaded or sent anywhere; files are read from local disk only.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from dfd.eval.scorefile import read_scores

from .common import (DATA_DIR, DEFAULT_DIRS, REPO_ROOT, ROLES, LabError, configured_dirs, file_id, jsonable,
                     repo_rel)

CACHE_VERSION = 1


def summarise_table(table) -> dict:
    """Summary of a parsed ScoreTable: blocks per (dataset, split) and candidates per (iteration, modality)."""
    blocks = []
    keys = sorted({(str(d), str(s)) for d, s in zip(table.dataset, table.split)})
    for dataset, split in keys:
        m = (table.dataset == dataset) & (table.split == split)
        sid = table.sample_id[m]
        _, first = np.unique(sid, return_index=True)
        labels = table.label[m][first]
        blocks.append({
            "dataset": dataset, "split": split, "role": ROLES[split],
            "n_samples": int(first.size), "n_fake": int(labels.sum()), "n_real": int(first.size - labels.sum()),
            "n_groups": int(np.unique(table.group_id[m]).size),
            "runs": sorted({str(r) for r in table.run_id[m]}),
            "seeds": sorted({int(x) for x in table.seed[m]}),
        })
    candidates = []
    for iteration, modality in sorted({(str(i), str(mo)) for i, mo in zip(table.iteration, table.modality)}):
        m = (table.iteration == iteration) & (table.modality == modality)
        runs = sorted({str(r) for r in table.run_id[m]}, key=lambda r: (int(table.seed[table.run_id == r][0]), r))
        seeds = [int(table.seed[table.run_id == r][0]) for r in runs]
        ckeys = sorted({(str(d), str(s)) for d, s in zip(table.dataset[m], table.split[m])})
        candidates.append({"iteration": iteration, "modality": modality, "run_ids": runs, "seeds": seeds,
                           "blocks": [[d, s] for d, s in ckeys]})
    return {
        "sha256": table.sha256,
        "n_rows": int(len(table) + table.n_frame_rows),
        "iterations": sorted({str(i) for i in table.iteration}),
        "modalities": sorted({str(i) for i in table.modality}),
        "blocks": blocks,
        "candidates": candidates,
        "warnings": list(table.warnings),
    }


class Catalogue:
    def __init__(self, root: Path = REPO_ROOT, dirs=None, cache_dir: Path | None = None):
        self.root = Path(root)
        self._dirs = dirs
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_DIR / "cache"
        self._lock = threading.RLock()
        self._mem: dict | None = None

    # ------------------------------------------------------------------ cache
    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "catalogue.json"

    def _load_cache(self) -> dict:
        if self._mem is not None:
            return self._mem
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            files = data.get("files", {}) if data.get("version") == CACHE_VERSION else {}
        except (OSError, ValueError):
            files = {}
        self._mem = files
        return files

    def _save_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "files": self._mem}), encoding="utf-8")
        os.replace(tmp, self.cache_path)

    # ------------------------------------------------------------------- scan
    def dirs(self) -> list[Path]:
        out = []
        for d in configured_dirs(self._dirs):
            p = Path(d)
            out.append(p if p.is_absolute() else self.root / p)
        return out

    def _csv_files(self) -> list[Path]:
        seen, out = set(), []
        for d in self.dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*.csv")):
                rp = p.resolve()
                if rp not in seen and p.is_file():
                    seen.add(rp)
                    out.append(p)
        return out

    def scan(self) -> dict:
        with self._lock:
            cache = self._load_cache()
            changed = False
            score_files, skipped = [], []
            live_keys = set()
            for p in self._csv_files():
                rel = repo_rel(p, self.root)
                live_keys.add(rel)
                st = p.stat()
                ent = cache.get(rel)
                if not ent or ent.get("mtime_ns") != st.st_mtime_ns or ent.get("size") != st.st_size:
                    try:
                        ent = {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "summary": summarise_table(read_scores(p))}
                    except Exception as e:                 # noqa: BLE001 - any unreadable file is just not a score file
                        ent = {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "error": f"{type(e).__name__}: {e}"}
                    cache[rel] = ent
                    changed = True
                if "summary" in ent:
                    score_files.append({"id": file_id(rel), "path": rel, **ent["summary"]})
                else:
                    skipped.append({"path": rel, "reason": ent["error"]})
            for stale in [k for k in cache if k not in live_keys and self._under_dirs(k)]:
                del cache[stale]
                changed = True
            if changed:
                try:
                    self._save_cache()
                except OSError:
                    pass                                   # the cache is an optimisation only
            return {
                "score_files": score_files,
                "roles": dict(ROLES),
                "default_dirs": list(DEFAULT_DIRS),
                "dirs": [repo_rel(d, self.root) for d in self.dirs()],
                "skipped_files": skipped,
            }

    def _under_dirs(self, rel: str) -> bool:
        p = (self.root / rel).resolve()
        for d in self.dirs():
            try:
                p.relative_to(d.resolve())
                return True
            except ValueError:
                continue
        return False

    # ----------------------------------------------------------------- lookup
    def find(self, fid: str) -> dict:
        """The catalogue entry of a file id (with an absolute `abs_path`), or LabError."""
        for sf in self.scan()["score_files"]:
            if sf["id"] == fid:
                p = Path(sf["path"])
                return {**sf, "abs_path": str(p if p.is_absolute() else self.root / p)}
        raise LabError("UNKNOWN_SCORE_FILE", f"score_file_id {fid!r} is not in the catalogue")


def public(cat: dict) -> dict:
    return jsonable(cat)
