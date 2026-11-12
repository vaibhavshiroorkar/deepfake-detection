"""Score-table loading with a small in-memory cache, and the raw extra columns the reader ignores."""

from __future__ import annotations

import csv
import threading
from collections import OrderedDict
from pathlib import Path

from dfd.eval.scorefile import read_scores

from .common import BREAKDOWN_COLUMNS, HASH_COLUMNS


class TableCache:
    """Keeps the last few parsed score tables, keyed by (path, mtime, size)."""

    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path):
        p = Path(path)
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        table = read_scores(p)          # may raise ScoreFileError
        with self._lock:
            self._items[key] = table
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)
        return table


def read_extras(path, run_ids, modality: str) -> dict:
    """Read the columns scorefile ignores, for the rows of the given runs and modality.

    Returns {"rows": {"run_id","dataset","split","sample_id": [..]}, "hash_column": name|None,
    "hashes": [..]|None, "breakdown": {column: [..]}}.
    """
    keep = set(run_ids)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        pos = {c: i for i, c in enumerate(header)}
        hcol = next((c for c in HASH_COLUMNS if c in pos), None)
        bcols = [c for c in BREAKDOWN_COLUMNS if c in pos]
        out = {"run_id": [], "dataset": [], "split": [], "sample_id": []}
        hashes = [] if hcol else None
        brk = {c: [] for c in bcols}
        i_run, i_mod, i_ds, i_sp, i_sid = (pos[c] for c in ("run_id", "modality", "dataset", "split", "sample_id"))
        for row in reader:
            if not row or len(row) != len(header):
                continue
            if row[i_run].strip() not in keep or row[i_mod].strip() != modality:
                continue
            out["run_id"].append(row[i_run].strip())
            out["dataset"].append(row[i_ds].strip())
            out["split"].append(row[i_sp].strip())
            out["sample_id"].append(row[i_sid].strip())
            if hcol:
                hashes.append(row[pos[hcol]].strip())
            for c in bcols:
                brk[c].append(row[pos[c]].strip())
    return {"rows": out, "hash_column": hcol, "hashes": hashes, "breakdown": brk}
