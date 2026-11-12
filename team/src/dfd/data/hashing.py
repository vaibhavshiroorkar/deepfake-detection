"""SHA-256 of files with an optional on-disk cache (path, size and mtime must match to reuse an entry).

The cache is a JSON file, normally data/_manifests/content_hash_cache.json (git-ignored). It only avoids
re-reading large image sets; a changed size or modification time forces a new hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HashCache:
    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path
        self.entries: dict[str, list] = {}
        self.hits = 0
        self.misses = 0
        self._dirty = False
        if cache_path is not None and cache_path.exists():
            try:
                self.entries = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.entries = {}

    def sha256(self, file: Path, key: str) -> str:
        """Hash of ``file``; ``key`` is the stable name used in the cache (the manifest path)."""
        st = file.stat()
        e = self.entries.get(key)
        if e is not None and e[0] == st.st_size and e[1] == st.st_mtime_ns:
            self.hits += 1
            return e[2]
        digest = sha256_of(file)
        self.entries[key] = [st.st_size, st.st_mtime_ns, digest]
        self.misses += 1
        self._dirty = True
        return digest

    def save(self) -> None:
        if self.cache_path is not None and self._dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.entries), encoding="utf-8")
            self._dirty = False
