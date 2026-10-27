"""The cache index: which cache entry holds each clip.

A two-column CSV, `clip_id,cache_path`. Paths are written relative to the
index's own directory when they sit under it, so a run directory can move as a
whole, and read back relative to the same directory.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


def read_cache_index(path: Path) -> dict[str, Path]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = tuple(csv.DictReader(handle))
    return {
        row["clip_id"]: (
            Path(row["cache_path"])
            if Path(row["cache_path"]).is_absolute()
            else (Path(path).parent / row["cache_path"]).resolve()
        )
        for row in rows
    }


def write_cache_index(path: Path, entries: Mapping[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("clip_id", "cache_path"))
        writer.writeheader()
        for clip_id, cache_path in sorted(entries.items()):
            writer.writerow(
                {"clip_id": clip_id, "cache_path": str(_relative(cache_path, path))}
            )


def _relative(cache_path: Path, index: Path) -> Path:
    resolved = Path(cache_path).resolve()
    try:
        return resolved.relative_to(index.parent.resolve())
    except ValueError:
        return resolved
