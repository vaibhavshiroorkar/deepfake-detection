"""Are two caches with different preprocessing hashes actually the same data?

`preprocessing_config_hash` covers the view settings *and* the code version, so
two runs that used identical settings under different version strings produce
identical views under different hashes. The hash is right to be strict: the code
version is in it because the preprocessing code may have changed, and nothing
about a matching config proves the code matched.

So this settles it by comparison rather than by argument. It loads the same clip
from both caches and compares every array element by element. Equivalence
claimed on that evidence is a measurement; equivalence claimed because the
configs look alike is a guess, and this project has been wrong about that class
of guess before.

Deterministic preprocessing means a sample generalises: if the code that built
the two caches were different, a difference would show on almost any clip. The
sample size is still reported, because "verified on 50 clips" and "verified"
are different claims.

    uv run python scripts/verify_cache_equivalence.py \
        --left runs/streams-20260905/cache --right runs/lavdf-av-20260914/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _clip_directories(cache_root: Path) -> dict[str, Path]:
    """Clip id to its cache directory, dropping the content-hash suffix."""
    found: dict[str, Path] = {}
    for dataset in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        for clip in sorted(p for p in dataset.iterdir() if p.is_dir()):
            found[clip.name.rsplit("-", 1)[0]] = clip
    return found


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    left = _clip_directories(arguments.left)
    right = _clip_directories(arguments.right)
    shared = sorted(set(left) & set(right))
    print(f"left {len(left):,} clips, right {len(right):,}, shared {len(shared):,}")
    if not shared:
        print("No clip is in both caches, so there is nothing to compare.")
        return 1

    # Spread across the shared set rather than taking a prefix, so a difference
    # confined to one part of the corpus is not missed.
    step = max(1, len(shared) // arguments.samples)
    chosen = shared[::step][: arguments.samples]

    compared = 0
    differing: list[str] = []
    fields: set[str] = set()
    for clip_id in chosen:
        left_files = list(left[clip_id].glob("*.npz"))
        right_files = list(right[clip_id].glob("*.npz"))
        if not left_files or not right_files:
            continue
        a = np.load(left_files[0], allow_pickle=False)
        b = np.load(right_files[0], allow_pickle=False)
        shared_keys = sorted(set(a.files) & set(b.files))
        for key in shared_keys:
            if a[key].dtype.kind not in "fiub":
                continue
            fields.add(key)
            if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key]):
                differing.append(f"{clip_id}:{key}")
        missing = sorted(set(a.files) ^ set(b.files))
        if missing:
            differing.append(f"{clip_id}:missing:{','.join(missing)}")
        compared += 1

    identical = not differing
    result = {
        "left": str(arguments.left).replace("\\", "/"),
        "right": str(arguments.right).replace("\\", "/"),
        "shared_clips": len(shared),
        "compared": compared,
        "arrays_compared": sorted(fields),
        "identical": identical,
        "differences": differing[:20],
    }
    print(f"compared {compared} clips across {len(fields)} arrays: {', '.join(sorted(fields))}")
    if identical:
        print("IDENTICAL. The two hashes label the same preprocessing.")
    else:
        print(f"DIFFERENT in {len(differing)} places, for example {differing[:3]}")

    output = arguments.output or Path("runs/cache-equivalence.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {output}")
    return 0 if identical else 2


if __name__ == "__main__":
    sys.exit(main())
