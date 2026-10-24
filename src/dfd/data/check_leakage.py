"""CLI: leakage check.

    python -m dfd.data.check_leakage --manifest data/_manifests/ffpp.manifest.csv --split data/_splits/ffpp-<hash>.csv
    python -m dfd.data.check_leakage --manifest a.csv b.csv --split sa.csv sb.csv --hash-files

Exit code 0 when clean, 1 when any leakage is found, 2 for input errors. See dfd.data.leakage for the checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .hashing import HashCache
from .leakage import check, load_inputs
from .manifest import ManifestError, default_data_root
from .splits import SplitError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True, type=Path)
    ap.add_argument("--split", nargs="+", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--hash-files", action="store_true", help="SHA-256 every file (slow)")
    ap.add_argument("--strict-duplicates", action="store_true", help="duplicate content inside one split also fails")
    ap.add_argument("--cache", type=Path, default=None, help="hash cache file (default: <data>/_manifests/content_hash_cache.json)")
    ap.add_argument("--merge-family", action="append", default=[], metavar="A,B",
                    help="treat these families as one (repeatable), for example lavdf,fakeavceleb")
    args = ap.parse_args(argv)
    try:
        recs, splits, rows = load_inputs(args.manifest, args.split)
    except (ManifestError, SplitError, OSError) as e:
        print(f"INPUT ERROR: {e}", file=sys.stderr)
        return 2
    root = args.data_root or default_data_root()
    cache = HashCache(args.cache or (root / "_manifests" / "content_hash_cache.json")) if args.hash_files else None
    rep = check(
        recs, splits, split_row_count=rows,
        merge_families=[tuple(x.split(",")) for x in args.merge_family],
        data_root=root, cache=cache,
        hash_files=args.hash_files, strict_duplicates=args.strict_duplicates,
    )
    if cache is not None:
        cache.save()
    print(rep.text())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
