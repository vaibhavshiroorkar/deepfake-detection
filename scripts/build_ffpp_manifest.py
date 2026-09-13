"""Write the FaceForensics++ manifest, then validate it like any other.

Two steps on purpose. The first derives rows from the drop, which is filename
parsing and can be wrong in ways that only show up as a leaked identity later.
The second runs the result through `load_manifest`, the same quarantine and
validation pass every other corpus goes through, so a malformed row is rejected
here rather than in the middle of a caching run.

    uv run python scripts/build_ffpp_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.data.faceforensics import manifest_from_ffpp
    from deepfake_detection.data.manifest import load_manifest, write_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/FaceForensics++/FaceForensics++_C23")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/faceforensics.csv")
    )
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Restrict the manipulation families, for leave-one-family-out.",
    )
    arguments = parser.parse_args(argv)

    frame = manifest_from_ffpp(
        arguments.root,
        arguments.data_dir,
        methods=tuple(arguments.methods) if arguments.methods else None,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)

    result = load_manifest(arguments.output, dataset="FaceForensics++")
    write_manifest(result.records, arguments.output)

    audit = arguments.audit or arguments.output.with_suffix(".json")
    counts: dict[str, int] = {}
    for record in result.records:
        counts[record.method] = counts.get(record.method, 0) + 1
    audit.write_text(
        json.dumps(
            {
                "records": len(result.records),
                "identities": len({r.source for r in result.records}),
                "by_method": dict(sorted(counts.items())),
                "quarantined_paths": [str(p) for p in result.quarantined_paths],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(result.records):,} rows -> {arguments.output}")
    print(f"{len({r.source for r in result.records}):,} source identities")
    for method, count in sorted(counts.items()):
        print(f"  {method:28} {count:,}")
    if result.quarantined_paths:
        print(f"quarantined {len(result.quarantined_paths):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
