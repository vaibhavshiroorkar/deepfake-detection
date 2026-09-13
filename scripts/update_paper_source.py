"""Refresh the generated tables in the paper source book.

Run this after any training or scoring run. It reads the artifacts under
`runs/` and rewrites only the marked blocks in `docs/research/paper-source.md`,
so the prose around them is untouched.

    uv run python scripts/update_paper_source.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.documentation.paper_source import (
        missing_blocks,
        update_paper_source,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", type=Path, default=Path("docs/research/paper-source.md")
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/design-b-20260910"))
    parser.add_argument(
        "--program-run", type=Path, default=Path("runs/program-20260906")
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("docs/research/result-traceability.md")
    )
    arguments = parser.parse_args(argv)

    if not arguments.path.is_file():
        print(f"No paper source at {arguments.path}")
        return 1

    absent = missing_blocks(arguments.path)
    if absent:
        print(f"{arguments.path} is missing markers: {', '.join(absent)}")
        return 1

    changed = update_paper_source(
        arguments.path,
        run_dir=arguments.run_dir,
        program_run=arguments.program_run,
        registry=arguments.registry,
    )
    print(f"{arguments.path}: {'updated' if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
