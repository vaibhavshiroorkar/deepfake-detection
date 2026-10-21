"""Build the final report PDF from the evaluation files.

Renders the same data the web app shows into one HTML page and prints it with
headless Edge (or Chrome). Run `scripts/score_visual_ensemble.py` first so the
ensemble rows exist.

    uv run python scripts/build_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from deepfake_detection.webapp.report import PDF_PATH, build_pdf
    from deepfake_detection.webapp.server import _research

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PDF_PATH)
    arguments = parser.parse_args(argv)
    path = build_pdf(_research(), output=arguments.output)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
