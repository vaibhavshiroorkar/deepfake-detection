"""Fill {{number.key}} placeholders in a deliverable source from results/numbers.json.

A placeholder whose key is missing stops the build and names the key, so a
deliverable can never ship with a gap or a stale value. `{{key|pct}}` prints a
fraction as a percentage; `{{key|int}}` adds thousands separators.

    uv run python scripts/fill_numbers.py deliverables/slides/slides.md deliverables/build/slides.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?:\|\s*(pct|int|3))?\s*\}\}")


def render(value, style: str | None) -> str:
    if style == "pct":
        return f"{100 * float(value):.0f}%"
    if style == "int":
        return f"{int(value):,}"
    if style == "3" or isinstance(value, float):
        return f"{float(value):.3f}"
    return str(value)


def fill(text: str, numbers: dict) -> str:
    missing = sorted({m.group(1) for m in PATTERN.finditer(text) if m.group(1) not in numbers})
    if missing:
        raise SystemExit("missing numbers: " + ", ".join(missing))
    return PATTERN.sub(lambda m: render(numbers[m.group(1)]["value"], m.group(2)), text)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: fill_numbers.py SOURCE TARGET")
    numbers = json.loads((ROOT / "results" / "numbers.json").read_text(encoding="utf-8"))
    source, target = Path(argv[0]), Path(argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(fill(source.read_text(encoding="utf-8"), numbers), encoding="utf-8")
    print(f"filled {source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
