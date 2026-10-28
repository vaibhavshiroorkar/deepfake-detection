"""E-027 preregistration record: appends the preregistration block to docs/EXPERIMENTS.md (once) and writes its SHA-256 and the time.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/modern_video/e027_prereg.py <block.md>     # append the block, then hash it
    PYTHONPATH=src .venv/Scripts/python.exe scripts/modern_video/e027_prereg.py --hash-only    # hash the block already in the file

Commits are not authorised (D-081), so data/_results/modern/e027/prereg.json stands in for a commit (docs/RESEARCH_INTEGRITY.md section 2).
The hashed span runs from the line "## E-027:" to the line "- Result: (to be appended below after the run)", inclusive.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "docs" / "EXPERIMENTS.md"
OUT = ROOT / "data" / "_results" / "modern" / "e027"
END = "- Result: (to be appended below after the run)"


def block_span(text: str) -> str:
    i = text.index("## E-027:")
    j = text.index("\n" + END, i) + 1 + len(END)      # the line itself, not the quotation of it in the Status line (fixed 2026-10-27 00:55)
    return text[i:j]


def main(argv) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "prereg.json").exists():
        print("prereg.json exists; refusing to overwrite")
        return 2
    text = EXP.read_text(encoding="utf-8")
    if argv and argv[0] != "--hash-only":
        if "## E-027:" in text:
            print("E-027 block already present; use --hash-only")
            return 2
        block = Path(argv[0]).read_text(encoding="utf-8")
        assert "—" not in block, "em-dash in block"
        with open(EXP, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(block if block.startswith("\n") else "\n" + block)
        text = EXP.read_text(encoding="utf-8")
    span = block_span(text)
    rec = {"experiment": "E-027", "written_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "committed": False,
           "e027_block_sha256": hashlib.sha256(span.encode("utf-8")).hexdigest(), "e027_block_chars": len(span),
           "span_rule": "from '## E-027:' to the first LINE equal to '" + END + "' inclusive, UTF-8",
           "correction": "a first prereg.json (kept as prereg_v0_wrong_span.json) hashed only 658 characters because the span ended at the quotation of the end line inside the Status line; re-hashed at once, before any feature or head existed",
           "note": "written before any E-027 feature was extracted or any head fitted"}
    (OUT / "prereg.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
