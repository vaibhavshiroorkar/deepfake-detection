"""Ledger of final evaluations (PROTOCOL.md section 7), one JSON object per line.

Default file: ledger/final_ledger.jsonl at the repository root (tracked in git).
Entry fields: date, iteration, dataset, split, run_ids, modalities,
score_file_sha256, and repeat_reason (only on a repeat). Tests must pass a
temporary path; nothing here writes to the real ledger unless asked to.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "ledger" / "final_ledger.jsonl"


class LedgerError(ValueError):
    pass


def read_ledger(path=DEFAULT_LEDGER) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(f"{path} line {ln}: not valid JSON ({exc})") from None
            if not isinstance(e, dict) or "iteration" not in e or "dataset" not in e:
                raise LedgerError(f"{path} line {ln}: entry lacks iteration or dataset")
            entries.append(e)
    return entries


def append_entries(entries: list[dict], path=DEFAULT_LEDGER) -> None:
    """Append entries (never rewrites existing lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True, ensure_ascii=True) + "\n")
