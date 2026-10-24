"""Final-test guard (CLAUDE.md rule 3, PROTOCOL.md section 7).

check_guard(table, ...) is called before anything is computed from a score
table. Rules:
  1. A table with test_in or test_x rows is refused unless final=True and an
     iteration tag is given.
  2. The tag must equal the `iteration` column of every run that has final rows
     (so a tag cannot be made up to pass the guard).
  3. If the ledger already has an entry with the same iteration and dataset,
     the evaluation is refused unless a non-empty repeat reason is given.
A first final evaluation is accepted. The guard only plans the ledger entries;
the caller writes them with ledger.append_entries after the evaluation has
succeeded, so a crash does not use up the single final evaluation.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np

from .ledger import DEFAULT_LEDGER, read_ledger
from .scorefile import FINAL_SPLITS, ScoreTable


class GuardError(RuntimeError):
    """The evaluation is refused."""


@dataclass
class GuardDecision:
    entries: list = field(default_factory=list)  # ledger entries to append on success
    is_repeat: bool = False
    final_blocks: list = field(default_factory=list)  # [(dataset, split), ...]


def final_blocks(table: ScoreTable) -> list[tuple[str, str]]:
    m = np.isin(table.split, FINAL_SPLITS)
    return sorted({(str(d), str(s)) for d, s in zip(table.dataset[m], table.split[m])})


def check_guard(table: ScoreTable, *, final: bool = False, iteration: str | None = None,
                repeat: str | None = None, ledger_path=DEFAULT_LEDGER,
                today: _dt.date | None = None) -> GuardDecision:
    blocks = final_blocks(table)
    if not blocks:
        return GuardDecision()
    names = ", ".join(f"{d}/{s}" for d, s in blocks)
    if not final:
        raise GuardError(f"refused: the scores contain final splits ({names}); "
                         "pass --final and --iteration <tag>")
    if not iteration:
        raise GuardError("refused: --final needs --iteration <tag>")
    m = np.isin(table.split, FINAL_SPLITS)
    in_file = sorted({str(i) for i in table.iteration[m]})
    if in_file != [iteration]:
        raise GuardError(f"refused: --iteration {iteration!r} does not match the iteration "
                         f"column of the final rows {in_file}")
    repeat = (repeat or "").strip()
    prior = read_ledger(ledger_path)
    seen = {(e["iteration"], e["dataset"]): e for e in prior}
    datasets = sorted({d for d, _ in blocks})
    clash = [d for d in datasets if (iteration, d) in seen]
    if clash and not repeat:
        when = ", ".join(f"{d} on {seen[(iteration, d)].get('date', '?')}" for d in clash)
        raise GuardError(f"refused: iteration {iteration!r} was already evaluated finally for "
                         f"{when}; a second evaluation needs --repeat \"<reason>\"")
    date = (today or _dt.date.today()).isoformat()
    entries = []
    for d, s in blocks:
        sel = (table.dataset == d) & (table.split == s)
        e = {
            "date": date,
            "iteration": iteration,
            "dataset": d,
            "split": s,
            "run_ids": sorted({str(r) for r in table.run_id[sel]}),
            "modalities": sorted({str(r) for r in table.modality[sel]}),
            "score_file_sha256": table.sha256,
        }
        if d in clash:
            e["repeat_reason"] = repeat
        entries.append(e)
    return GuardDecision(entries=entries, is_repeat=bool(clash), final_blocks=blocks)
