# Legacy team records

These files come from the team final-year project, which was submitted on 2026-10-30 and folded into this repository on 2026-11-13. They are historical evidence. They are read, never appended to.

| File | What it holds |
|---|---|
| `final_ledger.jsonl` | Every look at a sealed test set, one JSON line each, append-only while the project ran |
| `EXPERIMENTS.md` | E-001 to E-036: preregistrations, amendments and results |
| `DATA_LOG.md` | Where each dataset came from, its licence and its archive hashes |
| `RESEARCH_INTEGRITY.md` | What git can and cannot prove about the order of work |
| `NUMBERS*.json` | Every number the team deliverables cited, with its source |

These numbers used the team splits (`data/legacy-team/splits/`), not the root splits. Do not compare them with new results unless the same split is rerun.

The sealed outputs these files refer to now live in `data/legacy-team/` (results, final scores, manifests, splits) and `runs/legacy-team/` (checkpoints). Both are git-ignored. The submitted code and documents are at the tag `team/submission-final`.
