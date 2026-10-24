# FILES

Last updated: 2026-10-24 (Session 1)

One row per file or folder. Status is `exists` or `planned`. A `planned` row appears only when a decision in DECISIONS.md covers it. This file is updated whenever files change.

| Path | What it contains | What it contributes to the project | Status |
|---|---|---|---|
| .git/ | Local git history, branch `main` | Version history required by D-008 | exists |
| .gitignore | Patterns for the virtual environment, data, checkpoints, downloaded papers, model weights and secrets | Keeps large files, copyrighted papers and secrets out of the repository | exists |
| pyproject.toml | Project name, Python requirement (3.13 or newer), direct dependencies (torch 2.14.1, numpy) and the PyTorch cu130 package index | Declares the environment | exists |
| uv.lock | Exact resolved versions of every package | Reproducible environment, the lock file required in Step 1 | exists |
| .venv/ | The uv virtual environment (CPython 3.13.15, PyTorch 2.14.1+cu130) | The runtime used for all experiments; not committed | exists |
| docs/PLAN.md | Stage table, iteration slots I-0 to I-8, fixed constraints | Tracks progress against D-004 | exists |
| docs/DECISIONS.md | Decision method, D-001 to D-010 (user), open decisions O-001 to O-011, contradictions found | The record a reviewer reads to see why things were done | exists |
| docs/FILES.md | This table | Map of the repository | exists |
| docs/EXPERIMENTS.md | Environment record ENV-DESKTOP-GPU, the experiment template, the experiment log | Evidence log; every result is traceable here | exists |
| docs/DATA_LOG.md | Dataset table template; no dataset downloaded yet | Provenance, licence and role of each dataset | exists |
| docs/RESEARCH.md | Search protocol, literature matrix R-001 to R-020, unverified further reading list | Evidence base for decisions and for the paper | exists |
| docs/VIVA_PREP.md | Question bank linked to decisions and experiments, answers pending | Preparation for the viva | exists |
| docs/MATH_NOTES.md | Background definitions and formulas: ROC and AUC, EER, precision and recall, balanced accuracy, ECE, Brier score, Platt scaling, bootstrap and paired bootstrap. Labelled candidate metrics, not adopted | Reference for the metric decision O-005 and for the viva; decides nothing | exists |
| docs/STATUS.md | Next step, blockers, open questions, mistakes and corrections | Entry point for every new session | exists |
| CLAUDE.md | Not present. The Session 0 prompt calls it binding but it does not exist in the repository | To be provided by the user, or created on the user's instruction | missing |
| Research paper (IEEE format, LaTeX) | Not yet created; location and layout not decided | Deliverable under D-003 | planned |
| Technical report | Not yet created; location and layout not decided | Deliverable under D-003 | planned |
| Presentation | Not yet created; location and layout not decided | Deliverable under D-003 | planned |
| Software (detection system) | Not yet created; form is open decision O-011 | Deliverable under D-003 | planned |
| Git tags per frozen iteration | None yet | Required by D-008 | planned |
