# FILES

Last updated: 2026-10-24 (Session 1)

One row per file or folder. Status is `exists`, `planned` or `missing`. A `planned` row appears only when a decision in DECISIONS.md covers it. This file is updated whenever files change.

| Path | What it contains | What it contributes to the project | Status |
|---|---|---|---|
| CLAUDE.md | The binding project memory: goal, hard rules, living documents, session protocol, git rules | Governs every session; supplied by the user at the start of Session 1 | exists |
| .git/ | Local git history, branch `main`; no remote yet | Version history required by D-008 | exists |
| .gitignore | Patterns for the virtual environment, data, checkpoints, model weights, LaTeX and preview build outputs, secrets, and the supplied `Document Template/` folder | Keeps large files, copyrighted material and secrets out of git | exists |
| pyproject.toml | Project name, Python 3.13 requirement, direct dependencies (torch 2.14.1, numpy) and the PyTorch cu130 index | Declares the environment (D-017) | exists |
| uv.lock | Exact resolved versions of every package | Reproducible environment | exists |
| .venv/ | The uv virtual environment (CPython 3.13.15, PyTorch 2.14.1+cu130); not committed | Runtime for experiments and data tools | exists |
| docs/PLAN.md | Stage table, deadline and constraints, measurements to schedule, day-by-day schedule with cut lines, iteration slots, risks | Tracks progress against D-004, D-009, D-016 | exists |
| docs/DECISIONS.md | Decision method, D-001 to D-021, open decisions, contradictions found | The record a reviewer reads to see why things were done | exists |
| docs/FILES.md | This table | Map of the repository | exists |
| docs/EXPERIMENTS.md | Environment record ENV-DESKTOP-GPU (including the download-speed measurement), the experiment template, the experiment log (no experiments yet) | Evidence log; every result is traceable here | exists |
| docs/DATA_LOG.md | Requirements-driven data plan (D-021), keep, drop or defer table, bandwidth budget, manual-download sheets, stream B fact tables, downloaded-datasets table | Provenance, licence and role of each dataset | exists |
| docs/RESEARCH.md | Search protocol, literature matrix R-001 to R-067, NR table, notes that bear on decisions, further reading | Evidence base for decisions and for the paper | exists |
| docs/VIVA_PREP.md | Question bank linked to decisions and experiments, answers pending, standard definitions | Preparation for the viva | exists |
| docs/WORKSTREAMS.md | Parallel plan for three roles: project in plain English with decided/proposed/open tags, progress audit, data versus work, six workstreams, dependency map, action plan, workflow | Lets three people work without waiting on downloads (D-030, Proposed) | exists |
| docs/REQUIREMENTS.md | Stage 0 problem statement, requirements, success criteria | Stage 0 output | exists |
| docs/PROTOCOL.md | Stage 2 evaluation protocol and acceptance test A1 to A8 | Stage 2 output; spec for src/dfd/eval | exists |
| docs/MATH_NOTES.md | Background definitions and formulas of candidate metrics (not adopted) | Reference for the metric decision O-005 and the viva | exists |
| docs/STATUS.md | Next step, blockers, open questions, mistakes and corrections | Entry point for every new session | exists |
| docs/_agents/ | Raw reports of the Session 1 subagent streams (A.md templates, B.md datasets, C1.md, C2.md, C3.md literature) and MERGE_MAP.md mapping stream IDs to R-IDs | Audit trail for what was merged into the living documents (D-015) | exists |
| scripts/data/fetch_open.py | Resumable downloader that checks size and SHA-256, extracts, checks counts and writes manifests for the open datasets of D-021; data tooling only | Executes the approved data plan reproducibly | exists |
| data/ | Downloaded datasets, incoming archives, logs (`data/_logs/`) and manifests (`data/_manifests/`); not committed | Data for the experiments (D-021) | exists (growing) |
| paper/main.tex, paper/refs.bib | IEEEtran conference-style paper skeleton with Related Work from the matrix, placeholders marked "to be filled from E-xxx", and a bibliography of 18 verified entries | The research paper (D-003, D-014) | exists |
| report/DeepFake_Detection_Technical_Report_draft.docx | Technical report built from the supplied Word template: cover with logo, certificate, declaration, abstract, contents, chapters 1 to 5 with placeholders, references | The technical report (D-003, D-014) | exists |
| report/build_report.py, report/template_base_converted_from_doc.docx | Script that builds the report and the converted template base it uses | Reproduces the report | exists |
| slides/DeepFake_Detection_Presentation_draft.pptx | 19-slide deck built on the supplied deck's master, logos, colours and bars; all old content removed; speaker notes; result slides tagged "RESULT: TO BE FILLED" | The presentation (D-003, D-014) | exists |
| slides/build_slides.py | Script that builds the deck | Reproduces the deck | exists |
| assets/branding/ | The college and trust logos and two campus pictures, extracted unchanged from the supplied deck | Identity elements the user asked to be reused (D-014) | exists |
| Document Template/ | The supplied Word report template and the PowerPoint and PDF of an earlier project, used only for look and identity; not committed | Source of the templates (D-014); its body content is not reused (D-001) | exists (untracked, ignored) |
| Git tags per frozen iteration | `setup` exists locally; none for iterations | Required by D-008 | planned |
| Software (detection system) | Not yet created; form is open decision O-011 | Deliverable under D-003 | planned |
