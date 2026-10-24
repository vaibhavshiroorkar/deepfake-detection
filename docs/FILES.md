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
| src/dfd/infer/ | The inference library (`library.py`: file in, D-027 response out; `schema.py`: response contract checker; `adapters.py`: adapter interface; `mock_adapter.py`: MOCK stand-in, D-029; `real_adapter.py`: stub that fails until a checkpoint exists; `factory.py`) | Software deliverable, S4 |
| src/dfd/web/ | The local Flask web page (D-031, D-043): `app.py` routes and jobs, `templates/index.html`, `static/app.js`, `static/style.css`, `__main__.py` | Software deliverable, S4 |
| src/dfd/sweep/ | Frozen encoders (`encoders.py`), E-001 feature extraction (`extract_video.py`) and fine-tuning step-time measurement (`step_time.py`) | Sweep I-0 and the I-1 budget |
| scripts/sweep/, scripts/data/lavdf_job.py | Encoder weight fetch with hash check, the gated E-001 queue, the LAV-DF verify job | Reproducible sweep and data steps |
| tests/infer/, tests/web/, tests/conftest.py | Contract, mock, real-stub, web route and banner tests with synthetic media | Evidence that the response and the MOCK rule hold |
| docs/ITERATIONS.md | The proposed iteration ladder I-0 to I-8 with trigger, alternative, equation, prediction and failure line per slot, the rPPG thresholds and the freeze (D-041) | The plan the experiments and the paper follow |
| docs/O007_SWEEP.md | The pre-declared protocol, selection rule and fine-tuning plan of the encoder comparison (iteration I-0 and I-1), awaiting the user's approval of the shortlist | Design of the measured comparison for O-007 (D-036, Proposed) | exists |
| docs/WORKSTREAMS.md | Parallel plan for three roles: project in plain English with decided/proposed/open tags, progress audit, data versus work, six workstreams, dependency map, action plan, workflow | Lets three people work without waiting on downloads (D-030, Proposed) | exists |
| docs/REQUIREMENTS.md | Stage 0 problem statement, requirements, success criteria | Stage 0 output | exists |
| docs/PROTOCOL.md | Stage 2 evaluation protocol and acceptance test A1 to A8 | Stage 2 output; spec for src/dfd/eval | exists |
| docs/MATH_NOTES.md | Background definitions and formulas of candidate metrics (not adopted) | Reference for the metric decision O-005 and the viva | exists |
| docs/STATUS.md | Next step, blockers, open questions, mistakes and corrections | Entry point for every new session | exists |
| docs/_agents/ | Raw reports of the Session 1 subagent streams (A.md templates, B.md datasets, C1.md, C2.md, C3.md literature) and MERGE_MAP.md mapping stream IDs to R-IDs | Audit trail for what was merged into the living documents (D-015) | exists |
| scripts/data/fetch_open.py | Resumable downloader that checks size and SHA-256, extracts, checks counts and writes manifests for the open datasets of D-021; data tooling only | Executes the approved data plan reproducibly | exists |
| scripts/data/ffpp_parallel.py, scripts/data/run_ffpp.py | Threaded and retrying downloaders for the FF++ and DFD videos from the dataset's own server | Completed FF++ c23 and DFD after the authors' script proved slow and fragile | exists |
| scripts/data/ingest_zip.py, scripts/data/verify_videos.py | Verify, extract and count a hand-downloaded zip; ffprobe and SHA-256 check of a video folder | Completeness evidence for Celeb-DF v2, FF++ and DFD | exists |
| src/dfd/eval/ and tests/eval/ | Evaluation harness: metrics, threshold, temperature scaling, group bootstrap, final-test guard, ledger, command line; 80 tests | Stage 2 deliverable (V-001) | exists |
| src/dfd/data/ and tests/data/ | Manifests, group-aware splits, leakage check, content de-duplication; 49 tests | Stage 1 deliverable (V-002) | exists |
| src/dfd/prep/ and tests/prep/ | Video decoding, face detection and cropping (YuNet), audio extraction, resumable pipeline, benchmark tool; 26 tests | Shared preprocessing of D-028; speed in M-001 | exists |
| ledger/ | The final-evaluation ledger (empty until the first final evaluation) | Enforces one final test per iteration (PROTOCOL section 7) | exists |
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
