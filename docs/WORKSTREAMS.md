# WORKSTREAMS

Parallel plan for three people, written 2026-10-24 (evening) from the repository state: git log (36 commits), DECISIONS.md, PLAN.md, DATA_LOG.md, the data/ folder, the download logs and the test suites. Plain English. Everything undecided stays an open decision. Nothing in this file is implemented yet; the proposals are logged in DECISIONS.md as Proposed (D-027 to D-030) and need approval.

Note on decision numbers: in this repository D-017 is the Python and PyTorch environment and D-018 is the image scope; the access policy is D-020 and the data selection is D-021. All four are respected below.

## Update, later on 2026-10-24

After this plan was presented the user decided: D-027 to D-030 approved as proposed; D-031 the software is a local web page only (O-011 settled, so the command-line stream S4 narrows to the inference library behind a small local web service, framework still open as O-013); D-032 the MVP starts from frozen pretrained features with a linear head (the extractor itself is the first item of O-007); D-033 a still image goes through a face crop when a face is found and the whole image otherwise (O-002, image path). The repository was pushed to the private GitHub repository. In-the-Wild finished downloading at 21:16. LAV-DF is still blocked: the Hugging Face account has not yet accepted the dataset's terms (HTTP 403). The audit and the dependency map below are otherwise unchanged.

## 1. The project in plain English

We are building a program that takes an image, a video or an audio clip and says how likely it is to be fake, and says which kind of evidence (picture or sound) the verdict rests on. It is a student project, so the point is also the path: a simple first version, then at least 7 improvements, each started from a written hypothesis, each ending in a measured result, and a fair test that never uses people or videos seen in training.

| Part | What it is | Status |
|---|---|---|
| Inputs | Still images (face-manipulated or fully generated), video with its audio, standalone audio | Decided (D-002, D-018, D-019) |
| Output | A probability of fake per kind of evidence present, plus one combined score; no locating of fake segments in version 1 | Decided (D-011) |
| How a user would use it | Upload a file, wait, read the result | The form (command line, local web page or both) is Open (O-011) |
| Inside, step 1: check the file | Type and readability checks, reject unsupported input | Proposed (D-027) |
| Inside, step 2: prepare it | Video: pick frames, find the face, crop. Audio: extract 16 kHz mono sound. Image: face crop or whole image | Tooling partly built; face detector Open (to be logged from S2-C measurements); image handling Open (O-002) |
| Inside, step 3: score each kind of evidence | A model per modality | Open (O-006 to O-008); nothing exists |
| Inside, step 4: make the number honest | Temperature scaling fitted on validation data; a threshold chosen on validation data | Decided (D-024) |
| Inside, step 5: combine | Image gives the picture score, audio file gives the sound score, video combines both | The combining rule is Open (O-009) |
| Inside, step 6: answer | A structured response with scores, status, warnings | Proposed (D-027) |
| How we know it works | Group-disjoint splits, headline AUC on datasets never used in training, 3 seeds, bootstrap intervals, final test touched once | Decided (D-012, D-024); the harness is built |
| Heartbeat-in-skin-colour (rPPG) | An experiment, not part of the system unless a feasibility test says so | Decided as an experiment (D-006); test design Open (O-010) |

## 2. Progress audit

| Area | Status | Evidence |
|---|---|---|
| Environment | Complete and verified | EXPERIMENTS.md ENV-DESKTOP-GPU (RTX 5070 12 GB, PyTorch 2.14.1+cu130, matmul and mixed-precision tests); uv.lock; commit 5b47da3 |
| Data acquisition | Partial | 8 datasets verified in data/_manifests/downloads.jsonl (ProGAN val and test, diffusion test, CNN subset, ASVspoof 2019 LA, Celeb-DF v2, FF++ c23, In-the-Wild); DFD running (1,897 of 3,431 files at 21:28); pending: LAV-DF (needs the user's token), FakeAVCeleb and WildDeepfake (approvals), Celeb-DF++ (not fetched) |
| Preprocessing | Partial (implemented, not yet verified by the main agent) | src/dfd/prep (about 1,250 lines) and tests/prep exist but are uncommitted and stream S2-C has not reported; speed benchmark folders exist under data/_crops |
| Data loaders | Not started for model input. Manifests and splits are complete and verified | src/dfd/data, 49 tests passing (pytest tests/data), commit 425b6f7; nothing turns crops into tensors yet |
| Models | Not started | no module defining a network in src; open decisions O-006 to O-009 |
| Training | Not started | no training code; no experiment run (EXPERIMENTS.md has the template only) |
| Evaluation | Complete and verified for the harness; not yet exercised on real model scores | src/dfd/eval, 80 tests passing, V-001 in EXPERIMENTS.md, commit c56e819; paired comparison exists but has no command-line option |
| Inference | Not started | no code; response format only Proposed (D-027) |
| Dashboard or interface | Not started | O-011 open |
| Experiment tracking | Partial | PROTOCOL.md score-file format, ledger/ folder (empty), EXPERIMENTS.md template; no run registry, no config or checkpoint metadata yet (Proposed D-028) |
| Paper | Partial scaffold, no results | paper/main.tex compiles to 4 pages; refs.bib has 18 entries; commit 4710462 |
| Report | Partial scaffold, no results | report/...draft.docx, 26 pages; only the cover and structure were viewed |
| Slides | Partial scaffold, no results | slides/...draft.pptx, 19 slides; two slides viewed |
| GitHub | Connected, nothing pushed | gh is logged in; origin points to a private repository that is empty (gh repo view: isEmpty true) |
| Stage 2 gate (PROTOCOL.md A1 to A8) | All eight pass, but only A1 to A5 and A8 are logged (V-001) | A6 and A7 results are in docs/_agents/S2B.md and still need a V-002 entry; PLAN.md stage table still says "not started" and needs updating |

## 3. Data versus work

| Dataset | State | Building software and testing the pipeline | Training | Final experiments |
|---|---|---|---|---|
| FF++ c23 | Done, 5,000 videos | Yes, now | Video path training set (REQ-a) | In-dataset only, never the headline |
| ASVspoof 2019 LA | Done | Yes, now | Audio path train and val | In-corpus test |
| ProGAN val | Done | Yes, now | Generated-image train and val (class-disjoint 14 and 6) | n/a |
| ProGAN test | Done | Yes | n/a | In-dataset image test |
| CNN subset, diffusion test | Done (1,200 duplicates excluded, D-026) | Plumbing smoke tests only, no metric tuning | Never | Cross-generator image tests, touched once |
| In-the-Wild | Done at 21:16; manifest builder still a stub | After its builder is verified | Never | Audio cross-corpus test |
| Celeb-DF v2 | Done, counts verified | Plumbing only | Never | Video cross-dataset test |
| DFD | Downloading | Plumbing only, after it lands | Never | Video cross-dataset test and pulse sanity check |
| LAV-DF | Pending: needs the user's Hugging Face token | Not for plumbing: FF++ video plus ASVspoof audio stand in | Audio-visual training (REQ-c) | In-dataset only |
| FakeAVCeleb | Pending approval | No | Never | Audio-visual cross-dataset test |
| WildDeepfake (test shards) | Pending approval | No | Never | One extra video test |
| Celeb-DF++ (21 new generators) | Not fetched; optional | No | Never | Extra cross-generator test and pulse-by-generator evidence |

Which downloads truly block something: only LAV-DF blocks real work (audio-visual training and the fusion experiments), and it is blocked on the user's token, not on bandwidth. FakeAVCeleb, WildDeepfake and Celeb-DF++ each block one experiment, not the build. Nothing that is still downloading blocks the first working version: FF++, ASVspoof 2019 and ProGAN are all here and verified.

## 4. Parallel workstreams

Six streams, grouped into three roles (the roles are suggestions; Role 3 is the lightest on data and the heaviest on integration).

| Stream | Role | Objective | Status | Depends on | Can proceed while a download or training run is blocked | Interface it must deliver | Acceptance | Open decision needed first |
|---|---|---|---|---|---|---|---|---|
| S1 Data pipeline and loaders | Role 1 | Turn manifests and splits into model-ready batches; finish builders for In-the-Wild, DFD, LAV-DF | Manifests verified; prep partial; loaders not started | Face detector choice; frame count and crop size as parameters | Yes: everything on FF++, ASVspoof 2019, ProGAN | A dataset object that yields (input, label, group_id, sample_id) from a manifest and a split ID, reading the shared preprocessing (D-028) | Reproduces manifest counts; deterministic with a seed; throughput at or above the measured prep speed; tests pass | O-002 for the image path (face crop or whole image); the face detector is logged first |
| S2 Modelling and training | Role 2 | Build the MVP, the candidate models and the iterations | Not started | S1 loaders; decisions on MVP and backbones | Yes: write the hypothesis and criteria for I-0, a tiny from-scratch reference, a cost-measurement script for parameters, FLOPs and VRAM | A training run that writes a score file (PROTOCOL section 2) and a checkpoint with metadata (D-028) | I-0 runs with 3 seeds on the frozen protocol; cost axes recorded | O-006 (MVP), O-007 (backbones), O-008 (fine-tuning) before any code |
| S3 Evaluation and experiment tracking | Role 1 | Keep the protocol honest; add the paired-comparison command, the table generator and the run registry; measure cost axes | Harness complete | D-025 (definition of comparable) and the margin for it | Yes: all of it with fake scores | Reports as JSON and tables with intervals; the ledger | The A1 to A8 gate stays green; every table traces to a run ID | D-025 needs review |
| S4 Inference library and command line | Role 3 | One function: file in, response out, using the same preprocessing and calibration as training | Not started | D-027, D-028; real models later | Yes: write it against the mock adapter, clearly labelled | The response structure of D-027 | Contract tests pass for both the real and the mock adapter; mock never reports `real` | O-013 (how the interface calls it) |
| S5 User interface | Role 3 | Upload a file and show the response | Not started | D-027; S4 | Yes: build against the mock adapter with a visible MOCK banner | Calls S4 only through the contract | A user can upload each of the three input kinds and see per-modality scores, status and warnings; the banner shows when inference_source is mock | O-011 (form) |
| S6 Writing and integration | All, led by Role 3 | Fill paper, report and slides from the logs; keep traceability; run the integration checks | Scaffolds only | Results for numbers; structure for the rest | Yes: Data, Protocol and Method sections now; figures limited by the Celeb-DF and FakeAVCeleb terms | Documents that build; a check that every number has an E-xxx | Every number traces; builds clean; claims discipline (CLAUDE.md rule 8) | none |

## 5. Interfaces first (Proposed, see D-027 to D-030)

The response structure, the mock rules and the shared metadata are in DECISIONS.md as D-027, D-028 and D-029. The key points:
- The response carries the input description, a fused score, a score per modality present, status with timing, warnings, errors, and a field `inference_source` that is `real` or `mock`.
- A mock adapter lives in its own module, always sets `inference_source` to `mock`, makes every interface show a MOCK banner, and is refused by the evaluation harness and by the paper, report and slide builds.
- Training, evaluation and inference share one preprocessing package, one configuration whose hash is stored with every checkpoint, and one checkpoint metadata file; inference refuses a checkpoint whose preprocessing does not match.

## 6. Dependency map and critical path to 30 October

| Kind | Items |
|---|---|
| Hard dependencies (nothing downstream can start) | Decisions O-006 and O-007 then the I-0 hypothesis, then S1 loaders, then S2 training, then score files, then evaluation, then freeze and the one-time final test, then numbers in the documents. LAV-DF then the audio-visual MVP then fusion (O-009) |
| Soft dependencies (can start from an agreed interface or a small sample) | S4 and S5 from the response structure and the mock adapter; S6 sections from the logs; S3 additions from fake scores; S1 builders for datasets that have landed |
| Independent | Cost-measurement script, hypothesis writing, citation checks, GitHub push, test-suite upkeep, rPPG feasibility design (O-010), the manual downloads |

Critical path: user decisions, then loaders, then the MVP on day 3, then candidates and iterations on days 4 to 6, then freeze on day 6. It is limited by decisions and GPU time, not by downloads. Running downloads: DFD is not on the critical path (needed only for the final cross-dataset tests and a pulse sanity check on days 5 to 6); In-the-Wild is finished. LAV-DF is on the critical path only for the audio-visual MVP and fusion, and it waits on the user. FakeAVCeleb, WildDeepfake and Celeb-DF++ are off the critical path.

## 7. Immediate action plan

### A. Start now

| # | Task | Priority | Deliverable | Depends on | Acceptance | Hand off now? |
|---|---|---|---|---|---|---|
| A1 | Decide O-011, O-006 and O-002 (the user, by multiple choice) | P0 | Logged decisions | none | Entries in DECISIONS.md | No (user) |
| A2 | Approve or change D-027 to D-030 | P0 | Status Decided | none | Entries updated | No (user) |
| A3 | Finish and verify the preparation tooling; log the face detector choice | P0 | Verified tests, a logged decision | S2-C report | Tests pass; numbers from the measurements | Role 1 |
| A4 | Write the I-0 hypothesis, success and failure criteria | P0 | EXPERIMENTS.md entry before any run (CLAUDE.md rule 2) | A1 | Present before the first run | Role 2 |
| A5 | Log the Stage 2 gate (V-002 for A6 and A7) and update PLAN.md stage statuses | P1 | Updated docs | none | PLAN and EXPERIMENTS consistent | Anyone |
| A6 | Push the repository to the private GitHub repo | P1 | Commits visible on GitHub | the user's approval | git log matches | Anyone |
| A7 | Builders for In-the-Wild and DFD (layouts verified first) | P1 | Manifests and splits | downloads landed | Counts match the documentation | Role 1 |

### B. Continue in parallel

DFD download (monitor only); S2-C measurements; the data and evaluation test suites on every change; Related Work and Data sections of the paper.

### C. Blocked

| Task | Blocked by | What unblocks it | Honest temporary alternative |
|---|---|---|---|
| Audio-visual MVP and fusion | LAV-DF | The user's Hugging Face token | Build the audio and video plumbing with FF++ video and ASVspoof audio as stand-ins, labelled as stand-ins and never used for a result |
| Audio-visual headline | FakeAVCeleb approval | The owners | Report audio-visual results in-dataset only and say so |
| Any model code | O-006, O-007, O-008 | The user's decisions | Write hypotheses and the cost-measurement script |
| Interface build | O-011 | The user's decision | Build against the mock adapter only |

### D. Integration milestones

| Date | Milestone |
|---|---|
| Sat 24 Oct | Data core and evaluation harness verified; plan and proposals logged |
| Sun 25 Oct | M1: decisions A1 and A2 made; Stage 0 to 2 closed and logged; loaders and the I-0 hypothesis done; response contract frozen |
| Mon 26 Oct | M2: MVP for audio, generated images and FF++ video with 3 seeds logged; gated data arrives by drive; mock interface demo |
| Tue 27 Oct | M3: candidate comparison table; iterations I-1 to I-3; inference library runs a real MVP checkpoint |
| Wed 28 Oct | M4: iterations I-4 and I-5 including fusion; rPPG decision; interface calls the real library |
| Thu 29 Oct | M5: all iterations frozen, final test touched once each; documents complete; end-to-end demo with inference_source real |
| Fri 30 Oct | M6: assembly, proofreading, submission |

### E. Final validation evidence

All tests green; ledger entries for every final evaluation; the leakage check passing on the final splits; a check that every number in the paper, report and slides traces to an E-xxx and to a run with inference_source real; a recorded demo from upload to answer; git tags per frozen iteration; the repository pushed; commands to reproduce each table.

## 8. Workflow from here on

- Tracking: STATUS.md carries the one-sentence NEXT STEP; PLAN.md the stage and iteration statuses; EXPERIMENTS.md every run; DECISIONS.md every choice; commit messages `type(scope): summary` (CLAUDE.md section 5).
- Ownership: each stream writes only inside its own folders (src/dfd/<area>/, tests/<area>/); shared files change only through the integrator (the main agent), who runs the whole test suite before each commit.
- Integration: a short daily integration pass; contract tests for the response structure; a branch per stream and a pull request into main once the user approves the GitHub routine (open).
- Subagents: independent streams run as subagents under the D-015 rules: a self-contained brief with the CLAUDE.md hard rules, their own folders, no commits, spot-check before merge, no GPU jobs started without the main agent.
- Testing: the three test suites (eval, data, prep) plus the contract tests; a stream is not done until its acceptance criteria are logged.
