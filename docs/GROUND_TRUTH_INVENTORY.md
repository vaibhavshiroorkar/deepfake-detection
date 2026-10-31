# GROUND TRUTH INVENTORY

Written 2026-10-30 (Phase 0 of the final-submission preparation). Read-only reconstruction of what the repository actually contains, at branch `main`, commit e27de9a (tag `submission-final`). Nothing was run except read-only listings, two read-only inspections of checkpoint metadata, and file hashing. No model, server, test or experiment was run for this file.

Method: source code was read first, then the records (docs/EXPERIMENTS.md, docs/DECISIONS.md, docs/PROTOCOL.md, docs/DATA_LOG.md, docs/TESTING_RECORD.md), then the deliverables. Where code and documents disagree, the code wins and the disagreement is listed in section 6. The detailed evidence notes behind this file (with file:line for every claim) were written to the session scratchpad during the audit; the key line references are repeated here.

Status vocabulary: **implemented** (code exists, is wired into the running application or harness, and has tests); **partially implemented**; **experimental** (code runs but the decisions forbid it from making a finding); **proposed** (designed, not built); **unavailable** (not in the repository or not on the machine); **unverified** (claimed in a document but not confirmed by code or files in this audit).

"Demonstrate" means: can be shown live on this machine with the local checkpoint and the authorised demo inputs. "Claim as measured" means: a logged experiment (E-xxx) supports a number, under the stated conditions.

## 1. Protected and frozen state (do not change)

| Item | Where | Rule | Source |
|---|---|---|---|
| Frozen detector checkpoint | `checkpoints/I-5-freeze-candidate/` (git-ignored; id `I-5-freeze-candidate-20261025-1608`) | never retrain, re-threshold or replace | D-061, D-067, D-092 |
| Sealed score files and reports | `data/_final/` (git-ignored) | read once per frozen iteration; the Evaluation Lab may only re-analyse a file whose hash is in the ledger | PROTOCOL.md section 7, D-088 |
| Ledger of sealed reads | `ledger/final_ledger.jsonl` (tracked, 76 lines, 0 repeats) | append-only; a repeat needs `--repeat "<reason>"` and a STATUS mistakes entry | src/dfd/eval/guard.py:64-106, src/dfd/eval/ledger.py |
| Frozen v1 inference path | `src/dfd/infer/`, `src/dfd/web/` | unchanged; v1 page must not be demonstrated (its wording predates D-087) | docs/DEMO.md section 7, TESTING_RECORD.md |
| Generated-video approval table | `APPROVED_HEADS = {}` in src/dfd/platform/modules/generated_video.py:39-41 | stays empty; a finding needs an approved head hash plus a preregistered independent validation | D-074 |
| Report-level keys forbidden | `schema.FORBIDDEN_REPORT_KEYS` (src/dfd/platform/schema.py:35) | no fused verdict, no overall score or probability | D-087 |
| Lip-sync offset band | `OFFSET_BAND_FRAMES = (-6, 2)` (src/dfd/platform/modules/lip_sync.py:41-44) | fitted in E-025, not to be refitted | D-073, E-025 |
| Archived original deliverables | `paper/`, `report/`, `slides/` | unchanged archive; not to be submitted | D-090 |
| Submission package | `submission/` (manifest regenerated 2026-10-29) | any change requires regenerating MANIFEST.txt | D-093 |
| Restricted data | Celeb-DF, Celeb-DF++, FakeAVCeleb | no frame, crop or per-video score may be published | docs/DATA_LOG.md, DOC_FACTS.md section 3 |
| Licensing questions | E-029 heads, SyncNet weights, FakeAVCeleb form terms | parked by the user; research-only sentences stay | D-092 (f) |

## 2. Application and software components

| Component or claim | Evidence (file, function) | What the code establishes | Status | Dependencies | Limitations and caveats | Demonstrate? | Claim as measured? |
|---|---|---|---|---|---|---|---|
| v2 website (the product front end) | deliverables_v2/website/serve_v2.py `build_app`; static/index.html, app.js, evallab.js | Wraps the unchanged backend `create_app`, replaces only the static routes; binds 127.0.0.1; default port 8012; `--live` enables analysis | implemented | Flask, the backend below | Without `--live` shows invented fixtures under a MOCK banner | yes | no (software behaviour only) |
| Original platform dashboard | src/dfd/platform/dashboard/serve.py `create_app` | Same API; port 8002 | implemented, superseded UI | Flask | kept unchanged | possible, not recommended | no |
| v1 MVP web page | src/dfd/web/app.py, `python -m dfd.web` | Single-score page; default port 8000 (not 8001 as some docs say); shows "Likely fake / Likely real" | implemented, frozen, superseded | dfd.infer | wording predates D-087 and contradicts it | **no** (DEMO.md section 7) | no |
| HTTP API | serve.py: `GET /api/mode`, `/api/vocabulary`, `/api/fixtures[/<name>]`; live only `POST /api/analyse` (202 job id; 400 NO_FILE; 429 QUEUE_FULL at 4 pending), `GET /api/jobs/<id>`; 413 over 512 MB; `/api/evallab/*` | Upload is accepted before validation; type errors surface as a failed job | implemented | Flask | one worker thread; jobs kept in memory (50) | yes (curl or browser) | no |
| Input validation | src/dfd/infer/library.py `inspect` (lines 30-95) | Extension whitelist (image png jpg jpeg bmp webp; video mp4 mov avi mkv webm; audio wav flac mp3 m4a ogg), decode check, ffprobe, SHA-256 | implemented | ffmpeg, ffprobe on PATH | typing by extension only, no MIME check | yes (files 7 and 8 of the demo set) | no |
| Pipeline | src/dfd/platform/pipeline.py `analyse_file`, `default_modules` | inspect, then probe, then the coordinator over 8 modules in fixed order | implemented | all modules | no overall timeout (`total_timeout_s` not passed) | yes | no |
| Coordinator | src/dfd/platform/coordinator.py `Coordinator.analyse`, `_run_one`, `build_report` | Runs modules one at a time with per-module timeouts in daemon threads; exceptions become `failed`, timeouts `timed_out`, contract breaks `INVALID_RESULT`; every module appears once; report state ok, partial or failed | implemented, tested (tests/platform/test_coordinator_reliability.py) | schema, explain | a timed-out thread cannot be killed, only abandoned | yes | no |
| Report schema 2.1 and explanations | src/dfd/platform/schema.py, explain.py `attach` | Fixed fields, coverage lists, conflict note, template-built explanations, banned words list ("authentic", "proves", "%") | implemented | none | explanation text is template, not model-generated | yes | no |
| Verdict layer | src/dfd/platform/verdict.py `assess_module`, `_PROFILES`; verdict_evidence.py | Per-module labels only; `is_probability` always false; experimental modules can only give "Experimental signal observed"; "Likely AI-generated" unreachable | implemented (D-087) | stored evidence values in `data/_results/` | profile wording rests on post hoc rates for some modules | yes | the evidence sentences quote logged numbers |
| Mock mode | src/dfd/platform/mocks.py, fixtures/*.json (11 scenarios) | Scripted modules with `source="mock"` through the real coordinator; MOCK labels | implemented | none | invented data | yes, labelled as mock | **never** |
| Evaluation Lab | src/dfd/platform/evallab/ (runner, analysis, api, store) | Re-analyses stored score CSVs with the project metrics and group bootstrap; final splits only as re-analysis of a ledger-registered file; refuses leakage | implemented (D-088) | sealed score files in `data/_final` (local only) | runs no model; cannot approve a detector | yes | reproduces logged numbers (for example 0.8218), it does not create new ones |
| Checkpoint loader | src/dfd/infer/checkpoint.py | Verifies file hashes and the preprocessing identifier before loading | implemented | metadata.json | missing checkpoint gives module `failed`, not `unsupported` | yes | no |
| Real adapter | src/dfd/infer/real_adapter.py | Video: 16 frames, YuNet crop, frame probabilities averaged, temperature applied; image: face crop through T-V or whole image through T-I; audio: trimmed, peak-normalised 64,600 samples through three encoders | implemented, frozen | encoder weights verified by hash (data/_logs/encoder_weights.json) | relative paths: start the server from the repository root | yes | no |
| Test suite | tests/ (51 files, 20 folders) | Last record 2026-10-29: 713 passed, 2 skipped (opt-in GPU), 13 deselected (data-dependent) | implemented | pytest, CUDA hidden | tests show software behaviour, not accuracy; 13 data tests not re-run recently | yes (run on camera) | as a software-test count only |
| Docs consistency checker | scripts/check_docs.py, scripts/build_numbers.py | Checks numbers in deliverables against docs/NUMBERS.json | implemented | | describes the archived originals in part (D-090) | optional | no |

## 3. Detector and analysis modules (registry order)

Thresholds below were read from `checkpoints/I-5-freeze-candidate/metadata.json` on 2026-10-30: T-V 0.6139 (temperature 0.832), T-I 0.2073 (temperature 0.636), T-A 0.6954 (temperature 0.4543), each "maximum balanced accuracy on the validation split, on calibrated probabilities (D-024)".

| Module | Evidence | Input | Model and source | Status | Threshold and provenance | Measured results (sealed unless marked) | Limitations | Demonstrate? | Claim? |
|---|---|---|---|---|---|---|---|---|---|
| `face_video` | modules/face_video.py | video with a video stream | SigLIP base (google/siglip-base-patch16-224), fully fine-tuned on FF++ c23 (I-1, E-005), largest face, 16 frames | implemented, user-facing; label "Possible manipulation" (high uncertainty) | 0.6139, FF++ validation (D-024) | Celeb-DF family 0.8218 [0.8020, 0.8393] (E-014, T1, T3); Celeb-DF v2 0.8999; DFD 0.8797; FF++ test_in 0.9899; FakeAVCeleb 0.9552 with sensitivity 0.480 (E-015); LAV-DF 0.3226 (below chance) | false-alarm rate on real videos 0.1098 (Celeb-DF family) and 0.1405 (DFD), post hoc, no interval (E-032); audio-only edits invisible | yes (demo files 1, 2, 3) | yes, with conditions |
| `image_detector` | modules/image_detector.py | still image | face found: same T-V head on the face crop; no face: frozen OpenAI CLIP ViT-L/14 with linear head (T-I) | implemented, limited tier; positive labels carry "(weak evidence)", a low score reads "Inconclusive" | T-V 0.6139 or T-I 0.2073 | T-I: ProGAN test_in 1.0000, CNN-synthesis test_x 0.9316, diffusion test_x 0.9354 (E-014); 21 modern generators macro AUC 0.7039, misses 0.7395 (E-028) | T-V was never validated on still photographs; CLIP weights research-only | yes (demo file 5) | yes for the listed sets; no for modern generators |
| `speech` | modules/speech.py | audio, or video with sound | mean of logits of HuBERT base, WavLM base plus, Whisper small (frozen) with linear heads | implemented, limited tier | 0.6954, ASVspoof 2019 LA dev | In-the-Wild 0.9148 [0.8896, 0.9383] (E-014, T2); ASVspoof 2019 LA eval 0.9835; EER 6.94 percent (ASVspoof) and 15.78 percent (In-the-Wild), post hoc (E-030); FakeAVCeleb specificity 0.637 (E-015) | scores only the first 4.04 s; part of the gain removes a silence and level shortcut (0.061 of 0.163); scores any sound, including a tone | yes (demo files 3, 4, 6) | yes, with conditions |
| `generated_image` | modules/generated_image.py | image on the face-crop path only | same CLIP T-I head on the whole image | **experimental** (D-083), always "Experimental signal observed" | threshold not applied (null) | E-028: on images with a face it missed 81 percent of generated ones | no finding possible | yes, as an experimental card | no |
| `lip_sync` | modules/lip_sync.py, platform/syncnet/ | video with sound and one face, 2 to 30 s | SyncNet v2 port (MIT code, weights licence not stated), YuNet tracking | implemented heuristic, limited tier | offset band (-6, 2) frames from E-025; confidence 3.0 from E-022 (fitted on its own test cases) | E-025: false alarms 3 of 230 real clips (1.3 percent); flagged 0 of 160 Wav2Lip clips | measures timing only; does not separate Wav2Lip fakes; verdict profile has no evidence entry; `model.evaluation` reads "not evaluated" | yes (demo file 3) | only the E-025 rates, as calibration and validation of a timing rule, not as detection |
| `generated_video` | modules/generated_video.py, platform/genvideo/ | video at least 2 s | DINOv2-small mean CLS of 8 frames plus logistic regression (head E020-C2-seed0, an appearance baseline, NOT ReStraV), `validated: false` | **experimental**, `APPROVED_HEADS` empty (D-072, D-074) | stored 0.9623, not applied | DF26 held-out 0.4957 (chance, E-020); flags 338 of 440 real clips (E-026) | source-confounded; no finding possible | yes, to show the honest "no finding" card | no (negative findings only) |
| `media_metadata` | provenance/metadata.py, modules.py | every file | ffprobe | implemented, information only | stream duration mismatch heuristic | none | not evidence either way | yes | no |
| `content_credentials` | provenance/c2pa_verify.py, modules.py | every file | c2pa-python 0.38.0 if installed, else a marker-byte scan | implemented, provenance tier | none | none | with no trust anchor configured, a validly signed file reads Inconclusive (`C2PA_VALID_SIGNER_UNTRUSTED`); sidecar or cloud credentials and watermarks not checked; mapping tagged "needs user review" | yes (all demo files show "No content credentials found") | no |

## 4. Research components

| Component or claim | Evidence | What it establishes | Status | Limitations | Claim as measured? |
|---|---|---|---|---|---|
| Identity-disjoint splits with content de-duplication | src/dfd/data/splits.py, leakage.py, dedup.py, identity_exclusion.py; data/_splits/*.json | No person, speaker or source video on both sides; Celeb-DF++ samples of dev identities excluded; byte duplicates counted once | implemented | LAV-DF has no identity ids (assumed from its paper); 13 data tests deselected recently | yes, as protocol |
| Sealed-test guard and ledger | src/dfd/data/loaders.py:35-53, src/dfd/eval/guard.py, ledger.py | `--final --iteration` required; repeats need a reason; ledger appended after success | implemented | software discipline, not access control (a script reading the CSV bypasses it) | yes, as protocol |
| Metrics | src/dfd/eval/metrics.py | AUC (Mann-Whitney, ties half), EER, balanced accuracy, log loss, Brier, ECE (15 bins) | implemented | TPR at fixed FPR not in the package | n/a |
| Group bootstrap | src/dfd/eval/bootstrap.py, run.py:286-294 | percentile cluster bootstrap, B = 2,000, 95 percent, seed 20261024; paired differences; 98.33 percent Bonferroni line | implemented | intervals governed by the number of groups (about 44 identities on the Celeb-DF family) | n/a |
| Threshold and temperature | src/dfd/eval/threshold.py, calibration.py | max balanced accuracy on validation; temperature by bisection on NLL | implemented | calibration is in-domain only; ECE worsened on dev_x for T-V and T-A (E-004) | n/a |
| Encoder sweep | src/dfd/sweep/, E-001 to E-003 | tie-set rule, cheapest wins | implemented, exploratory | dev slices small (15 identities, 12 speakers, 2 generator units) | exploratory numbers only |
| Fine-tuning | src/dfd/train/finetune.py, E-005 | full SigLIP fine-tune, lr 1e-5, 1,800 steps of batch 32 | implemented | one seed per rate in the search | exploratory dev gain +0.1700 |
| Confirmatory tests T1 to T3 | E-014, D-062 (b), D-091 | T1 +0.1980, T2 +0.1633 improved and robust; T3 non-inferior (margin 0.0100), superiority at 95 percent does not survive Bonferroni | done once | baseline ran at 224 px | **yes, the only confirmatory claims** |
| Audio-visual fusion (LAV-DF) | E-010 to E-012, D-059 | fused AUC 0.9935 rests on a production cue; visual at chance within produced videos | experimental, negative | | no AV detection claim |
| rPPG | src/dfd/rppg/pulse.py, E-006, M-005 | S-1 failed (0.301 against 0.6) | negative, not adopted | | negative result only |
| Shortcut audits | E-018, E-019, E-021, E-023, E-028 | codec metadata alone 0.9985 on FakeAVCeleb; header metadata 1.0 on DeepAction; file format 0.93 on Community Forensics | done | | yes, as findings about datasets |
| Modern-generator image heads | E-028, E-029 (checkpoints/modern-E029) | retrained head ranks better (0.8839) but flags 0.5213 of real images | research artefact, not installed (D-086) | CC BY-NC-SA data | exploratory |
| DF26 generated-video heads | E-027 (checkpoints/modern-E027) | source confound persists (0.8023) | research artefact, not installed (D-085) | | exploratory |
| Gemini image detection | E-031 | proposed only | **proposed, not run** | licence, provenance, matched real controls unresolved | **no** |
| Emotion analysis, watermark or logo detection | D-069, D-077 | excluded by decision | not built | | no |

## 5. Environment and external dependencies

| Item | Verified value | Source |
|---|---|---|
| Machine | Windows 10 Pro 10.0.19045; RTX 5070 12,227 MiB; driver 595.79; Ryzen 9 7900X; 31 GB RAM | docs/EXPERIMENTS.md lines 14-18, docs/SETUP.md section 1 |
| Python and packages | Python 3.13.15 under uv 0.12.12; torch 2.14.1+cu130; transformers, timm, opencv-python-headless, flask, soundfile, numpy; optional c2pa-python 0.38.0 | pyproject.toml |
| External tools | ffmpeg and ffprobe 9.0.1 on PATH (present on 2026-10-30) | `which ffmpeg` |
| Local artefacts present on 2026-10-30 | checkpoints/I-5-freeze-candidate (metadata.json, weights_T-V.pt, head_T-I.npz, head_T-A_AU-4/6/8.npz); data/_demo_inputs (8 files plus SOURCE.json); data/models/syncnet/syncnet_v2.model; .venv | directory listing |
| Undeclared runtime imports | scipy (lip-sync path) and Pillow arrive only transitively | audit of imports against pyproject.toml |

## 6. Discrepancies found (code against documents)

| # | Discrepancy | Evidence | Effect | Suggested handling |
|---|---|---|---|---|
| 1 | v1 page default port is 8000; serve_v2.py docstring and site README say 8001 | src/dfd/web/__main__.py:13 | none if `--port 8001` is passed | say "8001 when started with --port 8001" or avoid the v1 page |
| 2 | SETUP.md says a missing checkpoint gives "unsupported or failed"; code always gives `failed` | modules/base.py:45-53 | wording only | correct the sentence when docs are next touched |
| 3 | `score.calibrated: True` on face_video and the whole-image path, while the verdict sets `is_probability: false` | face_video.py:49, verdict.py:137 | two fields read differently | in the viva: "calibrated in-domain only, therefore not a probability elsewhere" |
| 4 | Threshold fitted "on calibrated probabilities" (checkpoint) against PROTOCOL.md section 5 wording | metadata.json; PROTOCOL.md | negligible: temperature scaling is strictly increasing, so the same validation files fall on each side of the line; only scores lying between two adjacent validation scores could differ | explain, do not change |
| 5 | `lip_sync` not marked `requires_gpu` | modules/lip_sync.py | GPU guard does not cover it | note as limitation |
| 6 | Relative paths for YuNet and the encoder log | prep/face.py:28, sweep/encoders.py:20 | server must start from the repo root | runbook says so |
| 7 | DATA_LOG.md DF26 row says "NOT used for any experiment" but E-020 evaluated it | DATA_LOG.md:345 against :350 and the ledger | stale row | correct when docs are next touched |
| 8 | PLAN.md last updated 2026-10-25; the freeze was planned for 2026-10-29 but ran on 2026-10-25 | PLAN.md; ledger dates | stale plan | the paper must not date the freeze to 2026-10-29 |
| 9 | Author order: IDENTITY.md (Vaibhav, Shivansh, Adithya) against v2 deliverables (Shivansh, Vaibhav, Adithya) | D-089 | known and recorded | use the D-089 order in the submission |
| 10 | Generated-video head is the `cls_mean` appearance baseline, not ReStraV | head.npz metadata | wording risk | say "appearance baseline"; paper/main.tex states it correctly |

## 7. What git can and cannot tell us about people

All 168 commits are authored by the account "Shivansh Gupta <shivansh.gupta@somaiya.edu>", and the records show that much of the work was carried out by an AI coding agent working in that account under the user's decisions. No document assigns any work to a named member: D-030 and docs/WORKSTREAMS.md define three roles without names, and deliverables_v2/SPEC.md says "no contribution statements". Individual contributions therefore cannot be established from the repository; they are collected by interview (docs/TEAM_WORK_ALLOCATION.md, after approval).
