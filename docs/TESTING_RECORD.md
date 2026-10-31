# TESTING RECORD

Engineering verification of the software: does the code do what it is specified to do, and does the application run end to end. None of this measures detector accuracy; accuracy is reported only by the logged experiments in `docs/EXPERIMENTS.md` (for example the sealed test staircase and the FakeAVCeleb sealed half), and a passing test says nothing about how well a detector separates real from manipulated media.

Every count below comes from a log saved in `docs/test_evidence/2026-10-26/` in this repository. Older counts quoted for comparison are marked as history; they were not re-run.

## 1. Environment of the runs (2026-10-26)

| Item | Value |
|---|---|
| Machine | Windows 10 Pro 10.0.19045, NVIDIA RTX 5070 (12 GB), development machine of the project |
| Environment | project `.venv`, Python 3.13.15, synced with `uv sync --locked --extra c2pa` (100 locked packages; `uv lock --check` exit 0, `uv_lock_check.txt`) |
| Source state | git HEAD `5af239a9427339890bb63169476de12099fc5f47` plus uncommitted changes (the pytest header of each log records "tracked files MODIFIED vs HEAD (27 files)"): the reproducibility remediation, the dashboard fix of section 4, and the report, slides and paper update |
| External tools | ffmpeg 9.0.1 and ffprobe on PATH; pdftotext 4.00; Playwright Chromium build 1248 from `D:\cache\playwright` (set by `tests/web/test_web_browser.py`), Microsoft Edge 156 and Google Chrome 155 as fallbacks |
| Local artifacts | the git-ignored frozen checkpoint, SyncNet weights, generated-video head and DINOv2-small weights were present (docs/SETUP.md section 5) |

## 2. Automated tests

| Command (from the repository root, `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`) | Result | Log |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs` with 13 data-dependent items deselected (list below) | 497 passed, 2 skipped, 13 deselected, exit 0 | `full_safe_suite.txt` |
| `RUN_PARITY=1 RUN_LIVE_FACE=1 .venv/Scripts/python.exe -m pytest tests/platform_modules/test_parity.py tests/platform_live_face -p no:cacheprovider -rs` (GPU visible) | 5 passed, exit 0 | `optin_gpu_tests.txt` |

- The 2 skips of the safe suite are exactly the two opt-in tests (they need `RUN_PARITY=1` / `RUN_LIVE_FACE=1`); the second command runs them and the rest of their files.
- `CUDA_VISIBLE_DEVICES=-1` hides the GPU; an empty value does not on this machine (docs/SETUP.md section 1).
- The opt-in parity test compares the three wrapped detectors with the stored scores of the freeze parity check on 11 non-sealed files (training or validation split only, the test splits are refused by the loader); it rewrote `data/_results/platform/module_parity.json`: 11 cases, largest absolute difference 0.001919, tolerance 0.02, passed.
- Deselected (they read the real dataset inventories under `data/`, one hashes real files including sealed test samples): `tests/data/test_data_builders.py::test_A7_ffpp_c23_documented_counts`, `::test_A7_asvspoof2019_la_documented_counts`, `::test_A7_progan_documented_counts`, `::test_real_cnn_and_diffusion_structure`, `::test_celebdf_v2_real_files_counts`, `::test_ffpp_complete_synthetic_tree_with_the_real_official_split_files`, `tests/data/test_data_dedup.py::test_real_image_sets_pass_hash_check_after_dedup`, `tests/data/test_data_devx.py::test_real_dev_x_slices_are_group_disjoint` (3 parameters), `tests/data/test_data_dfd_itw.py::test_dfd_real_files_counts`, `::test_in_the_wild_real_files_counts`. They are not verified in this record.
- What the suite covers, by area: data loaders and split guards, the frozen inference library and v1 web page, the platform schema and coordinator (timeouts, failure isolation, late results), the wrappers of the frozen detectors with a fake adapter, the generated-video validation gate (approval table empty, metadata cannot authorise a finding, non-finite values, swapped files), lip-sync decision mapping and missing weights, C2PA with an ephemeral test PKI and without the package, provenance metadata, the dashboard in a real browser (rendering, live upload flow, error states).

## 3. Documentation and dependency checks

| Command | Result | Log |
|---|---|---|
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers` | 71 pass, 0 fail, exit 0 | `check_docs_venv.txt` |
| `uv lock --check`; `uv sync --locked --extra c2pa --dry-run` | exit 0; "Would make no changes" | `uv_lock_check.txt` |
| `git diff --check` | exit 0 | (not saved; run at the end of the session) |

The documentation check now runs in the project environment because `lxml`, `python-docx` and `python-pptx` were declared in the `dev` group on 2026-10-26.

## 4. Application runs (live dashboard and the v1 page)

Server: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.platform.dashboard.serve 8002 --live` (GPU visible). Inputs: `data/_demo_inputs/`, built by `scripts/platform/make_demo_inputs.py` from non-sealed files only (docs/DEMO.md section 2).

- HTTP API: every input uploaded with `POST /api/analyse` and polled with `GET /api/jobs/<id>`; the six media files ended `done` with all seven modules reported, the `.txt` file ended `failed` with `UNSUPPORTED_MEDIA_TYPE`, the corrupt `.mp4` `failed` with `UNREADABLE_FILE`. Full per-module output: `demo_api_run.txt`; summary table in docs/DEMO.md section 4.
- Browser: the page was driven in headless Chromium (Playwright): choose the file, press Analyse, wait for Done or Failed. Six reports rendered with the LIVE ANALYSIS banner and seven module cards, no page errors; the two failures show the error code and no report. `ui_results.json`, screenshots in `ui_screens/`.
- v1 page: `python -m dfd.web --adapter real --checkpoint checkpoints/I-5-freeze-candidate --port 8001` answered `/api/health` (`adapter: real`) and `/api/analyse` for a video with audio, an image and an audio file with `inference_source: real` and no errors (console output only, not saved).

### Defect found and fixed during these runs

- Dashboard, live mode: after a failed or refused analysis (unsupported type, unreadable file), the MOCK scenario shown at page load (with a card reading "Indicates manipulation", tagged MOCK) stayed on screen next to the failure message. Fix in `src/dfd/platform/dashboard/dashboard.js`: a new analysis hides the report on screen before it starts (`hideReport`). Regression test `tests/platform_dashboard/test_dashboard_live_ui.py::test_failed_or_refused_analysis_does_not_leave_a_mock_or_old_report_on_screen`; re-checked in the real browser (the two failure cases now show no report).

### Behaviour observed and documented, not changed

- The speech module (a wrapper of the frozen speech detector) has no speech check of its own: on a synthetic 440 Hz tone it reports "no indication within tested scope" with score 0.0. The lip-sync module on the same file reports the audio as not speech-like. Adding a speech gate would change a frozen detector's decision with a heuristic not evaluated for it, so it was not done (docs/DEMO.md, decided autonomously, needs user review).
- The generated-video score is 1.0 on a real LAV-DF clip and on a synthetic test pattern; the module is experimental and always inconclusive, so no finding is made from it.

## 5. Not verified

- The 13 deselected data-dependent tests (section 2).
- A run from a fresh clone without the git-ignored artifacts, and a fresh browser profile (`playwright install chromium` was not run).
- The live E2E scripts `scripts/platform/live_full_e2e.py` and `live_c2pa_e2e.py` were not re-run on this tree; their last recorded passes (60 checks, 0 failed; 5 of 5) are history from earlier on 2026-10-26.
- The reproducibility check `scripts/genvideo/repro_check.py` was not re-run (it scores DF26 clips, which must not be touched); its recorded result (16 of 16 identical frames, score difference 0) is history.
- CPU-only speed, other operating systems, other Python versions.

## 6. History (earlier records, not re-run here)

- Full suite at the post-audit remediation: 508 passed, 2 skipped (empty `CUDA_VISIBLE_DEVICES`, so the GPU was visible), docs/STATUS.md.
- Clean environment from the committed lock (2026-10-26, outside the repository): 495 passed, 2 skipped, 13 deselected, with the GPU visible and with `-1`; documentation check 16 pass, 2 fail before `lxml` and `python-pptx` were declared (docs/SETUP.md section 3).

## 7. Redesign and explanation layer (2026-10-27)

Source state: git HEAD `5af239a9427339890bb63169476de12099fc5f47` plus the uncommitted work of 2026-10-26 and 2026-10-27 (65 changed or
untracked entries at the browser run). Logs in `docs/test_evidence/2026-10-27_redesign/`. Commands from the repository root with
`PYTHONPATH=src` and `PYTHONDONTWRITEBYTECODE=1`.

| Command | Result | Log |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs` with the same 13 deselected items as section 2 | 551 passed, 2 skipped (the two opt-in tests), 13 deselected, exit 0 | `full_safe_suite.txt` |
| `RUN_PARITY=1 RUN_LIVE_FACE=1 .venv/Scripts/python.exe -m pytest tests/platform_modules/test_parity.py tests/platform_live_face -p no:cacheprovider -rs` (GPU visible) | 5 passed (second run; the first failed, see below) | `optin_gpu_tests.txt` |
| parity output of that run (11 non-sealed files, real detectors) | largest absolute difference 0.001919, tolerance 0.02, all passed (as on 2026-10-26: the wrapped detector scores are unchanged) | `module_parity_2026-10-27.json` |
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers` after `scripts/build_numbers.py` | 71 pass, 0 fail, exit 0 | `check_docs.txt` |
| `.venv/Scripts/python.exe scripts/platform/ui_demo_run.py --out docs/test_evidence/2026-10-27_redesign` (Edge 156, live server in-process, real detectors) | 8 of 8 inputs behaved as specified, 0 page errors, every real report valid under the schema 2.1 checker, every key figure equal to the JSON score after the declared rounding, no horizontal scroll at 390 px | `ui_redesign_run.json`, `screens/` |

New tests in the suite: `tests/platform_explain/test_explain.py` (38: numbers from cited fields, status wording, experimental and mock
labelling, missing values, fabricated-text detection, lip-sync curve and frame strip, decisions unchanged), `tests/platform_dashboard/test_dashboard_design.py`
(8: contrast of every text pair in light and dark mode, control edges, palette, reduced motion, focus ring), and in the dashboard browser tests:
upload-first opening with no invented report, file preview before analysis, group order and experimental de-emphasis on a real-shaped report,
raw values only under Technical details, the explanation text on screen equal to the JSON, keyboard reach of the file input and every disclosure.
Migrated (same intent, new page structure): the live-UI ready condition, the human progress names, the partial-report wording, the stale-report
regression (now starts from a scenario chosen on purpose), the observation-kind check (reads Technical details), the exact `/api/mode` body.

Per-file outcomes of the browser run are in docs/DEMO.md section 4. Screens: `0_landing_desktop.png`, `0_landing_dark.png`, and for each input
`*_desktop.png`, `*_phone.png`, and for the videos `*_explanations_open.png` (offset curve, frame strip). Before the redesign:
`docs/test_evidence/2026-10-26/ui_screens/`.

Protected paths: the SHA-256 of the 221 files under `src/dfd/infer`, `src/dfd/web`, `checkpoints/`, `ledger/`, `data/_final`,
`data/_results/genvideo` (as present at the start of the session) and `data/models/syncnet` were recorded before any edit and compared at the end
(result in the final report of the session); `git diff` shows no change under `src/dfd/infer`, `src/dfd/web` or `ledger/`.

Not verified on 2026-10-27: the 13 deselected data-dependent tests; `scripts/platform/live_full_e2e.py` and `live_c2pa_e2e.py` (selectors
migrated, not run, because they overwrite the recorded result files in `data/_results/platform`); a fresh clone; screen readers (only keyboard
reach and semantic structure were tested); browsers other than Edge 156.

Defects found and fixed on 2026-10-27: (1) a CSS rule overrode the `hidden` attribute, so the report shell showed at page load (fixed before
any demonstration; test added); (2) the generated-video explanation would have said "14 frames analysed" while the encoder sees 8 (fixed in the
template; module field unchanged); (3) the live-face browser check read fields that moved under Technical details (migrated). The first opt-in
run also overwrote the git-ignored screenshot `data/_results/platform/live_face_dashboard.png` without a backup (docs/STATUS.md, Mistakes).

## 9. Final verification of the modern-image work (2026-10-27 evening, session 3)

Code state: git HEAD `5af239a` plus the uncommitted work of 2026-10-26 and 2026-10-27 (nothing committed). Evidence folder: `docs/test_evidence/2026-10-27_final/` (logs and screenshots written by the commands below; they correspond to the final code of this session). Environment as section 1 (Windows 10 19045, Python 3.13.15, PyTorch 2.14.1+cu130, RTX 5070 12,227 MiB, driver 595.79, Edge 156 for the browser run).

| Command | Result | Evidence |
|---|---|---|
| `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs` with the same 13 data-dependent items deselected as section 2 | 568 passed, 2 skipped (the two opt-in tests), 13 deselected, 0 failed, exit 0, 55 s | `full_safe_suite.txt` |
| `RUN_PARITY=1 RUN_LIVE_FACE=1 .venv/Scripts/python.exe -m pytest tests/platform_modules/test_parity.py tests/platform_live_face -p no:cacheprovider -rs` (GPU visible) | 5 passed, exit 0, 36 s | `optin_gpu_tests.txt` |
| Detector parity (written by the opt-in test) | 11 cases, 0 skipped, largest absolute difference 0.001919, tolerance 0.02, passed | `module_parity_2026-10-27.json` |
| `PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/ui_demo_run.py --out docs/test_evidence/2026-10-27_final` (Edge, live server in-process, real detectors) | 8 of 8 inputs behaved as specified (6 completed, the unsupported .txt and the corrupt video failed with their error codes), 0 page errors, every report valid, key figures equal to the report JSON; 22 screenshots (desktop, phone width, explanations expanded for four inputs, dark landing page) | `ui_redesign_run.json`, `screens/` |
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers` | 71 pass, 0 fail (paper, report, slides: identity, dashes, placeholders, internal IDs, number provenance against the rebuilt docs/NUMBERS.json) | `check_docs.txt` |
| `.venv/Scripts/python.exe report/layout_check.py` | 26 of 26 rows within 2 mm of the template | `report_layout_check.txt` |
| `.venv/Scripts/python.exe slides/check_slides.py` | 923 pass, 0 fail (56 slides) | `slide_check.txt` |
| `cd paper; latexmk -g -pdf main.tex` | 31 pages, 0 undefined references or citations in main.log | `paper_build_check.txt` |

Additional checks of this session (outputs quoted in docs/EXPERIMENTS.md, not saved as files): independent recomputation of E-027 (closed AUCs, 353 of 440 real clips flagged), E-028 (all 21 per-generator AUCs, pooled and face-routed metrics), E-029 (freeze hashes, split hash, macro AUC of three seeds, all 21 per-generator AUCs, real-source false-positive rates) and E-030 (EER of I-0 and I-4 from the stored sealed scores, score-file hash against its provenance record); preregistration hashes of E-027, E-028 and E-029 re-derived from the document with the Status line restored. Protected paths: `git status` shows no change under `src/dfd/infer`, `src/dfd/web`, `ledger`, `checkpoints` (tracked part) or `data/_final`; the frozen T-I head and checkpoint metadata hashes equal those recorded in E-028; `APPROVED_HEADS` is `{}`; the lip-sync band is (-6, +2).

Product checks of D-083 on the final code (live run on `data/_demo_inputs/5_face_image_frame_ffpp_val.png`): image analysis `indicates_manipulation`, 0.999983, decision line 0.61 (unchanged); the experimental card `completed`, `inconclusive`, group experimental, score 0.020518, threshold null, no key figure, warning code `EXPERIMENTAL_PORTRAIT_SCOPE`; its limitations quote the E-028 face measurement; the whole page took 1.3 s for this image (1.7 s in the earlier run of this session; one run each, so no latency claim). A seventh unit test pins the limitation wording. This is behaviour on one demonstration image and says nothing about reliability.

Not run or not verified on 2026-10-27 evening: the 13 deselected data-dependent tests; `live_full_e2e.py` and `live_c2pa_e2e.py` (they overwrite recorded result files); a fresh clone; browsers other than Edge 156; screen readers; the E-028 parity rescoring and the E-028 re-encoding control were not repeated by the main agent (they need the images and the GPU path; their stored outputs were read). The earlier session's feature-cache loop (`extract_loop.sh`) was still running at the start, was left alone, and exited by itself once all 92 Eval and 15 Small shards were cached.

## 10. Recovery verification of the verdict layer, Evaluation Lab and UI redesign (2026-10-28)

Context: the session of 2026-10-27 22:30 to 23:02 (D-087, D-088, three parallel workstreams: benchmark audit, Evaluation Lab backend, UI redesign) ended at a usage limit before it recorded tests. This section re-verifies the integrated tree on 2026-10-28. Code state: git HEAD `5af239a` plus the uncommitted work (101 entries in `git status --short` at the end of this session); nothing committed. Environment as section 9. Evidence folder: `docs/test_evidence/2026-10-28_recovery/`.

| Command | Result | Evidence |
|---|---|---|
| `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_verdict tests/platform_evallab -q -p no:cacheprovider` (at the start, on the tree as left) | 112 passed | terminal output of this session (not saved) |
| Full safe suite (command and 13 deselections as section 2), at the start, on the tree as left | 708 passed, 2 skipped (the two opt-in tests), 13 deselected, exit 0 | terminal output of this session (not saved) |
| Same full safe suite after the two fixes below | 710 passed, 2 skipped, 13 deselected, exit 0, 69 s | `full_safe_suite.txt` |
| `PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/ui_demo_run.py --out docs/test_evidence/2026-10-28_recovery/ui_live` (Edge, live server in-process, real detectors) | 8 of 8 inputs as specified (6 completed with valid schema 2.1 reports; the unsupported .txt and the corrupt video ended in the "Unsupported input" category), 0 page errors, no horizontal overflow at 390 px. Run before fix 1 below (its full-page screenshots predate the header change; the fix changes only the scroll offset and header pinning) | `ui_live/ui_redesign_run.json`, `ui_live/screens/` |
| `PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/recovery_browser_check.py docs/test_evidence/2026-10-28_recovery/browser data/_demo_inputs/1_face_swap_video_ffpp_val.mp4` (new script: viewport screenshots after a real analysis at 1366x900 and 390x844, header geometry, then the Evaluation Lab flow against the real API: choose I-4.csv, run a celebdf_v2 dev_x block, open the sealed celebdf_family block's configuration) | after the fix: overview top 141 px below a 129 px sticky header (desktop), header not pinned at 390 px; 0 console or page errors; lab run finished, metrics, confusion matrix and ROC drawn from the API; sealed block shows the "re-analysis of a logged evaluation" notice | `browser/browser_check.json`, `browser/*.png` |
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers` | first 68 pass, 3 fail (number provenance stale: EXPERIMENTS.md and RESEARCH.md changed after the last build); after `python scripts/build_numbers.py` 71 pass, 0 fail | `check_docs.txt` (the second run) |

Evaluation Lab on the real stored score files (in-process Flask client, runs written to the session scratch folder, not to data/_evallab; script in the session scratch folder): the catalogue lists the 15 files of data/_final/scores; a sealed block (I-4, visual, celebdf_family test_x) with purpose "development" is refused with FINAL_PURPOSE; the same block with purpose "final_reanalysis" runs, its final_guard names the ledger entry of 2026-10-25 with the same score-file SHA-256 (e0c78a8c...f383f), the val_fit thresholds are the three per-seed validation thresholds, and the AUC point estimate is 0.8218, equal to E-014 (the interval differs from E-014 because this check used 500 resamples); a paired comparison of two development runs on the same block is available, a comparison of runs on different samples is refused with its reason; the SHA-256 of ledger/final_ledger.jsonl is unchanged after all runs; `APPROVED_HEADS` is `{}`. The development-slice values of these engineering checks are not results and are not quoted anywhere.

Defects found and fixed in this session (both in files of the UI workstream):
1. After a live analysis the page scrolled the report overview to a fixed 84 px offset (`scroll-padding-top`), while the sticky header was 129 px tall at 1366 px width (two rows) and 213 px at 390 px, so the file name of the report was hidden under the header (screenshot `browser` run before the fix, session scratch only; geometry 84 px against 129 px and 213 px). Fix: dashboard.js measures the header and sets `--head-h`, used by `scroll-padding-top`; at 720 px and below the header is no longer sticky (it took a quarter of a phone screen). Regression test `test_report_overview_is_not_hidden_under_the_header_after_analysis` (two viewports) in tests/platform_dashboard/test_dashboard_live_ui.py.
2. The lab's confusion matrix caption said "counts of samples" while the cells show the mean over three seeds (fractional values such as 1282.33). The caption now appends the server's `aggregation` text when it is not a single run.

Protected paths (checked 2026-10-28): `git status` shows no change under `src/dfd/infer`, `src/dfd/web`, `ledger`, `checkpoints` or `data/_final`; `schema.FORBIDDEN_REPORT_KEYS` is identical to HEAD; `APPROVED_HEADS` is `{}`; the lip-sync band is (-6, 2).

Not run on 2026-10-28: the opt-in GPU tests (`RUN_PARITY=1` rewrites data/_results/platform/module_parity.json and `RUN_LIVE_FACE=1` writes a screenshot under data/_results/platform/, an existing results folder that is protected in this session; the detectors and their wrappers were not changed since section 9, where both passed); the 13 deselected data-dependent tests; a fresh clone; browsers other than Edge; screen readers; the paper, report layout and slide checkers beyond what check_docs runs (the deliverables were not changed in this session).

## 11. Deliverables update for the verdict layer and the Evaluation Lab (2026-10-28, afternoon)

Changed: paper/main.tex (platform subsection: per-module verdicts and the Evaluation Lab; problem statement and Figure 1 caption: what "calibrated" means; software paragraph: 2026-10-28 test counts; comparison subsection: comparability classes, the RawNet2 row, no directly comparable face-video row, why the experimental heads stay unapproved; one limitation item), report/ch4_platform.py (subsections "Per-module verdicts and uncertainty" with a label table, and "Evaluation Lab"; testing figures; comparison paragraphs), report/ch1_2.py (one sentence on what "calibrated" means), slides/build_slides.py (slides 34 "Verdicts and the Evaluation Lab" and 35 "Benchmark, limits, unapproved heads"; 58 slides). The report and slide text was drafted by two document agents from a facts brief verified by the main agent; the main agent reviewed both and corrected slide 35 (the Xception row had no baseline value, one bullet called the installed whole-image model "not installed", the published EERs had no unit, and its notes cited the wrong experiment for the DF26-trained head) before any check below. A stray carriage return in paper/main.tex, present since session 3, had turned one cross-reference into the visible text "Section ef sec:later" in the PDF; it was fixed (the PDF now shows the section number).

| Command | Result | Evidence |
|---|---|---|
| `cd paper; latexmk -g -pdf main.tex` | exit 0, 31 pages, 0 undefined references or citations | `paper_build.txt` |
| `.venv/Scripts/python.exe report/build_report.py` (Word updates fields and exports the preview PDF) | exit 0, 119 pages, 0 placeholders, 53 citations, 14 figures, 45 tables, 42 equations | `report_build.txt` |
| `.venv/Scripts/python.exe report/layout_check.py` | 26 of 26 rows within 2 mm | `report_layout_check.txt` |
| `.venv/Scripts/python.exe slides/build_slides.py`; `.venv/Scripts/python.exe slides/check_slides.py` | 58 slides; 955 pass, 0 fail (one builder warning: the slide 34 table is 2.68 in against a 2.46 in guide; the checker passes and the PowerPoint render shows no overlap) | `slide_check.txt` |
| `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers` | 71 pass, 0 fail; result-like numbers traced: paper 1070 of 1070, report 1996 of 1996, slides 949 of 949 | `check_docs.txt` |
| `pytest tests/platform_verdict tests/platform_evallab` | 112 passed, exit 0 | `focused_verdict_evallab.txt` |
| Full safe suite (as section 2) | 710 passed, 2 skipped, 13 deselected, exit 0, 83 s | `full_safe_suite_final.txt` |

Visual checks: slides 34 and 35 exported through PowerPoint and inspected (no overlap or cut text). Not regenerated: slides/DeepFake_Detection_Presentation_preview.pdf (git-ignored; no LibreOffice on this machine; the PPTX is the deliverable). No application file changed in this step, so no browser run was repeated. Not run, as in section 10: opt-in GPU tests, the 13 data-dependent tests.

## 12. Pre-commit audit and the face-video wording (2026-10-28, late)

The user decided on 2026-10-28 that the face-video flag reads "Possible manipulation" and not "Likely manipulated" (D-087 addendum), accepted D-083 as an experimental card, and asked for a pre-commit audit. Code state: git HEAD `5af239a` plus uncommitted work; nothing committed. Baseline of this task: 101 `git status --short` entries, 0 staged, ledger SHA-256 `0f60d7eb7e241d461f4ecf15d4be3717797fdaff5bdd1fa7144f907534ea52d4`.

Changed for the wording: src/dfd/platform/verdict.py (vocabulary entry `possible_face_manipulation` with the label "Possible manipulation"; `likely_manipulated` removed; uncertainty of a flag "high"; reason text states that the false-alarm rate differed between test sets and that a flag is a possible sign, not a determination), the regenerated mock fixtures (`python -m dfd.platform.mocks`), tests/platform_verdict/test_verdict.py (code names; 3 new tests: label and text, the retired label absent from vocabulary and fixtures, threshold and experimental exclusion unchanged), tests/platform_dashboard/test_ui_v2.py (code name), the paper, report and slide wording, VIVA_PREP, and a D-087 addendum. Not changed: model, frozen threshold, splits, stored scores, sealed results, ledger, `APPROVED_HEADS`, `FORBIDDEN_REPORT_KEYS`.

| Command | Result | Evidence |
|---|---|---|
| `pytest tests/platform_verdict tests/platform_explain tests/platform_pipeline` (after the change) | 112 passed | terminal output (not saved) |
| `pytest tests/platform_modules/test_generated_image.py` (D-083 constraints: finding always inconclusive, threshold None, not calibrated, EXPERIMENTAL warning, image analysis unchanged) | 7 passed | terminal output (not saved) |
| Full safe suite (as section 2) | 713 passed, 2 skipped, 13 deselected, exit 0, 81 s | `full_safe_suite_label.txt` |
| `scripts/build_numbers.py`, then `check_docs.py --numbers` | 71 pass, 0 fail; numbers traced: paper 1072 of 1072, report 1998 of 1998, slides 949 of 949 | `check_docs_2.txt` |
| `latexmk -g -pdf main.tex` | exit 0, 31 pages, 0 undefined references | `paper_build_2.txt` |
| `report/build_report.py`; `report/layout_check.py` | exit 0, 119 pages; layout 26 of 26 | `report_build_2.txt`, `report_layout_check_2.txt` |
| `slides/build_slides.py`; `slides/check_slides.py` | 58 slides; 955 pass, 0 fail (the slide 34 table height warning of the builder remains; the PowerPoint render of slide 34 was inspected and shows no overlap) | `slide_check_2.txt` |
| `scripts/platform/ui_demo_run.py --out .../ui_live_label` (Edge, live server, real detectors, 8 inputs) | 8 of 8 as specified, 0 page errors, every report valid | `ui_live_label.log`, `ui_live_label/ui_redesign_run.json` |
| `scripts/platform/recovery_browser_check.py` (face-swap demo video, 1366 and 390 px) | the face-video card reads "Possible manipulation" with the new uncertainty text; no console errors, no horizontal overflow | `browser_label/browser_check.json` |

Searches for the retired label: no occurrence remains in src, tests, the built paper PDF, the report preview PDF or the slide deck (text, tables, notes). Remaining occurrences are historical on purpose: the D-087 contract text and the superseded NEXT STEP lines in docs/STATUS.md, the screenshots and run records under docs/test_evidence/ made before the change, and the wording history in docs/VIVA_PREP.md.

Line endings: paper/main.tex differs from HEAD by 362 added and 8 removed lines (content only); `git ls-files --eol` shows the index as LF and the working copy as CRLF for 39 modified files (`core.autocrlf=true`, no .gitattributes), and 38 of them were already CRLF at the start of this task. The stray carriage return fixed earlier is gone. Nothing was normalised; the staged diff is LF.

Test evidence in git (selection): every `.txt` and `.json` log of docs/test_evidence/, plus six screenshots of the final UI state in `2026-10-28_recovery/browser_label/` that show no dataset media. The other screenshots (about 55 MB, including previews and frame strips of FF++ and LAV-DF material, whose licences restrict redistribution, and superseded layouts) stay on this machine only and are not staged.

Not run: opt-in GPU tests and the 13 data-dependent tests (they write under protected result folders or read sealed data); any new evaluation or experiment.

## 13. Final staged-diff review (2026-10-28, night)

Baseline of this task: branch feat/generated-video-sprint, HEAD `5af239a`, 191 staged files (30,208 insertions, 439 deletions), no unstaged change, untracked only the local screenshots and `scripts/modern_image/extract_loop.sh`; ledger SHA-256 `0f60d7eb7e241d461f4ecf15d4be3717797fdaff5bdd1fa7144f907534ea52d4` (unchanged at the end). Staged list and statistics at the start: `baseline_staged_names.txt`, `baseline_staged_stat.txt`.

Changed (wording only, no code, model, threshold, split, score or record): the paper contribution item and the software paragraph no longer present the frozen v1 sentence "calibrated probability" as the meaning of every score; the software paragraph now quotes it as the wording of the frozen page and says it holds only for content like the validation split, that the audio score carries a ranking-score warning, and that the platform states the narrower meaning. The same claim was corrected in the report abstract, the report scope table (image, video without audio and standalone audio rows; the speech score has `calibrated: False` and the face-crop path is not calibrated for still images), the report pipeline caption, and the slide 2 card and notes. The caption of the dashboard figure (report and slide 27) now says the screenshot was taken before the per-module verdict labels; the image itself was not changed. The slide 34 notes now give the post hoc false-alarm rates (about 0.11 Celeb-DF family, 0.14 DFD) behind the "Possible manipulation" label. Kept on purpose: the report table of the v1 response fields quotes the fixed sentence as a field description of the frozen library.

| Command | Result | Evidence |
|---|---|---|
| `cd paper; latexmk -g -pdf main.tex` | exit 0, 31 pages, 0 undefined references; the one overfull vbox (4.2 pt, page 4) is the same as in `paper_build_2.txt` | `paper_build.txt` |
| `report/build_report.py`; `report/layout_check.py` | exit 0, 119 pages, 0 placeholders; layout 26 of 26 | `report_build.txt`, `report_layout.txt` |
| `slides/build_slides.py`; `slides/check_slides.py` | 58 slides; first run 954 pass, 1 fail (the longer slide 27 caption wrapped); caption shortened; second run 955 pass, 0 fail (the slide 34 table height warning of the builder remains) | `slides_build.txt`, `slide_check.txt` |
| `check_docs.py --numbers` | 71 pass, 0 fail; numbers traced: paper 1072 of 1072, report 1998 of 1998, slides 949 of 949 | `check_docs.txt` |
| `pytest tests/platform_verdict tests/platform_explain tests/platform_pipeline` | 112 passed | `focused_verdict.txt` |
| `pytest tests/platform_modules/test_generated_image.py` | 7 passed | `focused_d083.txt` |
| `pytest tests/platform_evallab` | 57 passed | `focused_evallab.txt` |
| Full safe suite (as section 2, GPU hidden) | 713 passed, 2 skipped, 13 deselected, exit 0 | `full_safe_suite.txt` |
| `uv lock --check` | exit 0 | `uv_lock_check.txt` |

Visual checks: page 16 of the built paper PDF (the software paragraph) was rendered and inspected. The report preview PDF and the slide deck were not rendered or inspected in this task; their checks are the layout check and the slide checker above. Not run: opt-in GPU tests, the 13 data-dependent tests, the live browser demo (no application file changed).

## 14. Commit-readiness gate (2026-10-28, night, second review)

Baseline of this task: branch feat/generated-video-sprint, HEAD `5af239a`, 204 staged files (32,281 insertions, 450 deletions), no unstaged change; ledger SHA-256 unchanged (`0f60d7eb7e241d461f4ecf15d4be3717797fdaff5bdd1fa7144f907534ea52d4`).

Found by rendering the deliverables (not by a checker): the text inside the pipeline figure (paper Figure 1, report Figure 4) and one slide 6 note still read "scores are probabilities of fake", and a report paragraph under Table 1 still said the system itself states the "calibrated probability" sentence. Changed: scripts/make_figures.py (one text line), assets/figures/fig01_pipeline.png and .pdf regenerated with `scripts/make_figures.py --only fig01` (same size and layout, one line of text differs) and copied to paper/figures/fig01_pipeline.pdf, report/ch1_2.py (the frozen web page of the working version states the sentence), slides/build_slides.py (slide 6 note). The paper, report and slides were rebuilt. Whitespace: one blank line at EOF removed from docs/MODERN_GENERATOR_RESEARCH.md; the same fix to docs/DATA_LOG.md was reverted because docs/NUMBERS*.json hash that file (number provenance went stale) and regenerating them would be unrelated churn.

| Command | Result | Evidence |
|---|---|---|
| `cd paper; latexmk -g -pdf main.tex` | exit 0, 31 pages, 0 undefined references (same single overfull vbox as before) | `paper_build_summary.txt` (extract of main.log; the verbose log embeds local paths) |
| `report/build_report.py`; `report/layout_check.py` | exit 0, 119 pages; layout 26 of 26 | `report_build.txt`, `report_layout.txt` |
| `slides/build_slides.py`; `slides/check_slides.py` | 58 slides; 955 pass, 0 fail (slide 34 table height warning remains) | `slides_build.txt`, `slide_check.txt` |
| `check_docs.py --numbers` | 68 pass, 3 fail once (provenance stale after the DATA_LOG edit, since reverted); then 71 pass, 0 fail | `check_docs.txt` (the second run) |

Rendered and inspected: paper PDF pages 10, 12 (software paragraph, platform subsection), 20, 21 and the page with Table I of the staircase; report preview pages 3, 5, 41, 69, 70, 71, 88; slides 2, 6, 27, 34 and 35 exported through PowerPoint (read-only open). Slide 34: the table (2.68 in) ends above its card with a clear gap, no overlap or cut text; the 2.46 in figure is the builder's layout budget, so the warning is conservative. Not inspected: the other pages and slides; the stored slide preview PDF is older than the deck and was not regenerated (git-ignored). Not run: tests (no implementation file changed), opt-in GPU tests, the 13 data-dependent tests, the live browser demo.

## 15. Phase B of the deliverables (2026-10-29)

Branch trim/docs. Records commit eb8a4f9. Claim fixes A1 to A10 of deliverables_v2/PHASE_B_FIXLIST.md applied to the v2 paper, report and deck; length cuts B1 to B3 applied. No logged number, threshold, ledger byte or protected path changed.

| Command | Result |
|---|---|
| `cd deliverables_v2/research_paper; latexmk -g -pdf main.tex` | 16 PDF pages in the college build: 8 front matter, 8 from title to references (limit 9); 0 undefined references; abstract 250 words |
| `deliverables_v2/project_report/build_report.py`; `layout_check.py` | 37 PDF pages (28 body, Appendix A), 39 references all cited; layout 26 of 26 |
| `deliverables_v2/presentation/build_deck_v2.py`; `check_deck_v2.py` | 25 slides (14 main, divider, 10 backup); 428 pass, 0 fail |
| `deliverables_v2/check_v2.py --write deliverables_v2/check_v2_output.txt` | 0 superior sentences without the D-091 qualifier; author order OK in all three; 713 present; every 3-decimal number in the record except 0.4543 (speech temperature in checkpoint metadata, not a result) |
| originals: `build_numbers.py`, `check_docs.py --numbers`, `report/layout_check.py`, `slides/check_slides.py` | 71 pass 0 fail; 26 of 26; 955 pass 0 fail (the archive is unchanged) |

Viewed by eye: paper pages 1 to 8 of the paper proper (first, results, last pages, every table and figure); report cover, contents, chapters 3 and 6 to 8, references (agent also viewed the other page types); deck slides 7 and 14 by the main agent, all 25 by the deck agent. A sealed-results table of the paper that overflowed the right margin was fixed. Report text sits about 3 mm nearer the right border than the left because of the template margins (left 25.4 mm, right 19.8 mm, border 8.5 mm); not changed, open question for the user.

## 16. Final pass (2026-10-29)

Branch trim/docs. D-092 logged (answers of the user, margin decision, trace of 0.4543). Nothing merged.

| Command or action | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs -q` with the 13 data-dependent items deselected | 713 passed, 2 skipped, 13 deselected, exit 0 (`docs/test_evidence/2026-10-29_final/full_safe_suite_pre_merge.txt`) |
| `uv run ... deliverables_v2/project_report/layout_check.py` | 28 of 28 rows within 2 mm (26 rows as before, text-edge rows against the centred reference, one new row for equal gaps) |
| `latexmk -g -pdf main.tex` (college build) and a clean rebuild from submission/sources/paper | 16 pages, 0 undefined references, identical text |
| `deliverables_v2/presentation/check_deck_v2.py` | 428 pass, 0 fail |
| originals: `check_docs.py --numbers`, `report/layout_check.py`, `slides/check_slides.py` | 71 pass 0 fail; 26 of 26; 955 pass 0 fail |
| `deliverables_v2/check_v2.py` | 0 superior sentences without the qualifier; every 3-decimal number of the paper, report, deck and notes is in the record (0.4543 traced in D-092) |
| Website on port 8012, real checkpoint (image, video with sound, audio) | 3 of 3 completed; labels as in D-087 and its addendum; no MOCK banner on live results; the face-video flag label was not exercised because no input flagged; collapsed technical panels hold decision and experiment ids |
| Render review | paper 16 pages, report 37 pages and deck 25 slides rendered and looked at by the main agent; problems found and fixed: right-shifted front matter (report and paper), missing space after the abstract label, pipeline figure too small, one table wider than the page (paper); none left |

Left unchanged: the report footer rule sits 1.6 mm closer to the right border than to the left (it must stay within 2 mm of the template), the screenshots of the website stay out of git.

Merge into main (user answer 2026-10-29: merge as is): `git merge --no-ff trim/docs` on main (cf84b99), merge commit db38f98; the tree of main equals the tree of trim/docs, so the archive files (paper/, report/, slides/, src/dfd/platform/dashboard/) are exactly as on trim/docs (they differ from the older main by the commits 0704b3a, 5af239a and f417011 of feat/generated-video-sprint, which the merge brings in). On main: safe suite 713 passed, 2 skipped, 13 deselected (`docs/test_evidence/2026-10-29_final/full_safe_suite_on_main.txt`); `check_docs.py --numbers` showed 68 pass, 3 fail (provenance stale, because the checkout rewrote line endings of EXPERIMENTS.md and DATA_LOG.md), 71 pass, 0 fail after `build_numbers.py` (one hash line of NUMBERS.json committed); report layout 26 of 26; slides 955 pass; v2 report layout 28 of 28; deck 428 pass; check_v2: 0 qualifier gaps, 0 numbers missing from the record. Tag `submission-v2`.

## 17. Final clean-up after the audit (2026-10-29)

Branch main (from 7a61675), D-093. No experiment run, no number, threshold, ledger byte, sealed file, checkpoint or file under src/ changed. Logs in `docs/test_evidence/2026-10-29_cleanup/`.

| Command or action | Result |
|---|---|
| Safe suite, as in section 2 (13 data-dependent items deselected) | 713 passed, 2 skipped (opt-in GPU tests), 13 deselected, exit 0 (`full_safe_suite_cleanup.txt`) |
| `tests/platform_genvideo/test_generated_video.py -k real_dinov2` on this machine (weight record present) | 1 passed: the real DINOv2-small path still runs; the D-093 change only adds a skip when data/_logs/encoder_weights.json is absent |
| Fresh clone of origin/main (7a61675) with the changed test file copied in, the project .venv, `pytest tests` with nothing deselected | 695 passed, 33 skipped, 0 failed (before the change the same clone gave 1 failure in that test); the skips are tests of ignored local data, weights and result files, each with its reason |
| `build_numbers.py`, then `check_docs.py --numbers` (Git Bash, xpdf pdftotext 4.00) | 71 pass, 0 fail, exit 0 (`check_docs_numbers.txt`); 0.9731 now also traces to the E-028 addendum |
| The same check with the MiKTeX poppler pdftotext 24.04 first on the PATH (PowerShell) | 70 pass, 1 fail: one en dash in the archive paper PDF (paper/main.pdf, the page range 4828 to 4832 of one reference, typeset with an en dash), which xpdf 4.00 drops in its default Latin-1 output (`check_docs_numbers_poppler24.txt`). The archive is not submitted and is unchanged (D-090); the three submitted PDFs and both v2 PDFs have 0 em or en dashes with poppler 24.04 |
| `deliverables_v2/check_v2.py` (Git Bash) | output identical to the tracked deliverables_v2/check_v2_output.txt: 0 sentences without the D-091 qualifier, 0 numbers missing from the record (`check_v2.txt`). Under poppler 24.04 the column order differs and one cut sentence was flagged; the full sentence in main.tex line 365 carries the Bonferroni pointer (`check_v2_poppler24.txt`) |
| `deliverables_v2/presentation/check_deck_v2.py` | 428 pass, 0 fail (`check_deck_v2.txt`) |
| `deliverables_v2/project_report/layout_check.py` | 28 of 28 rows within 2 mm (`report_v2_layout_check.txt`) |
| Paper rebuilt from submission/sources/paper (latexmk twice, scratch folder) | 16 pages, 0 undefined references, pdftotext output identical to the submitted paper PDF; no paper source or PDF changed |
| Word comments removed from submission/sources/report/*.docx only (36 comment ranges, 36 reference runs, 3 comment parts with their relationships and content types; every other zip part byte-identical) | 1,022 paragraphs and 1,247 text runs identical, 6 images, 24 tables, 4 sections; Word (COM, read-only): comments 36 then 0, pages 37 and 37, words 10,479 and 10,479; Word PDF export: per-page text identical on all 37 pages; pixel differences of anti-aliasing only on pages 23 (about 1,000 pixels) and 28 (22 pixels), no change of line breaks, smaller than the differences between the tracked report PDF and a fresh export of the unchanged DOCX (4 pages) (`docx_comment_removal.txt`). The working DOCX and both report PDFs are unchanged |
| Ledger | git blob 32ae3585b9c1accf4f510f65f9669f3ca6e4a069 unchanged; worktree SHA-256 47e166a7...b006 before and after; rebuilt from the blob, lines 1 to 75 CRLF and line 76 LF give 0f60d7eb...52d4, the value cited in sections 12 to 14 (docs/STATUS.md, Mistakes and corrections, 2026-10-29) |
| Protected and sealed files | 339 files under data/_final, src/ and checkpoints/ plus the ledger hashed before the switch to main and after the work: identical |

Not run: the opt-in GPU tests, the original dashboard and the v2 website in a browser (no code they serve changed), a rebuild of the report or the deck.
## 18. Post-freeze evaluations and Evaluation Lab groups (2026-10-31, D-094)

Logs in docs/test_evidence/2026-10-31_posteval/. Environment ENV-DESKTOP-GPU (Windows 10 19045, Python 3.13.15, torch 2.14.1+cu130, RTX 5070 12,227 MiB).

| Check | Command | Result | Log |
|---|---|---|---|
| Safe suite (CPU, 13 data-dependent items deselected as in section 2) | `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs` with the deselect list | 737 passed, 2 skipped (the two opt-in GPU tests), 13 deselected, exit 0 (713 before; +16 group tests, +8 v2 website lab tests) | full_safe_suite.txt |
| Lab and dashboard tests (after the last front-end edit: 34 of test_ui_v2.py re-run, 34 passed) | `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/platform_dashboard tests/platform_evallab -q -p no:cacheprovider -rs` | 160 passed | lab_and_ui_tests.txt |
| Live browser check of the redesigned lab on the real E-033 and E-034 score files (Edge through Playwright, server deliverables_v2/website/serve_v2.py on port 8013, mock mode for analysis) | scratchpad script, results copied here | 34 of 34 checks passed at 1,366 px and 390 px: dropdown lists the three datasets; no POST on open or on selection; advanced section collapsed; no horizontal scroll; the whole pool is not a group card and needs a second click; the headline AUROC equals results.json for T-V G2 (0.3715), T-I G5 (0.5693) and T-A FULL (0.6458); experimental banner and threshold provenance on every result; no "overall score" or "approved for" text | lab_live_check.txt, screens/ |
| Protected artefacts unchanged | SHA-256 of every file under ledger/, data/_final, checkpoints/, data/_results/modern/e028, data/_results/e015, data/cf_eval/_meta at 01:32:13 and at 02:20:02 (+0530) | 332 files, identical (diff empty); ledger 76 entries; APPROVED_HEADS = {}; no tracked file under src/dfd/infer, src/dfd/web, ledger, checkpoints, paper, report, slides changed | protected_diff.txt; data/_results/posteval/protected_before.sha256 and protected_after.sha256 |
| Preregistration hashes | re-derived from docs/EXPERIMENTS.md after the results were appended | E-033 5a6d0077...743f and E-034 b8817c16...054b both re-derive | (inline) |
| Reproducibility of stored predictions | score_e033.py reproduce (20 clips), score_e034.py reproduce (50 images) | largest absolute difference 0.0 for both (tolerance 0.02) | data/_results/posteval/E-03x/reproducibility.json |
| Traceability of the comparison document | script comparing every E-033 and E-034 interval string in docs/POST_FREEZE_BENCHMARK_COMPARISON.md with results.json; derived F1 recomputed from the stored rates | 19 of 19 found; 6 derived F1 values equal | (inline) |

Not run: the opt-in GPU tests (RUN_PARITY, RUN_LIVE_FACE; they rewrite data/_results/platform), scripts/check_docs.py (the deliverables were not changed), the 13 data-dependent tests.
## 19. Continuation audit of E-033, E-034 and the Evaluation Lab (2026-10-31)

Read-only verification of the previous session's claims; nothing rescored or retrained. Logs in docs/test_evidence/2026-10-31_continuation/.

| Check | Result | Log |
|---|---|---|
| Independent recomputation of E-033 and E-034 (scikit-learn AUROC, own threshold metrics and own group bootstrap, scratchpad script) | 88 of 88 checks passed. Preregistration hashes re-derive and each RESULT follows its end line. Checkpoint hashes and thresholds match. Pool, exclusions, labels, groups and seeded memberships reconcile with the source metadata. All 17 group results equal results.json to 1e-9. Bootstrap intervals agree within 0.007. A follow-up check (11 of 11) covered the E-034 group rules, exclusion of the duplicates, single scoring passes and the secondary analyses | (inline; script in the session scratchpad) |
| Evaluation Lab over HTTP on the real score files (server serve_v2.py, port 8013, stopped afterwards) | 98 of 98 checks passed. All 17 groups (five per dataset, plus FULL for the two LAV-DF heads) reproduce every stored metric and interval exactly, use the checkpoint threshold (source fixed) and read the descriptor's score file. Listed counts equal the stored results. Refusals: FULL without confirmation, a client threshold, an unknown group or dataset, FULL on CF-Eval | lab_api_check.txt |
| Live browser check (Edge through Playwright, 1,366 and 390 px) | 34 of 34 checks passed (same checks as section 18) | lab_live_check.txt, screens/ |
| Safe suite, run fresh | 737 passed, 2 skipped (opt-in GPU tests), 13 deselected, exit 0. Six "moov atom not found" stderr lines after the summary come from OpenCV's FFmpeg reader on the junk .mp4 files that tests/prep writes on purpose (unchanged since 2026-10-24); no test failed | full_safe_suite.txt |
| Protected artefacts | 332 files identical to the baseline data/_results/posteval/protected_before.sha256 (its own SHA-256 934a31ae...b0ee equals the value recorded in section 18); ledger 76 entries, no git diff; APPROVED_HEADS = {}; FORBIDDEN_REPORT_KEYS unchanged; no diff in src/dfd/platform/schema.py, verdict.py, modules/, src/dfd/infer, src/dfd/web | protected_now.sha256, protected_diff.txt (empty) |
| Literature trace | every published number of docs/POST_FREEZE_BENCHMARK_COMPARISON.md occurs in docs/LITERATURE_BENCHMARK_EVIDENCE.md; all 13 cited papers are registered (GenD added as R-088 after a Crossref check) | (inline) |

Not run: the opt-in GPU tests; scripts/check_docs.py (the deliverables were not changed); the 13 data-dependent tests.

## 20. Recommended demonstrations in the Evaluation Lab (2026-10-31, D-095)

| Check | Result | Log |
|---|---|---|
| Backend tests tests/platform_evallab (new test_evallab_showcase.py: per-seed derivations against hand computation, read-only endpoint leaves the tree unchanged and creates no run, every displayed AUROC and count equals the logged file) | 77 passed | (inline) |
| Browser tests tests/platform_dashboard (mocked API; 7 new tests: cards, counts and kind labels before the dataset list, modality filter, stored-result detail with derived labels, "Open in Lab evaluation" preselects without a POST, load error does not block the guided flow, no horizontal scroll at 390 and 1,280 px; dropdown labels now carry the modality) | 94 passed together with the evallab tests (171) | (inline) |
| Live browser check against the real API (serve_v2.py mock mode, port 8016; Chromium through Playwright, 1,366 and 390 px) | all 14 entries show the logged AUROC and real, fake and total counts; LAV-DF T-A G4 preselected by the button; 0 POST requests; horizontal overflow 0 at both widths | deliverables_v2/website/test_evidence/2026-10-31_reco/ |
| Safe suite, run fresh (13 data-dependent tests deselected as in section 2) | 748 passed, 2 skipped, 13 deselected, exit 0 (737 before plus the 11 new tests) | (inline) |
| Protected artefacts | 332 of 332 files identical to data/_results/posteval/protected_before.sha256 | (inline) |

## 21. Sealed re-analysis groups and the demo set (2026-10-31, D-096)

- Unit and API tests: tests/platform_evallab and tests/platform_dashboard, 175 passed. New: a sealed_reanalysis group runs as final_reanalysis with val_fit thresholds and ledger entries; the same descriptor without a ledger entry is refused (FINAL_GUARD); a sealed descriptor without the declaration is still refused; two datasets in one score file are counted separately (regression for the cache bug in STATUS.md).
- Reproduction check (real data, synchronous lab): G4 of In-the-Wild, Celeb-DF v2 and CNN gave AUROC 0.9148, 0.8999, 0.9316, balanced accuracy 0.8422, 0.8037, 0.8102, recall 0.8560, 0.7172, 0.6562, precision 0.7272, 0.9727, 0.9514, equal to data/_final/eval/I-4.json to four decimals. Run times: In-the-Wild G4 115 s, CNN G4 103 s, Celeb-DF G4 4 s, small groups 1 to 2 s after the file is cached.
- Browser check against the real server (mock mode, real Lab API), deliverables_v2/website/test_evidence/2026-10-31_sealed_groups/: all six datasets listed with modality prefixes; the Celeb-DF card preselects G4; group counts shown as in the groups file; Celeb-DF G1 run finished with the sealed re-analysis banner; horizontal overflow 0 at 1,366 and 390 px.
- Demo set: 23 files written, none skipped; every pool resolved through select_records(final=False).
- Full suite: `pytest -q` 765 passed, 2 skipped (this run did not deselect the 13 data-dependent tests). Afterwards, 332 of 332 files listed in data/_results/posteval/protected_before.sha256 were still identical; no diff in src/dfd/platform/verdict.py, schema.py, modules/, src/dfd/infer, src/dfd/web or ledger/.

## 22. Simple report view (2026-10-31, D-097)

- tests/platform_dashboard: 94 passed.
- Live browser check (server --live, Chromium, 1,366 and 390 px), deliverables_v2/website/test_evidence/2026-10-31_simple_view/: uploads of demo files 21 (LAV-DF video edit), 02 (FF++ Deepfakes) and 16 (ProGAN image). Each report had one section with the cards face_video, speech, media_metadata, generated_video in that order; the headline and the outcome chart counted these four only; horizontal overflow 0 at both widths.

## 23. Analysed file kept visible in the report (2026-10-31, D-098)

- tests/platform_dashboard: 94 passed.
- Live browser check (server --live, Chromium, 1,366 and 390 px), deliverables_v2/website/test_evidence/2026-10-31_report_preview/: demo files 02 (video), 09 (FLAC audio), 16 (PNG image). In all 6 cases the report showed the file (video and audio with controls, media loaded, 420 px wide on desktop, full width on phone), horizontal overflow 0; after "Analyse another file" no preview element remained in the report.
