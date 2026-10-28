# HANDOVER: Phase B of the deliverables work (paused 2026-10-28, about 21:20 IST, session limit)

This file is a ready-to-paste PROMPT for a new Claude Code session, followed by the facts it needs. Paste everything from "PROMPT START" to "PROMPT END". Preferred model: Sonnet 5.5, high effort (the user asked for this on 2026-10-28; use the same for any subagent).

---------------------------------------------------------------- PROMPT START

You are the implementer of a final-year project (multimodal deepfake detection: images, video, audio) in `D:\Final Year Project`, continuing work that a previous session paused when it hit its usage limit. Work under `CLAUDE.md` (read it first: hard rules, no em dashes or en dashes anywhere, real citations only, no unmeasured numbers, decisions logged), `docs/DOC_FACTS.md` and `docs/DECISIONS.md`. Then read `docs/STATUS.md` (NEXT STEP), `deliverables_v2/SPEC.md`, `deliverables_v2/PHASE_B_FIXLIST.md` (the authoritative list of claim fixes and cuts, with exact wording), `deliverables_v2/CONTENT_MAP.md` (verified facts, with a Phase B addendum at the end) and `deliverables_v2/audit/` (the hostile review: paper_audit.md, report_audit.md, slides_audit.md, claims_stress.md). Finish Phase B below. Do NOT merge into main. Stop after pushing `trim/docs` and reporting.

### 1. Where things stand (verify with git before trusting this)

- Branch `trim/docs`, created from commit `f417011` ("feat(platform): verdict layer, Evaluation Lab, ...", 214 files), which is pushed to `origin/feat/generated-video-sprint`. `trim/docs` itself is NOT pushed yet and has NO commits of its own yet.
- Everything below is uncommitted in the working tree of `trim/docs`: the Step 2 records, the whole `deliverables_v2/` tree (untracked) and the local test evidence.
- The ORIGINAL paper, report and deck (`paper/`, `report/`, `slides/`) are an unchanged archive (D-090). Do not edit them. Protected: `src/` (all), `ledger/` (SHA-256 must stay `0f60d7eb7e241d461f4ecf15d4be3717797fdaff5bdd1fa7144f907534ea52d4`), `data/`, `checkpoints/`, `scripts/`, `tests/`. `APPROVED_HEADS = {}`, `FORBIDDEN_REPORT_KEYS` unchanged, lip-sync band (-6, +2), no Gemini work.
- Safe suite measured 2026-10-28 evening: 713 passed, 2 skipped (opt-in GPU tests), 13 deselected (data-dependent tests); log `docs/test_evidence/2026-10-28_phase_b/full_safe_suite.txt`.

### 2. DONE (do not redo)

1. Git: the 214 staged files committed (`f417011`) and pushed; `trim/docs` created.
2. Records (working tree, uncommitted, all verified):
   - `docs/EXPERIMENTS.md`: E-032, the post hoc false-positive rates of the frozen video threshold (Celeb-DF family 0.1098 on 744 real videos, DFD 0.1405 on 363), "post hoc, re-analysis of sealed scores already in the ledger, not confirmatory". The script `scripts/eval/benchmark_audit_2026-10-27/frozen_rates.py` was replayed read-only; every number reproduces (the I-4 score file hash `e0c78a8c...` matches docs/PERFORMANCE_BENCHMARK.md section 12.4). Disclosed in E-032: section 12.4 had input hashes but no output hash, and the stored `frozen_rates_output.txt` has every line cut at 170 characters (only the run-id list is lost). Complete replay: `docs/test_evidence/2026-10-28_phase_b/frozen_rates_I4_replay.txt` (SHA-256 `46e012d6e207b8ce869fc8b1c7bc483e8e4d7d1e899b1d5ebb68120b505d2e11`).
   - `docs/DECISIONS.md`: D-089 (deliverables v2, author order Shivansh Gupta, Vaibhav Shiroorkar, Adithya Jithesh, no subtitle, maroon, honest pie chart), D-090 (the submission set is deliverables_v2/, originals are an unchanged archive), D-091 (T3 wording, supersedes D-068; a status line was added to D-068).
   - `docs/PROTOCOL.md` addendum and `docs/DOC_FACTS.md` section 7 correction: three confirmatory tests plus the 98.33 percent line, from D-062 (b) committed 2026-10-25 18:44, before the first sealed read (20:16) and the sealed run (ledger committed 22:16). Old text kept.
   - `docs/VIVA_PREP.md` (new section "added 2026-10-28 (night)"; the sentence on the 0.11 and 0.14 now cites E-032), `docs/PERFORMANCE_BENCHMARK.md` (correction note under section 12.4), `docs/NUMBERS.json` (rebuilt by `scripts/build_numbers.py`).
   - `scripts/check_docs.py --numbers` gave 71 pass, 0 fail after the rebuild. It checks the ORIGINAL deliverables, which are unchanged. RE-RUN `build_numbers.py` and `check_docs.py` after any further edit to EXPERIMENTS, DECISIONS, PERFORMANCE_BENCHMARK, DATA_LOG or RESEARCH (it goes stale: 68 pass, 3 fail until rebuilt).
3. Phase A hostile review (analysis only, finished): `deliverables_v2/audit/` (four files plus `occ_classified.tsv`). Findings that drive the fix list: T3 "superior" must carry the Bonferroni qualifier; speech gain is partly a removed silence/level shortcut; FakeAVCeleb "ranks well" is wrong (codec 0.9985); AASIST and DF26 are context only; unlogged numbers (0.11, 0.14 now E-032; "1.732 s per 1,000 inputs" and "0.999983" must be removed); stale test counts.
4. The v2 deliverables exist in a PRE-Phase-B state (built 2026-10-28 before the review), all independently verified earlier:
   - `deliverables_v2/research_paper/` (main.tex, main.pdf): 21 PDF pages, 13 pages of content, 0 undefined refs, 42 cited keys all in refs.bib.
   - `deliverables_v2/project_report/` (build_report.py, chapters.py, content.py, front.py, DOCX and PDF): 35 PDF pages, layout check 26 of 26, `622423` brown absent, maroon #B7202F used, old subtitle absent.
   - `deliverables_v2/presentation/` (build_deck_v2.py, check_deck_v2.py, the .pptx): 14 slides, checker 250 pass 0 fail, slide 10 uses screenshots of the redesigned website.
   - `deliverables_v2/website/`: redesigned cream/ivory/beige/warm-brown dashboard (serve_v2.py, static/), isolated over the unchanged backend; its Playwright check reported 0 problems; the automatic dark theme was REMOVED on 2026-10-28 so the cream theme always applies; README.md has the design tokens and contrast table. Start: `PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live`, open http://127.0.0.1:8012/ (and http://127.0.0.1:8012/#lab). The server is STOPPED now. NOTE: http://127.0.0.1:8002 is the ORIGINAL dashboard (dark green in OS dark mode) and is deliberately unchanged; the redesign is on 8012. See open question 1.
   - `deliverables_v2/figures/` shared figures (platform figure with eight modules and a colour key, dashboard crops), `deliverables_v2/check_v2.py` (the consistency checker, see section 5).

### 3. IN PROGRESS when the session stopped (inspect before continuing; nothing is trustworthy until rebuilt)

Three Sonnet subagents were launched in parallel (one per document, each told to edit only its own folder) and were stopped mid-work:
- PAPER: `main.tex` was NOT yet modified in Phase B (last edit 20:54; no D-091 wording, no "224 px" qualifier, still 21 pages). Start this workstream from scratch.
- REPORT: `deliverables_v2/project_report/chapters.py` WAS partly edited (21:16; it now contains the D-091 wording "does not survive", "224 px", "713 passed", an E-032 comment and the speech-shortcut sentence) but was NOT rebuilt (the PDF is the stale 21:00 build) and `front.py` (abstract) has NOT been fixed yet. Re-read chapters.py against PHASE_B_FIXLIST.md section A item by item, finish the rest, then rebuild.
- DECK: `build_deck_v2.py` is UNCHANGED (14 slides, pre-Phase B). Start this workstream from scratch.

### 4. PENDING WORK, in this order (fix claims BEFORE cutting length)

Step 3, claim fixes in all v2 documents (paper, report, deck and speaker notes, website text if it states results; the website text was checked and states no result): follow `PHASE_B_FIXLIST.md` section A exactly (A1 T3 wording; A2 speech shortcut sentence; A3 fine-tuning headline 0.7396 to 0.9096 [paired +0.1700 [+0.1228, +0.2114]], exploratory, or out of the abstract; A4 LAV-DF shortcut instead of 0.99 in abstracts and conclusions; A5 FakeAVCeleb without "well", codec 0.9985, thresholds do not transfer; A6 AASIST and published numbers context only, no "5.81 points worse", no causal sentence; A7 DF26 class C, fix "used for nothing else"; A8 false-positive rates only in results and limitations, pointing to E-032, remove "1.732 s per 1,000 inputs" and "0.999983"; A9 test counts 713/2/13, remove 497, 304, 710; A10 the other rewordings and the wording audit: every "improved", "robust", "confirmed", "generalise" maps to a declared rule and interval or goes).

Step 4, length cuts (PHASE_B_FIXLIST.md section B):
- Paper: at most 9 PDF pages from the title/abstract page to the end of the references (8 pages plus references; now 13). Use the 248-word abstract of `audit/paper_audit.md` section 5, updated for the fixes (at most 250 words, no development-slice number). Software and platform become one sentence pointing to the report; cost record, encoder table and figures that repeat tables go; post-freeze work is one paragraph (keep the short Gemini sentence group: no Gemini data or experiment exists, nearest proxy Imagen 3 inside E-028 and E-029, not a capability); published comparison and Celeb-DF++ arithmetic a few sentences each. Keep the CORE items: protocol, encoder rule, ladder, sealed staircase, T1 to T3, LAV-DF negative result, rPPG negative result, DF26 at chance, thresholds not transferring, main limitations. If 9 pages cannot be reached without removing a CORE item, STOP and report which.
- Report: body near its present size (35 PDF pages now; keep at most 40; template range 30 to 60). Check every table and figure is referenced in the text and supports a claim. If v2 dropped evidence that `audit/report_audit.md` marks "appendix" AND a question in `docs/VIVA_PREP.md` needs it, restore it as a short appendix with the VIVA_PREP question named in a Word comment; otherwise it stays out. Trim the 39 references only if no text cites the removed ones; otherwise keep them and say so.
- Deck: 14 MAIN slides and at most 10 BACKUP slides after a divider; one message per main slide, speaker notes on each. Main titles decided in PHASE_B_FIXLIST.md B3 (autonomous, needs user review): 1 Title and authors; 2 Problem, scope and objectives; 3 Protocol and data; 4 Method: file to score; 5 Working version and what we learned; 6 Iteration ladder; 7 Sealed tests T1 to T3; 8 Speech: the T2 gain and its shortcut; 9 Image analysis; 10 Platform and what the tests after the freeze allow it to claim; 11 Dashboard (live demo screenshots from the redesigned site); 12 Gemini-generated images; 13 Published comparison (context only) and limitations; 14 Conclusion and demo summary. Backup titles: see B3 and `audit/slides_audit.md` section 5.

Step 5, rebuild and check (the originals' checkers still apply to the unchanged originals and must still pass; run the v2 equivalents too):
- Paper: `cd deliverables_v2/research_paper && latexmk -g -pdf -interaction=nonstopmode main.tex` (0 undefined references).
- Report: `uv run --no-project --with python-docx --with lxml python deliverables_v2/project_report/build_report.py`, then `uv run --no-project --with numpy --with pillow python deliverables_v2/project_report/layout_check.py` (26 of 26). Word exports the PDF.
- Deck: `python deliverables_v2/presentation/build_deck_v2.py`, `python deliverables_v2/presentation/check_deck_v2.py` (update its slide-count rule to 14 main + divider + at most 10 backup). Render slides through PowerPoint COM, read-only (see `_render.ps1`).
- Originals: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/build_numbers.py` then `... scripts/check_docs.py --numbers` (71 pass, 0 fail), `.venv/Scripts/python.exe report/layout_check.py`, `.venv/Scripts/python.exe slides/check_slides.py` (these three check the archive; all must pass).
- Consistency: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe deliverables_v2/check_v2.py --write deliverables_v2/check_v2_output.txt` (greps for superior, robust, confirmed, ranks well, 0.99, 497, 304, 710, calibrated probability, Likely, dashes, 1.732, 0.999983; checks that every "superior" sentence has the D-091 qualifier, the author order, "713", and that every decimal with 3 or more places is in the record). The abstract, conclusion and title/closing slide of all three documents must state the same claims with the same numbers and qualifiers.
- Look by eye (render and open the images; do not claim what you did not view): the paper's first page, results page and last page; the report cover, results chapter, conclusion and any appendix; deck slide 1, the sealed-tests slide and the conclusion.
- Update `docs/STATUS.md` (NEXT STEP, what was done) and `docs/FILES.md` (deliverables_v2/ and its subfolders, docs/test_evidence/2026-10-28_phase_b/), add a short section to `docs/TESTING_RECORD.md`, then commit on `trim/docs` and push (`git push -u origin trim/docs`). Suggested commits: (1) records only (E-032, D-090, D-091, PROTOCOL, DOC_FACTS, VIVA_PREP, PERFORMANCE_BENCHMARK, NUMBERS, STATUS, FILES, phase_b evidence), commit this FIRST and early so the work survives another limit; (2) `deliverables_v2/` (exclude `_render`, `_charts`, `__pycache__`, LaTeX aux files, and any screenshots of dataset media; check `git status` and stage by explicit path; no `git add .` or `-A`). Message format `type(scope): summary`, end with the Co-Authored-By line given in the session's attribution reminder. Never merge into main; never force-push; no `git reset`, `clean`, `stash` or `checkout --`.

### 5. STOP CONDITIONS (stop and report instead of continuing)

- A fix would change a logged number.
- The false-positive script output no longer reproduces (it did on 2026-10-28; E-032 has the hashes).
- The paper cannot reach 9 pages including references without removing a CORE item.
- A checker fails after two fix attempts.
- You find another number in a v2 document with no E-entry. Already judged and NOT a stop: `0.4543` is the speech temperature stored in `checkpoints/I-5-freeze-candidate/metadata.json` (0.45432135...), a calibration parameter and not a result; it appears in the frozen-system table of the paper and report (the paper cut will probably drop that table; otherwise cite the checkpoint metadata). The speech EER percent values (6.94 [5.94, 7.79]; 15.78 [12.75, 18.70]) are the freeze-report fractions x 100.

### 6. OPEN QUESTIONS FOR THE USER (ask, do not decide silently; tag autonomous choices "decided autonomously - needs user review")

1. The user saw the ORIGINAL dashboard at http://127.0.0.1:8002 in dark green and asked for cream, ivory, beige and brown. The redesign lives on port 8012 and now always renders cream. Does the user want the redesign served on 8002 as well (that would touch `src/dfd/platform/dashboard/`, a protected path), or is 8012 enough?
2. The 14-slide main list and the backup list are the implementer's choice (needs review). The user's brief wanted "platform and after the freeze on one slide", "published comparison with limitations on one slide", "maths into method or backup".
3. Gemini: the earlier brief required a concise Gemini account in paper, report and deck. The 9-page paper cut keeps only a short sentence group inside the post-freeze paragraph. Confirm this is acceptable.
4. Per-module verdict labels (D-087) and D-084 to D-086, D-088, D-089 to D-091 implementation details still await the user's review.
5. Unresolved licensing (blocks distributing, not committing source): E-029 heads (Community Forensics CC BY-NC-SA 4.0 data), SyncNet weights, FakeAVCeleb terms.

### 7. Practical lessons from the previous session (save yourself time)

- Do not pass long Python through a bash heredoc that contains quotes or `\n`: it failed once at parse time and once turned `\n` into real newlines. Write the script with the Write tool, then run it.
- The docs have mixed line endings (DOC_FACTS, STATUS, FILES, PERFORMANCE_BENCHMARK, VIVA_PREP are CRLF; EXPERIMENTS, DECISIONS, PROTOCOL LF). Edit by bytes and keep each file's ending, or use the Edit tool.
- `scripts/check_docs.py` and the original checkers read `paper/`, `report/`, `slides/` (the archive), not deliverables_v2/.
- PowerPoint (version 16) is installed: open read-only through COM from PowerShell and `Slides.Item(n).Export(path, 'PNG', 1280, 720)`. LibreOffice is not installed. The report PDF is exported by Word through `word_tools.ps1`.
- Subagents: they cannot always write files into the repository; spot-check their output (D-015) before merging, and if one reports by message, write the file yourself. Give each a single folder. Subagent runs hit the session limit together; commit records early.
- Run the safe suite only with the GPU hidden: `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests -p no:cacheprovider -rs -q --deselect ...` (the 13 deselected ids are listed in docs/TESTING_RECORD.md section 2). Never run RUN_PARITY or RUN_LIVE_FACE tests, `live_*_e2e.py`, or anything that writes under `data/_results`, `data/_final` or `ledger/`.
everything int he report and the research paper look right shifted check alignment 
### 8. Final report format (when done)

1. Commits with hashes (records commit, deliverables commit) and the decision and E-entry ids logged (D-089, D-090, D-091, E-032).
2. Before/after table: pages, words, figures, tables, main and backup slides per v2 document.
3. Every claim fix with its location in each document.
4. The final abstract with its word count and the 14 main slide titles.
5. Checker results and exactly what you viewed by eye.
6. The consistency grep results.
7. Completed and remaining project steps; open questions of section 6.

---------------------------------------------------------------- PROMPT END
