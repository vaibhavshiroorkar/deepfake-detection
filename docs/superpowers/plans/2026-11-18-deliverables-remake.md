# Deliverables Remake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the paper, report and PPT around one idea, what survives the upload, in a standard order, with one figure style and prose that reads as written by a person.

**Architecture:** The sources stay where they are (`deliverables/paper/paper.tex`, `deliverables/report/report.tex`, `deliverables/slides/build_deck.py`, `scripts/make_figures.py`). Every number stays a `{{key}}` placeholder filled from `results/numbers.json`. Each task rewrites one deliverable and ends with its build and checks passing.

**Tech Stack:** LaTeX (IEEEtran; the college report class), python-pptx on the college slide master, matplotlib.

**Spec:** `docs/superpowers/specs/2026-11-18-deliverables-remake.md`

## Global Constraints

- No em or en dashes, no decorative Unicode, no emoji (`~/.claude/CLAUDE.md`).
- Plain English: common words, one idea per sentence, active voice, point first; no "robust", "crucial", "leverage", "delve", "furthermore", "notably", no "not only X but Y".
- Every number is a `{{key}}` placeholder; `overleaf.py` and `build_deck.py` must report 0 missing.
- No result, claim or limitation changes; only order and wording.
- H1 to H4 keep their names. Amendment numbers appear only in the appendix table.
- Paper abstract at most 250 words. Paper title: *What Survives the Upload? Evaluating Audio-Visual Deepfake Detectors After Removing File Shortcuts*.
- Report and PPT title unchanged (the registered project title).
- Figure colours: face `#2a78d6`, voice `#eb6834`, lip sync `#1baf7a` (validated with the dataviz script, 11-18).

## Review Focus

- A claim lost in the move: every result in today's paper (H1 to H4, cross-corpus, per method, per generator, A11 to A19, limitations) must still be in the rewrite. Check with a list taken from today's section headings before deleting anything.
- A placeholder key mistyped while moving text: caught by `overleaf.py` reporting a missing number.
- A `\ref` to a label that moved or was renamed: caught by the reference check.
- Text overflowing a figure box or slide box: check the rendered PNG of every figure and slide, not only the layout warnings.
- Overleaf compile errors from LaTeX constructs that `overleaf.py` cannot see (an unbalanced brace, a table column count): check brace balance and column counts; Vaibhav's Overleaf compile is the final check.

## Verification commands (used by every task)

```bash
.venv/Scripts/python.exe scripts/build_numbers.py
.venv/Scripts/python.exe scripts/make_figures.py
.venv/Scripts/python.exe deliverables/overleaf.py        # expect: 0 missing, for both
.venv/Scripts/python.exe deliverables/slides/build_deck.py  # expect: 13 slides, 0 numbers pending, no layout lines
python check_refs.py deliverables/build/overleaf/paper/main.tex   # expect: missing targets: none, braces balanced
git diff -U0 | grep "^+" | grep -cP "[\x{2013}\x{2014}]"           # expect: 0
```

---

### Task 1: One figure style, and the stage figure redrawn

**Files:** Modify `scripts/make_figures.py` (all seven figure functions and the module docstring).

- [ ] Redraw `pipeline()` so no text leaves its box: stage boxes wide enough for their two lines at 8.5 pt and 6.8 pt; the file-audit box spans its sentence; the answer box holds three short lines. Five numbered stages: Input, Clean, Check (face, voice, lip sync stacked), Compare, Answer; the file audit above, fed by the raw and the cleaned file.
- [ ] Relabel the result figures in plain English: `census` y axis "File-clue score (AUC)", legend "Original files" and "After cleaning"; `cross_corpus` y axis "Lip-sync check (AUC)"; `sync_methods`, `h1`, `h4`, `h2` the same way. Keep values and colours.
- [ ] Run `make_figures.py`; open every PNG; nothing overflows, labels do not collide.
- [ ] Commit: "Figures: one style; the stage figure redrawn".

### Task 2: The paper, rewritten in the standard order

**Files:** Modify `deliverables/paper/paper.tex`.

- [ ] Before editing, write the list of every result and limitation in today's paper (from its headings and tables) into a scratch checklist.
- [ ] New title and abstract (at most 250 words), led by the upload idea.
- [ ] Introduction: open with the codec example (real FakeAVCeleb clips are H.264, its cloned voices MP3, its face swaps MPEG-4; a model that reads only that scores 0.994); state the upload problem; the contribution list, each item ending with a forward reference (`Section~\ref{...}`).
- [ ] Method by stage, with Fig.~\ref{fig:pipeline}: file audit; cleaning (and why the first version failed: AAC priming); face and voice checks; lip-sync check; comparing with real clips (validation threshold, real-only threshold, fusion).
- [ ] Experimental setup: data and locked targets, splits by person, seeds and intervals, measures, the four preregistered tests H1 to H4 in one table, one sentence pointing to the appendix for dated changes.
- [ ] Results in the order of the spec, every table and figure kept; exploratory checks named by what they do.
- [ ] Related work moved after Results, trimmed to what separates us from each work.
- [ ] Discussion and limitations; Conclusion; Appendix table of the dated changes (ID, date, what changed, why, before which result).
- [ ] Tick every item of the scratch checklist against the rewrite.
- [ ] Run the verification commands; count abstract words; commit: "Paper: rewritten around what survives the upload".

### Task 3: The report, in the same order and voice

**Files:** Modify `deliverables/report/report.tex`.

- [ ] Same checklist step as Task 2, from today's report.
- [ ] Abstract, Introduction (objectives and contributions reframed), Literature review (trimmed), Project history (kept; tightened).
- [ ] Merge today's "Data and the Shortcut Audit", "Shortcut-Neutral Protocol" method parts and "Real-Only Sync Path" method into one Method chapter by stage, with the stage figure.
- [ ] Experimental setup chapter: data, splits, preregistration with H1 to H4, the amendments table (the only place amendment numbers appear).
- [ ] Results chapter: today's audit results, H1, sync results, cross-corpus, H4, H2 and further checks, in the paper's order.
- [ ] Discussion, limitations and future scope; Conclusion; Appendix (reproducibility).
- [ ] Replace every running-text amendment number with its plain description.
- [ ] Tick the checklist; run the verification commands; commit: "Report: rewritten in the paper's order and voice".

### Task 4: The PPT, aligned

**Files:** Modify `deliverables/slides/build_deck.py`.

- [ ] Introduction slide: the upload question ("Platforms re-encode every upload. What still works after that?").
- [ ] Flowchart slide: the stage figure from Task 1 as the slide's visual, full width.
- [ ] Results and conclusion headlines and speaker notes in the new framing; the face, voice and lip-sync colours match the figures.
- [ ] Build, export the PDF, render all 13 slides to PNG and look at each one; commit: "PPT: aligned with the upload framing and the new figure".

### Task 5: A human-voice pass, then the full rebuild

**Files:** the three sources above; `docs/PENDING.md`; `docs/CHANGELOG.md`.

- [ ] Read the paper's abstract, introduction and conclusion, the report's abstract and conclusion, and every slide's notes against the humanizer skill's patterns; fix what it finds (words only).
- [ ] Run every verification command, the poster build (unchanged, must still build), and the full test suite.
- [ ] Move the remake from the pending list to the changelog; commit; push.
