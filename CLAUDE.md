# CLAUDE.md : project memory (read this first, every session)

This file is the binding memory of the project. Read it at the start of every session, then read `docs/STATUS.md` for the NEXT STEP line.

## 1. Binding goal and scope

- **Project:** Multimodal deepfake detection system (images, videos, and the audio inside videos), built from zero in this repository.
- **Deliverables:** (1) working software, (2) an IEEE-format research paper in LaTeX (Overleaf compatible), (3) a technical report, (4) a presentation. All four are built in this repo and kept in sync.
- **Purpose of the build:** a final-year college project. The point is to show a clear, documented path: a simple MVP first, then at least 7 iterations that each improve something for a stated reason, ending at results comparable to published detectors on a protocol we define and run ourselves. We are not trying to build the best model in the world.
- **Audience for the docs:** a professor who is strong in mathematics and will ask "why this, not that". Every choice must be traceable to a decision entry and, where possible, to an experiment.

## 2. Hard rules (never break these)

1. **Nothing unmeasured is written as a result.** Numbers in any document come from a logged experiment (E-xxx) in this repo. Observations that are not measurements are labelled "quick look, not a measurement".
2. **Hypothesis before run.** Every iteration and every comparison writes its hypothesis, success criterion and what would count as failure in `docs/EXPERIMENTS.md` before the run starts. Results are appended afterwards. Failed runs stay in the log.
3. **No test-set tuning.** Data is split once, by source identity (no person, video source or recording appears on both sides of a split). Tuning uses only train and validation. The final test split is touched only when the iteration is frozen. A guard in the evaluation script refuses test data unless the run is flagged `--final`.
4. **Fair comparison.** Candidate models are compared on the same data, split, budget and metric, with at least 3 seeds, reporting mean, standard deviation and a bootstrap interval. If intervals overlap, prefer the cheaper or simpler candidate and say so.
5. **Dates are real.** Never backdate a decision, experiment, commit or file. Use the actual current date.
6. **Citations are real.** Only cite a paper that is in `docs/RESEARCH.md` with status `READ`, and verify author, year and venue from the source itself. If unsure, write `NR` (not retrieved) instead of guessing.
7. **Licences and provenance.** Every dataset and pretrained weight gets its licence, source URL, version and file hash recorded in `docs/DATA_LOG.md` before use. Research-only weights are labelled as such.
8. **Claims discipline.** No "real-time", "state of the art", "better than X" unless a head-to-head run on the same protocol supports it. Write differences, not boasts.
9. **Mistakes are documented.** If you overwrite something, hit a bug that changes results, or find an earlier result was wrong, write it in `docs/STATUS.md` under "Mistakes and corrections" and fix the affected docs.
10. **Autonomy tag.** When the user is not available and you must decide, decide, and tag the entry `decided autonomously - needs user review`. Never silently decide.
11. **No em-dashes** in any document or comment. Use commas, colons or full stops.

## 3. Living documents (update after every milestone)

| File | What it holds | When to update |
|---|---|---|
| `CLAUDE.md` | This memory file | Only when rules or scope change (log the change in DECISIONS) |
| `docs/PLAN.md` | Stages, iterations, status | After every milestone |
| `docs/DECISIONS.md` | Numbered decision log and decision method | Every real choice |
| `docs/FILES.md` | Every file: what it contains, what it contributes | Every time a file is added, moved or removed |
| `docs/RESEARCH.md` | Literature matrix and reading status | Every paper read |
| `docs/EXPERIMENTS.md` | Experiment IDs, environments, hypotheses, results | Before and after every run |
| `docs/DATA_LOG.md` | Datasets, licences, hashes, splits | Every data change |
| `docs/VIVA_PREP.md` | Likely cross-questions with evidence-backed answers | After each decision or result |
| `docs/STATUS.md` | NEXT STEP, blockers, mistakes and corrections | End of every session |
| `paper/`, `report/`, `slides/` | The written deliverables | Whenever the underlying evidence lands |

## 4. Session protocol

Start: read `CLAUDE.md`, `docs/STATUS.md`, `docs/PLAN.md`. State the NEXT STEP and carry it out.
During: small steps, commit after each logical change (see section 5).
End: update `docs/STATUS.md` (what was done, NEXT STEP in one concrete sentence, open questions), update `docs/FILES.md` and any affected living docs, commit, push.

If a usage limit or crash interrupts a session, the NEXT STEP line must be enough for a fresh session to continue.

## 5. Git and GitHub

- Remote: GitHub repository supplied by the user (`origin`). Authenticate with `gh auth login` on the user's machine. Never write tokens into any file.
- Commit small and often. Message format: `type(scope): summary`, for example `docs(decisions): add D-012 backbone shortlist`. Types: `docs`, `feat`, `fix`, `exp`, `data`, `paper`, `chore`.
- Push at the end of each work block. Tag each frozen iteration: `iter-0-mvp`, `iter-1`, and so on.
- Large data, checkpoints and videos stay out of git (`.gitignore`); record their hashes instead.

## 6. Environments

Name every environment and record it in `docs/EXPERIMENTS.md` (OS, Python, CUDA, PyTorch, GPU model and VRAM from `nvidia-smi`, driver). Stated by the user on 2026-10-24: RTX 5070 class GPU (VRAM stated as 8 or 12 GB, to be verified with `nvidia-smi`), AMD Ryzen 7900K, 32 GB DDR5, local fine-tuning. Plan every experiment inside the verified VRAM.

## 7. How to ask the user

Use multiple-choice questions for real decisions. Ask only what blocks the next step. Record the answer in DECISIONS the same session.
