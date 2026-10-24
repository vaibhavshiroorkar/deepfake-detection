# STATUS

Last updated: 2026-10-24 (Session 1 in progress)

NEXT STEP: Resume Session 1 at Step 3 by reading docs/_agents/ (A.md, B.md, C1.md, C2.md, C3.md, one file per stream, none yet merged), spot-check each against its primary sources (title pages for papers, official pages for datasets), merge verified material into docs/RESEARCH.md, docs/DATA_LOG.md and docs/FILES.md, and then show the user the dataset access plan for approval before any download starts.

## Session 1 progress (in progress, updated mid-session)

- Done and committed locally: CLAUDE.md added; D-001 to D-010 reasons filled and D-009 revised (deadline 2026-10-30); D-011 to D-019 logged; PARTIAL status added and all 18 opened papers re-marked; day-by-day schedule in PLAN.md; download throughput measured (4.09 MB/s); docs/MATH_NOTES.md written (background, candidate metrics, not adopted).
- Five subagent streams were started (A templates, B datasets, C1/C2/C3 literature). C1 and C2 have reported; their files are in docs/_agents/ and are NOT yet merged or fully verified (title pages and a sample of numbers of C1 were checked). A, B and C3 had not reported when this note was written.
- Nothing has been pushed: no `origin` remote and `gh` is not logged in on this machine.

## Where the project stands

- Session 0 steps done: machine verified and PyTorch environment working (Step 1); eight living documents written (Step 2); first literature pass, 17 papers read from primary sources (Step 4); handoff prepared (Step 5).
- Step 3 (GitHub) is incomplete: no remote exists and the GitHub CLI is not logged in. Nothing has been pushed and no `setup` tag exists.
- No dataset downloaded, no model or training code written, no experiment run. This is as the Session 0 prompt requires.

## Blockers

1. GitHub login. `gh` 2.103.2 was installed in this session (it was absent), but `gh auth status` reports no login and `git remote -v` is empty. The user must run `gh auth login` in a terminal. Then:
   - `gh repo create <name> --private --source=. --remote=origin` (or add an existing repository with `git remote add origin <url>`)
   - `git push -u origin main`
   - `git tag setup` then `git push origin setup`
   The implementer did not create a repository on the user's account without being told its name and visibility.
2. CLAUDE.md is missing. The prompt calls it binding and refers to its commit message format and document set. A provisional commit message format was used: a first line "S0 step N: summary", a body, and the co-author trailer given in the session instructions. Please provide CLAUDE.md or confirm this format.

## Open questions for the user

1. What GitHub repository name and visibility (private or public) should be used?
2. What is the submission deadline? (D-009 leaves dates open; a date lets the plan be checked.)
3. Should the reasons and rejected alternatives for D-001 to D-010 be recorded? The prompt gave the decisions but not their reasons.
4. Is Python 3.13 acceptable? It was chosen over the system Python 3.14 because more machine-learning libraries support it; the choice is small and can be reversed.
5. The open decisions O-001 to O-011 in DECISIONS.md need answers in order, starting with the problem statement (O-001, O-002).

## Mistakes and corrections

| Date | Mistake | Correction |
|---|---|---|
| 2026-10-24 | The prompt's first instruction, to read CLAUDE.md, could not be followed: the file does not exist and the repository folder was empty. | Reported as a blocker and as a contradiction in DECISIONS.md; work that did not depend on it proceeded; a provisional commit format was used and labelled as such. |
| 2026-10-24 | While adding the PyTorch dependency, the first attempt left duplicated sections in pyproject.toml, which made the file unparsable. | The duplicated block was removed and `uv sync` was rerun; the resolved lock file was checked by importing PyTorch and running the GPU tests. |
| 2026-10-24 | An arXiv identifier for the audio-visual dissonance paper (2009.14232) was written from memory. The downloaded file turned out to be an astrophysics paper. | The file was discarded without being used. The correct paper (arXiv:2005.14405) was found by search and its title page was checked. All other identifiers in RESEARCH.md were checked against title pages the same way. |
| 2026-10-24 | The first attempts to read PDFs through the web-fetch tool failed because the tool cannot parse PDF binary. | PDFs were downloaded to a scratch folder outside the repository and read with pdftotext. |
| 2026-10-24 | A tool-based text extractor (pymupdf through uvx) timed out because of a network error. | pdftotext, which was already installed, was used instead. |
| 2026-10-24 | The disk-space readings for drive C: changed between 46 GB, 67 GB and 103 GB during the session. | ENV-DESKTOP-GPU records the range rather than one value. Drive D:, where the repository lives, was stable at 404 GB. |

## Provisional items to confirm

- Commit message format (see Blockers).
- Python 3.13 and the CUDA 13.0 build of PyTorch 2.14.1. The official install page could not be read by the fetch tool, so the wheel index was inspected directly. The choice is recorded in ENV-DESKTOP-GPU.
