# Research presentation materials

Use this package for the research presentation covering July through November 2026.
Vaibhav Shiroorkar, Shivansh Gupta and Adithya Jithesh share equal credit, one third each.
Dr. Tejashree Phatak is the project guide.

| Material | Purpose |
|---|---|
| [Research report](research-report.md) | Research question, history, methods, results, corrections and next experiments |
| [Presentation source](slides.json) | Fifteen slides with evidence references and speaker notes |
| [Rehearsal](rehearsal.md) | Equal speaking time, demonstration and likely questions |
| [Project handbook](../../HANDBOOK.md) | Everything in one place, including the full history and every mistake |
| [Credit policy](../../../AUTHORS.md) | Equal credit across all current work and deliverables |
| [Report PDF](Research_Report.pdf) and [editable Word report](Research_Report.docx) | Nine sections covering the full research narrative |
| [Presentation PDF](Research_Presentation.pdf) and [editable PowerPoint deck](Research_Presentation.pptx) | Fifteen slides with speaker notes in PowerPoint |

The generated Word report, PowerPoint deck and matching PDFs live in this folder.
Run `powershell -ExecutionPolicy Bypass -File docs/research/presentation/build.ps1` from the repository root to rebuild them.
The Windows builder requires `uv` and LibreOffice. Pass `-Soffice` with the full path to `soffice.com` if it is not on PATH.
Python dependencies run in an isolated `uv` environment. They do not change either research project's environment.

## Evidence boundaries

The root study and the team project (now folded in) use different implementations, splits and evaluation rules.
Their scores must stay separate. The report identifies evidence as measured, historical, exploratory or planned.
The root [result registry](../result-traceability.md) controls whether an old result remains usable.
The [team experiment log](../../../results/legacy-team/EXPERIMENTS.md) controls team experiment status.

This package summarizes existing records. Preparing it did not rerun experiments or validate unavailable checkpoints.
No new literature priority claim is made. The full references remain in the existing paper sources and literature register.

## Historical submissions

The earlier deliverables (`submission/`, `deliverables_v2/`, `paper/`, `report/`, `slides/` and `Exhibition/`) are kept at the git tag `team/submission-final`.
Their PDFs and Office files describe earlier evidence cutoffs. Use this package for the current research presentation.
Original signatures, DOI proofs and submission checksums remain historical records.
The current credit statement supersedes earlier task allocations for present-day attribution.

## Reproducing research

Start with the root [reproducibility guide](../../reproducibility.md) or the team setup guide (`docs/SETUP.md` at tag `team/submission-final`).
Use each project's own environment and dataset manifests.
Raw data, licensed weights and many run outputs are not distributed in Git.
Confirm local artifacts before attempting a live demonstration.
Never rerun a sealed test merely to prepare slides.
