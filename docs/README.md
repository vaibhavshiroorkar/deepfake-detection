# Documentation

## Start here

- [Final project summary](FINAL.md): the clean version, only what the final project uses.
- [Pending work](PENDING.md): everything still open, in order.
- [Project log](PROJECT-LOG.md): every decision and discussion behind the work, 13 to 15 November.
- [Project handbook](HANDBOOK.md): the one file to read first. What the project is, what it measured, every mistake, what to do next, the paper plan and the full history.
- [Preregistration P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md): the current study's hypotheses, fixed before any run.
- [Authors and equal credit](../AUTHORS.md)
- [Project overview](../README.md)
- [Generated inventory](handoff.md): datasets on disk, evaluation reports and MLflow runs, written by `ddf handoff update`.
- [Deliverables](../deliverables/README.md): the paper, report, slides and poster.
- [Findings](research/findings.md): every claim the measurements support, by
  F-number. Other documents cite these numbers rather than restate them.
- [Obstacles](obstacles.md): known traps by area.

## Deep dives

One chapter per pipeline stage, written 2026-10-29, from the [chapter index](handbook/README.md). Chapter 08 describes the old ResNet-18 sync branch, which AV-HuBERT has replaced:

- [Problem and research question](handbook/01-problem-and-research-question.md)
- [Deep learning foundations](handbook/02-deep-learning-foundations.md)
- [Audio-video foundations](handbook/03-audio-video-foundations.md)
- [Data and leakage](handbook/04-data-and-leakage.md)
- [Preprocessing pipeline](handbook/05-preprocessing-pipeline.md)
- [Visual artifact branch](handbook/06-visual-branch.md)
- [Audio spoof branch](handbook/07-audio-branch.md)
- [Audio-video synchronization branch](handbook/08-sync-branch.md)
- [Fusion, calibration, and abstention](handbook/09-fusion-and-calibration.md)
- [Cross-modal streams](handbook/10-crossmodal-streams.md)

## Technical reference

- [CLI reference](reference/cli.md)
- [Dashboard](dashboard.md)

## Research evidence

- [Research design](research-design.md)
- [Model selection](model-selection.md)
- [Related work and novelty check](research/related-work.md)
- [Result traceability](research/result-traceability.md)
- [Findings](research/findings.md)
- [Corpora](research/datasets.md)
- [P-002 design](research/P-002-design.md): how to rebuild for maximum performance after P-001
- [Paper](../deliverables/paper/paper.tex) and [report](../deliverables/report/report.tex)

## Decisions and project controls

- [Data card](data-card.md)
- [Model card](model-card.md)
- [Reproducibility](reproducibility.md)
- [Threat model](threat-model.md)
- [Changelog](CHANGELOG.md): finished work, by version
- [ADR-001: Local MLflow tracking](decisions/ADR-001-local-mlflow.md)
