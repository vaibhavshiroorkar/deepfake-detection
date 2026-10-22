# Documentation

## Start here

- [Project overview](../README.md)
- [Project handoff](handoff.md): what is measured, what is next.
- [Findings](research/findings.md): every claim the measurements support, by
  F-number. Other documents cite these numbers rather than restate them.
- [Obstacles](obstacles.md): traps to read before touching an area.

## Learning handbook

Start with the [handbook index](handbook/README.md):

- [Learning path](handbook/00-learning-path.md)
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
- [Architecture for generated media](research/architecture.md)

## Research evidence

- [Research design](research-design.md)
- [Model selection](model-selection.md)
- [Experiment matrix](research/experiment-matrix.md)
- [Result traceability](research/result-traceability.md)
- [Findings](research/findings.md)
- [Predictions](research/predictions.md)
- [Corpora](research/datasets.md)
- [Contributions](research/contributions.md)
- [Paper draft](research/paper.md) and the
  [generated tables it is written from](research/paper-source.md)

## Decisions and project controls

- [Data card](data-card.md)
- [Model card](model-card.md)
- [Reproducibility](reproducibility.md)
- [Threat model](threat-model.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)
- [ADR-001: Local MLflow tracking](decisions/ADR-001-local-mlflow.md)

## Update rules

Keep links limited to existing files. Update the CLI reference generated block
whenever the parser changes, then run `uv run ddf-docs`.
