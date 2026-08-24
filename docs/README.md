# Documentation

## Start here

- [Project overview](../README.md)
- [Research design](research-design.md)
- [Data card](data-card.md)
- [Documentation command reference](reference/cli.md)

## Learning handbook

Start with the [handbook index](handbook/README.md). Live chapters are:

- [Learning path](handbook/00-learning-path.md)
- [Problem and research question](handbook/01-problem-and-research-question.md)
- [Deep learning foundations](handbook/02-deep-learning-foundations.md)
- [Audio-video foundations](handbook/03-audio-video-foundations.md)
- [Data and leakage](handbook/04-data-and-leakage.md)

Future chapters are:

`handbook/05-preprocessing-pipeline.md`
`handbook/06-visual-branch.md`
`handbook/07-audio-branch.md`
`handbook/08-sync-branch.md`
`handbook/09-fusion-and-calibration.md`
`handbook/10-training-system.md`
`handbook/11-evaluation-and-statistics.md`
`handbook/12-inference-and-dashboard.md`
`handbook/13-reproducing-the-project.md`
`handbook/14-viva-preparation.md`

## Technical reference

- [CLI reference](reference/cli.md)

Planned references: `reference/architecture.md`, `reference/configuration.md`,
`reference/artifact-contracts.md`, `reference/testing.md`, and
`reference/hardware-and-compute.md`.

## Research evidence

- [Research design](research-design.md)
- [Model selection](model-selection.md)

Planned evidence files: `research/questions-and-hypotheses.md`,
`research/experiment-matrix.md`, `research/metrics-and-statistics.md`,
`research/result-traceability.md`, `research/findings.md`,
`research/error-analysis.md`, and `research/paper-outline.md`.

## Decisions and project controls

- [Reproducibility](reproducibility.md)
- [Threat model](threat-model.md)
- [Roadmap](../ROADMAP.md)
- [Changelog](../CHANGELOG.md)

Planned decisions: `decisions/ADR-001-local-mlflow.md`,
`decisions/ADR-002-source-disjoint-splits.md`,
`decisions/ADR-003-calibrated-late-fusion.md`,
`decisions/ADR-004-quality-aware-abstention.md`, and
`decisions/ADR-005-detector-bakeoff.md`.

## Documentation ownership

| Package | Handbook owner | Reference owner |
|---|---|---|
| `data` | `04-data-and-leakage.md` | `artifact-contracts.md` |
| `views` | `05-preprocessing-pipeline.md` | `architecture.md` |
| `branches` | `06-visual-branch.md` through `08-sync-branch.md` | `architecture.md` |
| `training` | `10-training-system.md` | `configuration.md` |
| `fusion` | `09-fusion-and-calibration.md` | `artifact-contracts.md` |
| `evaluation` | `11-evaluation-and-statistics.md` | `testing.md` |
| `inference` | `12-inference-and-dashboard.md` | `architecture.md` |
| `dashboard` | `12-inference-and-dashboard.md` | `architecture.md` |
| `documentation` | `00-learning-path.md` | `testing.md` |

## Update rules

Update this index when a document becomes available. Keep links limited to
existing files. Update the CLI reference generated block whenever the parser
changes, then run `uv run ddf-docs`.
