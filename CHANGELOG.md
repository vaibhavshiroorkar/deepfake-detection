# Changelog

This file records material changes to the software and research protocol.
Experiment metrics belong in the experiment tracker, not this file.

## Update rules

- Add current work under `Unreleased`.
- Record changes that affect behavior, data, models, evaluation, or users.
- Do not record routine experiment runs or daily progress.
- Move entries into a dated version when the project creates a release tag.

## Unreleased

### Fixed

- MLflow tracking now redacts sensitive camel-case and punctuation-delimited
  configuration and tag keys, and records failed finalization attempts as
  failed runs without replacing their original errors.

### Added

- Training-only detector review sampling and JSONL annotation contracts with
  source-disjoint calibration groups, fixed evidence gates, frame hashes,
  multi-face labels, independent double-review checks, whole-face disagreement
  audits, explicit adjudication, and deterministic gold-label resolution.
- A deterministic constant-velocity face association challenger with bounded
  gap recovery and stable one-to-one matching. Greedy IoU remains the default.
- A deterministic five-landmark lower-face view with a versioned template,
  fixed crop region, strict geometry checks, nearest-frame fill, quality
  coverage, and cache identity. The existing box crop remains the default.
- Landmark-aware MTCNN and YuNet face detector adapters with a pinned,
  integrity-checked YuNet model asset.
- Windows CI that installs the full local environment, checks lint, format,
  lock, documentation, tests, and the configured tracked smoke run.
- A deterministic CPU late-fusion smoke command with source-disjoint,
  class-balanced fixture groups, held-out validation metrics, byte-hashed
  artifacts, and optional MLflow evidence. Its metrics are software fixture
  evidence only.
- Configured `ddf run` execution with layered YAML, an explicit project root,
  optional MLflow tracking, and failed-run status for nonzero command exits.
- MLflow training evidence for branch histories, stage metrics, elapsed time,
  byte-hashed checkpoints, fusion artifacts, and metadata outputs.
- Optional local MLflow tracking with SQLite metadata, local artifacts, runtime
  snapshots, resolved configuration artifacts, and failed-run records.
- MLflow-safe tracking keys and bounded parameter, tag, runtime, and run-name
  values with deterministic hash suffixes.
- MLflow-compatible rejection and encoding of ambiguous dot-path tracking keys.
- Versioned layered YAML configuration for reproducible local experiments.
- Runtime environment snapshots with Git, package, hardware, memory, and FFmpeg
  details, plus deterministic shared training seeds.
- A beginner handbook with a 15-chapter learning path and live foundations for
  the research problem, deep learning, audio-video timing, data leakage, and
  the implemented preprocessing pipeline.
- Handbook chapters for the current visual, audio, and synchronization model
  branches, plus calibrated late fusion, missing-evidence rejection, and
  planned candidate comparisons.
- A public CLI parser contract, generated CLI command reference, and CLI drift
  validation in `ddf-docs`.
- Repository-owned documentation validation for Markdown, local links, change
  contracts, and optional external links.
- A live roadmap with phase gates for the final-year project.
- Model selection rules for controlled component comparisons.
- A reproducibility contract for local experiments and future MLflow tracking.
- A threat model for research scope, failure modes, and misuse.
- A documentation index in the README.
- An approved specification for the project handbook and research evidence
  system.

## 0.1.0 - 2026-08-24

### Added

- Source-disjoint manifest and split tooling.
- Shared audio-video preprocessing with quality gates and cache hashes.
- Visual, audio, and synchronization model branches.
- Source-grouped cross-fitting and calibrated late fusion.
- Bootstrap evaluation, corruption tests, and subgroup reports.
- Video inference and a thin Streamlit dashboard.
- Research design and data card documents.
