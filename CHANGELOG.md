# Changelog

This file records material changes to the software and research protocol.
Experiment metrics belong in the experiment tracker, not this file.

## Update rules

- Add current work under `Unreleased`.
- Record changes that affect behavior, data, models, evaluation, or users.
- Do not record routine experiment runs or daily progress.
- Move entries into a dated version when the project creates a release tag.

## Unreleased

- Required paired MTCNN and YuNet research reports, usable tracking evidence,
  clean pinned environments, exact input report hashes, and source run IDs for
  detector decisions. Disabled the unbound downstream scalar tie-break.
- Restricted detector review sampling to identity-strict training rows and
  bound the identity-strict subset hash through benchmark evidence.
- Required candidate bytes and strict aggregate report content to match the
  supplied benchmark report before MLflow logging begins.
- Bound detector review evidence to the verified frozen training split and its
  hash. Enforced the 500-frame, 100-clip gate after calibration removal.
- Bound benchmark reports to the reviewed sample and annotation audit. Added
  strict nested aggregate and candidate validation before MLflow upload.
- Derived MTCNN provenance from loaded weights and clarified visible-face
  annotation rules.

### Fixed

- Detector evidence now binds thresholds to calibration data, hashes sequence
  identity, uses fixed complete source bootstraps, counts stable-track identity
  events, and records backend-derived CPU runtime metadata.
- Root output-directory ignores no longer hide Python packages under `src`.
- MLflow tracking now redacts sensitive camel-case and punctuation-delimited
  configuration and tag keys, and records failed finalization attempts as
  failed runs without replacing their original errors.

### Added

- Detector CLI commands for the pinned YuNet asset, training-only review
  sampling, annotation audits, benchmark runs, and frozen comparisons.
- A shared cache and prediction preprocessor factory with explicit detector,
  tracker, crop, model path, and expected model hash inputs. Existing MTCNN,
  greedy IoU, and box-crop defaults remain unchanged.
- MLflow-safe detector evidence logging for aggregate reports, hashes, and
  path-free prediction JSONL. Raw review data and model binaries remain local.
- A deterministic CI detector comparison smoke whose fixture scope cannot
  select a real detector.
- A source-disjoint detector benchmark evaluator with fixed threshold
  calibration, all-face matching, landmark and tracking metrics, source
  bootstraps, deterministic raw evidence, and frozen selection rules.
- Training-only detector review sampling and JSONL annotation contracts with
  source-disjoint calibration groups, fixed evidence gates, frame hashes,
  multi-face labels, independent double-review checks, whole-face disagreement
  audits, canonical reviewer identities, disagreement-only adjudication, and
  deterministic gold-label resolution.
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
