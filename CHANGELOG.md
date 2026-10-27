# Changelog

This file records material changes to the software and research protocol.
Experiment metrics belong in the experiment tracker and in
[findings](docs/research/findings.md), not this file; entries cite a finding
number instead of restating it.

## Update rules

- Add current work under `Unreleased`.
- Record changes that affect behavior, data, models, evaluation, or users.
- Do not record routine experiment runs or daily progress.
- Move entries into a dated version when the project creates a release tag.

## Unreleased

### Added

- Public demo start, status and stop commands. The demo serves an isolated
  visual-model runtime through a temporary HTTPS tunnel and excludes datasets.

### Fixed

- Preserve explicit zero-start windows, initialize silent-media audio counts,
  retain dataset identities, and keep pilot evaluation distributions intact.
- Split cross-fitting into training, inner validation and untouched outer holdout.
  Record checkpoint-selection sources and reject contaminated OOF exports.
- Bind audio masks to checkpoint metadata. Preserve reused audio normalization
  across evaluation, feature export and serving.
- Restore recorded stream architectures, add temporal positions for new streams,
  and bind deep fusion to its exact serving checkpoints.
- Refuse partial dashboard checkpoint loads, invalidate replaced-media caches,
  preserve request identity in the web app, and bound incoming upload streams.
- Retain coverage reports when abstention leaves fewer than two scored classes.
- Keep optional model imports inside the cache and prediction handlers. Resolve
  training-history type annotations without importing Torch for CLI help.
- Bind research-stage reuse to configuration and artifact digests. Propagate
  child failures and retain historical evidence while corrected runs are pending.
- Mark affected research claims as requiring reruns. Extend CI to research
  scripts, browser behavior, base installs and Python 3.11 through 3.13.

### Removed

- The detector benchmark package (`benchmarks/`), its `ddf detector sample`,
  `validate-annotations`, `run` and `compare` commands, its MLflow logger, its
  tests and its CI smoke. It needed 625 human-reviewed frames that nobody was
  going to review. `ddf detector fetch-yunet` stays. ROADMAP Phase 2 records the
  decision.
- The dashboard's Evidence gate and everything only it used: the gate page, the
  video input, preprocessing, visual model and prediction sections, `runtime`,
  `state`, `workflow`, `view_model`, `lib/locked`, and the visual-only
  `VisualPredictionEngine` with its loader. The web app is now the only path that
  issues a verdict.
- `dashboard/lib/cross_modal.py` (its numpy reference formulas moved into
  `tests/test_cross_attention.py`), `data/download.py` and the `tqdm`
  dependency, `views/timeline.sliding_window_starts`,
  `experiments/scopes.is_generalization`, and the unused scripts
  `supervise_run.ps1`, `watch_run.ps1`, `make_run_configs.py`, `probe_stream.py`
  and `score_branches_folder.py`.
- `docs/superpowers/` (dated plans and specs) and `.github/workflows/test.yml`,
  which duplicated `ci.yml` with fewer checks.

### Changed

- `cli.py` is now the `cli/` package: one module per command group, each owning
  its handlers and its part of the parser. `build_parser`, `main` and
  `python -m deepfake_detection.cli` are unchanged.
- Every trainer selects its checkpoint on validation ROC-AUC through one
  `training/selection.BestEpoch`. The binary branch, sync and stream-fusion
  trainers still selected on loss, which is how a never-fine-tuned ConvNeXt
  checkpoint shipped (F29). The sync branch selects on misalignment AUC and logs
  offset accuracy separately (F31).
- Checkpoints record their `architecture`, and `branches/registry.py` rebuilds a
  branch from its checkpoint. `features export` and `predict` used to build
  EfficientNet whatever the file held.
- Core code no longer imports the dashboard: `paths.py` and `media_kind.py` moved
  into the package root (`DDF_PROJECT_ROOT` overrides the root) and clip
  playback moved to `views/playback.py`.
- One `digest.file_sha256` replaces six copies, one `data/cache_index.py`
  replaces the CLI's reader and writer and nine script copies, and
  `evaluation/branch_report.py` holds the report `ddf evaluate branch` writes.
- `pandas` and `joblib` are declared core dependencies; both were imported and
  arrived only through other packages.
- The web app's pages share one design system: a sidebar in place of the tab
  bar, a three-step check page whose result stays in view, colour, type, spacing
  and radius tokens in one block of `app.css`, no inline styles, and the admin
  page on the same stylesheet. Asset links carry a version so browsers fetch the
  new files.
- `ddf train stream --stream emotion` builds the emotion stream through its own
  builder. Both streams used the lip-sync builder; for the default encoders the
  two build the same model, so earlier checkpoints load and score as before.

### Fixed

- Frozen probe encoders apply their own input statistics. Perception Encoder
  and CLIP were fed ImageNet-normalised pixels, so `A-encoder-comparison` and
  the DF26 headline were measured on mis-scaled inputs (F33). Feature files now
  record the normalisation.
- The web app reports a familiarity gate that cannot load instead of silently
  skipping it, compares the admin token as bytes so a non-ASCII token is a 401
  rather than a 500, refuses an oversized upload from its declared length before
  the body is read, and caches sample playback.
- Test fixtures seeded from `hash()` of a string, which changes per process,
  made one training test flaky. They use `zlib.crc32`.
- `run_corruption_eval.py` loads checkpoints through the registry instead of a
  bare `torch.load`.
- `fusion/stream_loading.py` rebuilt the cross-modal streams without reading
  their encoder from the history file; it now does.
- `.gitattributes` keeps shell scripts at LF line ends; Windows checkouts gave
  them CRLF, which bash on Linux and WSL rejects.

### Added

- `scripts/codec_shortcut_check.py` and F35: the served visual models read the
  video codec, not only the face.
- Visual training augmentation (`training/augment.py`): random JPEG, H.264 and
  MPEG-4 round trips on both classes, self-blended fakes from genuine clips, and
  checkpoint selection on codec-swapped validation clips
  (`--augment-compression`, `--self-blend`, `--select-on-codec-swap`).
  `scripts/run_visual_recipes.py` trains and scores the recipes.
- `scripts/face_crop_probe.py`: a frozen encoder over the cached face crops with
  a linear probe, the face-crop cell F13 left empty.
- Registry rows `A-visual-ensemble`, `B-gate-fusion` and `A-codec-shortcut`, so
  the web app's default model and served head trace to files, and F36 for the
  ensemble. The ensemble report records each member's leave-one-out score.
- `tests/test_webapp_content.py`: every research number the web app shows must
  appear in the findings or the result registry.
- F37 and registry row `A-visual-recipes`: four visual recipes against the
  codec shortcut, none better than the shipped one on DFDC.
- AV-HuBERT and HSEmotion, the encoders the stream design named:
  `streams/avhubert.py` restates AV-HuBERT Base in plain torch (fairseq cannot
  run here), and `streams/hsemotion.py` loads HSEmotion's expression weights
  into timm through an allowlist. `ddf train stream --encoder avhubert` and
  `--encoder hsemotion` select them; the released weights go under `models/`
  (`paths.py`). `scripts/check_encoder_ports.py` holds both to the official code
  (they match to 2e-6) and chose the lip crop; `scripts/setup_encoder_reference.sh`
  builds the environment the official code runs in.
- `scripts/avhubert_sync_score.py`: lip-sync scored with no training, as the
  agreement between lips-only and audio-only AV-HuBERT features.
- `scripts/run_deep_ablation.py --streams --sizes`, to compare chosen stream sets
  without fitting every subset.
- `python_speech_features` as a dev dependency, the reference the AV-HuBERT
  filterbank test checks against.

### Earlier unreleased work

The work between 0.1.0 and this cleanup, grouped by area. The measurements it
produced are findings F1 to F34.

- **Web app.** `ddf-web` with three models, a movable threshold and measured
  operating points, the familiarity check, a token-guarded admin page writing
  `configs/webapp.yaml`, and a PDF report (`scripts/build_report.py`). The
  six-checkpoint visual ensemble is its default (F36).
- **Design B streams.** Configurable visual streams (`ddf train visual-stream`),
  audiovisual cross-attention streams with `diagonal_mass` (`ddf train stream`),
  a feature-level fusion head (`ddf train fusion --model deep`), stream export
  into the shared feature store, per-media-kind thresholds, and LAV-DF and DFDC
  adapters (F1 to F9).
- **Evaluation.** `ddf evaluate branch` and `ddf manifest usable`, both reporting
  abstentions; identity-clustered intervals everywhere with one single-class
  bootstrap policy; corruption, error-analysis, generalization-slice, ablation,
  seed-spread and family-holdout scripts (F16 to F34).
- **Provenance.** Out-of-fold purity checked from `trained_source_identities`;
  a generated result registry (`scripts/update_result_registry.py`) with
  `provisional (1 seed)` status; `views/equivalence.py` for cache hashes proven
  to label the same data; deterministic density-gate fitting (F17).
- **Training.** Stream checkpoints selected on ROC-AUC (F3), frozen BatchNorm on
  fine-tuning (F2), `--architecture convnext-tiny`, `AutoModel` audio builders
  so WavLM is not truncated (F28), audio padding masks, resumable and sharded
  cache builds, cross-fitting per seed.
- **Generated media.** DF26 fetching and frame-rate normalisation, the frozen
  probe pipeline with leave-one-generator-out (F15, F33), the authenticity
  density gate (F17), and `docs/research/architecture.md`.
- **Dashboard.** The teaching pages, the MLflow reader, dataset discovery over
  `data/`, the checkpoint picker that reads architecture from tensor shapes, and
  the trained Lip-Sync and Emotion pages.
- **Documentation.** The handbook, findings, paper draft and generated paper
  source, model card, data card, threat model, datasets survey, obstacles, and
  a drafted `CITATION.cff` and `LICENSE.md` (not yet in force).
- **Fixed.** Cache index paths relative to the index; deterministic sync token
  resizing; MLflow key redaction; a clean-environment install that could not
  collect the test suite; `score_by_family.py` label mixing; NaN spread on
  one-window clips.

## 0.1.0 - 2026-08-24

### Added

- Source-disjoint manifest and split tooling.
- Shared audio-video preprocessing with quality gates and cache hashes.
- Visual, audio, and synchronization model branches.
- Source-grouped cross-fitting and calibrated late fusion.
- Bootstrap evaluation, corruption tests, and subgroup reports.
- Video inference and a thin Streamlit dashboard.
- Research design and data card documents.
