# Changelog

Finished work, grouped by version. Open work is in [PENDING.md](PENDING.md), and the reasons behind each decision are in the [project log](PROJECT-LOG.md).

When a pending item is done, it moves here under **Unreleased**. A stage ends with a version: the Unreleased heading becomes the version number and date, `pyproject.toml` takes the same version, and the commit is tagged `vX.Y.Z`. Comparing a version with the pending list before it shows what was planned and what was finished.

Experiment numbers are not repeated here. They live in `results/` and the documents built from them.

## Unreleased

- Amendments A12 to A14 fixed before any result (commit `e80f089`). A12 done (`scripts/run_a12.py`, with tests; `results/p001-a12/`): on DFDC and HiFi-AVDF even a fusion fit on the target's own labels does not beat the best single path; on LAV-DF a fusion fit on the other corpora beats A8 by 0.12 but trails the audio path. No fixed fusion of these paths transfers. In the paper (new subsection and table), the report (new section), the H2 notes, `docs/FINAL.md` and the READMEs, replacing the untested guess about the FakeAVCeleb weights.
- Two post-hoc checks on H2 (exploratory, `h2_checks` in `scripts/build_numbers.py`, with a test). Without the cloned-voice clips the FakeAVCeleb gain is +0.001 to +0.004. On LAV-DF the visual path scores even unedited-video fakes below the reals (0.175 to 0.243), so it reads a file difference, not the edit. The abstract, the H2 sections, the report summary, the slide verdict and notes, the poster, both READMEs and `docs/FINAL.md` now say H2 passes its rule but shows no gain on fakes the visual path can see.
- H2 ran on its Holm family on 11-16 at 22:56, and its preregistered rule says "H2 holds". Fusion minus visual-only: FakeAVCeleb strict +0.088, LAV-DF (window label) +0.170, DFDC +0.003 (p = 0.305). Output in `runs/p001/h2/`; HiFi-AVDF is not in it yet.
- H2 with HiFi-AVDF (reported only): +0.056 [+0.049, +0.063] on 3,485 clips. Fusion beats a visual path near chance but trails the audio path alone, as on LAV-DF. Added to the H2 tables, the interpretation, the notes, `docs/FINAL.md` and both READMEs. The full run's family targets match the family run byte for byte (handbook mistake 72).
- H2 results committed: `results/p001-h2/` (fusion predictions, `h2.json`, receipt) and `results/p001-branches/` (the six visual and audio models on the test set and the targets, with identity-bootstrap intervals), each with a README.
- H2 is interpreted in the paper, the report and the H2 speaker notes, with a new table of each path's AUC on the clip label. The gain on LAV-DF is over a visual path that is below chance, and fusion there trails the audio and sync paths alone; on DFDC it adds nothing.
- `scripts/build_numbers.py` records a one-class set (the audio path on DFDC) as counts with no AUC instead of failing, and adds each path's AUC on the clip label for H2 (`h2.{target}.auc.{path}`), checked against scikit-learn on all 60 values. Both with tests.
- The visual and audio results on the targets are shown through that table and `results/p001-branches/README.md`, not a separate report table.
- The PC hung a second time (handbook mistake 71). The H2 chain now skips finished stages and caches HiFi-AVDF alone at four shards; A11 waits for the whole H2 chain; a resource monitor logs RAM and GPU use every minute.
- Project log section 5 describes the 3-day plan.
- `scripts/build_numbers.py` reads the visual and audio results and H2 into `results/numbers.json`, with tests. Pass and fail print as "yes" and "no".
- The H2 section is written into the paper, the report (new chapter), the slides (new slide 14) and the poster. Fusion is no longer listed as "not run".
- `scripts/make_figures.py` draws `fig_h2.png` (fusion minus visual-only per target, with intervals and single seeds) once H2 exists; the paper and the report include it.
- The sync scorer leaves out and counts windows shorter than 50 frames instead of failing (handbook mistake 67). HiFi-AVDF scored again with the fix: 4 windows left out.
- Amendment A9: neutral LAV-DF clips are labelled by their kept window (`scripts/lavdf_window_labels.py`); 1,307 of 2,823 fake clips had no edited time in it. H2 uses the window label; the sync path and H4 on LAV-DF are reported under both labels. Handbook mistake 68.
- The paper, report, slides, poster and figures show LAV-DF under both labels, explain A9, and use the window label for H2.
- Amendment A10: the mouth cropper can locate small faces with YuNet before landmarking (`--locator`), used only on windows below the 0.8 face share. DFDC had found a face in 35% of windows. Handbook mistake 69.
- The 2-day chain finished (sync path on the three targets, H4, neutral audits). Neutral ceilings: HiFi-AVDF 0.999 to 0.629, LAV-DF 0.936 to 0.557, DFDC 0.588 to 0.526. The cache build then moved to normal priority.
- Results committed: `results/p001-sync-targets/` and `results/p001-h4/` (after A10, with READMEs), the results from before A10 beside them, and the neutral audits of the three targets plus the LAV-DF window label. H4 is met on HiFi-AVDF and on LAV-DF under the file label in all seeds; under the window label 2 of 3 LAV-DF seeds miss the bar. `numbers.json` holds 953 numbers.
- The cross-corpus and H4 figures no longer overlap their labels.
- The paper and report interpret the cross-corpus and H4 results, including coverage before and after A10. Target tables count clips scored (`.scored`) instead of all clips (`.n`), which had included clips with no usable window.
- The paper and report state the HiFi-AVDF clips without audio and the short windows left out.
- `docs/research/P-002-design.md`: a design for rebuilding the study for maximum performance after the submission, from what P-001 showed.
- The paper, report and slides state the one-training-corpus limitation; the report's full-run plan now leads with leave-one-corpus-out training under P-002.
- The README describes P-001 and its results, each linked to its results folder, instead of the older study.
- `docs/FINAL.md` is complete except H2. Every main claim is labelled verified (with its results folder) or assumed (with what would test it).
- Handbook section 5.1c and the project log record the target results, H4, A9, A10 and the 11-16 decisions.
- Handbook section 5 opens with a table of how sure each part is (verified, assumed, not results).
- P-002 ingestion checks (`data/ingestion_checks.py`, `scripts/check_ingestion.py`, with tests): face coverage, full span, label in window, shortcut ceiling. On the P-001 data they fail exactly where the hand-found problems were:
  - DFDC face coverage is 0.28 before A10 and 0.936 after;
  - LAV-DF label in window is 0.537;
  - HiFi-AVDF full span is 0.999;
  - the FakeAVCeleb ceiling is 0.632.
- `docs/research/P-002-backbones.md`: visual pick CLIP ViT-L/14 with LayerNorm-only tuning (mean cross-dataset video AUROC 94.2), audio pick frozen XLS-R (mean EER 17.4, where the current wav2vec2 Base is 28.4), with criteria, labels and sources. AV-Deepfake1M++ is 1.46 TB, more than this PC's free disk.
- `results/preregistration/P-002-draft.md`: the P-002 questions (Q1 to Q6), data, ingestion checks, paths and draft bars. Not frozen.
- Amendment A11 (exploratory): frozen CLIP ViT-L/14 and XLS-R backbones on the P-001 protocol. Visual streams now convert the cache's ImageNet normalization to their backbone's own, read from the backbone's config (tested). EfficientNet and DINOv3 are unaffected. `ddf train visual-stream --backbone clip` is available.
- `scripts/score_visual_stream.py` scores a visual-stream checkpoint on cached clips, rebuilt from its history file, in the branch prediction format (tested).
- `scripts/run_h2.py` runs before a reported-only target is scored and lists it under `not_scored_yet`; the Holm targets stay mandatory (tested). The H2 chain was relaunched to cache DFDC and LAV-DF in parallel, because full-HD targets cached at about 290 clips an hour.
- The six P-001 visual and audio models are trained (3 seeds each). The PC hung under four concurrent heavy jobs (handbook mistake 70); the chains resumed one heavy job at a time.
- The poster build stops when a column overflows, instead of writing a file PowerPoint calls corrupted.

## 0.3.0 - 2026-11-16: neutral protocol, sync path, deliverable builders

- **Neutralizer v4.** Audio is handled as samples: level to -23 dBFS RMS, cut at the first sound on a whole frame, exactly 3.00 s, dither at -66 dBFS, then one encode. Version 1 had put leading silence back through AAC priming (P-001 amendment A4).
- **H1 run twice and reported as not met.** The neutral FakeAVCeleb ceiling is 0.632 against a bar of 0.60.
- **Real-only sync path.** Frozen AV-HuBERT features and an alignment head trained on real clips only, three seeds, with receipts. In-domain results are in `results/p001-sync/`.
- **Locked targets prepared.** HiFi-AVDF manifest (amendments A5 and A7, one frame size per corpus), LAV-DF sampled by original recording (A3), and neutral copies of DFDC, LAV-DF test and HiFi-AVDF.
- **Frozen target scoring and H4.** `scripts/score_sync_targets.py` and `scripts/run_h4.py` (split conformal and Neyman-Pearson umbrella thresholds from real target clips).
- **H2 fixed in advance and coded.** Amendment A8 fixes the fusion fit and test before any visual or audio model existed; `fusion/h2.py` and `scripts/run_h2.py` implement it, with tests.
- **Numbers and figures from result files.** `scripts/build_numbers.py` writes `results/numbers.json`; `scripts/fill_numbers.py` fills every `{{key}}`; `scripts/make_figures.py` draws the figures.
- **Deliverables.** The report in the college LaTeX format, the IEEE paper, a 19-slide deck on the college slide master, an A0 poster, and Overleaf zips. All 33 references were checked against their sources.
- **Resume fixes.** Transient extraction and neutralization failures are retried instead of skipped. Neutralization resumes from the cut log alone, so a half-written video is redone. Receipted stages resolve relative program paths on Windows.
- **Docs.** `PENDING.md`, `PROJECT-LOG.md`, `FINAL.md` and three new mistakes in the handbook. Stale docs and the October Word and paper builders were removed (git keeps them). This changelog moved into `docs/`.

## 0.2.0 - 2026-11-13: one project, protocol fixes, shortcut audit


### 2026-11-13: one project, protocol fixes, shortcut-neutral pipeline

The plan behind these changes is preregistered in
[P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

Merged:

- The team project is folded in and `team/` is deleted. The submitted tree stays at the tag `team/submission-final`. Its ledger, experiment log, data log and integrity record are in [`results/legacy-team/`](../results/legacy-team/README.md). Its data moved to `data/` (ASVspoof2019_LA, In-the-Wild, `legacy-team/`) and its checkpoints to `runs/legacy-team/`. A second byte-identical Celeb-DF copy (9.5 GB) and a duplicate AV-HuBERT Base checkpoint were removed.
- `streams/avhubert.py` now takes the team version, which loads Base and Large. The old one loaded Base only.
- New `avsync/`: the real-only `AlignmentHead`, the supervised `WindowHead`, MediaPipe mouth crops and AV-HuBERT feature caching, ported from the team project.
- New `data/speech.py`: ASVspoof 2019 LA (121,461 rows, matches the official counts, no speaker in two partitions) and In-the-Wild (31,779 clips) manifests for the audio path.

Fixed:

- **FF++ identity leak.** `build_source_split(group_by="component")` keeps every identity a clip names in one partition. The old source-only grouping let 226 identities cross train and test. `ddf split build` uses component grouping for FaceForensics++ by default and records the mode in `audit.json`.
- **Target column.** FakeAVCeleb's metadata puts `fake.mp4` in the target column of all 499 RealVideo-FakeAudio rows. It was read as a person, so the identity-strict subset dropped that whole category. File names are no longer read as identities.
- **Locked test sets.** DFDC, Celeb-DF v2, DeepSpeak v2, In-the-Wild, KoDF and HiFi-AVDF are now evaluation-only, as MNW already was. DFDC had been used to choose the BatchNorm arm and the visual ensemble.
- **DFDC scope.** Its reports used scope `generalization_celebdf`. A new `generalization_cross_corpus` scope replaces it in the drivers.
- **Threshold provenance.** `ddf threshold` now needs `--split val|oof` and records the clip ids it selected on. `ddf evaluate` takes `--threshold-file`, refuses a threshold selected on any evaluated clip, and writes `threshold_source` into every report. A bare `--threshold` is still allowed, and the report marks it `fixed`.
- **EER** now interpolates where FPR equals FNR. It used to take the nearest ROC point.
- **Hardcoded path.** `scripts/train_family_holdout.sh` no longer changes into the old folder name.
- **Environment.** The editable install pointed at the old folder `deepfake-generalization`, so the package did not import and the test suite had not run since the rename.

Added:

- `scripts/shortcut_audit.py`. For each clip it reads container, codecs, bitrates, frame rate, size, duration, leading and trailing digital silence, and loudness. It then scores each cue alone, and all cues together, as a real-or-fake classifier, with folds grouped by identity. Output goes to `results/shortcut-audit/<dataset>/`.
- `preprocessing/neutralize.py` and `scripts/neutralize_dataset.py`. They cut audio and video together at the first and last sample above -60 dBFS, then re-encode to H.264 at 25 fps with AAC at 16 kHz mono. They also normalize loudness to -23 LUFS and strip metadata. They write a new manifest with shifted `sync_start_sec`, plus a sidecar of every cut.
- `evaluation/conformal.py`. False-alarm thresholds from real clips only: split-conformal, Neyman-Pearson umbrella, and the minimum number of real clips needed (59 for alpha = delta = 0.05).
- [Related work](research/related-work.md): which of the four objectives are new, with sources for every claim in `docs/research/related-work/`. Access steps for AV-Deepfake1M, DeepSpeak v2, KoDF and HiFi-AVDF are still to be added to the [dataset notes](research/datasets.md).

Later on 2026-11-13:

- **Shortcut audits committed** (`results/shortcut-audit/`).
  - FakeAVCeleb: file and audio-edge cues alone reach 0.994 AUC for real vs fake, 0.997 for fake video and 0.9996 for fake audio.
  - In FakeAVCeleb, 493 of 500 fake-audio clips carry MP3 audio where real clips carry AAC.
  - LAV-DF has no codec cue, but bitrate, trailing silence and duration together reach 0.936.
  - DFDC is close to clean (0.588).
- **LAV-DF file manifest.** `scripts/build_lavdf_file_manifest.py` labels each file by what was done to it. A first LAV-DF audit on the window manifest was deleted: 2,527 of 3,483 files supplied both a real and a fake window, so file cues could not separate them.
- **Feature extraction.** `scripts/extract_avhubert_features.py` caches AV-HuBERT lips, audio and fused features for up to three 2-second windows per clip.
- **Sync path trainer.** `scripts/train_sync_path.py` trains the real-only alignment head at three seeds. Both the training loss and the epoch choice use real clips only.
- **Visual and audio driver.** `scripts/run_p001_branches.sh` runs both paths at three seeds, with thresholds chosen on validation and recorded with each report.
- **P-001 amendment A1**, committed before training: the training subsample rule, three audio seeds, sync windows, and the per-file LAV-DF audit.

### Added

- A research report, presentation and rehearsal covering both project histories through November 2026. (Removed 2026-11-16; `git show 100c220:docs/research/presentation/README.md`.)
- A shared [credit policy](../AUTHORS.md) assigning one third to each student across both research lines and all current deliverables.
- Public demo start, status and stop commands. The demo serves an isolated
  visual-model runtime through a temporary HTTPS tunnel and excludes datasets.
- An in-browser demo of the best single visual model and the familiarity gate,
  hosted on a static Hugging Face Space. `hosting/browser/export.py` writes the
  ONNX models; `pipeline.js` ports the served preprocessing to JavaScript.
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
- `LICENSE.md` states the code terms (all rights reserved) and `CITATION.cff`
  is complete. The handbook drops its learning path chapter and its study
  sections (learning goals, exercises and viva questions). `docs/handoff.md` is
  titled Project status.

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
  density gate (F17), and `docs/research/architecture.md` (removed 2026-11-16; read it with `git show 100c220:docs/research/architecture.md`).
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
