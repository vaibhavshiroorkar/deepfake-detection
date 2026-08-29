# Project roadmap

The project tests whether cue-specific multimodal fusion generalizes better than
a strong visual baseline. Each phase has an exit gate. A later phase must not
change decisions frozen by an earlier phase.

The first phase with an unmet exit gate is the next body of work.

## Phase 0: Research platform

- [x] Define source-disjoint and identity-strict evaluation protocols.
- [x] Build manifests, split audits, shared views, and cache provenance.
- [x] Add visual, audio, and synchronization branch interfaces.
- [x] Add out-of-fold feature export and calibrated late fusion.
- [x] Add evaluation, one-video inference, and dashboard paths.
- [x] Cover the platform with automated tests.

Exit gate: a clean checkout can run the test suite and show every CLI command.

## Phase 1: Reproducible local experiments

- [ ] Add the layered project handbook and technical reference.
- [ ] Add research questions, an experiment matrix, and result traceability.
- [ ] Add automated documentation validation and package ownership checks.
- [x] Add MLflow as an optional local dependency.
- [x] Store MLflow metadata in SQLite and artifacts on the local filesystem.
- [x] Log the run contract from [reproducibility.md](docs/reproducibility.md).
- [x] Add versioned configuration files for preprocessing and training.
- [x] Add a CPU smoke configuration that finishes on a small fixture dataset.
- [x] Add CI for linting, formatting, tests, and documentation links.

Exit gate: one command reproduces a tracked smoke run from a clean environment.

## Phase 2: Face and mouth view integrity

- [x] Extend detections with five facial landmarks.
- [x] Add YuNet behind the detector interface.
- [x] Add versioned landmark-aligned crops while keeping box crops as default.
- [x] Add deterministic training-only review sampling and annotation tooling.
- [ ] Create the reviewed detector benchmark sample.
- [x] Add frozen detector, tracker, crop, and benchmark evaluation tooling.
- [ ] Compare MTCNN and YuNet using the frozen selection rules.
- [x] Add and test constant-velocity association against greedy IoU.
- [ ] Freeze the detector, tracker, and alignment configuration.

Exit gate: the selected view pipeline meets the quality rules in
[model-selection.md](docs/model-selection.md). Its configuration hash is frozen.

The software gate is complete. The evidence gate still needs at least 500
human-reviewed frames from at least 100 training clips, a second review of at
least 10 percent, paired detector runs, and acceptance of the measured choice.

## Phase 3: Strong branch baselines

- [ ] Add valid-length masks to every padded audio batch.
- [ ] Train EfficientNet-B0 plus GRU as the visual baseline.
- [ ] Compare ConvNeXt-Tiny under the same visual training budget.
- [ ] Train Wav2Vec2 Base as the audio baseline.
- [ ] Compare WavLM and AASIST under the same audio protocol.
- [ ] Train the current synchronization branch on authentic correspondence.
- [ ] Compare it with a published synchronization baseline.
- [ ] Run every accepted branch with three fixed seeds.

Exit gate: each branch beats its cue-label prior baseline. Every accepted run
has complete provenance and confidence intervals.

## Phase 4: Fusion and validation

- [ ] Export source-grouped out-of-fold predictions for every branch and seed.
- [ ] Fit calibrated logistic fusion from out-of-fold predictions only.
- [ ] Run branch, pair, calibration, sync-label, and abstention ablations.
- [ ] Compare logistic fusion with the small MLP.
- [ ] Select and freeze the decision threshold on validation data.
- [ ] Freeze model settings, checkpoints, and hashes before test access.

Exit gate: the final experiment manifest can reproduce every validation table.

## Phase 5: Locked evaluation and submission

- [ ] Run the frozen FakeAVCeleb test once.
- [ ] Run identity-strict and leave-one-method-family-out evaluations.
- [ ] Run the locked zero-shot Deepfake-Eval evaluation.
- [ ] Report subgroup, corruption, coverage, and abstention results.
- [ ] Complete error analysis without retuning the locked models.
- [ ] Create the final model card from measured results.
- [ ] Add `CITATION.cff` with the student's verified academic identity.
- [ ] Choose and add a license after checking university and dataset terms.
- [ ] Reproduce the main result from a clean environment.

Exit gate: the dissertation tables trace back to immutable run IDs and
artifacts. Claims include failures, limits, and confidence intervals.

## Scope controls

- Test at most one or two serious challengers per component.
- Prefer a cheaper model when results are statistically indistinguishable.
- Do not tune on the final test set or Deepfake-Eval.
- Do not add interface work until the evidence pipeline is complete.
- Treat a negative fusion result as a valid final result.
