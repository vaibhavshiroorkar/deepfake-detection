# Project roadmap

The project tests whether cue-specific multimodal fusion generalizes better than
a strong visual baseline. Each phase has an exit gate. A later phase must not
change decisions frozen by an earlier phase.

The first phase with an unmet exit gate is the next body of work.

That rule no longer describes the project on its own. Phases 3 through 5 ran
ahead of Phase 2's evidence gate, which is blocked on human review rather than
on code, and the work has since opened a second track on open-world generated
media that these phases never anticipated. Read
[the handoff](docs/handoff.md) for what is actually next; the phases below are
the record of what is and is not done.

## Phase 0: Research platform

- [x] Define source-disjoint and identity-strict evaluation protocols.
- [x] Build manifests, split audits, shared views, and cache provenance.
- [x] Add visual, audio, and synchronization branch interfaces.
- [x] Add out-of-fold feature export and calibrated late fusion.
- [x] Add evaluation, one-video inference, and dashboard paths.
- [x] Cover the platform with automated tests.

Exit gate: a clean checkout can run the test suite and show every CLI command.

## Phase 1: Reproducible local experiments

- [x] Add the layered project handbook and technical reference.
- [x] Add research questions, an experiment matrix, and result traceability.
- [x] Add automated documentation validation and package ownership checks.
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

The software gate is complete. The evidence gate still needs a frozen sample
with at least 625 human-reviewed frames from 125 training clips. After
calibration sources are removed, at least 500 frames from 100 clips must remain.
It also needs a second review of at least 10 percent, paired detector runs, and
acceptance of the measured choice.

## Phase 3: Strong branch baselines

- [x] Add valid-length masks to every padded audio batch.
- [x] Train EfficientNet-B0 plus GRU as the visual baseline.
- [x] Compare ConvNeXt-Tiny under the same visual training budget, three seeds each.
- [x] Train Wav2Vec2 Base as the audio baseline.
- [x] Compare WavLM under the same audio protocol (undecided: both at 1.0000).
- Descoped: compare AASIST. WavLM was the audio challenger, and F28 shows the
  in-domain partition cannot separate two encoders that both read 1.0000. A
  third cannot be decided either until a cross-corpus audio partition exists.
- [x] Train the current synchronization branch on authentic correspondence.
- Descoped: compare it with a published synchronization baseline. The branch
  is below chance at all three seeds (F26) and its objective is not learnable
  at this budget (F32), so a baseline comparison has nothing to compare.
- [x] Train every branch on three fixed seeds.
- [x] Score them: `scripts/score_sync_branch.py` gives sync one, and the audio and visual seeds are scored. Fusion still needs per-seed folds.

Exit gate: each branch beats its cue-label prior baseline. Every accepted run
has complete provenance and confidence intervals.

Audio masking is implemented and tested, but no shipped checkpoint was trained
with it. The mask needs a per-clip valid sample count, which the cache only
started recording now, so entries built earlier report `None` and a batch
holding one of them is scored unmasked exactly as before. Masked training needs
a rebuilt audio cache, and the shortcut it closes is named in the threat model.

All three branches trained at full scale in `program-20260906` on seed 17, and
every headline number carries an identity-clustered interval. The open items are
the three challengers and the seeds, and all four are now GPU time rather than
code: `train visual --architecture convnext-tiny` builds the VIS-01 challenger
through the same branch, which was always architecture-agnostic.

## Phase 4: Fusion and validation

- [x] Export source-grouped out-of-fold predictions for every branch.
- [x] Repeat the test and DFDC export for the remaining two seeds.
- [ ] Cross-fit the folds per seed, which the head's own three seeds need. `pwsh scripts/run_crossfit_seeds.ps1`, six to eight hours.
- [x] Fit calibrated logistic fusion from out-of-fold predictions only.
- [x] Run branch, pair, calibration, cue-label, abstention and sync-label ablations.
- [ ] Run the leading-silence ablation. It needs the cache rebuilt with
  `--keep-leading-silence`, which re-runs face detection over 59 GB. F25's
  half-second shift at inference found no silence shortcut, but that is not
  the cache-level ablation the research design names.
- [x] Compare logistic fusion with the small MLP.
- [x] Select and freeze the decision threshold on validation data.
- [x] Freeze model settings, checkpoints, and hashes before test access.

Exit gate: the final experiment manifest can reproduce every validation table.

The ablations are measured and recorded as `A-branch-ablation`,
`A-fusion-model-ablation`, `A-generalization-slices` and `B-fusion-ablations`.
The logistic head ships: the MLP matches it in-domain and loses 0.1508 on DFDC.

The gate is not met, and seeds are still what holds it, but the diagnosis has
moved. All three branches now have all three seeds and are scored. The fusion rows
stay `provisional (1 seed)` because the head reads out-of-fold features exported
from seed-17 fold models, so the head itself has one seed however many the
branches have. `run_crossfit_seeds.ps1` is the remaining work.

What the seeds already show is why this matters rather than being bookkeeping:
cross-corpus, the fused number moves 0.1057 with the branch seed alone, and the
seed every result reports is the best of three.

## Phase 5: Locked evaluation and submission

- [x] Run the frozen FakeAVCeleb test once.
- [x] Run the identity-strict evaluation.
- [ ] Run the leave-one-method-family-out evaluation.
- [x] Run the locked zero-shot Microsoft-Northwestern-WITNESS evaluation.
- [x] Report subgroup, coverage, and abstention results.
- [x] Report corruption results.
- [x] Complete error analysis without retuning the locked models.
- [x] Create the final model card from measured results.
- [ ] Verify the academic identity in `CITATION.cff` (drafted, three TODOs).
- [ ] Confirm the code license in `LICENSE.md` with the university (drafted, not in force).
- [x] Reproduce the tracked smoke run and the suite from a clean environment.
- [ ] Reproduce the main result from a clean environment (needs the cache).

Exit gate: the dissertation tables trace back to immutable run IDs and
artifacts. Claims include failures, limits, and confidence intervals.

A clean-environment check on 2026-09-23 found the test suite could not even be
collected with the install set CI itself uses: `streamlit` lives only in the
`dashboard` extra, `ci.yml` did not install it, and nine test modules imported
the dashboard at module level. That aborted the whole suite. The modules now
guard themselves the way the project's other optional-dependency tests already
did, `ci.yml` installs the extra so those tests run rather than skip, and four
further tests that depended on gitignored `runs/` artifacts now skip with a
reason instead of failing. A fresh copy with no `runs/`, no `data/` and no
checkpoints installs from `uv.lock`, reproduces the tracked smoke run in one
command, and passes the full suite.

Subgroup results are in `A-generalization-slices` and report no gap by race or
gender in-domain, at a ceiling where no gap could appear. That is the honest
reading and it is recorded as such in F20.

Corruption results are in `A-corruption-visual` and `A-corruption-audio`, and
they bound the classifier rather than the pipeline: the face track is held fixed,
so detector failures under corruption are not measured. The headline is F24, that
JPEG 75 costs 0.0054 of ROC-AUC and takes false alarms from 0 to 51 percent.

Leave-one-method-family-out is the one still genuinely open. The head-only
holdout in the same result refits the fusion head alone, which bounds the head's
method-specificity and nothing wider, because the branch checkpoints saw every
method. The real study retrains the branches per fold through
`split method-holdout` and is GPU work.

## Scope controls

- Test at most one or two serious challengers per component.
- Prefer a cheaper model when results are statistically indistinguishable.
- Do not tune on the final test set or the evaluation-only MNW benchmark.
- Do not add interface work until the evidence pipeline is complete.
- Treat a negative fusion result as a valid final result.
