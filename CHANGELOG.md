# Changelog

This file records material changes to the software and research protocol.
Experiment metrics belong in the experiment tracker, not this file.

## Update rules

- Add current work under `Unreleased`.
- Record changes that affect behavior, data, models, evaluation, or users.
- Do not record routine experiment runs or daily progress.
- Move entries into a dated version when the project creates a release tag.

## Unreleased

- Ran the first source-grouped leave-one-generator-out measurement. A frozen
  DINOv3 probe over full frames is at chance on FaceForensics++, macro 0.5387
  with all six intervals containing 0.5, against 0.7077 to 0.7839 for face crops
  through a fine-tuned model on the same corpus and protocol. The region of
  interest, not the encoder, is what decides it: FF++ manipulates the face and a
  720p frame resized to 224 leaves the face a few dozen pixels across.
- That corrects how the full-frame choice was argued here. It is right for media
  generated whole and wrong for face manipulation, so a system covering both
  needs both paths routed by content rather than one region chosen on principle.
- Fixed a NaN in the extractor: a clip short enough to yield one window produced
  a sample standard deviation over a single element. Zero is the honest value.

- Made leave-one-generator-out the only headline metric. There is no in-domain
  column any more: detecting a generator you trained on is not the problem, and
  a strong in-domain number has three times in this project turned out to be a
  shortcut rather than skill, most recently a 0.9997 on Veo 3 that was pure
  content confound. Model selection uses the held-out generator, and any change
  that buys in-domain accuracy at the cost of transfer is refused by the
  protocol rather than by judgement.
- Measured per-branch coverage on 314 Veo 3 clips. The face-dependent branches
  answer on 52.5 percent of them and the audio branch on 100 percent, because
  generated video that is not a talking head has no face to track. That is the
  no-face constraint measured rather than argued, and it is the one result from
  that run worth keeping: every AUC in it is confounded by content.

- Pointed the near-term work at DF26, the benchmark that matches this project's
  actual target: 2,691 single-person public-speaking clips, 271 real and 2,420
  from seven modern generators including Veo 3.1, Kling 3.0 and Grok. Its real
  clips are content-matched to its fakes, which removes the confound that made
  the public per-generator sets unusable here, and it is 3.6 GB behind an
  automatic gate.
- Added `scripts/fetch_df26.py`, which downloads it and flattens the upstream
  `fake/<scenario>/<generator>/` layout into the `real/` and `fake/<generator>/`
  shape the folder scorer already groups by, keeping the scenario in a sidecar
  index so the per-scenario analysis stays possible.
- Amended the frozen-probe recommendation from DINOv3 to Perception Encoder. The
  same method with three encoders on DF26 reads 72.7 macro for PE, 57.0 for
  DINOv3 and 55.6 for CLIP, so the encoder choice is worth 15 points.

- Reorganised `docs/research/architecture.md` around the constraint that
  actually matters now: a live demo where people generate fakes on the spot and
  test them. Section 0 is the narrow detector to build for that, three layers of
  provenance, a specialist trained on the demo distribution, and an abstention
  gate. The open-world research design follows it, still valid and not what wins
  a demo next month.
- Added `scripts/normalize_media.py`, which re-encodes every collected clip
  identically, real and generated alike, and strips metadata. This is the step
  that decides whether a collected dataset teaches manipulation or teaches
  post-processing. The failure is measured here: the detector finds re-encoded
  in-the-wild clips at 86 percent and raw generator output at 23 percent, so it
  learned the encoder. Verified on twelve mixed clips, which all came out
  h264 640x360 at 25fps.

- LAV-DF did not need re-caching after all. Its cache and the program run's used
  the same `ViewConfig` and the same preprocessing code, differing only in the
  `code_version` string, so the two hashes label byte-identical data. Verified
  rather than assumed: `scripts/verify_cache_equivalence.py` compared 60 clips
  spread across the 1,099 that both caches held, and all five arrays matched on
  every one. That stopped a five-hour GPU job two hours in.
- Added `views/equivalence.py`, a narrow table of hash pairs proven to label the
  same data, each naming its evidence. It unblocks the project's only fully
  audiovisual corpus, which could not be compared against any other corpus while
  its hash differed. The check accepts equal hashes and verified pairs and
  nothing else, and the tests pin that a near-identical hash is still refused.
- Stopped the FaceForensics++ leave-one-family-out sweep after three of six
  arms. FF++ carries no audio, so the sweep is visual-only and off the
  audiovisual path. The harness and the three measured arms are committed and
  the remaining arms can be run at any time.

- Added `scripts/score_media_folder.py`, which scores a folder of clips through
  the served engine and groups the result by source directory. It exists for the
  case the system fails hardest at: a number on clips from Veo, Sora or
  Higgsfield, which is the baseline every later improvement gets compared
  against. Detection rates carry Wilson intervals, and the distribution gate is
  reported beside the verdict rather than folded into it.
- Measured leave-one-family-out for the first family. FaceForensics++ Deepfakes
  reads 0.8912 [0.8610, 0.9225] when seen in training and 0.7771 [0.7365,
  0.8216] when held out, intervals that do not overlap. That decomposes the
  failure usefully: an unseen manipulation family costs 0.11 and still works, an
  unseen corpus reads 0.4920, and an unseen media class reads 0 of 10. The
  problem is the capture pipeline and the media class, not the manipulation
  method.

- Added `docs/research/architecture.md`, the design for an open-world
  synthetic-media detector. It reorganises the system around one conclusion: a
  jointly generated clip is internally consistent, so cross-modal disagreement
  cannot answer "is this real". That question moves to whether the signal
  carries the statistics of a physical capture chain, and lip-sync and emotion
  drop to deciding which kind of manipulation it is.
- The design is argued from this project's own measurements, including the ones
  that contradict its earlier assumptions: the cross-modal streams never
  measured correspondence, frozen DINOv3 lost to a fine-tuned EfficientNet,
  late fusion beat the cross-attention head, and the detector keys on
  post-processing rather than on manipulation.

- Cached FaceForensics++ c23: 6,992 of 7,000 clips, 8 undecodable, under the
  same code version as the program run so the preprocessing hash is identical
  and a model trained here can be scored zero-shot against the already-cached
  DFDC and Celeb-DF without re-caching.
- Recorded that FF++ c23 carries no audio track at all, which `ffprobe` confirms
  and the cache audit recorded as `missing_audio` on every clip. It can train
  the visual stream and nothing else, and pointing an audio-bearing stream at it
  yields an empty usable manifest rather than a silent error.
- Face-track quality is worse than FakeAVCeleb's, as expected for uncontrolled
  footage: of 7,000 clips, 1,089 had an unstable track and 387 low face
  coverage, leaving 4,259 train, 902 validation and 954 test clips over 670,
  142 and 145 source identities.

- Scored the LAV-DF lip-sync stream, trained in an earlier run and never
  measured in ROC-AUC. It reads 0.9969 with diagonal mass at 0.0592 against a
  chance of 0.0600, on matched windows cut from the same file. The corpus that
  forbids every cue except correspondence is solved without correspondence, so
  the finding no longer depends on FakeAVCeleb being permissive.
- Put confidence intervals on the MNW detection counts, which exposed a counting
  error: one MNW clip is genuine and was being counted as a detection. The set
  is 84 manipulated clips, not 85, and detection is 24 of 84. Named generators
  reach 23 percent [15, 34] and in-the-wild clips 86 percent [49, 97], intervals
  that do not overlap.

- Added `docs/research/datasets.md`, a survey of the corpora that would close
  the generative-video gap, with licences, sizes and access routes. It records
  two traps found while surveying: commercial generators embed watermarks that a
  detector will learn instead of the content, which is why a de-watermarked Sora
  benchmark exists at all; and several corpora are evaluation-only, which
  `data/guards.py` has to enforce in code before the data is touched.

- Extracted FaceForensics++ c23, which had been downloaded and left as a 17.9 GB
  zip. 7,000 clips: 1,000 originals and 1,000 each from six manipulation
  families. Added `data/faceforensics.py` and
  `scripts/build_ffpp_manifest.py`, built a source-disjoint split of 4,840 train
  and 1,032 validation and 1,128 test over 720, 154 and 154 identities, and
  started caching under the same code version as every other corpus so the views
  are interchangeable. This is the corpus the published cross-dataset tables
  train on, so it is what makes any number here comparable to one.
- Recorded that fully generated video is a separate problem this system is not
  built for, with the per-generator evidence: 0 of 10 on `vasa_1`, 0 of 4 on
  `raskai`. The motion result compounds it, since the model reads smoothness as
  authenticity and generated video is smooth.

- Added `docs/research/paper-source.md`, the single document to write the paper
  from, and `scripts/update_paper_source.py` which regenerates its tables from
  the artifacts under `runs/`. Prose holds the judgement, generated blocks hold
  the numbers, and a table that says "not generated yet" names the command that
  would fill it. A hand-typed number goes stale the first time a run repeats.
- Retrained lip-sync at a fifth of the encoder learning rate to test whether
  unfreezing the encoders was what collapsed it. It was not: validation loss
  quadruples at epoch 2, while the encoders are still frozen. The head overfits
  on its own. Both runs select epoch 1, so the shipped model is unchanged, and
  the hypothesis is recorded as wrong.
- Both cross-modal streams, not just one, sit on chance diagonal mass for every
  epoch of every run: emotion 0.3434 to 0.3457 against a chance of 0.3438 at 8
  query steps, lip-sync 0.0592 to 0.0595 against 0.0592 at 50.

- Measured the BatchNorm trade at full scale, which sharpens the earlier
  1,400-clip result and changes what to do about it. Live BatchNorm moves
  EfficientNet by +0.1276 on DFDC and -0.5430 in-domain, to 0.4557, which is
  below chance. The frozen arm ships. The live arm is not a detector that
  generalizes better, it is a broken one that ranks slightly above chance out of
  domain, and the value of the measurement is the size of the lever.
- DINOv3 is the control and behaved as a frozen ViT must: identical to four
  decimals on every epoch of both arms, because it has no running statistics.
- Recorded that the visual stream has two correct in-domain numbers, 0.9987
  against `video_fake` and 0.9719 against `clip_fake`. A clip with a real video
  track and spoofed audio is one and not the other. They agree on DFDC, where
  every manipulated clip has a manipulated video track.
- Added `scripts/score_batchnorm_arms.py`. Both arms of both visual streams are
  kept under `checkpoints-frozen-bn/` and `checkpoints-live-bn/` and both are in
  MLflow.

- Added `docs/research/paper.md`, the draft. Its claim is the evaluation method
  rather than the architecture or the accuracy: the negative answer to the
  research question is reported, and so is how close the project came to
  reporting the opposite.
- Corrected the MNW claim while sourcing it for the paper. It was quoted as "29
  percent detection on in-the-wild generators"; the file says 25 of 85
  manipulated clips overall, and the in-the-wild clips are the ones it detects
  best at 7 of 8. The failure is concentrated in the named generators, worst on
  `vasa_1` at 0 of 10.

- Added `scripts/run_fusion_ablations.py` and ran the two ablations the research
  design specifies and that had never been run. Late fusion over calibrated
  logits reads 0.5687 [0.5279, 0.6052] on DFDC against the deep head's 0.5411
  [0.4984, 0.5828], and they are indistinguishable in-domain, so the embedding
  is not carrying what it was expected to carry. Abstaining on partial coverage
  moves DFDC by 0.0007 and costs 2.4 percent of the clips: the policy is still
  right, but not for accuracy reasons.

- The result registry in `docs/research/result-traceability.md` has rows, and
  they are generated by `scripts/update_result_registry.py` from the artifacts
  they point at rather than typed in. A hash typed by hand stays put while the
  artifact moves, and a stale hash still reads as evidence.
- Recorded a third registry status, `provisional (1 seed)`. The frozen protocol
  asks for three training seeds; every Design B stream was trained on one. The
  rows say so rather than rounding up to `accepted`.
- `docs/research/findings.md` records seven provisional findings and the five
  claims they supersede, each with the reason the first reading was wrong.
- Added `scripts/backfill_mlflow.py`. Design B was trained through direct CLI
  calls and tracking is wired to `ddf run --config`, so none of its streams had
  an MLflow run and none of its numbers was traceable. The backfill builds runs
  from the history files the trainer wrote, tagged `backfilled` with the source
  file and its SHA-256, so a backfilled run cannot be read as a live one.
- Added `scripts/batchnorm_sweep.py`, which reproduces the BatchNorm arms under
  `runs/` instead of in a scratch script. The finding was quoted in three
  documents with no artifact behind it.

- The Evidence gate takes a video, a photograph or a sound file, and routes each
  to the streams it can actually drive. It took four video suffixes before, so a
  photograph was refused at the uploader and the visual stream could never see
  one. The gate now says which streams will run before Analyze is pressed.
- The gate serves the five Design B streams through `MultimodalEngine` and the
  feature-level head, replacing the single frozen visual model. A checkout with
  no trained head still answers, through the visual baseline, and the result
  says which of the two produced it rather than leaving them
  indistinguishable.
- Added `scripts/fit_gate_fusion.py`, which fits the served head under the
  deployment presence patterns and chooses a decision threshold per media kind
  on rows it was not fitted on. One threshold cannot serve all three: an image
  drives the visual stream alone, and its cut-off came out at 0.76 against 0.50
  for a fused video and 0.26 for sound alone. The thresholds live in the
  checkpoint, so a head cannot be paired with someone else's cut-off.
- A stream the head knows about and the server has no checkpoint for no longer
  stops the gate. The head masks an absent stream to exactly zero, which is what
  it already does for a clip that could not be read, and the missing stream is
  named in the blockers on every verdict: a fusion of three where five were
  expected is a weaker claim and nothing else on the page shows it.
- The verdict panel now lists the streams the upload could have driven and
  whether each one produced a score. A stream that should have run and did not
  is the explanation for the verdict, and it was previously invisible: the panel
  showed a hardcoded single "visual" row whatever ran.
- Stream rebuilding moved from `scripts/score_streams.py` into
  `deepfake_detection.fusion.stream_loading`, so the dashboard and the scoring
  script rebuild a checkpoint the same way. A stream rebuilt with the wrong
  temporal model loads most of its tensors and then computes a different vector,
  silently, which is what a second copy of that logic would eventually do.

- Measured that freezing BatchNorm during fine-tuning trades cross-corpus
  transfer for in-domain accuracy, which an earlier entry recorded as a clean
  win. One variable at a time on 1,400 clips, each arm scored on the full DFDC
  set: frozen BatchNorm reads 0.9197 validation and 0.5005 DFDC, live BatchNorm
  reads 0.5267 and 0.6482. Changing the epoch budget from 10 to 3 moved DFDC by
  0.005, so the epoch count is not the lever.
- Added `scripts/stream_correlation.py` and `scripts/run_deep_ablation.py`, which
  report whether two streams fail on the same clips and whether fusing a subset
  beats the best single stream on each partition.

- Every manifest now carries a `label` column, derived on load when the file
  does not have one. Widening dataset discovery to accept `clip_fake` and
  `manipulation_type` manifests made four datasets visible in the picker and
  then crashed it with `KeyError: 'label'` when one was opened: discovery
  accepted the new shapes and the rendering still assumed the old one.
- A manifest with no label information gets -1, not 0. A clip nothing is known
  about must not read as confirmed real.

- The dataset picker now offers all five datasets on disk. It listed only
  FakeAVCeleb, because discovery recognised a dataset only through a raw
  `meta_data.csv` or a manifest carrying a `label` column, and only FakeAVCeleb
  has either. Celeb-DF, DFDC, LAV-DF and MNW were fully downloaded and
  invisible.
- Three things had to change. A manifest is recognised by `clip_id` and
  `video_path` plus any of `label`, `clip_fake` or `manipulation_type`, which
  are the three shapes this pipeline actually writes. Discovery takes extra
  manifest directories, because the pipeline writes splits into the run that
  produced them rather than into `data/`. And a manifest with no owning raw drop
  resolves to the top-level `data/` directory its clips live in, instead of to
  whatever folder the CSV sat in.
- A manifest whose clips are not under `data/` is skipped. Older runs carry
  these, and each was otherwise inventing a dataset named after its run
  directory.

- Choosing a checkpoint on the Visual stream page now sets the architecture
  controls to what that checkpoint was trained with, instead of printing what to
  set them to. Loading is non-strict by design, so a mismatch does not raise:
  the backbone lands, the temporal model silently stays randomly initialised,
  and the page then shows a confident number computed from half a trained model.
- Adoption happens once per file rather than on every rerun, so the architecture
  panel stays editable for anyone deliberately trying a different temporal model
  against the same weights.
- `checkpoints.architecture` also reports the embedding width, read from the
  projection layer. A Design A branch reports None there, correctly: it ends in
  a classifier and has no shared width.

- `CacheStore.load` now takes a `views` argument, and the datasets ask only for
  the views they read. A cached clip holds four and they are not small: 9.19 MiB
  for the visual view, 7.18 MiB for the mouth crops, about 20 MiB in total
  against the one view a branch actually uses. Default is still every view.
- This was a blocking failure, not an inefficiency. Visual stream training died
  three times inside the `sync_video_view` allocation of a run that never reads
  mouth crops: twice in a DataLoader worker, where the retry caught it, and once
  single-process after two and a half hours, where nothing did and the stream
  was lost.

- Moved the dashboard's tuning controls and reference detail behind `Advanced`
  expanders, so each page opens on the thing it answers. Fusion leads with a
  plain-language verdict and two numbers; Preprocessing leads with the steps and
  hides the knobs that tune them; the Visual stream page leads with the pictures
  and hides the architecture controls and the shape ladder. Every widget still
  exists and returns the same value, so no step's behaviour changed.
- The Fusion and Explainability pages now open with a sentence rather than a
  table: whether combining branches helped, and whether the model is even across
  manipulation methods.

- Unlocked the Fusion and Explainability pages. Both were dimmed and
  unclickable because the only fusion artifact was a software fixture and there
  was no trained model to explain. `runs/program-20260906` holds a fusion model
  fitted on out-of-fold branch features with an ablation over every subset, so
  both pages now read recorded results.
- The Fusion page leads with the ablation rather than the fused score, and
  reports both verdicts: all three branches beat every single branch in-domain
  and do not on DFDC. Showing only the in-domain verdict would answer the
  easier question. It also shows the raw branch scores, where sync ranks below
  chance.
- The Explainability page builds the one specified view that has data,
  per-method and per-manipulation-type accuracy with subgroup coverage, and
  states plainly that Grad-CAM and embedding shift are not built. Filling them
  with something that looked like an explanation would be worse than the lock.
- Added `dashboard/lib/results.py` to read run artifacts. Nothing raises for a
  missing file: a run that has not produced an artifact is a normal state, and
  the page names the file it wanted.
- `locked.LOCKED` is now empty. The machinery stays for the next unbuilt stage.

- Fusion training now accepts `holdout` feature rows as well as `oof`. Streams
  train on the training partition and have never seen validation, so validation
  rows are as leakage-free as cross-fitted ones while costing one training run
  per stream rather than one per fold. Cross-fitting the five Design B streams
  would have tripled a 13-hour run. Mixing the two roles in one store is
  refused, and the test partition is still refused outright.
- `scripts/score_streams.py` exports three partitions: `holdout` for fitting
  fusion, in-domain test, and DFDC.

- Implemented `StreamConfig.freeze_batchnorm_on_finetune`, which the config had
  documented and no code applied. It was the reason the EfficientNet visual
  stream sat at chance. Measured on 1,400 clips, one change at a time:

  | Variant | Best validation AUC |
  | --- | --- |
  | as shipped | 0.5154 |
  | no gradient checkpointing | 0.4620 |
  | BatchNorm frozen while fine-tuning | 0.9058 |
  | GRU rather than BiLSTM | 0.9028 |

  Fine-tuning rewrote all 49 running means and variances from batches of eight
  drawn by an inverse-frequency sampler, matching neither the ImageNet
  statistics the weights came from nor the distribution validation is drawn
  from. Training reads batch statistics and looked fine; validation reads the
  running statistics and did not.
- Visual stream training now runs with gradient checkpointing off. Peak VRAM
  measured 394 MiB of 16,303, so it saved nothing and perturbed BatchNorm.
- DINOv3 was never affected: a ViT has no running statistics, which is why it
  trained to 0.9559 through the identical path.

- Stream training now selects its checkpoint on validation ROC-AUC, not
  validation loss. The old criterion kept a checkpoint that scores below
  chance: `visual-efficientnet` reached its lowest BCE at epoch 1, while the
  backbone was still frozen, and that checkpoint measured 0.4668 in-domain.
  BCE scores calibration and punishes a confident mistake hard, so a model that
  ranks well but is overconfident loses to one that hedges and ranks badly.
  Every objective here is stated in ROC-AUC.
- Loss is still recorded every epoch. It shows a run that is diverging rather
  than merely miscalibrated, which AUC alone would hide.
- Added `training/ranking.py`. It averages ranks inside a tie group, so an
  untrained model emitting one logit for every clip scores 0.5 rather than
  whatever order argsort produced, and returns NaN for a single-class split
  rather than 0.5, which would read as a real measurement of a model at chance.
- `scripts/score_streams.py` rebuilds its feature store instead of appending, so
  re-scoring after training another stream no longer fails on a duplicate key.

- Added `scripts/score_streams.py`, which scores every trained Design B stream
  on the held-out in-domain test set and on DFDC, and writes the feature store
  `ddf train fusion --model deep` reads. Both come from the same forward pass.
  Training records loss only, and loss does not answer the question the
  objectives are stated in: the visual branch reached 0.9742 in-domain and
  0.7583 on DFDC from loss curves that looked alike.
- Each stream is rebuilt from its own history file rather than from flags. A
  stream rebuilt with the wrong temporal model would otherwise load most of its
  tensors and compute a different vector, which is the failure the dashboard
  checkpoint picker had to be taught to report.

- Added `fusion/stream_export.py`, which writes Design B stream embeddings into
  the same feature store the branches use, so `ddf train fusion --model deep`
  can read them. `export_features` could not be extended to cover it: it is
  welded to three branches with three different call signatures, while streams
  are uniform and take a loop.
- The exporter imports its view selection and waveform normalisation from
  `data.datasets` rather than restating them. A first draft restated the
  normalisation and got it wrong, using peak where the dataset centres and
  divides by standard deviation. A stream trained on one and scored on the other
  raises nothing; the only symptom is a disappointing number.

- Added `ddf train fusion --model deep`, which reads the whole embedding per
  stream instead of one calibrated scalar, and `training/fusion.py` to fit it.
  The validation slice it stops early against is held out by source identity,
  never by clip: FakeAVCeleb splits at random put the same speaker on both
  sides, which would undo the cross-fitting of the branch checkpoints one layer
  higher up.
- Deep fusion assembles rows non-strictly, so a clip one stream cannot read is
  zero-filled with a presence flag rather than dropped. The scalar path keeps
  strict assembly, which is what stops a missing branch becoming a zero logit
  there.
- Added `scripts/train_design_b.ps1`, which trains the three visual streams and
  the two audiovisual streams in sequence. It attempts loader workers and
  retries single-process, matching `run_program.ps1`: at workers=0 the cache
  loads at 65 ms per clip, an epoch over 7,637 clips is 8 minutes of pure I/O,
  and the GPU measured 15 percent utilisation waiting for it.

- The dashboard's checkpoint picker now finds trained weights. It looked only in
  a top-level `checkpoints/<stream>/` directory that has never existed in this
  repository, while `ddf run` writes into `runs/<run>/checkpoints/` and nothing
  copies between them. Every stream page therefore offered untrained weights
  only, with twenty trained checkpoints on disk. Discovery now scans the runs
  tree and labels each file by the run that wrote it, because the same
  `fold0-visual.pt` name recurs in every run.
- The picker reads the temporal model out of a checkpoint's tensor shapes and
  names the setting to match. A GRU holds three gate matrices per layer and an
  LSTM four, which is the only record of which one a bare state dict carries.
  Loading is non-strict, so a wrong setting was reported as four tensors of the
  wrong shape and left the temporal model random.
- Added the two unidirectional temporal options. `ddf train visual` builds a
  unidirectional GRU, so no setting in the dashboard could match its
  checkpoints.
- A branch checkpoint loading into a stream is now reported as the design
  difference it is, not as a configuration error. `ddf train visual` ends in a
  classifier straight to one logit; a stream ends in a projection to the shared
  width. The backbone and temporal model transfer, 362 tensors of 364, and the
  head cannot.

- Added `ddf train visual-stream`, which trains any of the three visual
  backbones (DINOv3, EfficientNet-B0, Xception) through `streams/visual_stream.py`
  and projects each to a shared width, so several can be concatenated for
  feature-level fusion. `ddf train visual` remains welded to EfficientNet-B0 and
  is unchanged.
- `--freeze-backbone` keeps a backbone frozen for the whole run. This is the
  point of including DINOv3: a fine-tuned backbone can absorb the corpus it
  trains on, which is how the EfficientNet baseline reached 0.9742 in-domain and
  then called 130 of 155 genuine Celeb-DF videos fake.
- `parameter_groups` now names the modules it treats as encoders instead of
  hardcoding the two cross-modal ones, and drops the encoder group when nothing
  in it is trainable. An optimizer group with an empty parameter list raises,
  which every frozen-backbone run would have hit.
- Added `fusion/deep.py`: concatenate stream embeddings, project each to a
  common width, then an MLP. Late fusion collapsed each branch to one calibrated
  scalar and discarded 255 of every 256 dimensions; with three scalars it put a
  -3.019 coefficient on a branch scoring 0.4364 ROC-AUC, below chance, because
  calibration could invert it into an in-domain gain that did not survive a
  change of corpus. Stream dropout zeroes a whole stream at random during
  training, so the head cannot build its decision on one stream that way.
- `AssembledFeature` now carries `branch_embeddings` alongside the scalar
  logits. Additive, so the existing late-fusion path is unaffected.

- Added LAV-DF and DFDC adapters. LAV-DF cuts a matched pair from every forgery,
  a window on the manipulated span labelled fake and a window that misses it
  labelled real, both from the same file. The two share a speaker, a codec and a
  re-encoding pass, so a model cannot separate them by recognising the
  generator's compression. DFDC is the cross-corpus test: it is the only
  audio-bearing dataset here that is not built on VoxCeleb2, which FakeAVCeleb
  and LAV-DF both are.
- `load_manifest` now keys its duplicate quarantine on the window as well as the
  file, so two windows cut from one recording are kept as distinct clips. The
  original guard is unchanged for everything else, and FakeAVCeleb's duplicate
  listings still quarantine.
- `ddf train stream` gained `--encoder-learning-rate`, defaulting to 5e-6
  against the head's 1e-4. A single rate across a randomly initialised head and
  100M pretrained encoder parameters drove training loss to 0.039 while
  validation climbed to 2.56.

- Built the audiovisual cross-attention stream, the first half of the streams
  design the dashboard has specified since it was written. Audio queries video,
  and the residual between the query and what video explains is the mismatch
  signal fusion will read. `ddf train stream` trains it. Pooling the attended
  vector instead of the residual was tried first and moved the embedding by 2e-7
  when the audio was shifted 320 ms, so it could not have trained at all.
- Added `diagonal_mass`, the fraction of attention mass near the diagonal,
  recorded every epoch and never optimised. It checks whether a stream learned
  real synchronisation independently of whether it classifies well: a falling
  loss beside a flat diagonal mass is a shortcut.
- Wav2Vec2 rather than Whisper on the audio side for now. Whisper's encoder
  takes 30 seconds of log-mel and rejects anything else, so a 2 second window
  would be 28 seconds of padding. AV-HuBERT stays deferred because fairseq pins
  PyTorch to 2.0 and would break the CUDA stack.
- Added `ClipRecord.sync_start_sec`, so a dataset that knows where a
  manipulation sits can place the synchronisation window on it. LAV-DF hides a
  0.8 to 1.6 second forgery inside a longer clip, which the fixed window near
  the clip start would usually miss. The cache key gains the field only when it
  is set, so existing entries keep their fingerprints.
- Added `ddf cache build --skip-cached` and `--shard I/N` with `ddf cache
  merge`, making a long cache build resumable and shard-parallel.
- Added `ddf evaluate branch`, replacing an evaluation script that had been
  copy-pasted between run directories, and `ddf manifest usable`, replacing an
  untracked filter that removed clips a branch cannot read. Both report
  abstentions rather than dropping rows from the denominator.
- Added `ddf manifest from-meta`, `ddf split subsample` and `ddf handoff
  update`, and adapters for Celeb-DF-v2, MNW and LAV-DF. MNW is refused by
  `data/guards.py` anywhere outside external evaluation.
- Constrained `evidence_scope` to a validated vocabulary and moved the frozen
  baseline's identity out of Python into `configs/frozen-baseline.json`.

- Rebuilt the dashboard around the teaching frontend. Its pages walk a clip
  through the pipeline stage by stage with configurable models, and the frozen
  provenance-checked baseline now lives on one Evidence gate page inside that
  navigation. Fusion and Explainability stay registered but disabled, and audio
  and sync remain prototype pages.
- Added `deepfake_detection.preprocessing.ops`, the per-step preprocessing
  functions shared by the pipeline and the dashboard, adapting the existing face
  detectors rather than duplicating them.
- Added `deepfake_detection.streams`: one configurable visual stream over
  EfficientNet-B0, Xception and DINOv3, and an activation tracer that reports
  what each backbone stage computed.
- Added a read-only MLflow reader for the dashboard and replaced the Weights and
  Biases checkpoint path with an MLflow one. This project tracks with MLflow
  only.
- Added `librosa` to the media extra and `matplotlib` to the dashboard extra.
- Added an obstacles reference and a dashboard architecture document.
- Added a ten-page local teaching dashboard. The visual path is provenance
  checked. Experiments reads strict local FakeAVCeleb development-validation
  evidence. Audio and sync remain prototype pages, and fusion remains locked
  behind a software fixture.
- Updated the dashboard instructions and project handoff with local artifact,
  MLflow, and raw-data limitations.
- Replaced the former external benchmark with the evaluation-only
  Microsoft-Northwestern-WITNESS benchmark. Defined one ignored four-dataset
  data root and added versioned experiment and result traceability records.
- Added a resumable FaceForensics++ downloader that removes partial transfer
  files after failures and preserves completed payloads.
- Required CUDA for visual, audio, and synchronization research training. Added
  MLflow metrics for training throughput and peak allocated GPU memory.
- Added a maintained project handoff with dataset, artifact, MLflow, dashboard,
  verification, and resume instructions.
- Added a provenance-checked visual-only inference mode and made it the local
  dashboard default for the trained development baseline. The dashboard labels
  its evidence scope and cross-dataset limits.
- Required paired MTCNN and YuNet research reports, usable tracking evidence,
  clean pinned environments, exact input report hashes, and source run IDs for
  detector decisions. Disabled the unbound downstream scalar tie-break.
- Bound detector parsing and report hashing to one immutable, single-read byte
  buffer and removed caller-supplied report digests.
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

- Cache indexes now store paths relative to the index directory, so training
  can load caches created below that directory.
- Sync token resizing now uses deterministic nearest timestamp selection. This
  permits CUDA backward while deterministic algorithms remain required.
- Documentation checks now skip external files below the ignored data root.
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
