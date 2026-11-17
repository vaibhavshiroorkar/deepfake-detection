# P-001: Shortcut-neutral audio-visual detection, attribution and false-alarm control

Registered 2026-11-13, before any model in this study is trained or scored on neutral data.
Results go in `results/<experiment-id>/` in later commits. Changes are new dated amendments
at the end of this file, committed before the run they affect. Nothing above an amendment is edited.

## Questions

- **Q1 (protocol).** How much of the real-or-fake signal in FakeAVCeleb, LAV-DF and DFDC can a model get from file and audio-edge cues alone? Does neutralization remove it?
- **Q2 (detection).** After neutralization, does fusing the visual, audio and AV-alignment paths beat the visual path alone on held-out corpora?
- **Q3 (modality).** Can the model say which stream was manipulated, and does that transfer between FakeAVCeleb and LAV-DF?
- **Q4 (false alarms).** Does a threshold calibrated on real clips from the target domain hold the false-positive rate where a source-calibrated threshold fails?

## Data and splits

- **Training corpora:**
  - FakeAVCeleb v1.2. Neutral copy from `scripts/neutralize_dataset.py`.
  - The LAV-DF train partition. Neutral copy.
- **Split:**
  - 70/15/15 by source identity, seed 17, `build_source_split(group_by="source")`.
  - FakeAVCeleb's identity graph is fully connected, so we also report the identity-strict subset beside every in-domain number.
  - Manifest targets that are file names are ignored (fixed 2026-11-13).
- **Locked targets.** Never used for training, selection or thresholds (`data/guards.py`):
  - DFDC
  - Celeb-DF v2 (visual only)
  - MNW (read once, last)
  - DeepSpeak v2 and AV-Deepfake1M test, if access arrives before the freeze.
- **Seeds:** 17, 29, 43 for every trained model.

## Primary tests

All intervals are 95%, from 1,000 bootstrap resamples of source identity. "Holds" means the interval excludes the stated bound.

| ID | Test | Success | Failure |
|---|---|---|---|
| H1 | Shortcut ceiling (`scripts/shortcut_audit.py`, all cues, grouped 5-fold) on neutral copies, clip label | AUC <= 0.60 on each training corpus | AUC > 0.60: neutralization incomplete; report which cue remains |
| H2 | Fusion minus visual-only, ROC-AUC, paired, mean of 3 seeds | Holds above 0 on at least 2 of {DFDC, LAV-DF test, FakeAVCeleb test-strict} | Otherwise reported as a negative result; no further tuning |
| H3 | Manipulated-modality macro-F1 (audio-fake, video-fake as two binary heads), train FakeAVCeleb test LAV-DF and the reverse | Both directions above the clip-label-only baseline, interval excludes 0 | Report as not transferring |
| H4 | Realized FPR with the real-only conformal threshold, alpha = 0.05, n_cal = 200 real target clips | FPR <= 0.07 on every target | Report the targets where it fails |

Holm correction across H2 to H4. H1 is a precondition: if it fails, H2 to H4 still run and are reported with that caveat.

## Fixed choices

These are fixed now so no test set can choose them:

- **Visual path:** EfficientNet-B0 + GRU, live BatchNorm. The earlier choice of live BatchNorm was made on DFDC. It is kept to avoid a new selection, and the history is disclosed.
- **Audio path:** Wav2Vec2 Base, attentive pooling.
- **Sync path:** frozen AV-HuBERT Base, AlignmentHead trained on real clips only (`avsync/heads.py`).
- **Fusion:** logistic regression on out-of-fold branch logits (primary). The StreamFusion MLP with per-modality heads is secondary.
- **Thresholds:** balanced accuracy on validation (`ddf threshold --split val`), except H4.
- **No ensembles across seeds or architectures** in primary results.

## Reported for every run

ROC-AUC, PR-AUC, balanced accuracy, EER (interpolated), FPR and FNR at the frozen threshold, FPR at 95% TPR, Brier score, ECE, coverage. Per-clip predictions are committed under `results/`.

## Amendment A1, 2026-11-13, before any P-001 model is trained

- **Training subsample.** The training partition is thinned with `ddf split subsample --fake-ratio 3 --seed 17`. This keeps every clip with real video (RealVideo-RealAudio and RealVideo-FakeAudio). It samples video-fake clips at 3 per real-video clip, spread across (category, method) strata. Validation and test keep their natural mix. Reason: FakeAVCeleb has about 500 real identities against about 20,000 fakes, and the earlier study used the same rule.
- **Audio path: three seeds.** The audio path is trained at seeds 17, 29 and 43. The earlier driver trained it at seed 17 only.
- **Sync path input.** Each clip gives up to three 2-second windows at 0, 2 and 4 s, with a face in at least 80% of frames. A clip's score is the mean of its window scores. Clips with no usable window are abstentions and are counted in coverage.
- **LAV-DF in H1.** LAV-DF is audited per file (`scripts/build_lavdf_file_manifest.py`), not per window. A window manifest lets one file supply both labels.

## Amendment A2, 2026-11-13, before any P-001 result is read

The data section listed FakeAVCeleb plus the LAV-DF train partition as training data. H2 also treats the LAV-DF test set as a cross-corpus target, and H3 needs a model trained on each corpus alone. These contradict each other. Resolved as follows:

- **Primary models (H1, H2, H4)** train on neutral FakeAVCeleb only. LAV-DF test and DFDC are then both cross-corpus targets for H2.
- **Reverse models (H3 only)** train on the neutral LAV-DF train partition only, with the same recipe and seeds. They are scored on FakeAVCeleb test.
- **LAV-DF** is used per file. A file's labels come from `modify_video` and `modify_audio`. Its sync windows start at 0, 2 and 4 s, like FakeAVCeleb.
- **No model trains on both corpora** in P-001.

## Amendment A3, 2026-11-13, before any P-001 result is read

LAV-DF is sampled by original recording (`scripts/sample_lavdf.py`, seed 17). Each sampled original comes with all its edits, so real and fake clips share speaker, room and camera.

| Partition | Originals | Files | Use |
|---|---|---|---|
| test | 1,000 | 3,823 | Cross-corpus target for H2 and H3 |
| train | 1,500 | 5,684 | Reverse models for H3 |
| dev | 300 | 1,089 | Validation for the reverse models |

The DFDC target is the existing 3,032-clip subset (`runs/streams-20260905/dfdc-manifest.csv`), neutralized like the training data.

## Amendment A4, 2026-11-13, after H1 attempt 1 and before any model is trained

**H1 attempt 1 failed.** On the version-1 neutral copies, the shortcut ceiling was 0.933 (clip label). The record is in `results/shortcut-audit/fakeavceleb-neutral-v1`. Re-muxing exposed each source's AAC priming differently, so face-swapped clips gained 28 ms of leading silence. Duration and trailing silence also still carried the label.

The neutralizer is now version 4 (`preprocessing/neutralize.py`):

- Audio is decoded once, cut on sample indices and scaled to a fixed RMS level in numpy, then muxed as a WAV.
- Cuts start on a whole video frame.
- **Every clip is exactly 3.00 s long, starting at the first sound.** Clips with less than 3.00 s of sound are refused. Refusals are counted per category, because they remove short generated-speech clips more often than real ones.
- Dither at -66 dBFS removes the exact digital zeros that speech synthesizers leave in pauses.

On a 600-clip validation sample (150 per category, not used for anything else), version 4 gave ceilings of 0.598 (clip), 0.489 (video) and 0.702 (audio). 29 of 300 fake-audio clips and 0 of 300 others were refused.

H1 is re-run once on the full set with version 4. Its result is reported whether it passes or not. The audio-label ceiling is reported beside H3, because a residual audio cue weakens any modality-prediction claim.

**Sync windows** become 2 s windows starting at 0 and 1 s (`--step 1`), because every clip is now 3 s. This replaces the 0, 2 and 4 s rule of A1.

## Amendment A5, 2026-11-15, before HiFi-AVDF is downloaded

HiFi-AVDF (Peng et al., CVPR 2026 Findings) is added as a **locked cross-corpus target**. It is public (huggingface.co/datasets/JielunPeng/HiFi-AVDF, MIT licence, 6.5 GB). It pairs each real video with a generated copy from one of six commercial generators: Kling 2.5, PixVerse V5, Seedance 1.0, Sora 2, Veo 3.1 and Wan 2.5. The 2026-11-15 decision to skip it is reversed, because access turned out to need no request.

- **Use.** H2 (fusion vs visual-only) and H4 (real-only false-alarm control) are scored on it, with results per generator and pooled. It never trains, selects or calibrates anything, except the real-only calibration split that H4 defines.
- **Audit first.** `scripts/shortcut_audit.py` runs on the raw files before any model is scored, grouped by real-fake pair. The neutralized copy is audited the same way. Both ceilings are reported beside every HiFi-AVDF number.
- **Same pipeline.** Neutralizer v4 with fixed 3.00 s clips, with refusals counted per generator.
- **HAVIC baseline.** HAVIC's released fine-tuned weights (huggingface.co/JielunPeng/HAVIC) may be scored on HiFi-AVDF, DFDC and LAV-DF. They are not scored on our FakeAVCeleb test: HAVIC fine-tuned on a random, ungrouped FakeAVCeleb split, so its weights have almost certainly seen our test identities.
- KoDF stays skipped.

## Amendment A6, 2026-11-15, before any KoDF or AV-Deepfake1M data exists here

- **KoDF** is a locked cross-corpus target if access is granted. The 2026-11-15 skip is reversed. It enters with the same rules as HiFi-AVDF (A5): raw and neutral shortcut audits before scoring, neutralizer v4, and results per method and pooled. If access is refused, KoDF is reported as unavailable, not dropped silently.
- **AV-Deepfake1M and AV-Deepfake1M++** were requested on 2026-11-15, through the 2025 challenge and the gated Hugging Face repos.
  - AV-Deepfake1M is a locked target for H2. It is also a third corpus for H3, with reverse models trained on its training split under the A2 rules.
  - AV-Deepfake1M++ is a locked target for H4, the false-alarm test under shift.
  - Both get the shortcut audit first. Smeu et al. (CVPR 2025) already documented leading silence in AV-Deepfake1M.
- A dataset that arrives after the primary models are frozen is scored with those frozen models only. Nothing is retrained because a new target appeared.

## Amendment A7, 2026-11-15, before HiFi-AVDF is neutralized or scored

The raw HiFi-AVDF audit, while still running, showed frame size following the label. Real clips are 1280x720. Generated clips are 1920x1080 (Kling 2.5: 1920x1072). Neutralizer v4 kept native size, which was correct for FakeAVCeleb and LAV-DF (all 224x224), but would carry this cue into HiFi-AVDF.

- **Fixed frame size per corpus.** Every clip of a corpus is resampled to one frame size: the native size of its real clips (`--size`).
  - HiFi-AVDF: 1280x720, stretched, not padded. Padding would add black bars to Kling clips alone. Stretching changes Kling's aspect ratio by 0.4%.
  - FakeAVCeleb and LAV-DF: unchanged (already uniform), so existing neutral copies and the trained sync models stay valid.
  - DFDC: unchanged. Both classes mix 1920x1080 and 1080x1920 (portrait: 35% of fakes, 28% of reals). The neutral audit reports what orientation still carries.
- No model has been scored on HiFi-AVDF. Only the raw audit, which reads file properties, has run.

## Amendment A8, 2026-11-16, before any visual or audio model is trained and before any fusion is fit

The 2-day window became 3 days (11-16 to 11-18), so H2 runs now instead of in the full run. No visual or audio checkpoint exists yet, and no target has been scored by those paths. This fixes how the fusion is fitted and tested.

- **Inputs.** For each clip: the visual logit and the audio logit from `ddf evaluate branch`, and the sync logit, logit(p) of the sync probability. Fusion at seed s uses the three branches of seed s. Probabilities are clipped to [1e-6, 1 - 1e-6] before the logit, and each logit is standardized with the mean and SD of the fit set.
- **Fit set: FakeAVCeleb validation.** Training clips are excluded: each branch was fit on them, so their logits are in-sample. Validation chose each branch's epoch and threshold and nothing else. Branch logits there are slightly optimistic, and we report that.
- **What "out-of-fold" means here.** The fusion's own validation scores, which set its threshold, come from 5-fold cross-fitting with folds grouped by identity. The model applied to test and to the targets is refit on all of validation.
- **Model.** scikit-learn `LogisticRegression`, L2, C = 1.0, lbfgs, `class_weight="balanced"`. Nothing else is tuned.
- **Coverage.** A clip enters H2 only if all three branches score it (a face is found and the clip has audio). Visual-only is compared on exactly the same clips. Coverage is reported per target.
- **Statistic.** Per target and seed: ROC-AUC(fusion) minus ROC-AUC(visual-only). The H2 statistic is the mean of the three seed differences. Its interval comes from 1,000 bootstrap resamples of identities (seed 17). Each resample is applied to all three seeds at once. Identities are FakeAVCeleb `source`, the DFDC original video, the LAV-DF original recording, and the HiFi-AVDF real and generated pair.
- **Decision.** The one-sided p-value is the share of resamples with a mean difference at or below 0. Holm at 0.05 runs across the three H2 targets {DFDC, LAV-DF test, FakeAVCeleb test-strict}. H2 holds if at least 2 of the 3 survive. H4 is a bar check with no p-value, so the Holm family is H2's three targets; this narrows the earlier "H2 to H4" wording. HiFi-AVDF gets the same statistic, outside the success rule.
- **Dropped for this window.** The StreamFusion MLP (secondary) is not run.
- **Targets** are cached with the training preprocessing hash and scored by frozen checkpoints with their validation threshold files. They never fit, select or calibrate anything.

## Amendment A9, 2026-11-16, after the LAV-DF sync and H4 results were read, and before any H2 result exists

**What was found.** Neutralizer v4 keeps a fixed 3.00 s from the first sound. LAV-DF edits single words inside longer recordings, so the kept window often misses the edit. Of the 2,823 fake clips in the neutral LAV-DF test sample, 1,307 (46%) contain no manipulated time inside the kept window, 407 contain part of it, and 1,109 contain all of it (from `metadata.min.json` `fake_periods` and the neutralizer's cut log). Under the file label, the 1,307 are real footage labelled fake. FakeAVCeleb, DFDC and HiFi-AVDF manipulate whole clips and are not affected.

**Change.**

- **Window label for LAV-DF.** A neutral LAV-DF clip is fake if its kept window overlaps manipulated time by at least one video frame (0.04 s). Its video and audio labels follow `modify_video` and `modify_audio` under the same rule. Everything else about the clip is unchanged.
- **Primary for H2.** H2 on LAV-DF uses the window label. No H2 result exists yet, so this is decided before H2 is read.
- **Sync path and H4 on LAV-DF.** The file-label results were read before this amendment. Both are reported: the file label as preregistered, and the window label as a disclosed correction. Neither replaces the other in the record.
- **Audit.** The neutral LAV-DF shortcut audit is run under both labels.
- **Not changed.** Splits, models, thresholds, the calibration draw, and every other target.

**Why it was missed.** The cut log records each clip's offset, and A3 sampled LAV-DF by recording, but no check compared the kept window with the edit times. Handbook mistake 68 records it.

## Amendment A10, 2026-11-16, after the DFDC sync result was read, and before any H2 result exists

**What was found.** On neutral DFDC, the mouth cropper found a face in only 2,110 of 6,064 windows, so the sync path scored 921 of 3,032 clips and only 67 real ones. The cropper runs MediaPipe's Face Landmarker on the full frame. Its detector is built for faces that fill much of the frame, as in FakeAVCeleb's 224x224 clips. DFDC frames are 1920x1080 or 1080x1920 with faces of about 100 to 150 pixels. On 10 random windows with no face found, locating the face first with the YuNet detector already in the repository (`artifacts/models/face_detection_yunet_2026may.onnx`), then running the landmarker on a crop around it, found the face in all 10 (in 24 to 50 of 50 frames).

**Change.**

- **Located fallback.** When the landmarker alone finds a face in fewer than 80% of a window's frames (the existing `MIN_FACE` bar), YuNet locates the face on the first, middle and last frame. The landmarker runs again on a square crop 2.2 times the median face size around it, and the landmarks are mapped back to full-frame coordinates. The fallback is kept only if it finds a face in more frames. The mouth crop that follows is unchanged, and is scaled by eye distance either way.
- **Scope.** All three locked targets, all windows below the bar, with no use of labels. Windows that already passed are not touched, so their features stay byte-identical.
- **Training data unchanged.** The trained sync models are not retrained. FakeAVCeleb windows that fail the bar are not re-extracted; the models stay frozen.
- **Reporting.** The DFDC, LAV-DF and HiFi-AVDF sync and H4 results before A10 stay on record. Results after A10 are reported as the corrected version, with coverage per target. H2, which has not run, uses the A10 features.
- **H4 on DFDC.** Our DFDC subset has 194 real clips, fewer than the 200 that H4's calibration needs, so H4 still does not run on DFDC.

## Amendment A11, 2026-11-16, before any upgraded model exists: an exploratory backbone test

**Why.** A literature check for P-002 ([P-002 backbones](../../docs/research/P-002-backbones.md)) found published evidence that two frozen foundation backbones transfer far better than P-001's:
- CLIP ViT-L/14 for faces;
- XLS-R for speech. In one benchmark, the wav2vec 2.0 Base front end used by P-001's audio path scores near chance on In-the-Wild.

That evidence comes from raw files, which carry the shortcuts P-001 measured. This amendment asks whether the advantage survives on shortcut-neutral data. The idea came after P-001's visual and sync results were read. Every result under A11 is therefore **exploratory**: it supports no preregistered claim, and none of H1 to H4 changes.

**What runs.**

- **Visual:** CLIP ViT-L/14 (`timm` `vit_large_patch14_clip_224.openai`), frozen. A GRU head goes over 16 frames, through `ddf train visual-stream`. The cached frames are ImageNet-normalized, so they are converted to CLIP's own normalization before the encoder. A test checks the conversion.
- **Audio:** XLS-R 300M (`facebook/wav2vec2-xls-r-300m`), frozen for every epoch, with the existing attentive-pooling head, through `ddf train audio`.
- **Everything else as P-001:**
  - the same thinned training partition, validation and test splits, caches and preprocessing hash;
  - three seeds (17, 29, 43);
  - thresholds on validation;
  - the same locked targets, scored by frozen models;
  - the same identity-bootstrap intervals.
- **Deviations from the published methods,** fixed now: no LayerNorm tuning for CLIP, and no weighted sum of XLS-R layers. Both are reported as such.

**Reported.**
- Each upgraded path against its P-001 counterpart, on the identity-strict test and on each locked target, with paired identity-bootstrap intervals.
- No fusion is refit with the upgraded paths for the submission.

**Stop rule.** If the upgraded models are not scored on all three targets by 17 November, 23:59, A11 is reported as not completed. The submission does not wait for it.

## Amendments A12 to A14, 2026-11-17, after the H2 results and the two post-hoc H2 checks were read: three exploratory checks

**Why.** H2 holds by its rule, but two checks run after the verdict showed that no target shows fusion improving on fakes the visual path can see ([`results/p001-h2`](../p001-h2/README.md)). Three questions follow, each fixed here before its result exists. Every result under A12 to A14 is **exploratory**: none of H1 to H4 changes, and no threshold, model or verdict is refit on what they find.

### A12: does a fusion fit on other corpora transfer?

The paper says the A8 fusion trails single paths on LAV-DF and HiFi-AVDF *likely because* its weights were fit on FakeAVCeleb. A12 tests that guess with the branch logits that already exist (`runs/p001/h2-full/`).

- **Leave one corpus out.** For each target T in DFDC, LAV-DF (window label) and HiFi-AVDF: fit the A8 fusion (`fit_fusion`, unchanged) on the pooled clips of the other two targets, on the clip label, and score T. No corpus weighting.
- **Upper bound, not a detector.** On each T, 5-fold fusion cross-fit by identity on T itself (`cross_fit_scores`, unchanged). It uses T's labels, so it only shows whether the three paths hold complementary information on T.
- **Reported, per seed and as a mean over seeds with the A8 identity bootstrap (1,000 resamples, seed 17):** AUC of the leave-one-out fusion, of the upper bound, of the A8 fusion and of each single path on the same clips; the difference leave-one-out minus A8 fusion and leave-one-out minus each single path.
- **Reading, fixed now.** If the upper bound does not beat the best single path, the paths carry no complementary information on T and no weighting helps. If it does and the leave-one-out fusion comes close to it, the A8 weights were the problem. If it does and the leave-one-out fusion does not, transfer of fusion weights between corpora fails.

### A13: what does the visual path read on LAV-DF?

The P-001 visual path scores LAV-DF fakes whose video was never edited below the real clips (0.175 to 0.243). A13 measures what differs between those two groups in the frames the path sees.

- **Data.** The cached `visual_view` (16 face frames, converted back from ImageNet normalization) of every scored LAV-DF test clip in two groups: RealVideo-RealAudio and RealVideo-FakeAudio. FakeAVCeleb identity-strict test (real against FakeVideo clips) is measured the same way, to show which direction the path learned.
- **Six statistics per clip, fixed now:** sharpness (mean variance of the Laplacian), high-frequency share (spectral energy above half the Nyquist frequency), 8x8 blockiness (mean jump across 8-pixel boundaries over mean jump elsewhere), mean brightness, contrast (standard deviation of luminance), and temporal change (mean absolute difference between consecutive frames).
- **Reported:** for each statistic, the AUC separating the two groups with an identity-bootstrap interval, and its Spearman correlation with the visual path's probability (mean of three seeds) within the real-video clips; a logistic regression on all six, cross-fit by identity, as the share of the group difference these statistics can explain.
- **Reading, fixed now.** A statistic is a candidate only if its group AUC lies outside 0.4 to 0.6 and its correlation with the visual score is at least 0.3 in size. A candidate is a description, not a cause. If none qualifies, the difference is reported as unexplained.

### A14: how much of the audio path's FakeAVCeleb score is a leftover file cue?

The P-001 audio path scores 0.952 to 1.000 on the FakeAVCeleb test, while the neutral audio-label file-cue ceiling is 0.793. The shortcut audit names two remaining audio cues: loudness after AAC encoding and small differences at the clip edges.

- **Perturbations of the cached `audio_view`, applied at scoring time to the frozen P-001 audio models, fixed now:**
  - loudness: every clip's valid samples scaled to the same RMS level (-23 dBFS);
  - edges: 250 ms cut from both ends of the valid samples, then zero-padded back to the cached length;
  - both together.
- **Data:** FakeAVCeleb identity-strict test and full test, audio label, three seeds.
- **Reported:** AUC per seed under each perturbation, with identity-bootstrap intervals, beside the unperturbed AUC.
- **Reading, fixed now.** If AUC under both perturbations stays at 0.95 or above, the named cues do not drive the score. If it falls toward the 0.793 ceiling or below, part of the score came from them. Cues the audit did not name are not covered.

**Stop rule.** A12 to A14 are dropped from the submission if not finished by 19 November, 18:00.

## Amendments A15 to A18, 2026-11-17, 20:40, before any of their results exists: four exploratory checks run overnight

**Why.** Vaibhav chose these on 11-17 to close four gaps the paper names: H3 has not run, the sync mechanism is "evidence, not proof", speaker overlap with AV-HuBERT's pretraining is unmeasured, and H1 missed its bar with video bitrate as the leading residual cue. Every result under A15 to A18 is **exploratory**: none of H1 to H4 changes its verdict, and no threshold, model or earlier result is refit on what they find. Stop rule: 18 November, 08:00; anything unfinished is reported as unfinished.

### A15: H3 in one direction, FakeAVCeleb to LAV-DF

H3 needs both directions. The reverse needs models trained on LAV-DF, whose train partition is not yet neutralized, so A15 runs only FakeAVCeleb to LAV-DF and H3 stays **not run** as preregistered.

- **The two binary heads** are the P-001 visual path (trained on the video label) and audio path (trained on the audio label), at each seed's FakeAVCeleb validation threshold, as committed in `results/p001-branches/*-lavdf-test-predictions.csv` (`predicted`). Nothing is refit or rethresholded.
- **Data.** LAV-DF test, clips scored by both heads. Labels per stream from the window manifest (A9, `data/manifests/lavdf-test-window.csv`, primary) and from the file labels (secondary).
- **Statistic.** Macro-F1 = the mean of F1(visual head, video label) and F1(audio head, audio label), per seed, then the mean over seeds.
- **Baseline, the clip-label-only predictor:** it marks both streams fake exactly when the clip is fake, using the true clip label. It is generous, since it knows which clips are fake; the heads must beat it by saying *which* stream.
- **Reported:** both macro-F1 values, their difference per seed and as the mean over seeds with a 95% interval from 1,000 bootstrap resamples of LAV-DF original recordings (seed 17), the one-sided share of resamples at or below 0, and each head's F1, precision and recall per stream.
- **Reading, fixed now.** The direction transfers if the mean difference is above 0 and its interval excludes 0. Otherwise it does not.

### A16: does the sync score move when the audio is moved?

The paper reads the sync path's per-method order as evidence that it measures lip and audio agreement. A16 tests that on real clips directly.

- **Clips.** Every real (RealVideo-RealAudio) clip of the neutral FakeAVCeleb test partition whose 2 s window at 0 s has a face in at least 80% of its 50 frames.
- **Conditions, fixed now.** Lips always from 0 to 2 s. Audio from: 0 to 2 s (original); shifted later by 0.04, 0.08, 0.20, 0.40 and 0.80 s; and swapped, taking the 0 to 2 s audio of another real test clip with a different source identity, assigned by a derangement drawn with seed 17.
- **Scoring.** The frozen AV-HuBERT Base and the three P-001 sync heads, features cast to float16 as in the feature cache, window misfit as in P-001.
- **Reported, per condition:** AUC of manipulated against original windows per seed and as the mean over seeds with a 1,000-resample identity bootstrap (seed 17), and the share of clips whose misfit rises.
- **Reading, fixed now.** The score moves with the mechanism if the mean AUC interval lies above 0.5 for the 0.20, 0.40 and 0.80 s shifts and the swap, and the mean AUC at 0.20 s is above that at 0.04 s. Shifts of 0.20 to 0.80 s and swaps resemble the head's training negatives (5 to 20 frames, other clips), so they test the trained objective on unseen people; 0.04 and 0.08 s lie outside it. A16 does not test whether the voice-cloning row also carries the residual audio cue.

### A17: speaker overlap with AV-HuBERT's pretraining

- **Source of the split.** VoxCeleb2's dev and test speaker lists, taken from the public `yangwang825/vox2-veri-3s` dataset on Hugging Face, because the official metadata now needs a login. Its counts match the published split (5,994 dev and 118 test speakers, 36,237 test utterances). The file hashes are recorded with the result.
- **Assumption, stated now.** AV-HuBERT Base was pretrained on LRS3 and the English part of VoxCeleb2 dev, so a VoxCeleb2 test speaker was never in pretraining and a dev speaker may have been.
- **Reported:** how many of FakeAVCeleb's 500 source identities, and how many of its neutral test clips, belong to VoxCeleb2 dev, test or neither. LAV-DF has no speaker ids, so its overlap is reported as unknown.
- **Only if** at least 5 FakeAVCeleb test identities with real clips fall in each of dev and test: the sync path's test AUC on clips of each group, with identity-bootstrap intervals, as a description, not a test. Otherwise only the counts are reported.

### A18: H1 again with a fixed video bitrate

- **Change.** Neutralizer v4 with one difference: the video is encoded at a constant 160 kb/s (x264 `-b:v`, `-minrate` and `-maxrate` 160k, `-bufsize` 160k, `nal-hrd=cbr`) instead of constant quality (CRF 23). 160 kb/s is the median video bitrate of the v4 copies (157 kb/s). Everything else is v4. The copies are written beside v4 and replace nothing.
- **Data.** All of FakeAVCeleb, as H1. The 600-clip check sample runs first as a smoke test. If the full set cannot be neutralized and audited by 05:00, only the check sample is reported, labelled as such.
- **Reported:** the shortcut ceiling on the clip, video and audio labels with `scripts/shortcut_audit.py`, unchanged, the strongest single cues, and the clips refused.
- **Reading, fixed now.** If the clip-label ceiling is at most 0.60, a fixed bitrate removes the residual that made H1 miss; if not, something besides bitrate remains and the audit names it. H1's preregistered verdict stays "not met" either way.

## Amendment A19, 2026-11-17, 21:15, before any A19 model exists: AVH-Align's method on our protocol (exploratory)

**Why.** The paper compares with no published baseline. AVH-Align (Smeu et al., CVPR 2025; code `bit-ml/AVH-Align` at commit `55401de`, CC BY-NC-SA 4.0) is the closest published method: a real-only alignment model on frozen AV-HuBERT features.

**What cannot be run.** The released weights expect features from `self_large_vox_433h.pt` (AV-HuBERT Large fine-tuned on LRS3) and AVH-Align's own mouth preprocessing. That checkpoint is not here, and our Large checkpoint is pretrained only. So the published model is **not** evaluated.

**What runs instead: AVH-Align's head, loss and score, retrained on our protocol.**
- **Model.** Their `FusionModel` unchanged except the input width: 768 (our AV-HuBERT Base lips-only and audio-only passes) instead of 1024.
- **Data.** Exactly the real FakeAVCeleb training and validation windows the P-001 sync path used (`load_windows`, face in 80% of frames, 50 frames), seeds 17, 29 and 43.
- **Training, as their code.** Features L2-normalized per frame. For each lips frame, the audio frames 15 either side (zeros outside the 2 s window, where theirs uses the video's edges). Loss: minus the log-softmax at the true offset, summed. Adam, learning rate 1e-5, batch 1,024, ReduceLROnPlateau (factor 0.1, patience 5), early stopping after 10 epochs without a lower real validation loss, at most 500 epochs; the epoch with the lowest real validation loss is kept.
- **Score, as their code.** Per window, the log-sum-exp over frames of minus the aligned-pair output; a clip's score is the mean over its windows (P-001's rule).
- **Sets.** FakeAVCeleb test and identity-strict test, LAV-DF test (file and window labels), DFDC and HiFi-AVDF, from the existing feature caches, on the clips the P-001 sync path scored.
- **Reported:** AUC per seed and as the mean over seeds with a 95% identity-bootstrap interval; the difference against the P-001 sync path on the same clips, mean over seeds with one identity bootstrap shared by the seeds (1,000 resamples, seed 17, `mean_seed_difference`).
- **Reading, fixed now.** On each set, AVH-Align's method is better if the mean difference is above 0 with an interval that excludes 0, worse if below with an interval that excludes 0, and not different otherwise. This compares two heads on the same features and data, not the published system.
