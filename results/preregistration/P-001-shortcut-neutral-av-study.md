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
