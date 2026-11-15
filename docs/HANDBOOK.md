# Project handbook

Start here. This file is the single place to learn what the project is, what it has measured, what went wrong, and what to do next. Everything else in the repository is either code, raw evidence, or a deeper explanation that this file links to.

Authors: Vaibhav Shiroorkar, Shivansh Gupta and Adithya Jithesh, equal credit (see [AUTHORS.md](../AUTHORS.md)). Last revised 2026-11-13.

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [How the system works](#2-how-the-system-works)
3. [Data, splits and why they are built this way](#3-data-splits-and-why-they-are-built-this-way)
4. [How to run things](#4-how-to-run-things)
5. [What we have measured](#5-what-we-have-measured)
6. [Mistakes, start to finish](#6-mistakes-start-to-finish)
7. [What a hard reviewer would say](#7-what-a-hard-reviewer-would-say)
8. [What to do better, ranked](#8-what-to-do-better-ranked)
9. [The paper: objectives, related work, venue](#9-the-paper-objectives-related-work-venue)
10. [Where things live](#10-where-things-live)
11. [Glossary](#11-glossary)
12. [History](#12-history)

## 1. The project in one page

**The problem.** Audio-visual deepfake detectors report 95 to 99 AUC on their own benchmarks. They fail on clips from anywhere else. Our own best face model scores 0.999 in-domain and 0.67 on Celeb-DF.

**What we found first.** On FakeAVCeleb, the most used audio-visual benchmark, a classifier that never looks at a face or listens to a voice does almost as well as a real detector:
- Leading digital silence alone predicts audio-fake at about 0.99 AUC.
- The video codec alone predicts video-fake at about 0.97 AUC.

Real clips are H.264. Most face-swapped clips are MPEG-4 Part 2. Clips with generated audio end with about 1.2 s of digital silence, against 24 ms for real ones. A detector can learn the generator's file pipeline instead of the manipulation.

**What we are building.** A detector with three paths and a fusion step, trained and tested only on clips whose file-level cues have been removed:

- A visual path that looks for face artifacts.
- An audio path that looks for synthetic speech.
- A sync path that checks whether the lips match the sound. It is trained on real clips only, so it cannot learn a fake generator's habits.
- Fusion that outputs whether the clip is fake and which stream was edited.
- A threshold set from real clips of the target domain, so the false-alarm rate holds when the domain changes.

**What the paper will claim.** It does not claim that fusion is new. It claims four things, each measured, each reported even if negative. The details are in [section 9](#9-the-paper-objectives-related-work-venue) and the preregistration [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md).

1. A shortcut audit and a neutralized protocol for AV deepfake data.
2. Whether fusion still beats visual-only once the shortcuts are gone.
3. Whether "which modality was faked" transfers across corpora.
4. Whether a threshold calibrated on real target clips controls false alarms under codec and corpus shift.

**Where it stands (2026-11-13).**
- No result of the new study exists yet. Every older number is either historical, from the team project, or marked "requires rerun" because of a protocol defect (section 6).
- The protocol fixes, the neutralizer and the false-alarm module are written and tested.
- The shortcut audit is running.

## 2. How the system works

```text
clip (mp4)
  |
  +-- neutralize (v4): decode audio once, scale to -23 dBFS RMS, start at the
  |   first sound on a whole frame, keep exactly 3.00 s, dither -66 dBFS,
  |   re-encode H.264 25 fps + AAC, strip metadata (preprocessing/neutralize.py)
  |
  +-- views: face track -> 16 face crops (224 px)
  |          audio -> 4 s at 16 kHz
  |          mouth -> 88 px grey lip crops, 25 fps (views/, avsync/mouth.py)
  |
  +-- paths
  |     visual: EfficientNet-B0 per frame + GRU          -> video_fake logit
  |     audio:  Wav2Vec2 Base + attentive pooling        -> audio_fake logit
  |     sync:   frozen AV-HuBERT, lips-only and audio-only passes,
  |             AlignmentHead trained on real clips only -> misalignment score
  |
  +-- fusion: logistic regression on out-of-fold path logits (primary)
  |           StreamFusion MLP with per-modality heads (secondary)
  |
  +-- decision: threshold from validation, or from real target clips
                (evaluation/conformal.py); abstain when a view is missing
```

Deeper explanations, one stage per chapter, are in [docs/handbook/](handbook/README.md). They were written on 2026-10-29. Two of them describe the design before this revision:
- Chapter 08 describes the old ResNet-18 sync branch. That branch scored below chance across corpora, and AV-HuBERT replaces it.
- Chapter 05 has no neutralization step.

The code layout:

| Stage | Code |
|---|---|
| Manifests and splits | `src/deepfake_detection/data/` (`manifest.py`, `protocols.py`, `guards.py`, `speech.py`) |
| Neutralization | `src/deepfake_detection/preprocessing/neutralize.py`, `scripts/neutralize_dataset.py` |
| Views and cache | `src/deepfake_detection/views/` |
| Visual and audio paths | `src/deepfake_detection/branches/` |
| Sync path | `src/deepfake_detection/avsync/`, `src/deepfake_detection/streams/avhubert.py` |
| Fusion | `src/deepfake_detection/fusion/` |
| Metrics, bootstrap, false-alarm control | `src/deepfake_detection/evaluation/` |
| Command line | `ddf`, in `src/deepfake_detection/cli/` |

## 3. Data, splits and why they are built this way

| Corpus | On disk | Labels | Role |
|---|---|---|---|
| FakeAVCeleb v1.2 | 21,544 clips | clip, video, audio | Train, validation, in-domain test |
| LAV-DF | 136,304 clips | clip, video, audio, forged segment times | Train, and test for modality transfer |
| DFDC (subset) | 3,032 clips | clip only | Locked cross-corpus test |
| Celeb-DF v2 | 6,529 clips | clip only, no usable audio | Locked visual-only test |
| FaceForensics++ | 7,000 clips | clip, method | Visual experiments |
| MNW | 134 clips | clip | Locked external test, read once |
| ASVspoof 2019 LA | 121,461 utterances | audio | Audio path training |
| In-the-Wild | 31,779 utterances | audio | Locked audio test |
| HiFi-AVDF | downloading | clip, audio, video (real paired with generated) | Locked test on six modern generators (P-001 A5) |
| DeepSpeak v2, AV-Deepfake1M, AV-Deepfake1M++, KoDF | not on disk | | All requested; locked targets once granted (P-001 A6). See [access status](research/datasets.md#access-status-2026-11-13) |

**Rules, and the mistake each one prevents.**

- **Split by person, not by clip.** A model that sees the same face in train and test can learn the face. FakeAVCeleb is split 70/15/15 by source identity. Its identity graph is fully connected (every person appears as a target of someone else), so an identity-strict subset is reported beside every in-domain number.
- **FaceForensics++ splits by connected component.** The face donor is a real person who is also a source elsewhere. Splitting by source alone let 226 identities cross train and test. Component grouping gives 0.
- **Locked targets cannot train, choose or calibrate anything.** `src/deepfake_detection/data/guards.py` refuses DFDC, Celeb-DF, MNW, DeepSpeak, In-the-Wild, KoDF and HiFi-AVDF for training and selection. DFDC was once used to pick a model, and that made its numbers partly in-sample.
- **Thresholds come from validation, with a receipt.** `ddf threshold --split val` records the clips it chose on. `ddf evaluate --threshold-file` refuses to run if any of those clips is being evaluated.
- **Neutral copies before training.** Codec, silence, loudness and length differ by class in the raw files. Every training and test clip goes through `neutralize_dataset.py` first. Every neutral clip is exactly 3.00 s long. Clips with less sound than that are refused, and refusals are counted per category, because short clips are mostly generated speech.
- **Three seeds, identity bootstrap.** Seed alone moved DFDC AUC by 0.073 in earlier runs, which is larger than most effects we measured. Every claim uses seeds 17, 29 and 43 and 1,000 identity-level bootstrap resamples.

**Known limits of the data.**

- FakeAVCeleb has about 500 real videos against about 20,000 fakes, so its intervals are wide.
- Only FakeAVCeleb and LAV-DF carry per-modality labels, so modality transfer has two directions.
- FakeAVCeleb, LAV-DF and AV-HuBERT's pretraining all draw on VoxCeleb2 and LRS3. Some "cross-corpus" signal may be the encoder recognizing speakers it saw before. This is open; see [section 8](#8-what-to-do-better-ranked).

## 4. How to run things

Use the project interpreter. The `python` on PATH is a different install.

```bash
# tests (about 20 minutes on this machine)
.venv/Scripts/python.exe -m pytest -q

# manifests
.venv/Scripts/ddf.exe manifest from-meta --meta data/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/meta_data.csv \
    --root data/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 --data-dir data \
    --output data/manifests/fakeavceleb.csv --audit data/manifests/fakeavceleb.audit.json --dataset FakeAVCeleb

# shortcut audit: what file and audio-edge cues alone can do
.venv/Scripts/python.exe scripts/shortcut_audit.py --manifest data/manifests/fakeavceleb.csv \
    --dataset FakeAVCeleb --output results/shortcut-audit/fakeavceleb

# neutral copies and their manifest
.venv/Scripts/python.exe scripts/neutralize_dataset.py --manifest data/manifests/fakeavceleb.csv \
    --dataset FakeAVCeleb --output-root data/neutral --manifest-out data/manifests/fakeavceleb-neutral.csv

# split by identity (component grouping is automatic for FaceForensics++)
.venv/Scripts/ddf.exe split build --manifest data/manifests/fakeavceleb-neutral.csv \
    --output-dir runs/<run>/split --dataset FakeAVCeleb --seed 17

# threshold on validation, then evaluate with it
.venv/Scripts/ddf.exe threshold --predictions <val-predictions.csv> --split val --output threshold.json
.venv/Scripts/ddf.exe evaluate predictions --predictions <test-predictions.csv> \
    --threshold-file threshold.json --output metrics.json
```

The full list of commands is in [docs/reference/cli.md](reference/cli.md). How runs are tracked and checked is in [docs/reproducibility.md](reproducibility.md).

**How results are stored.**

- `runs/<run>/` is ignored by git and holds large files: caches, checkpoints, logs.
- `results/<experiment>/` is tracked and holds small files: metrics, per-clip predictions for test sets (so anyone can recompute a number), and run receipts.
- `results/preregistration/` is tracked. Each hypothesis is committed before its run starts, so git order shows what was decided first.
- `results/legacy-team/` holds the team project's ledger and experiment log. It is read only.

## 5. What we have measured

### 5.1 The shortcut audit (new study, 2026-11-13)

`scripts/shortcut_audit.py` reads each clip's file metadata and audio edges and nothing else. It then scores how well each cue alone separates real from fake. Folds are grouped by identity.

**Shortcut ceiling.** This is the AUC of a gradient-boosted model given only file and audio-edge cues. Folds are grouped by identity. Raw files.

| Corpus | Clips | Real vs fake | Fake video | Fake audio | Strongest single cue |
|---|---|---|---|---|---|
| FakeAVCeleb | 21,544 | **0.994** | **0.997** | **1.000** (0.9996) | Video codec (0.971 clip label); leading silence (0.992 audio label) |
| DFDC (our subset) | 3,032 | 0.588 | 0.588 | no audio fakes | Bitrate, 0.548 |
| LAV-DF | 8,000 files (stratified sample) | **0.936** | 0.738 | 0.751 | No single strong cue: audio bitrate 0.775, trailing silence 0.696, duration 0.635 |

**What FakeAVCeleb's files say about their labels** (results/shortcut-audit/fakeavceleb):

| Category | Clips | Video codec | Audio codec | Median leading silence | Median trailing silence |
|---|---|---|---|---|---|
| RealVideo-RealAudio | 500 | H.264 (all) | AAC (all) | 0 ms | 24 ms |
| RealVideo-FakeAudio | 500 | H.264 (all) | **MP3 (493)**, AAC (7) | 31.5 ms | **1,191.6 ms** |
| FakeVideo-RealAudio | 9,709 | **MPEG-4 Part 2 (8,979)**, H.264 (730) | AAC (all) | 0 ms | 17 ms |
| FakeVideo-FakeAudio | 10,835 | **MPEG-4 Part 2 (all)** | AAC (all) | 31.4 ms | 173 ms |

Three things follow:

- FakeAVCeleb's labels can be read off its files. Any detector score on raw FakeAVCeleb has to be compared with 0.994, not with 0.5.
- Two of these cues have not been reported for this dataset: the MP3 audio track on generated speech, and the per-category codec split.
- DFDC is close to clean on these cues. That is one reason detectors trained on FakeAVCeleb fall toward chance there: the cue they learned is absent.

LAV-DF is different from FakeAVCeleb. Every category is H.264 with AAC audio, so the codec carries nothing. But weak cues add up: audio bitrate, trailing silence and duration together reach 0.936. Neutralization fixes the audio bitrate and removes edge silence. It cannot fix duration, because LAV-DF's edits replace words and change a clip's length. Our paths read fixed 2 to 4 second windows and never see the length, but any model given whole clips could learn it.

A first attempt on LAV-DF used a window-level manifest in which 2,527 of 3,483 files supply both a real and a fake window. File cues cannot separate two windows of the same file, so that run understated the shortcut. It was deleted, and LAV-DF is being audited per file.


**H1: does neutralization remove the shortcut?** (preregistered bar: ceiling at most 0.60 on the clip label)

| Attempt | Data | Clip | Video | Audio | Result |
|---|---|---|---|---|---|
| 1, neutralizer v1 | all 21,544 FakeAVCeleb clips | 0.933 | 0.901 | 0.994 | **Failed.** Re-muxing added 28 ms of leading silence to face-swapped clips (mistake 59) |
| check, neutralizer v4 | 600-clip sample, not reused | 0.598 | 0.489 | 0.702 | Below the bar on clip and video labels; the audio label keeps a residual |
| 2, neutralizer v4 | 20,284 clips (1,260 short generated-speech clips refused, no real ones) | 0.632 | 0.617 | 0.793 | **Not met.** Main residual: video bitrate (0.57 to 0.64), which under constant-quality encoding follows image content |

### 5.1b The sync path (P-001, 2026-11-15)

Frozen AV-HuBERT features, with an alignment head trained only on real clips. Scored on neutral FakeAVCeleb, test AUC is **0.775, 0.809 and 0.806** for seeds 17, 29 and 43 (identity-strict test: 0.781 to 0.810). The old ResNet-18 sync branch scored 0.44 to 0.49. All three seeds sit above the shortcut ceiling on these copies (0.632).

The per-method pattern shows it reads lip-audio agreement: voice cloning 0.942, Wav2Lip 0.84 to 0.86, and face swap alone 0.63 to 0.68 (a face swap keeps the original lips and voice). At the threshold chosen on validation, it flags 27 to 39% of real test clips. That is the false-alarm problem H4 addresses. Details: [results/p001-sync](../results/p001-sync/README.md).

### 5.2 Older numbers, and why none of them is a result yet

These come from earlier stages. Each is listed so nobody cites it by mistake. "Requires rerun" means a defect in section 6 touches it.

| Claim | Number | Status | Source |
|---|---|---|---|
| Visual branch, FakeAVCeleb in-domain | 0.9988 AUC | Measured on raw files that carry the codec shortcut | [result-traceability](research/result-traceability.md) |
| Visual branch, DFDC, seeds 17 / 29 / 43 | 0.7611 / 0.6963 / 0.6878 | Seed 17 was the one reported, and it was the best of three. Mean 0.7151 | same, row `A-visual-seeds` |
| Six-checkpoint visual ensemble, DFDC | 0.7739 | Membership chosen by looking at DFDC, so partly in-sample | same, row `A-visual-ensemble` |
| Audio branch, DFDC | 0.5054 [0.4747, 0.5371] | Chance | same, row `A-audio-crosscorpus-gap` |
| ResNet-18 sync branch, three seeds | 0.4364, 0.4570, 0.4931 | Below chance; replaced by AV-HuBERT | same, row `A-sync-seeds` |
| Fusion minus visual, DFDC | -0.0085 [-0.0196, +0.0027] | Requires rerun: fold checkpoints chose epochs on the outer holdout | same |
| Team project, video model on the Celeb-DF family (sealed, 3 seeds) | 0.8218 [0.8020, 0.8393] | Valid under the team's own split; not comparable to the new study | [legacy EXPERIMENTS.md](../results/legacy-team/EXPERIMENTS.md), E-014 |
| Team project, codec metadata alone on FakeAVCeleb | 0.9985 | The first measurement of the codec shortcut | same, E-021 and E-023 |

The pattern behind all of them: the models did very well on the corpus they trained on, and close to chance elsewhere. Section 6 explains how much of that comes from shortcuts and protocol errors.

### 5.3 How the numbers are computed, and what changed

Old and new numbers in this project are **not directly comparable**. This list says what changed in the maths and why. None of these changes was chosen after seeing the result it affects.

| What | Before | Now | Why | Effect on comparison |
|---|---|---|---|---|
| EER | FPR and FNR averaged at the nearest ROC point | Linear interpolation where FPR = FNR (`evaluation/metrics.py:equal_error_rate`) | On small sets the nearest point can be off by several points (test case: 0.417 vs exact 0.333) | Old EERs can differ by a few points, most on small sets such as MNW |
| Decision threshold | Fixed 0.5 | Maximum balanced accuracy on validation. The file records the clip ids used, and evaluation refuses any overlap | 0.5 means nothing on 10:1 fake-heavy data | Old FPR, FNR and balanced accuracy at 0.5 are not comparable |
| FF++ split | Grouped by source identity | Grouped by connected identity component | Donor faces crossed train and test (226 identities) | Old FF++ in-domain numbers were optimistic |
| Identity-strict subset | Read "fake.mp4" as a person | File names are not identities | The old strict test had zero fake-audio clips | Old strict numbers covered three of four categories |
| Input data | Raw files | Neutralizer v4: one codec, fixed 3.00 s, cut at first sound, level -23 dBFS RMS, dither -66 dBFS | Raw files leak the label (ceiling 0.994) | Neutral and raw numbers measure different things. 1,260 short generated-speech clips are refused, so the neutral test set has fewer fakes |
| Sync clip score | Not used | Mean window misfit s, reported as 1 - exp(-s) | Puts the score in [0, 1) for the evaluation tools | Monotone, so AUC is unchanged. It is not a calibrated probability |
| Shortcut ceiling | Did not exist | Gradient-boosted trees on file and audio-edge cues, 5 folds grouped by identity | The bar any detector must beat | Single-cue AUCs flip direction in-sample (max of AUC and 1 - AUC), so they are slightly optimistic. The ceiling itself is out-of-fold |
| False-alarm threshold (H4) | Did not exist | Split conformal rank ceil((n+1)(1 - alpha)); Neyman-Pearson umbrella for an (alpha, delta) bound | Thresholds from the source domain break under shift | Standard published formulas (Vovk et al. 2005; Bates et al. 2023; Tong et al. 2018), not new maths |

**Unchanged:** ROC-AUC itself; the 95% intervals from 1,000 bootstrap resamples of identities (not clips); three seeds; and Holm correction across H2 to H4, as preregistered.

## 6. Mistakes, start to finish

This log lists every methodological, data, engineering and documentation mistake
we could trace, from the first commit on 2026-07-04 to 2026-11-13. Entries run in
the order the mistake was made. The "How it was caught" line gives the date it
was found, which is often weeks later.

Each entry cites a file or a commit. Commit hashes resolve with `git show <hash>`.
Files under `runs/` and `data/` are git-ignored, so they exist only in the primary
checkout.

### Prototype (July to August)

#### 1. Validation and test sets were undersampled to 1:3

- **What went wrong:** The plan said validation and test keep FakeAVCeleb's
  natural class mix. `build_splits.py` undersampled fakes to 1:3 in every split
  instead.
- **How it was caught:** The 2026-07-23 snapshot listed it as open
  reconciliation item 2.
- **What it cost:** Every prototype precision, recall and F1 sits at an
  artificial prevalence. That includes the Stage 2 test accuracy 0.963 and F1
  0.974.
- **Fix:** The prototype was discarded. The 2026-08-24 rebuild kept source
  splits without undersampling.
- **Source:** `git show 926624a:docs/PROJECT_OVERVIEW.md`, section 14.

#### 2. Experiment tracking was decided but never built

- **What went wrong:** The plan named W&B and per-category validation logging.
  Neither existed. Tracking was a hand-edited `evaluation/experiments.csv`, and
  the trainer logged aggregate metrics only.
- **How it was caught:** The same snapshot, items 3 and 5.
- **What it cost:** No prototype number has a run record behind it. Per-category
  failures (for example on wav2lip) were invisible during training.
- **Fix:** Local MLflow and versioned configs in the rebuild (ROADMAP Phase 1).
- **Source:** `git show 926624a:docs/PROJECT_OVERVIEW.md`, section 14;
  `ROADMAP.md` (removed; read it with `git show 6d55756:ROADMAP.md`), Phase 1.

#### 3. FakeAVCeleb's mixed codecs were found and judged harmless

- **What went wrong:** On 2026-07-28 we found that wav2lip fakes are MPEG-4
  Part 2 while genuine clips are H.264. We treated it as a browser playback bug.
  The docs reasoned that OpenCV and PyAV decode both codecs to arrays, so models
  could not see it.
- **How it was caught:** Only in late October. F35 re-encoded test clips and
  watched the verdicts flip. The team project found the same thing in E-021.
- **What it cost:** This is the most expensive mistake in the project. Genuine
  clips re-encoded as MPEG-4 are called fake 93.3 percent of the time. Face-swap
  plus wav2lip fakes re-encoded as H.264 are caught 30.7 percent of the time
  instead of 100 percent. Every in-domain FakeAVCeleb number partly measures the
  encoder. Training the cue away (F37) dropped DFDC from 0.7611 to 0.5720.
- **Fix:** Open. P-001 preregisters neutralized copies and a shortcut ceiling
  test (H1) before any new training.
- **Source:** commit `ededbf7`; `docs/research/findings.md` F35 and F37;
  `docs/obstacles.md`, "FakeAVCeleb's codecs follow its labels";
  `results/preregistration/P-001-shortcut-neutral-av-study.md`.

#### 4. The dashboard's audio views were wrong in four places

- **What went wrong:** "Trim leading silence" cut the waveform but not the frame
  timeline, shifting audio by 64 ms on a test clip. The downmix toggle did
  nothing. Two plots had mislabelled time axes.
- **How it was caught:** 2026-07-28, by checking every plot against the
  operation it calls, on real clips.
- **What it cost:** Teaching views showed the exact desync the docs warned
  about. No model number depended on them.
- **Fix:** Fixed in the same commit.
- **Source:** commit `7fe2c91`.

#### 5. Face alignment pasted a mirrored second face into every crop

- **What went wrong:** Five-point alignment used reflected border padding. It
  put a mirrored face in every crop. An `align_inset` setting left a quarter of
  the canvas empty on pre-cropped frames. Its benefit was argued, never
  measured.
- **How it was caught:** 2026-08-01, while removing alignment.
- **What it cost:** The v3 cache and the `best_v3_aligned.pt` weights. The
  backbone comparison in commit `e659c71` was run on the v3 cache.
- **Fix:** Alignment removed. `PIPELINE_VERSION` went to 4 so stale crops are
  rebuilt.
- **Source:** commit `80f8b60`; `docs/obstacles.md`, "Five-point alignment was
  removed, and carried two traps".

#### 6. The DINOv2 stream could not load its own weights

- **What went wrong:** `dinov2_config()` had never been run with pretrained
  weights. A hardcoded `global_pool="avg"` built the wrong norm layer, so the
  weights did not fit.
- **How it was caught:** 2026-08-02, on the first training attempt.
- **What it cost:** Every visual-stream test passed because all of them used
  `pretrained=False`. The suite tested architecture, not loading.
- **Fix:** Pooling became a config field. New tests load real weights.
- **Source:** commit `e659c71`.

#### 7. Backbones were compared and chosen on the test split

- **What went wrong:** EfficientNet, Xception and DINOv2 were ranked on the
  300-clip held-out test split. The test split became a selection set.
- **How it was caught:** Not flagged at the time. The rebuild introduced a
  frozen test, and the team project later sealed its test with a ledger.
- **What it cost:** The prototype test AUCs (EfficientNet 0.99983, Xception
  0.99401, DINOv2 1.00000) are selection numbers, at a ceiling 300 clips cannot
  resolve. Two sets of test numbers exist for the same streams: 0.99983 and
  0.99401 in `e659c71` (v3 cache), and 0.99580 and 0.98960 in `81c04db` (v4
  cache). Neither commit says which to quote.
- **Fix:** Discarded with the prototype.
- **Source:** commits `77cf21f`, `e659c71`, `81c04db`.

#### 8. The rebuild cut the prototype out of main's history

- **What went wrong:** On 2026-08-24 main restarted from a new root commit. The
  July and August work, including Adithya Jithesh's pipeline and trainer, stayed
  on an unconnected branch.
- **How it was caught:** Recorded in `docs/obstacles.md`; repaired on
  2026-11-12.
- **What it cost:** For 80 days main showed no trace of the prototype or its
  author's work.
- **Fix:** Merge commit `1c4e279` links the history without changing files.
- **Source:** commit `1c4e279`; `docs/obstacles.md`, "`main` and `old` share no
  history".

### Team project (September to October)

The team project ran from 2026-10-24 to 2026-11-02. Its record is in
`results/legacy-team/`.

#### 9. Small record-keeping slips in the first sessions

- **What went wrong:** A download time was written as about 18:00 when file
  timestamps say 16:55 to 17:10. E-001's header named the commit before the run.
  The real share was written as 10.8 percent instead of 9.7, and the first
  correction (D-053) missed the report and slide sources.
- **How it was caught:** Independent verification passes on 2026-10-25.
- **What it cost:** No result changed. Each slip needed a correcting commit.
- **Fix:** Corrected in place with the old value stated.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, lines 78 and 315; commits
  `12f379a`, `192fd07`.

#### 10. Audio trained on ASVspoof learned its recording conditions

- **What went wrong:** ASVspoof fakes start earlier and are louder. In-the-Wild
  fakes start later and are quieter. A model trained on the first reads the
  second backwards.
- **How it was caught:** M-004 on 2026-10-25, checking for a label bug.
- **What it cost:** The from-scratch audio reference reads 0.99996 in-domain
  and 0.232, 0.219 and 0.286 on In-the-Wild dev_x, below chance at all three
  seeds.
- **Fix:** Reported as a diagnosis. Frozen pretrained speech encoders were used
  instead. The leading-silence cue came back in the 13 November audit (entry 48).
- **Source:** `results/legacy-team/EXPERIMENTS.md`, M-004.

#### 11. E-011 passed its preregistered test on a production-pipeline shortcut

- **What went wrong:** On LAV-DF every original is real and every produced
  video is fake or audio-fake. The success lines did not control for that. The
  visual head learned "original or produced" and E-011 read SUPPORTED.
- **How it was caught:** A post-hoc control the same day, scoring only produced
  videos.
- **What it cost:** Fused AUC 0.9935 and visual 0.7491 came from the shortcut.
  Within produced videos the visual head reads 0.5070. Retraining on produced
  videos only (E-012) gave 0.5169, still chance.
- **Fix:** E-012 recorded NOT SUPPORTED. The visual head has no audio-visual
  claim on LAV-DF.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-011 control and E-012.

#### 12. The published baseline was tuned only to the edge of its grid

- **What went wrong:** E-013 picked learning rate 1e-4, the top of the grid,
  while validation AUC was still rising (0.9646, 0.9809, 0.9883). An under-tuned
  baseline favours our own system.
- **How it was caught:** Review after E-013, the same day.
- **What it cost:** The first delta for the non-inferiority test (0.0143) was
  set on that baseline.
- **Fix:** E-013b added 3e-4 as a dated amendment. The final delta is 0.0100.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-013 and E-013b; commit
  `262d5b1`.

#### 13. Multiplicity correction was an amendment, not the primary rule

- **What went wrong:** Three confirmatory tests ran under 95 percent rules. The
  Bonferroni level was added later as a "robustness" line only.
- **How it was caught:** The 2026-10-25 amendment, before the sealed read.
- **What it cost:** The headline "superior to Xception" holds at 95 percent,
  +0.0176 [+0.0025, +0.0327], and fails at 98.33 percent, [-0.0011, +0.0355].
  The defensible claim is non-inferior only.
- **Fix:** The paper uses the Bonferroni wording. P-001 uses Holm across H2 to
  H4 as the primary rule.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-014; commit `b5d1144`.

#### 14. The sealed FakeAVCeleb test ran on a corpus with a codec shortcut

- **What went wrong:** E-015 scored the sealed half once and read 0.9552 for
  video. The prediction had been 0.55 to 0.80. The surprise was a shortcut, not
  skill.
- **How it was caught:** E-021 on the development half, 2026-10-26.
- **What it cost:** Codec metadata alone separates real from fake video at
  0.9985. Pixel statistics with no face reach 0.9585. A common re-encode drops
  the frozen score from 0.9732 to 0.8772, and wav2lip from 0.9548 to 0.7944.
  The frozen thresholds also failed: sensitivity 0.480, audio specificity
  0.637. E-023 found codec and method fully confounded.
- **Fix:** The sealed number stands as ranking on a shortcut dataset. No
  codec-matched control was run.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-015, E-021;
  `results/legacy-team/RESEARCH_INTEGRITY.md`, section 6.

#### 15. The canonical re-encode protocol leaked two colour tags

- **What went wrong:** Protocol v1 copied `color_transfer` and
  `color_primaries` from the input. All 100 Pexels reals carried bt709; no
  generated DeepAction clip did.
- **How it was caught:** The E-019 re-audit, 2026-10-26.
- **What it cost:** Metadata alone reached AUC 1.000 on canonical clips. Later,
  the v3 header audit passed only because the fields were constant by
  construction. Pixel statistics still reach about 0.77 to 0.79.
- **Fix:** Protocol v2 forced the tags, v3 fixed GOP structure. The pixel floor
  is open.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-019 and addenda;
  `results/legacy-team/RESEARCH_INTEGRITY.md`, section 6.

#### 16. The installed generated-video head was misnamed and did not generalize

- **What went wrong:** E-020's heading called it a "ReStraV-style" detector. The
  installed head was C2, a logistic regression on mean DINOv2-small embeddings.
- **How it was caught:** The 2026-10-26 audit.
- **What it cost:** On held-out DF26 it reads 0.4957 [0.4755, 0.5168], chance.
  Its DeepAction results are probably confounded by source.
- **Fix:** Heading clarified post-audit. The module stays experimental.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-020;
  `results/legacy-team/RESEARCH_INTEGRITY.md`, section 6.

#### 17. Lip-sync thresholds were tuned on the cases they were checked on

- **What went wrong:** E-022 failed two hypotheses, then set `MIN_CONFIDENCE`
  and a speech heuristic from the same 40 and 50 development cases. The AFTER
  pass follows by construction. E-025 then split LAV-DF by file, not speaker.
- **How it was caught:** Stated in E-022 itself; the E-025 speaker issue in the
  post-audit clarification.
- **What it cost:** No held-out check of the lip-sync module's thresholds.
  Speaker overlap in E-025 cannot be excluded.
- **Fix:** Open. E-025 records the limit.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-022 and E-025;
  `results/legacy-team/RESEARCH_INTEGRITY.md`, section 6.

#### 18. Git cannot show that preregistrations came before runs

- **What went wrong:** E-019 to E-025 and D-071 to D-073 first appear in one
  commit, `0704b3a`, with the scripts that ran them. Result files are
  git-ignored. Frozen artifacts have no trusted hash manifest. A ledger field
  named `score_file_sha256` hashes the feature table, not a score file.
- **How it was caught:** The merge-readiness audit, 2026-10-26.
- **What it cost:** "Registered before the run" is an in-document claim, not
  checkable evidence, for seven experiments.
- **Fix:** Rule: commit each preregistration alone before the run. P-001
  follows it.
- **Source:** `results/legacy-team/RESEARCH_INTEGRITY.md`, sections 1 to 4.

#### 19. Post-hoc numbers reached the deliverables

- **What went wrong:** E-032 false-positive rates (0.1098 on Celeb-DF family,
  0.1405 on DFD) were computed in an audit and quoted in the paper, report and
  slides before any log entry existed. E-030 was run on the false premise that
  EER was never computed. E-027's "closed" look reused the 271 DF26 reals from
  development.
- **How it was caught:** The 2026-10-27 benchmark audit.
- **What it cost:** E-030 duplicated an existing freeze result (0.0694 and
  0.1578). E-027's closed AUC measures new fakes against reused reals.
- **Fix:** E-032 logged and labelled post hoc. The freeze report is the primary
  EER record.
- **Source:** `results/legacy-team/EXPERIMENTS.md`, E-027, E-030 correction,
  E-032.

### Root study (September to November)

The root study restarted on 2026-08-24. Its first entries fall in late August.

#### 20. A human-review benchmark was built and then abandoned

- **What went wrong:** From 2026-08-25 to 09-01 we built annotation contracts,
  agreement gates and evidence binding for a 625-frame MTCNN against YuNet
  benchmark. Nobody was going to review the frames.
- **How it was caught:** Dropped by decision.
- **What it cost:** More than ten commits of tooling, later deleted. The face
  detector was frozen by decision, not measurement.
- **Fix:** Package removed; ROADMAP Phase 2 records the decision.
- **Source:** `ROADMAP.md` (removed; read it with `git show 6d55756:ROADMAP.md`), Phase 2; `CHANGELOG.md`, Removed; commits `8d3e669`
  to `1bbf27a`.

#### 21. The sync window ignored where LAV-DF hides its forgeries

- **What went wrong:** The preprocessor placed the sync window near the start
  of each clip. LAV-DF forges a short span anywhere in a longer recording.
- **How it was caught:** Measured when LAV-DF was added in September.
- **What it cost:** 56,582 of 99,873 LAV-DF forgeries fall entirely outside the
  window. Those fakes would have trained on genuine video with a fake label.
- **Fix:** Per-clip `sync_start_sec`.
- **Source:** `docs/obstacles.md`, "The synchronisation window is pinned near
  the start of a clip".

#### 22. Zero-start LAV-DF windows shifted into manipulated spans

- **What went wrong:** A default treated an explicit start of zero as "not
  set" and shifted those windows. Some "genuine" windows then overlapped
  manipulated spans.
- **How it was caught:** The audit against the team protocol, 2026-10-27.
- **What it cost:** The 0.9969 lip-sync score, called "the finding to lead
  with" in F1a, and the 0.7407 and 0.5059 transfer scores (F12). All marked
  `requires rerun`.
- **Fix:** Code fixed. The LAV-DF cache rebuild and retrain are open.
- **Source:** commit `5d4285d`; `docs/handoff.md`, Current state;
  `docs/research/result-traceability.md`, rows `B-lavdf-lipsync` and
  `B-lavdf-transfer`.

#### 23. Checkpoints were selected on validation loss

- **What went wrong:** Two visual streams kept their lowest-BCE epoch. One kept
  an epoch-1 checkpoint with the backbone still frozen.
- **How it was caught:** 2026-09-10, when loss and AUC moved in opposite
  directions on the same epoch.
- **What it cost:** VST-01 and VST-02 withdrawn. The kept checkpoint scored
  0.4668 in-domain, below chance.
- **Fix:** Stream trainers moved to ROC-AUC selection. The fix was partial (see
  entry 37).
- **Source:** commit `b63cf87`; `docs/research/findings.md` F3.

#### 24. Scripted edits broke the PowerShell runners three times

- **What went wrong:** A Python heredoc turned `split\train` into a tab
  character. The first fix used a PowerShell 7 `Join-Path` form that 5.1
  rejects. A later edit glued a comment onto code, and the tokenizer called the
  file clean. A chain script exited silently under `set -e`.
- **How it was caught:** Each failed at launch, 2026-09-11 and 09-13.
- **What it cost:** Three stream runs died within seconds. The chain script, if
  re-run, would have started a second training run beside the first.
- **Fix:** Paths with forward slashes, `Parser::ParseFile` checks, chain script
  deleted.
- **Source:** commits `5c1f87c`, `e176afc`, `f6dec51`.

#### 25. Freezing BatchNorm was recorded as a clean win

- **What went wrong:** It took a stream from 0.5154 to 0.9058 on validation,
  and validation was all we measured.
- **How it was caught:** The full-scale arms on 2026-09-12, and a controlled
  1,400-clip sweep later.
- **What it cost:** The full-scale trade looked like +0.1276 on DFDC. With
  epochs held fixed the effect is +0.0177, seven times smaller. The numbers
  also disagree across files: the CLI help and `scripts/train_design_b.ps1` cite 0.9197
  and 0.5005 frozen against 0.5267 and 0.6482 live, while F2 reports 0.9997 and
  0.5067 against 0.4909 and 0.6343. A `docs/obstacles.md` table came from a scratch
  run F2 does not reproduce.
- **Fix:** F2 supersedes the claim. The obstacles table was corrected on
  2026-10-22. The help text still carries the other numbers.
- **Source:** `docs/research/findings.md` F2; commit `1219401`;
  `src/deepfake_detection/cli/train.py`, `--live-batchnorm` help;
  `scripts/train_design_b.ps1`.

#### 26. Xception was removed on an argument, not a measurement

- **What went wrong:** We dropped Xception as redundant with EfficientNet. It
  never completed a run in the root study, so the redundancy was never measured.
- **How it was caught:** The commit says so itself, 2026-09-12.
- **What it cost:** One planned comparison. The experiment matrix records a
  decision, not a result.
- **Fix:** Recorded honestly. Not rerun.
- **Source:** commit `3284514`.

#### 27. Three readings of the stream results were wrong

- **What went wrong:** Emotion's diagonal mass of 0.3439 was called six times
  chance, against lip-sync's baseline. Lip-sync was called a redundant audio
  duplicate to drop. Emotion was called a duplicate of the visual stream.
- **How it was caught:** Computing chance per query length, and the 31-subset
  ablation (F4).
- **What it cost:** At 8 query steps 0.3438 is chance, so emotion was at chance.
  Lip-sync sits in the best DFDC combination. Emotion is the best single stream.
- **Fix:** All three moved to the superseded table.
- **Source:** `docs/research/findings.md` F1, F4, F5, Superseded findings.

#### 28. The motion mechanism in the drafts had the wrong sign

- **What went wrong:** Drafts claimed that sampling frames across a clip makes
  camera motion look like forgery flicker, causing false alarms.
- **How it was caught:** Measured on 2026-09-13.
- **What it cost:** False alarms rise only from 0.0 to 2.7 percent with motion
  on DFDC. Motion lowers the fake score on manipulated clips (correlation
  -0.316), so the real failure is missed detections.
- **Fix:** F6 supersedes the claim.
- **Source:** commit `8a4a64e`; `docs/research/findings.md` F6.

#### 29. One genuine MNW clip was counted as a detection

- **What went wrong:** MNW was treated as fake-only. Its one genuine clip was
  called fake, and the renderer counted that as a catch.
- **How it was caught:** Adding Wilson intervals, 2026-09-13.
- **What it cost:** The detection count read 25 of 85 instead of 24 of 84. The
  stale figure stayed in docs until 2026-10-22.
- **Fix:** Corrected in two passes.
- **Source:** commits `3ca6457`, `1219401`.

#### 30. The leave-one-generator-out probe leaked twice

- **What went wrong:** Real clips sat in train and test together. In FF++ and
  DF26 a held-out fake could match its own source video on the training side.
- **How it was caught:** A smoke test returned 1.0000 on every arm, 2026-09-14.
- **What it cost:** Nothing reported. The leak was closed before any result.
- **Fix:** Split on source video as well as generator. The leak warning stays.
- **Source:** commit `fbb16e4`.

#### 31. A near-perfect Veo 3 score was a content confound

- **What went wrong:** Veo 3 lab clips were scored against talking-head reals.
- **How it was caught:** The same night, 2026-09-14.
- **What it cost:** A 0.9997 that measured content, not generation. It was the
  third near-perfect in-domain score to collapse.
- **Fix:** Leave-one-generator-out became the only headline metric.
- **Source:** commit `1395e2d`.

#### 32. An encoder claim was written before any artifact existed, then lost

- **What went wrong:** A commit message said Perception Encoder was "about 15
  AUROC points better" than DINOv3 with no run behind it. The real comparison
  ran on 2026-09-27 and nobody recorded it.
- **How it was caught:** The claim was withdrawn for weeks. The run was found
  on 2026-10-13.
- **What it cost:** The real gap was +0.0741, half the claim. The registry
  showed the comparison as pending and blocked for two weeks.
- **Fix:** Recorded as F33. Then entry 40 invalidated it again.
- **Source:** commits `fbb16e4`, `4137347`; `docs/handoff.md`, Limits.

#### 33. Frame rate leaked through one DF26 generator

- **What went wrong:** Real DF26 clips run at 25 or 29.97 fps and nothing else
  does. Wan 2.6 alone runs at 30.
- **How it was caught:** A frame-rate control before the number was quoted,
  2026-09-15.
- **What it cost:** Macro fell from 0.6942 to 0.6786. Wan 2.6 fell 0.094.
- **Fix:** The controlled table is the one reported.
- **Source:** commit `c6d27e4`; `docs/research/findings.md` F15.

#### 34. A test passed by reading real results

- **What went wrong:** `build_blocks` gained a parameter whose default pointed
  at the real filesystem. The empty-run test read actual results and passed.
- **How it was caught:** 2026-09-15.
- **What it cost:** The test checked nothing until then.
- **Fix:** Fixed in the same commit.
- **Source:** commit `c6d27e4`.

#### 35. The density gate was calibrated in-sample, on a biased split, and not reproducibly

- **What went wrong:** The first threshold was set on the clips the Gaussian
  was fitted on. The held-out half was the next 150 files of a directory sorted
  by demographic folder. PCA used a randomized solver with no fixed seed path.
- **How it was caught:** Held-out clips flagged at 100 percent, then 26.7
  percent against a 5 percent target, on 2026-09-15. The solver issue on
  2026-10-13.
- **What it cost:** The reported 79.4 percent interception was one random draw.
  The exact fit reads 74.4 percent, and 67.3 percent averaged over twenty
  splits. The served model file and the report were different fits (thresholds
  6.7136 and 6.6959).
- **Fix:** Held-out half after shuffling, exact PCA, split spread in the report.
- **Source:** `docs/research/findings.md` F17; commit `a36bacd`.

#### 36. Only seed 17 was ever scored, and it was the best draw

- **What went wrong:** Seeds 17, 29 and 43 trained on 2026-09-08. Stage 5 of
  `scripts/run_program.ps1` named the seed-17 checkpoint directly, so two finished
  checkpoints sat unscored.
- **How it was caught:** Scoring them on 2026-10-13.
- **What it cost:** The visual DFDC baseline 0.7611 is the best of three; the
  mean is 0.7151. The fused headline 0.7500 is the best of three; the mean is
  0.6873, spread 0.1078. That spread is thirteen times the fusion difference it
  was used to judge.
- **Fix:** Three seeds, mean and range. Fusion rows still need reruns (entry
  41).
- **Source:** `docs/research/findings.md` F21, F27.

#### 37. Loss-based selection survived in three trainers

- **What went wrong:** Entry 23 moved stream trainers to AUC. The binary
  branch, sync and stream-fusion trainers still selected on loss.
- **How it was caught:** ConvNeXt at seed 43 read 0.9032 in-domain against its
  siblings' 0.999, October.
- **What it cost:** The shipped seed-43 ConvNeXt was never fine-tuned. It is
  the only seed where ConvNeXt separates from EfficientNet on DFDC, so it
  decided VIS-01. It is also one of the six web-app ensemble members.
- **Fix:** One `BestEpoch` on validation ROC-AUC for every trainer.
- **Source:** `CHANGELOG.md`, Changed; `docs/research/findings.md` F29.

#### 38. Challenger flags built the wrong models

- **What went wrong:** `--audio-model` loaded WavLM into a Wav2Vec2 body,
  silently dropping its relative position embeddings. `evaluate branch` loaded
  ConvNeXt into EfficientNet. `features export` and `predict` built
  EfficientNet whatever the checkpoint held.
- **How it was caught:** Parameter counts on 2026-10-13: 94,371,712 against
  94,381,936.
- **What it cost:** Any "WavLM result" would have measured a truncated
  Wav2Vec2. AUD-01 stayed undecided anyway, since both read 1.0000.
- **Fix:** `AutoModel` builders; checkpoints record their architecture.
- **Source:** commit `23c253a`; `CHANGELOG.md`, Changed;
  `docs/research/findings.md` F28.

#### 39. DFDC's audio placeholder was read as a measurement

- **What went wrong:** `audio_fake` counts 0 of 2,410 on DFDC. We wrote that
  DFDC has no manipulated audio, so no audio cross-corpus test was possible.
  The ingestion docstring says the label is a placeholder.
- **How it was caught:** Reading the ingestion module, October.
- **What it cost:** The audio branch went unscored cross-corpus for weeks. Its
  clip-level DFDC score, available all along, is 0.5054 [0.4747, 0.5371].
- **Fix:** F30 replaces the claim.
- **Source:** `docs/research/findings.md` F30.

#### 40. PE and CLIP were fed ImageNet-normalized pixels

- **What went wrong:** The probe extractor used ImageNet statistics for every
  encoder. PE expects mean and std 0.5, so its inputs were scaled about 2.2
  times off.
- **How it was caught:** The audit, 2026-10-27.
- **What it cost:** The DF26 headline 0.7527, the encoder ranking, and the gate
  encoder comparison (PE 32.8 percent against DINOv3 67.3) are all `requires
  rerun`.
- **Fix:** Each encoder uses its own statistics, recorded in the feature file.
  Re-extraction is open.
- **Source:** `docs/research/findings.md` F33, F17; `CHANGELOG.md`, Fixed.

#### 41. Fusion fold checkpoints chose epochs on the holdout they scored

- **What went wrong:** Out-of-fold purity was a string the operator typed. Each
  fold model picked its epoch on labels from its own outer holdout.
- **How it was caught:** Purity checks in code on 2026-10-13, then the full
  leak in the 2026-10-27 audit.
- **What it cost:** Every fusion row: in-domain 0.9990, DFDC 0.7500, the seed
  spread, the branch ablation, error analysis and slices. Three of the four
  stated targets now wait on reruns.
- **Fix:** Inner validation folds; OOF exports that saw their labels are
  refused. Retraining is open.
- **Source:** commits `23c253a`, `5d4285d`; `docs/research/result-traceability.md`;
  `docs/handoff.md`, targets table.

#### 42. An exit criterion and two required ablations were never run

- **What went wrong:** The design requires a sync offset interval that
  excludes chance. Only total loss was logged, so it cannot be computed. The
  sync label-mode and leading-silence ablations were "exposed" in the CLI and
  read as done.
- **How it was caught:** Auditing the design against the registry, October
  (F31).
- **What it cost:** The only quantitative branch criterion is unmeasured. Sync
  validation loss is 2.1843 against a chance of 2.079.
- **Fix:** Label-mode ablation run (F32). The leading-silence ablation is still
  open, and entry 48 shows it mattered.
- **Source:** `docs/research/findings.md` F31, F32; `ROADMAP.md` (removed; read it with `git show 6d55756:ROADMAP.md`), Phase 4.

#### 43. AUC alone hid three threshold collapses

- **What went wrong:** We reported slices and robustness on ROC-AUC.
- **How it was caught:** Per-slice error counts, October.
- **What it cost:** Dropping rtvc costs 0.0437 of AUC but misses 73 of 75
  clips. JPEG 75 costs 0.0054 of AUC and calls 76 of 150 genuine clips fake.
  Noise at 20 dB costs 0.0227 and misses 857 of 859 fakes.
- **Fix:** Error counts at the operating point beside every AUC.
- **Source:** `docs/research/findings.md` F20, F24, F25.

#### 44. The visual branch was compared with fusion on a different label

- **What went wrong:** The branch scored RealVideo-FakeAudio as genuine.
  Fusion scored it as fake. We quoted 0.9988 beside 0.9990.
- **How it was caught:** The branch ablation, October.
- **What it cost:** Fusion looked worth +0.0002 in-domain. On one task it is
  +0.0159, and all of it is the 75 RVFA clips.
- **Fix:** The error-analysis script drops those clips automatically.
- **Source:** `docs/research/findings.md` F19, F23.

#### 45. Status docs went stale while the work moved

- **What went wrong:** The handoff said `data/` was empty while both primary
  datasets were on disk. Run costs were estimated two orders of magnitude too
  high. The paper still called FF++ absent after training on it. Dashboard
  pages called the audio and sync branches untrained a month after they had
  three seeds.
- **How it was caught:** Spot checks, 2026-09-13 to 2026-10-13.
- **What it cost:** The cost estimates "deterred the replication for weeks".
  Readers saw false status.
- **Fix:** The dataset and evidence tables are now generated. Costs are
  measured.
- **Source:** `docs/handoff.md`, Dataset inventory and "What a run actually
  costs"; commits `7b67715`, `2c3940f`.

#### 46. Two Celeb-DF false-alarm counts circulate

- **What went wrong:** Findings, the model card and the web app say 141 of 155
  genuine Celeb-DF clips are called fake. Code comments and the handbook say
  130 of 155. No file says which checkpoint gave 130.
- **How it was caught:** While drafting this log.
- **What it cost:** A reader cannot tell which number belongs to which model.
- **Fix:** Open.
- **Source:** `docs/research/findings.md` F16;
  `docs/handbook/10-crossmodal-streams.md`; `src/deepfake_detection/data/lavdf.py`.

#### 47. Loader workers grew to 9 GB each

- **What went wrong:** Visual training with prefetching workers took the host
  out of memory twice. Round-trip augmentation grew workers to about 9 GB each.
- **How it was caught:** Out-of-memory crashes.
- **What it cost:** Visual training runs with `--workers 0`. F37 has one seed
  per recipe.
- **Fix:** Open.
- **Source:** `docs/handoff.md`, "What a run actually costs";
  `docs/research/findings.md` F37.

### Audit (13 November)

All entries below were found on 2026-11-13. Items 48 to 57 are fixed in the
uncommitted working tree unless marked open.

#### 48. Silence and codec predict FakeAVCeleb labels on their own

- **What went wrong:** Fake-audio clips end with about 1.2 s of digital silence.
  Real clips end with about 24 ms. 92 percent of FakeVideo-RealAudio clips are
  MPEG-4 Part 2, and every real clip is H.264.
- **How it was caught:** `scripts/shortcut_audit.py` on a 300-clip sample and a
  partial 8,865-clip run.
- **What it cost:** Leading silence alone predicts audio-fake at 0.998 AUC.
  Video codec predicts video-fake at 0.974. F25's "silence shortcut is absent"
  only tested a 0.5 s shift on one checkpoint. The file on disk has since grown
  to 11,838 scored clips. My naive single-feature AUCs on it are 0.992 and
  0.967, and the medians are 1,191.6 ms (RVFA) and 24.0 ms (real) trailing
  silence, 92.5 percent MPEG-4 for FVRA. The numbers will move until the run
  ends.
- **Fix:** Open. P-001 preregisters neutralized copies with a ceiling of 0.60.
- **Source:** `results/shortcut-audit/fakeavceleb/clips.csv`;
  `scripts/shortcut_audit.py`; `results/preregistration/P-001-shortcut-neutral-av-study.md`.

#### 49. The test suite had not been running

- **What went wrong:** The repo was renamed from `deepfake-generalization` to
  `deepfake-detection`. The `.venv` editable install still pointed at the old
  folder, so the package would not import.
- **How it was caught:** A test run on 2026-11-13.
- **What it cost:** An unknown span of commits went untested locally. No green
  run since the rename can be trusted.
- **Fix:** The editable path now points at `deepfake-detection\src`.
- **Source:** `.venv/Lib/site-packages/_editable_impl_deepfake_generalization.pth`.

#### 50. A training script still changed into the old folder

- **What went wrong:** `scripts/train_family_holdout.sh` hardcoded a `cd` into
  `deepfake-generalization`.
- **How it was caught:** The same rename sweep.
- **What it cost:** The F34 family-holdout reproduction command failed on any
  other path.
- **Fix:** `cd "$(dirname "$0")/.."`.
- **Source:** `scripts/train_family_holdout.sh`.

#### 51. The FF++ split leaked donor identities across partitions

- **What went wrong:** The split grouped by source identity only. Donor faces
  listed in `targets` crossed partitions.
- **How it was caught:** Re-reading the split audit.
- **What it cost:** 226 train/test, 249 train/val and 61 val/test identity
  overlaps. F11 had concluded that donor leakage did not inflate the ordinary
  split; that needs a recheck. Component grouping gives 0 overlaps. It also puts
  all 18 DeepFakeDetection actors in one component, so they all land in train.
  F34 already notes that the donor-aware stress test is unmeasured.
- **Fix:** `identity_components` groups by connected component.
- **Source:** `runs/ffpp-20260913/split/audit.json`;
  `src/deepfake_detection/data/protocols.py`; `docs/research/findings.md` F11,
  F34.

#### 52. A file name was read as a person

- **What went wrong:** FakeAVCeleb's `meta_data.csv` puts `fake.mp4` in
  `target1` for all 499 RealVideo-FakeAudio rows. We parsed it as an identity.
- **How it was caught:** The identity-strict files had no RVFA clips.
- **What it cost:** `identity_strict_subset` dropped the whole RVFA category.
  The strict test has 0 RVFA clips against 75 in the ordinary test.
- **Fix:** Targets ending in `.mp4`, `.wav` or `.avi` are ignored.
- **Source:** `runs/full-20260904/split/test-identity-strict.csv`;
  `src/deepfake_detection/data/manifest.py`.

#### 53. DFDC was not locked, and it chose two models

- **What went wrong:** Only MNW was evaluation-only. DFDC chose the BatchNorm
  arm (0.5005 frozen against 0.6482 live) and the six-checkpoint ensemble,
  picked after comparing three groupings on DFDC.
- **How it was caught:** Reading the guard against F36 and the BatchNorm help
  text.
- **What it cost:** The ensemble's 0.7739 and the BatchNorm comparison are
  partly in-sample. P-001 keeps live BatchNorm to avoid a second selection and
  discloses it. MNW, the one locked set, was also scored by at least four
  checkpoints (pilot, baseline, program, ensemble).
- **Fix:** DFDC, Celeb-DF v2, DeepSpeak v2, In-the-Wild, KoDF and HiFi-AVDF are
  now evaluation-only.
- **Source:** `src/deepfake_detection/data/guards.py`;
  `docs/research/findings.md` F36; `docs/handoff.md`, Recorded evaluation
  reports.

#### 54. DFDC results carried the Celeb-DF scope

- **What went wrong:** The runners wrote DFDC metrics under evidence scope
  `generalization_celebdf`. The generated table also shows the fusion DFDC
  report labelled dataset FakeAVCeleb, and three sync reports with scope `?`
  and 0 rows.
- **How it was caught:** Reading the evidence table.
- **What it cost:** Seven DFDC visual reports are filed under the Celeb-DF scope.
  Filtering by scope mixes two cross-corpus tests.
- **Fix:** New scope `generalization_cross_corpus` in the runners. Existing
  report files and the fusion and sync labels are open.
- **Source:** `docs/handoff.md`, Recorded evaluation reports;
  `src/deepfake_detection/experiments/scopes.py`; `scripts/run_program.ps1`;
  `scripts/run_seeds.ps1`.

#### 55. Decisions used a fixed 0.5 threshold on 10:1 fake-heavy data

- **What went wrong:** The drivers passed `--threshold 0.5` instead of a
  threshold chosen on validation.
- **How it was caught:** Reviewing the evaluate command.
- **What it cost:** Every accuracy, recall and false-alarm rate at 0.5 depends
  on prevalence. For example DFDC fusion precision 0.9904 with recall 0.2820
  (F6). F16 already said Youden's J is the wrong rule for deployment.
- **Fix:** `--threshold-file` records its split and clip ids, and the command
  refuses a threshold chosen on the clips it scores.
- **Source:** `src/deepfake_detection/cli/evaluate.py`;
  `docs/research/findings.md` F6, F16.

#### 56. EER used the nearest ROC point

- **What went wrong:** EER took the ROC point closest to FPR = FNR, with no
  interpolation.
- **How it was caught:** Code review.
- **What it cost:** On small sets such as MNW (134 clips) the nearest point can
  be several points off. Every root-study EER is affected.
- **Fix:** Linear interpolation in `equal_error_rate`.
- **Source:** `src/deepfake_detection/evaluation/metrics.py`.

#### 57. DeepSpeak v2 was described as on disk

- **What went wrong:** DeepSpeak v2 was described as "on disk". The folder is
  empty and the access request still awaits author review.
- **How it was caught:** Listing `data/DeepSpeak_v2`.
- **What it cost:** Plans assumed a non-VoxCeleb2 audio-visual corpus we do not
  have. The committed handoff says access is pending, so the claim did not
  reach a committed doc.
- **Fix:** Corrected. Access still open.
- **Source:** `data/DeepSpeak_v2`; `docs/handoff.md`, Current state.

#### 58. The project history links to a folder that no longer exists

- **What went wrong:** Commit `fb463ba` removed `team/`. `docs/project-history.md`
  still links `team/README.md`, `team/docs/EXPERIMENTS.md`,
  `team/docs/DECISIONS.md` and `team/docs/POST_FREEZE_BENCHMARK_COMPARISON.md`,
  and says the submission is under `team/submission/`.
- **How it was caught:** Checking citations for this log.
- **What it cost:** Six dangling links and one stale path in the document readers start with.
  The submitted tree is reachable only through tag `team/submission-final`.
- **Fix:** Open. Point the links at `results/legacy-team/` or the tag.
- **Source:** `docs/project-history.md`, lines 140 to 242; commit `fb463ba`.

#### 59. The first neutralizer put the shortcut back

- **What went wrong:** Neutralizer v1 cut clips with an ffmpeg input seek and normalized loudness with `loudnorm`. Re-muxing exposed each source's AAC priming differently. Face-swapped clips gained 28 ms of leading silence that real clips did not. Duration and trailing silence still differed by class too.
- **How it was caught:** By preregistered test H1, the audit rerun on the neutral copies (2026-11-13). The ceiling was 0.933, where the bar was 0.60.
- **What it cost:** One full neutralization (about two hours) and the feature extraction already started on it. No model was trained on v1 copies.
- **Fix:** Neutralizer v4 handles the audio as samples in numpy, starts each cut on a whole frame, fixes every clip at 3.00 s, and dithers away synthetic digital zeros. It was checked on a 600-clip sample first: ceilings 0.598 (clip), 0.489 (video), 0.702 (audio). A regression test builds two differently encoded sources and requires identical edges.
- **Source:** `results/shortcut-audit/fakeavceleb-neutral-v1/summary.json`; P-001 amendment A4; commit `438fb23`.

#### 60. The preregistration contradicted itself

- **What went wrong:** P-001 listed LAV-DF train as training data and also used LAV-DF test as a cross-corpus target and as the target of a transfer test.
- **How it was caught:** While planning the cross-corpus runs, before any result was read.
- **What it cost:** Nothing measured. It would have made the H2 and H3 cross-corpus claims false.
- **Fix:** Amendment A2: primary models train on FakeAVCeleb only. Reverse models train on LAV-DF only, for H3.
- **Source:** P-001 amendment A2; commit `371e087`.

#### 61. The feature extractor probed every file before starting

- **What went wrong:** `extract_avhubert_features.py` asked ffprobe for each clip's length one file at a time before extracting anything. On 21,544 clips that took most of an hour.
- **How it was caught:** The run had written nothing after 50 minutes.
- **What it cost:** About an hour of an overnight run.
- **Fix:** Read lengths from the neutralizer's sidecar and probe the rest in parallel.
- **Source:** commit `3b83b43`.

#### 62. Resume treated temporary failures as finished

- **What went wrong:** Under memory pressure, ffmpeg could not start (Windows error 0xC0000142). The feature extractor and the neutralizer wrote those failures into their resume records, and on restart they would have skipped those clips for good.
- **How it was caught:** All 182 windows written in the last minutes before the memory stop were errors, and the same window extracted fine on its own.
- **What it cost:** Nothing measured. The bad index rows were removed before the rerun.
- **Fix:** Only final outcomes count as done: "ok", "no face", "no audio", and refusals for length.
- **Source:** commits `78dc2e3`, `033e3a9`.

#### 63. The receipt tool could not launch anything on Windows

- **What went wrong:** `scripts/research_artifacts.py` passed `.venv/Scripts/python.exe` to Windows as a relative path with forward slashes. Windows will not start a program from that, so every receipted stage failed at launch.
- **How it was caught:** The first receipted sync evaluation. The visual and audio driver would have failed the same way at its first training step.
- **What it cost:** Nothing measured.
- **Fix:** Resolve a relative program path before launching. The receipt still records the portable relative command.
- **Source:** commit `582a577`.

## 7. What a hard reviewer would say

These are the strongest objections we expect, with our current answer. "Open" means we have no answer yet and the paper must not pretend otherwise.

1. **"Fusion of audio, video and sync is not new."** True. AVFF (CVPR 2024), AVH-Align (CVPR 2025) and HAVIC (CVPR 2026 Findings) all fuse modalities. *Answer:* we do not claim the architecture. We claim the measurements in section 9, run under a protocol nobody else uses.

2. **"Your neutralization may remove real evidence along with the shortcut."** Re-encoding at CRF 23 can blur fine blending artifacts. Trimming silence can remove a real generator's trace. *Answer: open.* We must show what neutralization costs. Train on raw and test on raw, then train on neutral and test on neutral, on LAV-DF, where the codec is not confounded with the label. If the neutral model loses little there, the protocol removes the shortcut and keeps the signal.

3. **"FakeAVCeleb has about 500 real videos. Your intervals will be too wide to support anything."** *Answer:* partly. We report n and the interval beside every number, and we never rank on a difference whose interval spans zero. LAV-DF and, if access arrives, AV-Deepfake1M carry the statistical weight.

4. **"AV-HuBERT was pretrained on VoxCeleb2 and LRS3, the same sources as your data."** *Answer: open.* We will check speaker overlap between the pretraining lists and our test speakers, and report results on the non-overlapping subset.

5. **"Your history shows seed spread larger than your effects."** Earlier runs scored only seed 17, the best of three: DFDC 0.7611 against a mean of 0.7151. *Answer:* every new claim uses three seeds and reports their spread. No result is picked from a single seed.

6. **"Conformal guarantees need exchangeability. Real deployment traffic drifts."** *Answer:* true, and that is the point of H4. We measure how far the guarantee survives each shift, and we report how many real target clips it needs (59 for alpha = delta = 0.05).

7. **"Modality attribution is tested on two corpora only."** *Answer:* yes. That is a stated limit. AV-Deepfake1M would add a third.

8. **"You do not compare with the state of the art on its own protocol."** *Answer: must do.* Report our model on the common random FakeAVCeleb split too, and run AVH-Align (public code and weights) on our protocol. A reader then sees both views.

9. **"A codec census is trivial."** *Answer:* it takes an afternoon, and no published AV deepfake paper reports one. The shortcut ceiling it gives is the bar every detector on this data should beat.

10. **"Earlier versions of this project made claims that were later withdrawn."** *Answer:* yes. Section 6 lists every one. The new study is preregistered, and its results are committed with per-clip predictions.

## 8. What to do better, ranked

Ranked by how much each step strengthens the paper per day of work.

| Rank | Step | Why | Cost |
|---|---|---|---|
| 1 | Finish the shortcut audit on FakeAVCeleb, LAV-DF and DFDC; rerun it on neutral copies (H1) | It is the protocol's foundation and the cheapest new result | hours, CPU |
| 2 | Sign the AV-Deepfake1M EULA | A third corpus with per-modality labels and unseen speakers; AVH-Align's own training data | 15 minutes of your time |
| 3 | Train the three paths and fusion on neutral data, 3 seeds (H2) | The core comparison | about 2 GPU days |
| 4 | Run AVH-Align on our protocol | A reproduced baseline beats a quoted one | half a day |
| 5 | Neutralization cost on LAV-DF (objection 2) | Without it the protocol is open to "you threw the signal away" | half a GPU day |
| 6 | Real-only threshold under codec and corpus shift (H4) | The deployment result; CPU only once scores exist | a day |
| 7 | Modality transfer, FakeAVCeleb <-> LAV-DF (H3) | The second new claim | a GPU day |
| 8 | Speaker overlap with AV-HuBERT pretraining (objection 4) | Closes an obvious hole | a day |
| 9 | HiFi-AVDF (public, added back by P-001 A5) as a locked test on six modern generators; run public baselines (AVH-Align, HAVIC) on our data. KoDF requested | A run baseline is fairer than a quoted table | 6.5 GB download |
| 10 | Trailing-silence and loudness cues on LAV-DF and DFDC | Extends the audit beyond FakeAVCeleb | hours |

What not to do:
- Do not add a new backbone or a bigger model before steps 1 to 5 are done. It would make the paper look like every other one and answer none of the open objections.
- Do not tune anything on a locked target.

## 9. The paper: objectives, related work, venue

Working title: *Which stream lied? Shortcut-neutral audio-visual deepfake detection with modality prediction and real-only false-alarm control.*

The objectives are preregistered in [P-001](../results/preregistration/P-001-shortcut-neutral-av-study.md). The related-work check that shaped them is in [research/related-work.md](research/related-work.md), with sources in [research/related-work/](research/related-work/).

| Objective | What is new | What already exists |
|---|---|---|
| O4 Shortcut-neutral protocol | Per-class codec and silence census for AV data; a shortcut ceiling; neutral copies of train and test; a trailing-silence fix | Leading silence documented and fixed at test time only (Smeu et al., CVPR 2025); DeepFake Doctor (2025) on silence and method splits |
| O1 Fusion vs visual-only | The comparison after neutralization, with three seeds and identity bootstrap | Many fusion detectors, none evaluated on shortcut-free data |
| O2 Which modality was faked | Transfer between FakeAVCeleb and LAV-DF; whether clip labels hide what modality labels show | Dual-label heads in-dataset (Yu et al. 2023/2025, Multimodaltrace 2023, MRDF 2024) |
| O3 False alarms under shift | Real-only calibration for AV video under codec re-encoding, with clip counts and per-modality cost | Concurrent work for images (Mahdaoui, Nov 2026) and speech (PGA-G, Oct 2026); the conformal method itself is standard |

Call O2 "manipulated-modality prediction", not "attribution". AMDD (2026) uses "attribution" to mean naming the generator.

**Venue.** Check each deadline on the venue's own site before planning around it. The audio-heavy findings (silence, loudness, the audio path) fit Interspeech. The protocol and the AV results fit a CVPR or ICCV media-forensics workshop. A journal (IEEE TIFS, or IEEE Signal Processing Letters for a short version) has no deadline and suits a negative or mixed result. Decide after H1 and H2 are in.

## 10. Where things live

| What | Where |
|---|---|
| This handbook | `docs/HANDBOOK.md` |
| Preregistration | `results/preregistration/` |
| New results | `results/<experiment>/` (tracked) and `runs/<run>/` (large, ignored) |
| Older root-study results and their status | [research/result-traceability.md](research/result-traceability.md), [research/findings.md](research/findings.md) |
| Team project records | [results/legacy-team/](../results/legacy-team/README.md); submitted tree at tag `team/submission-final` |
| Research design and protocol | [research-design.md](research-design.md) |
| Data card, model card, threat model | [data-card.md](data-card.md), [model-card.md](model-card.md), [threat-model.md](threat-model.md) |
| Dataset notes and access | [research/datasets.md](research/datasets.md) |
| Change record | [CHANGELOG.md](../CHANGELOG.md) |
| Paper draft | [research/paper.md](research/paper.md), [main.tex](main.tex) |

## 11. Glossary

- **AUC (ROC-AUC).** The chance that a random fake scores higher than a random real clip. 0.5 is guessing, 1.0 is perfect. It ignores the threshold.
- **Balanced accuracy.** The mean of the true-positive and true-negative rates. It is not fooled by a 10:1 class imbalance.
- **Bootstrap, identity-level.** Resample whole people, not clips, a thousand times, and read the 95% interval from the spread. Clips of one person are not independent, so resampling clips would give intervals that are too narrow.
- **Conformal threshold.** A cut-off set from scores of real clips alone, so that a new real clip is flagged with probability at most alpha.
- **Cross-corpus.** Trained on one dataset, tested on another the model never saw.
- **EER.** The error rate where false positives equal false negatives.
- **Evaluation-only (locked).** A dataset that may be scored but never used to train, pick a model, or set a threshold.
- **Identity-strict subset.** Test clips whose every named person belongs to the test partition only.
- **Neutralization.** Re-encoding every clip identically, cutting edge silence and normalizing loudness, so file-level cues carry no label information.
- **Out-of-fold.** A score for a clip from a model that never trained on that clip, used to train the fusion layer without leakage.
- **Preregistration.** Writing down the hypothesis, the test and the success rule, and committing it before the run.
- **Shortcut.** A cue that predicts the label in a dataset but is not the manipulation, such as the codec.
- **Shortcut ceiling.** The best AUC a classifier gets from shortcut cues alone. A detector must beat it to show it sees the manipulation.

## 12. History

How the project got here, from the first commit on 2026-07-04. The failures along the way are listed one by one in [section 6](#6-mistakes-start-to-finish); this is the story that connects them. Historical numbers below stay subject to the corrections in the [result registry](research/result-traceability.md).

| Dates | Phase |
|---|---|
| 2026-07-04 to 07-23 | First plan and prototype |
| 2026-08-01 to 08-05 | First trained visual streams |
| 2026-08-24 to 09-04 | Rebuild as a research platform |
| 2026-09-09 to 09-15 | Full pipeline, first results |
| 2026-10-13 to 10-22 | Seeds, challengers and the paper draft |
| 2026-10-24 to 10-29 | Team project with a sealed test |
| 2026-10-27 to 10-29 | Audit of the root study, claims withdrawn |
| 2026-10-31 | Exploratory post-freeze evaluations E-033 and E-034 |
| 2026-11-02 | AV-HuBERT experiments E-035 and E-036 preregistered |
| 2026-11-12 | Both histories joined; research materials and equal credit aligned |


### H1. The first plan (July)

The first design had five streams: three visual models (Xception, EfficientNet
and DINOv2) and two cross-modal streams (lip-sync and emotion). The idea was
that a lip-sync fake changes only the mouth, so the frames look real. The fake
lives in the mismatch between sound and face. The primary dataset was
FakeAVCeleb, with identity-disjoint splits from day one.

On 2026-07-09 the design changed in three ways (commit `d06fd6b`):

- **Feature-level fusion instead of score averaging.** The streams pass
  embeddings, not scores, to one joint model.
- **Embedding cross-attention for lip-sync.** We dropped the Bohacek and Farid
  method, which lip-reads, transcribes and compares words. Our stream compares
  audio and mouth embeddings directly, with no text in between.
- **No Swin.** DINOv2 became the third visual stream, because a self-supervised
  backbone was expected to transfer better than two artifact CNNs.

Adithya built the Stage 1 data pipeline and the first EfficientNet stream
(`d30cc0a`). On 2026-07-23 the code was snapshotted and taken apart so it could
be rebuilt one step at a time (`926624a`). That snapshot records five places
where the plan and the code disagreed. The full plan of that time is still in
history: `git show 926624a^:PROJECT_OVERVIEW.md`.

In early August Adithya added the trainer, resume support, and trained Xception
and DINOv2 streams (`bf09253` to `81c04db`). DINOv2 was then swapped for DINOv3
(`60b0f0f`).

### H2. Rebuild as a research platform (late August)

The prototype could train models but could not prove anything. So on
2026-08-24 the project restarted from a new root commit with evaluation as the
first concern:

- source-disjoint and identity-strict splits, checked in code;
- a preprocessing hash and a split hash stored with every checkpoint;
- local MLflow tracking, versioned configs and a CPU smoke run in CI;
- a handbook that teaches each part from first principles.

The first baseline, EfficientNet-B0 plus a GRU, scored 0.9992 ROC-AUC on
FakeAVCeleb validation. Four more runs at the same settings showed that the seed
alone moves validation loss by 2.8 times ([status](handoff.md), "Run
comparison"). That was the first sign that one run is one draw, not a result.

A planned human-reviewed face-detector benchmark was dropped. Nobody was going
to label the 625 frames it needed, so the decision is recorded as a decision,
not a measurement (roadmap, Phase 2).

### H3. Full pipeline and the first surprises (September)

All three branches (visual, audio, sync) and the fusion head were trained at
full scale. Three results changed how the project worked:

- **The selection rule was the result.** Two streams picked their checkpoint by
  validation loss. One kept an epoch-1 checkpoint that scored 0.4668, below
  chance. Both results were withdrawn and rerun with ROC-AUC selection (F3,
  `b63cf87`).
- **In-domain numbers were shortcuts.** Three times a near-perfect in-domain
  score collapsed elsewhere: 0.9990 against 0.4920 cross-corpus, a 0.9969
  matched-pair stream reading 0.5059 under new capture conditions, and 0.9997 on
  Veo 3 that was entirely a content confound. After that, leave-one-generator-out
  became the only headline metric (`1395e2d`).
- **False alarms on real video range from 0 to 91 percent**, depending on
  capture conditions (F16). A density model over genuine footage cut false
  accusations by about two thirds (F17).

Other results from this phase: training on FaceForensics++ did not improve
transfer (F10). Xception was removed on an argued redundancy with EfficientNet,
and the commit says it was argued, not measured (`3284514`). DF26's audio was
audited before use and found unusable for audio-visual work (F14).

### H4. Seeds, challengers and the central finding (October)

Every branch and the fusion head were retrained at seeds 17, 29 and 43.

- **Seed beats every effect.** Cross-corpus on DFDC, the fused score moves 0.1078
  with the seed. The 0.7500 headline was the best of three; the mean is 0.6873
  (F27). At three seeds, fusion never beats the visual branch on DFDC (F19).
- **The intended mechanism is unproven.** The cross-attention streams logged diagonal
  attention mass every epoch, never as part of the loss. It sat at chance in
  every run while the emotion stream reached 0.9991 ROC-AUC (F1). The model is
  accurate in those runs, but diagonal attention does not establish its use of timing.
- **Two of three branches do not transfer.** The sync branch is at or below
  chance at all three seeds (F26). The audio branch reads 0.5054 on DFDC against
  0.7739 in-domain (F30).
- **An unseen manipulation family costs 0.12.** On FaceForensics++, macro
  ROC-AUC falls from 0.8743 to 0.7511, and FaceSwap alone falls to chance (F34).
- **The visual model reads the codec.** The served models respond to the video
  encoding, not only the face (F35). Training that shortcut away cost more on
  DFDC than it gained (F37).

A second track tested a frozen encoder with a linear probe on generated video
(DF26). In DF26, whether a clip has an audio track nearly predicts its label.
A user web app, an in-browser demo and a LaTeX paper draft followed.

### H5. The team project (late October)

From 2026-10-24 Shivansh Gupta, Vaibhav Shiroorkar and Adithya Jithesh built a
second system for the final-year submission, supervised by Dr. Tejashree Phatak.
It lives in [`team/`](../results/legacy-team/README.md). Both research lines receive equal team credit.
The team experiments used a separate protocol and reused available datasets.

It was built to a stricter protocol than this project had used:

- each hypothesis and its pass line written down before the run;
- a test split sealed and read once, after the system was frozen, with a ledger
  of every read;
- three confirmatory tests declared in advance, with a Bonferroni correction.

Its main results ([team experiments](../results/legacy-team/EXPERIMENTS.md), E-014):

- video, Celeb-DF family: 0.6238 at the MVP to 0.8218 for the final system;
- speech, In-the-Wild: 0.7516 to 0.9148;
- the advantage over an Xception baseline holds at the 95 percent level but not
  at the corrected 98.33 percent level, and the paper says so.

It found the same two problems this project found, with different code, data
splits and encoders:

| Problem | This project | Team project |
|---|---|---|
| The audio-visual signal is not audio-visual | attention at chance while accuracy is high (F1) | visual head at chance within produced LAV-DF videos, 0.5032 (E-014) |
| File format beats content | visual models read the codec (F35) | codec metadata alone separates real from fake FakeAVCeleb video at 0.9985 (E-021) |

These are related observations from different protocols, not a controlled replication.
The root study's LAV-DF window defect also prevents treating its old score as valid corroboration.

The historical submission is preserved under `team/submission/` and tag `team/submission-final`.
Its checklist dates the final files to 2026-10-30. File preparation does not establish an external submission date.

### H6. Audit and withdrawn claims (late October)

Reading this project against the team protocol exposed three faults
(`5d4285d`, [status](handoff.md)):

- **Fold leakage.** Fusion fold checkpoints chose their epoch using labels from
  the holdout they were later scored on.
- **Shifted LAV-DF windows.** A default shifted windows that started at zero, so
  some "genuine" windows overlapped manipulated spans. The 0.9969 lip-sync score
  must be rerun before it is read.
- **Wrong input normalization** for the Perception Encoder and CLIP features
  (F33). The old 0.7527 open-world score does not stand.

The code was fixed and every affected row is marked `requires rerun` in the
[result registry](research/result-traceability.md). Three of the four stated
targets now wait on reruns. The withdrawn numbers stay in the record with the
reason.

### What we changed because of mistakes

Each fault became a rule, so it cannot recur silently:

| Mistake | Rule now |
|---|---|
| Checkpoint chosen on validation loss kept a below-chance model | Select on the metric that is reported |
| One seed reported as the result | Three seeds, mean and spread, best-of-three named as such |
| In-domain near-perfect scores read as success | Leave-one-generator-out and cross-corpus are the headline |
| Fold checkpoints saw holdout labels | Code rejects any OOF export that saw its own labels |
| Test data reachable during tuning | Sealed split, read once, ledger entry per read (team protocol) |
| A number in a doc with no file behind it | Every number resolves to a registry row and a file hash |

### Still open

- Rerun the cross-fit seeds, rebuild the LAV-DF cache, and re-extract the PE and
  CLIP features. Commands are in [status](handoff.md).
- The MNW visual scores sit near 0 ROC-AUC on 85 clips. A ranking that is almost
  perfectly inverted usually means a flipped label. Quick look, not a
  measurement: check label polarity before reading it as a result.
- Test the audio-visual claim causally: shift or swap the audio at test time and
  see whether the score moves. Attention at chance shows the model is not
  aligned; it does not yet show the model ignores timing.
- The team's preregistered E-035 (zero-shot AV-HuBERT on LAV-DF windows) and
  E-036 (trained heads) are the next audio-visual runs.

### H7. Post-freeze evaluations and the current research package

E-033 evaluated the frozen video and speech heads on previously unread LAV-DF samples.
E-034 evaluated the image head on previously unread Community Forensics-Eval shards.
Both have appended results in the [team experiment log](../results/legacy-team/EXPERIMENTS.md).
Their preregistered blocks still say PLANNED because those blocks preserve the ordering record.
These are exploratory evaluations on seen distributions, not new confirmatory tests.

The post-freeze comparison (`docs/POST_FREEZE_BENCHMARK_COMPARISON.md` at tag `team/submission-final`) records weak transfer and format sensitivity.
For example, the image head's pooled AUROC was 0.7909 on 7,209 images.
Its PNG-only diagnostic was 0.5693 on 368 images. The populations differ, so this does not prove a format effect.

On 2026-11-02, E-035 and E-036 preregistered zero-shot and trained AV-HuBERT heads.
Their result slots remain empty. Implementation and preregistration are progress, not measured accuracy.

On 2026-11-12, the repository joined both histories and linked the July prototype history.
The [current presentation set](research/presentation/README.md) brings these phases together.
It includes the post-freeze findings while keeping the historical submission identifiable.

