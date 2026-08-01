# Audio-Visual Deepfake Detection: Project Brief

A single self-contained account of the project: the problem, the research behind the
approach, the system as designed, what is built today, what remains, and how it will be
evaluated. Written to be read on its own, converted to slides or a document, or pasted
into a chat tool as context.

**Source of truth:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) remains the authoritative
engineering document. This brief is derived from it plus the ten stage plans, the
preprocessing reference, and the code. Where the two disagree, PROJECT_OVERVIEW.md wins.

**Status date:** 2026-08-01. Team of three. Academic research project.

---

## Contents

1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [Problem and motivation](#2-problem-and-motivation)
3. [The research hypothesis](#3-the-research-hypothesis)
4. [Related work and where we sit in it](#4-related-work-and-where-we-sit-in-it)
5. [What is novel here](#5-what-is-novel-here)
6. [System architecture](#6-system-architecture)
7. [Preprocessing: raw clip to tensors](#7-preprocessing-raw-clip-to-tensors)
8. [Data, splits, and the leakage discipline](#8-data-splits-and-the-leakage-discipline)
9. [The five streams in detail](#9-the-five-streams-in-detail)
10. [Fusion](#10-fusion)
11. [Evaluation plan](#11-evaluation-plan)
12. [Explainability](#12-explainability)
13. [Tooling, environment, and engineering discipline](#13-tooling-environment-and-engineering-discipline)
14. [The inspection dashboard](#14-the-inspection-dashboard)
15. [Roadmap: the ten stages](#15-roadmap-the-ten-stages)
16. [Current status and results so far](#16-current-status-and-results-so-far)
17. [Risks, open decisions, and known contradictions](#17-risks-open-decisions-and-known-contradictions)
18. [Repository layout](#18-repository-layout)
19. [Glossary](#19-glossary)
20. [References](#20-references)
21. [Suggested slide outline](#21-suggested-slide-outline)
22. [Anticipated questions](#22-anticipated-questions)

---

## 1. The one-paragraph version

We are building an audio-visual deepfake detector: one that examines a clip's video track,
its audio track, and crucially the relationship between the two. The reason is that
"deepfake" is not one thing. A face swap rewrites most of a face and leaves manufacturing
residue everywhere. A lip-sync repaint alters a few percent of the frame and leaves almost
none. A cloned voice over genuine footage alters no pixels at all. These are different
manipulations that hide in different places, and no single detector covers them. So the
system applies **three detection principles to the same clip**: visual artifact analysis,
audio-visual synchrony, and audio-visual affective consistency. Five streams implement
them, three visual and two cross-modal, and each emits an embedding rather than a score. A
fusion network concatenates those embeddings and learns from the combination, so it can
represent conjunctions like "artifact evidence is weak but synchrony evidence is strong",
which is what distinguishes one manipulation family from another and is something no
average of five scores can express.

---

## 2. Problem and motivation

### What a deepfake is

Media altered by AI to show something that never happened. Three common forms, which map
almost exactly onto the three detection principles:

1. A face swapped onto another person's body.
2. Mouth movement changed so a person appears to say words they never said.
3. A cloned voice speaking sentences the real person never spoke.

### One label, three different problems

The central design fact of this project is that these three forms leave evidence in
completely different places. Our primary dataset, FakeAVCeleb, contains all of them, and
the proportions matter:

| Manipulation family | Methods in FakeAVCeleb | Clips | Where the evidence is |
|---|---|---|---|
| **Face swap** | `fsgan`, `faceswap` | 4,694 | In the pixels. Blending seams, colour drift between face and neck, warping at the jaw, GAN upsampling patterns. Plenty of residue. |
| **Lip-sync repaint** | `wav2lip`, `fsgan-wav2lip`, `faceswap-wav2lip` | 15,872 | Barely in the pixels. Mostly in the timing relationship between visible articulation and sound. |
| **Voice clone on genuine video** | `rtvc` | 500 | Not in the pixels at all. Only in the relationship between voice and face. |
| Genuine | `real` | 500 | n/a |

Read the table as three separate detection problems wearing one label:

- **Face swaps are the tractable case.** The whole face is synthetic, so residue is
  everywhere, and a convolutional network trained on artifacts does well. Roughly 4,700
  clips here have no lip-sync manipulation whatsoever, and the visual streams are the only
  thing that can catch them. This is why three of the five streams are visual and why we do
  not treat visual detection as a solved warm-up.
- **Lip-sync repaints are the hard case.** A tool such as wav2lip is handed genuine video
  and a target audio track and repaints only the mouth region. Identity, hair, background,
  lighting, camera grain and the whole upper face are untouched original footage. Two
  consequences: there is very little artifact surface to work with, and what surface exists
  is low-amplitude high-frequency detail, exactly what H.264 re-encoding and social-media
  transcoding destroy first. A detector that scores well on face swaps can do poorly here
  for a simple reason: there is barely anything to see.
- **Voice cloning over real video is invisible to vision by construction.** FakeAVCeleb's
  `RealVideo-FakeAudio` category has a genuine video track and a synthetic audio track. To
  a visual-only stream these clips are real, and correctly so. Nothing that never listens
  can catch them.

### Why that forces an audio-visual design

Two of the three families defeat any purely visual approach, and the third is invisible to
any purely audio approach. What the last two families cannot repair is agreement between
the tracks. A real recording captures one physical event twice, once as light off a moving
mouth and once as the sound that mouth made, and the two views are causally locked
together. Synthesis breaks the lock: the generated mouth is plausible on its own and the
audio is plausible on its own, but their joint behaviour no longer belongs to one event.
That relationship is a higher-level property than pixel residue, so it also survives
compression better.

So the system measures three things rather than one, and the ablation in Stage 7 is what
establishes how much each contributes.

---

## 3. The research hypothesis

**Central claim.** Manipulation families leave evidence in different places, so a detector
built from complementary signals covers more of the space than any single signal can. Three
principles, applied to the same clip:

| Principle | Question it asks | Streams | Best against |
|---|---|---|---|
| **Visual artifacts** | Does this face show manufacturing residue? | Xception, EfficientNet-B0, DINOv2 | Face swaps, and any fake with pixel-level residue |
| **Audio-visual synchrony** | Do the mouth's movements and the sound belong to one event? | Lip-sync | Lip-sync repaints, voice clones |
| **Audio-visual affect** | Does the emotion on the face match the emotion in the voice? | Emotion | Voice clones, and any fake where affect drifts |

**Sub-claim, and the part that is genuinely being tested.** The cross-modal principles will
catch partial fakes that artifact detectors miss, and will degrade less under compression,
because correspondence between two tracks is a higher-level property than pixel residue.
Stage 7's ablation and Stage 9's robustness curves are where this is confirmed or refuted.

**Two corollaries we designed around.**

- *Streams must emit embeddings, not scores.* If each stream reduces its evidence to a
  single probability, the combination step can only average opinions. Averaging cannot
  represent "the visual evidence is weak, the sync evidence is strong, and that particular
  combination points at one manipulation family rather than another". Feature-level fusion
  can.
- *A standalone audio-only classifier is not the answer.* Asking "does this voice sound
  synthetic" is a different question from "do this voice and this face belong to the same
  event". Only the second one catches a real voice pasted onto a manipulated mouth, or a
  cloned voice on genuine video where the audio itself is high quality. We deliberately do
  not build an audio-only stream. It could be added later as one more fusion input if the
  ablation shows a gap, but it is excluded on purpose, not by oversight.

**A correction we keep on the record as a warning.** An earlier plan split the system into
separate spatial, temporal, and standalone audio models. That was wrong for this problem
and was reversed. A standalone audio model cannot tell whether audio and video agree, and
the disagreement is where the fake hides. The system is audio-visual by construction.

---

## 4. Related work and where we sit in it

### Surveys we work from

- Hashmi et al. (2024), *Understanding Audiovisual Deepfake Detection Techniques,
  Challenges, Human Factors and Perceptual Insights*.
- Khan, Khan and Ahmad (2025), *A Comprehensive Survey of DeepFake Generation and
  Detection Techniques in Audio-Visual Media*.

### The architectures we implement

| Work | What we take from it |
|---|---|
| Chollet (2017), Xception | Depthwise separable convolutions. Our low-level artifact detector: blending edges, colour inconsistency. |
| Tan and Le (2019), EfficientNet | Compound scaling. A second artifact-focused CNN view at a fraction of the parameters. |
| Oquab et al. (2023), DINOv2 | Self-supervised ViT features learned with no fake/real labels, so they describe images generally instead of memorizing one generator's fingerprint. Our generalization bet. |
| Vaswani et al. (2017), Attention Is All You Need | Scaled dot-product cross-attention, `softmax(QK^T / sqrt(d)) V`, the mechanism both cross-modal streams are built on. |
| Chung and Zisserman (2016), SyncNet | Closest prior art for the lip-sync stream: audio-visual synchrony as an embedding-space temporal check, not a semantic one. |
| Mittal et al. (2020), Emotions Don't Lie | Conceptual basis for the emotion-mismatch stream. We implement the idea with cross-attention on learned embeddings rather than their original scoring method. |

### Work we read and deliberately do not implement

- **Bohacek and Farid (2024), *Lost in Translation: Lip-Sync Deepfake Detection from
  Audio-Video Mismatch*.** They lip-read the video into text, transcribe the audio into
  text, and compare the words. We considered this and rejected it: our lip-sync stream is
  embeddings end to end, with no transcription or lip-reading-to-text anywhere. Their
  method depends on two error-prone semantic decoders and on the speech being intelligible
  language; an embedding sync check does not. Kept as a benchmark and a possible future
  stream.
- **Oorloff et al. (2024), AVFF: Audio-Visual Feature Fusion.** No public official code and
  a heavy two-stage pretraining recipe, too much to reproduce from zero. Used as a
  benchmark to compare against, and as the inspiration for the optional Stage 8
  self-supervised pretraining experiment.
- **Zhou and Lim (2021), Joint Audio-Visual Deepfake Detection.** Comparison point.

---

## 5. What is novel here

1. **The specific combination.** No paper in our reading list fuses cross-modal mismatch
   cues from *both* lip-sync and emotion, both via cross-attention on learned embeddings,
   at the *feature* level, alongside three strong visual backbones. The pieces exist
   separately; this arrangement of them does not.
2. **Feature-level rather than score-level fusion of mismatch cues.** Most multi-signal
   deepfake work averages or logistic-regresses per-model scores. We hand the fusion model
   the streams' internal representations so it can learn interaction effects between weak
   artifact evidence and strong mismatch evidence.
3. **An honest generalization test.** We evaluate on Deepfake-Eval-2024, real in-the-wild
   deepfakes with unseen generators and messy conditions, never touched during training,
   in addition to clean in-distribution FakeAVCeleb numbers. We also hold out entire
   manipulation methods and measure degradation under compression and noise.
4. **A full ablation that decides the architecture.** Each stream must earn its place in a
   table, not in a discussion. Dropping a stream because the data said so is a result.
5. **Per-stream attribution as explainability.** Because the feature store already holds
   every stream's embedding per clip, we can show which stream caught which fake, plus
   Grad-CAM on the surviving visual streams to show where on the face they look.

---

## 6. System architecture

```
                      one clip (mp4)
                            |
        +-------------------+-------------------+
        |          preprocessing (shared)       |
        |  16 timestamps, one video path and    |
        |  one audio path over the SAME stamps  |
        +-------------------+-------------------+
                            |
     faces [16,3,224,224]   mouth [16,3,96,96]   audio [16,5600]
        |         |    |            |                |     |
        |         |    +------------+----------------+     |
        v         v                 v                      v
  +-----------+ +-----------+  +-----------+  +--------------------+
  | Xception  | | Efficient |  | DINOv2    |  |  lip-sync stream   |
  |           | | Net-B0    |  |           |  |  video x audio     |
  +-----------+ +-----------+  +-----------+  |  cross-attention   |
        |             |              |        +--------------------+
        |             |              |                 |
        |             |              |        +--------------------+
        |             |              |        |  emotion stream    |
        |             |              |        |  face x voice      |
        |             |              |        |  cross-attention   |
        |             |              |        +--------------------+
        |             |              |                 |
        +------+------+------+-------+--------+--------+
                            |
                  each stream: Linear + LayerNorm
                    -> embedding of common_dim=256
                            |
                  feature store, keyed by clip_id
                            |
             concatenate k x 256  ->  fusion MLP  ->  sigmoid
                            |
                  P(fake) in [0,1], threshold 0.5
```

Three properties of this diagram are load-bearing:

- **The three visual streams never see audio.** They do not do lip-sync or emotion
  matching. That work belongs entirely to the cross-modal streams. Keeping this straight
  prevents a common misreading of the design.
- **Every stream ends in the same shape.** A `Linear` plus `LayerNorm` projection to
  `common_dim = 256` makes streams of different native dimensionality (2048, 1280, 384,
  512, 768, 1024) compatible with one concatenation.
- **The feature store is the seam.** Streams write embeddings keyed by `clip_id`; fusion
  reads them. Streams can therefore be trained separately, on different machines, at
  different times, and fusion never has to run five backbones at once.

---

## 7. Preprocessing: raw clip to tensors

Preprocessing is fully built. Every step is an individual pure function in
`preprocessing/ops/`, imported by both the batch pipeline (`preprocessing/extract_clip.py`)
and the inspection dashboard. There is exactly one implementation, never two.

### The output contract

```
faces  [16, 3, 224, 224]  float32, 5-point aligned, ImageNet-normalized
mouth  [16, 3,  96,  96]  float32, landmark-centered crop (for lip-sync, Stage 4)
audio  [16, 5600]         float32 waveform windows, 16 kHz, 0.35 s each
label  scalar int         1 = fake, 0 = real
```

`16 x 5600` is `NUM_FRAMES x (AUDIO_SR x AUDIO_WINDOW_SEC)`, that is 16 windows of 0.35
seconds at 16 kHz. Frame *i* and audio window *i* describe the same instant.

### Visual path, per sampled frame

1. **Frame sampling** (`audio.sample_timestamps`). 16 evenly spaced timestamps, inset by
   half an audio window so every frame's plus/minus 0.175 s of audio stays inside the clip.
2. **Face detection** (`faces.detect`). MTCNN from `facenet-pytorch`, most-confident face
   only, returning box, 5 landmarks and probability. A face must clear a confidence
   threshold, default 0.90.
3. **5-point alignment** (`faces.align_face`). A partial-affine transform (rotation, scale,
   translation, no shear) warps the detected landmarks onto a canonical ArcFace-style
   template. This pose-normalizes the face so the temporal model sees a stable face across
   frames rather than one that rolls and rescales with head motion. This was the single
   largest quality upgrade available without adding a heavier detector, and it used MTCNN's
   own landmarks, so it cost no new dependency.
4. **Crop and resize** (`faces.crop_and_resize`), the fallback path when alignment is off
   or landmarks are missing. Margin-padded bbox crop to 224x224. Everything stays RGB end
   to end.
5. **ImageNet normalization** (`faces.imagenet_normalize`). Applied once here for all three
   visual backbones, since all three are ImageNet-pretrained in `timm`.

**A bug worth telling in the write-up.** Aligned crops originally padded the region outside
the source frame by *reflection*. FakeAVCeleb clips are already tight face crops, so the
aligned canvas always overshoots the frame edge, by a measured 14 to 45 percent. Reflection
therefore pasted a mirrored, upside-down second face into essentially every training crop.
The fix was to pad black instead, and to stop insetting the alignment template. It is a
good slide: a plausible-looking preprocessing default silently corrupting the entire
dataset, caught by looking at the pictures in the dashboard.

### Mouth ROI

`faces.mouth_roi` returns a 96x96 crop centred on the two mouth-corner landmarks. It is a
*parallel* output, not a replacement: the visual and emotion streams keep the full face.
Note for Stage 4: AV-HuBERT-style preprocessing expects a grayscale, 68-landmark mean-face
aligned mouth ROI. Matching that exactly needs a 68-landmark model, a new dependency, so it
is deferred; today's landmark-centred RGB crop is the best available from MTCNN's 5 points.

### Audio path

1. **Decode** (`audio.decode`) via PyAV, which bundles its own ffmpeg, so no system ffmpeg
   is required anywhere in the project.
2. **Mono downmix** (`audio.downmix`), channel mean.
3. **Resample to 16 kHz** (`audio.resample`) via librosa.
4. **Leading-silence handling** (`audio.leading_silence_sec`). FakeAVCeleb has a known
   shortcut bug: fake-audio clips carry extra silence at t=0, which a model can learn to
   detect instead of learning anything about deepfakes. Trimming the audio would desync it
   from the frames. Instead, the measured offset is fed into `sample_timestamps` so *both*
   frame and audio sampling start past the silence. The shortcut is removed and the
   alignment is preserved.
5. **Window extraction** (`audio.extract_windows`). One 0.35 s window centred on each frame
   timestamp, clamped and zero-padded to a fixed length.

Frame-to-audio alignment is by **timestamp, not sample index**, so any later stage can
trace a frame back to exactly the samples it is paired with.

### Extras: the robustness layer

A separate, off-by-default set of ops used for robustness experiments and for the
dashboard's what-if toggles. Baseline, with all extras off, reproduces the real pipeline
exactly.

- Visual enhancement: sharpen, denoise, CLAHE.
- Visual degradation: gaussian blur, JPEG re-compress, downscale-upscale. These simulate
  the compression and quality loss seen in the wild and feed the Stage 9D robustness
  curves.
- Audio: spectral denoise, RMS normalize, bandpass, add noise, and mel-spectrogram as a
  visualization view.

### Caching

`extract_clip.py` writes `frames.npy`, `audio.npy` and `timestamps.npy` under
`data/processed/<clip_id>/` plus a `version.txt` stamped with `PIPELINE_VERSION`, currently
**3**. Bumping the version invalidates stale caches so they re-extract transparently.
Version history: v1 plain MTCNN crop, v2 alignment plus silence-aware sampling, v3 black
padding and no template inset.

---

## 8. Data, splits, and the leakage discipline

### Datasets

| Dataset | Role | Why |
|---|---|---|
| **FakeAVCeleb v1.2** | Primary train and test | The rare set with real *audio* manipulation, which is what makes cross-modal evaluation possible. 21,544 clips in our extraction. Roughly 500 real against 19,500 fake. |
| **Deepfake-Eval-2024** | Final in-the-wild test only, never trained on | Real-world deepfakes, unseen generators, messy conditions. This is the honesty test. |
| FaceForensics++, Celeb-DF | Optional visual-only baselines | Both are visual-only with no manipulated audio, so they cannot test cross-modal mismatch. |

### FakeAVCeleb categories

| Category | Video | Audio | Note |
|---|---|---|---|
| RealVideo-RealAudio (RVRA) | real | real | The only genuine class. |
| RealVideo-FakeAudio (RVFA) | real | fake | Invisible to visual-only streams by construction. |
| FakeVideo-RealAudio (FVRA) | fake | real | |
| FakeVideo-FakeAudio (FVFA) | fake | fake | |

Manipulation *method* is a separate and equally important axis: `real`, `faceswap`, `fsgan`,
`wav2lip`, `fsgan-wav2lip`, `faceswap-wav2lip`, `rtvc`. The category tells you which track
was touched; the method tells you which detection principle should have caught it. See the
family breakdown in §2.

We report per-category *and* per-method accuracy, never aggregate alone. Aggregate accuracy
is dominated by the 15,872 wav2lip-derived clips and can look excellent while an entire
family, the 4,694 pure face swaps or the 500 voice clones, is being missed. Per-method
reporting is how we tell whether all three detection principles are actually working, which
is the claim the project has to defend.

### Label semantics per stream

A visual-only stream sees the authenticity of the **video track**, not of the clip:
`FakeVideo-*` is fake, `RealVideo-*` is real, *including* `RealVideo-FakeAudio` whose
fakeness is audio-only. A visual stream missing those clips is not a bug. It is the reason
the cross-modal streams exist, and it is the cleanest way to explain the architecture in
one sentence.

### Identity-disjoint splits

The most important single constraint in the project. The same identities appear across
categories in FakeAVCeleb. A real clip in train and its fake derivative in test share
background, lighting, framing and wardrobe, and a model will happily exploit that to
produce a spectacular, meaningless AUC. This is the most common way deepfake projects
generate invalid results.

Splits are built once, by `preprocessing/build_splits.py`, grouped on the `source` identity,
and never re-split randomly downstream. `verify_splits.py` asserts zero identity overlap and
zero file overlap and is run as a gate.

**Current splits** (verified, 500 source identities, all identity-disjoint):

| Split | Clips | Real | Fake |
|---|---|---|---|
| train | 1400 | 350 | 1050 |
| val | 300 | 75 | 225 |
| test | 300 | 75 | 225 |

**Class balance decision (settled 2026-07-23).** `build_splits.py` undersamples fakes to a
1:3 real:fake ratio in *every* split, including val and test. The alternative, keeping the
natural roughly 40:1 ratio in val and test, was considered and rejected: on a 300-clip
evaluation set, 40:1 makes precision, recall and F1 unstable, and 1:3 keeps rebuilt numbers
directly comparable to the previous build's results, which were measured the same way.
Train-time class weighting and `WeightedRandomSampler` still layer on top for training.
Every reported precision, recall and F1 depends on this choice, so it is stated wherever
numbers are reported.

---

## 9. The five streams in detail

### Common template for the visual streams

All three visual streams are the same config-driven module,
`models/streams/common/visual_stream.py`, differing only by a `StreamConfig`:

```
16 face crops -> backbone (per frame) -> 16 per-frame embeddings
             -> temporal model (BiLSTM, GRU or mean-pool, configurable)
             -> one clip-level embedding
             -> Linear + LayerNorm -> 256-dim
             -> [development only] temporary Linear + sigmoid head
```

The temporary head exists so a stream's standalone discriminative power can be confirmed
before fusion exists. It is a development-time check, is never what gets written to the
feature store, and is discarded once the stream folds into fusion.

| Stream | timm id | Params | Native dim | Character |
|---|---|---|---|---|
| **EfficientNet-B0** | `tf_efficientnet_b0.ns_jft_in1k` | ~5M | 1280 | Lightest, built first because it fits the 6 GB laptop GPU. Artifact-focused. |
| **Xception** | `legacy_xception` | ~22M | 2048 | Depthwise separable convolutions. Classic deepfake-detection baseline, artifact-focused. |
| **DINOv2 (ViT-S/14)** | `vit_small_patch14_dinov2.lvd142m` | ~22M | 384 | Self-supervised. Built with an explicit `img_size=224`, so a face crop becomes a 16x16 patch grid. Trained frozen first, on purpose. |

**Why DINOv2 is trained differently.** Its features were learned with no fake/real labels,
so they describe images generally instead of latching onto one generator's fingerprint.
Fine-tuning it end to end on FakeAVCeleb artifacts risks destroying exactly the property it
was chosen for. So it starts frozen with a lightweight probe (temporal model plus temporary
head) on top, and is only unfrozen if the frozen probe underperforms the CNNs by a wide
margin. That is the opposite default from Xception and EfficientNet's staged fine-tuning,
and deliberately so.

**Temporal modelling note.** Frame-to-clip aggregation is done by the temporal model, not by
pooling per-frame predictions. That is a deliberate change from mean/max pooling of frame
scores, made because feature-level fusion needs one embedding per clip, not a pooled score.

### Cross-modal stream 1: lip-sync (Stage 4, designed, not built)

- **Video encoder (Key/Value):** `torchvision.models.video.r2plus1d_18`, Kinetics-400
  pretrained, run over the 16-frame mouth sequence. A 3D CNN consumes the whole clip
  natively, so no separate LSTM is needed here. Output 512-dim.
- **Audio encoder (Query):** `facebook/wav2vec2-base-960h` over the aligned audio. Output
  768-dim.
- **Cross-attention:** `softmax(QK^T / sqrt(d)) V`, audio attends to video by default, with
  the direction config-driven rather than hardcoded.
- **Output:** projected to `common_dim = 256`, same as every other stream.
- **Both encoders frozen by default**, configurable to unfreeze if standalone performance
  is weak.

No transcription, no lip-reading to text, anywhere in the path. This is a SyncNet-style
temporal and embedding synchrony check, not a semantic word-level comparison.

### Cross-modal stream 2: emotion (Stage 5, designed, not built)

- **Face-emotion encoder (Key/Value):** `trpakov/vit-face-expression`, a FER-trained ViT,
  using the penultimate layer's embedding rather than the classification logits. Run per
  frame, then combined across the 16 frames, starting with mean-pooling. Output 768-dim.
- **Voice-emotion encoder (Query):** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`,
  a dimensional speech-emotion model. Output 1024-dim.
- **Cross-attention:** the same module as Stage 4 with the direction flipped by config, not
  by a code fork. Voice is Query, face is Key/Value.
- **Output:** projected to 256.

Emotion mismatch is expected to be a weaker and noisier signal than lip-sync, and to earn
its place mainly inside fusion rather than standalone. We record its standalone AUC anyway,
for the ablation table.

> **Open discrepancy to resolve before Stage 4.** PROJECT_OVERVIEW.md §2 names AV-HuBERT
> plus Whisper for lip-sync and HSEmotions for face emotion. The stage-4 and stage-5 plans
> and the 2026-07-09 architecture-pivot spec name R(2+1)D-18 plus Wav2Vec2 and
> `trpakov/vit-face-expression`. The stage plans and the pivot spec agree with each other
> and are the more recent, more concrete decision, so they are what is written above.
> PROJECT_OVERVIEW.md §2 should be updated to match, or the choice re-decided explicitly.

---

## 10. Fusion

**Feature-level, not score averaging.** This is the deliberate, harder choice.

1. Each stream produces a clip-level embedding.
2. Each embedding is projected (`Linear` plus `LayerNorm`) to `common_dim = 256`.
3. Every projected embedding is written to the shared feature store, keyed by `clip_id`.
4. Fusion reads that table, concatenates the included streams into one vector of
   `k x 256` (1280 with all five), passes it through an MLP with config-driven width and
   depth, and applies a sigmoid. Threshold at 0.5 for the label.

**Discipline:** fit on train, select on validation, report only on test. Fitting or tuning
fusion on the test set is the same leakage mistake as identity leakage, one level up.

**The trade-off, stated honestly.** Feature-level fusion does not give a single
interpretable "how much did stream X contribute" number the way per-stream scores plus a
logistic regression's learned weights would. We recover that interpretability differently,
through the Stage 7 ablation, which runs subsets of streams and compares fused metrics.

**Open decision for Stage 6:** frozen backbones with only the fusion head trained, versus
end-to-end fine-tuning. This must be justified against the actual compute budget. On a 6 GB
laptop GPU, frozen backbones with cached embeddings is the realistic default; end-to-end
fine-tuning of five backbones is likely out of reach without a bigger machine.

**Stretch comparison, must not block Stage 9:** build late fusion too, a weighted average
and then logistic regression over calibrated per-stream scores, and evaluate it identically.
The feature-level-versus-late comparison is itself a reportable result.

---

## 11. Evaluation plan

### Metrics reported everywhere

Accuracy, AUC-ROC (the primary, threshold-free metric), LogLoss, precision, recall, F1,
confusion matrix, EER. Plus, at every epoch and in every final table, the **per-category and
per-method breakdown** (`val_acc_FVFA-WL`, `val_acc_FVFA-FS`, `val_acc_RVFA`, and so on).
Aggregate accuracy alone hides whether all three detection principles are working. A system
that catches every wav2lip clip and no pure face swap, or the reverse, can post a strong
headline number while failing the project's actual claim.

### Stage 7: the ablation table

Each stream alone, all visual, visual plus one cross-modal, all five, and leave-one-out,
all on the same test split with the same metric. From that table we make explicit keep/drop
calls:

- Do Xception and EfficientNet overlap enough that one is redundant? Both are
  artifact-focused CNNs, so this is a real possibility.
- Does DINOv2 survive by catching different fakes than the two CNNs, as its
  self-supervised nature predicts?
- How much does each cross-modal stream add over visual-only?

Dropping a stream because the data said so is a genuine finding, not a failure.

**A build note that matters.** Late-fusion ablation is trivial: drop a score column and
re-fit a tiny logistic regression. Feature-level ablation is not. A stream subset changes
the fusion MLP's *input dimension*, so the first layer has to be reconfigured and a fusion
MLP trained per subset. You cannot mask embedding slots with zeros in a fixed-size input and
expect a meaningful number, because the MLP never saw that pattern in training. If
retraining per subset proves too slow, we prioritize the combinations that answer real
questions (all-visual versus all-five, leave-one-cross-modal-out) over an exhaustive power
set.

### Stage 9: four evaluations

| # | Evaluation | Question it answers |
|---|---|---|
| 9A | In-distribution FakeAVCeleb | The headline number. Fused AUC, accuracy, LogLoss, confusion matrix, per-category breakdown, positioned against standalone streams and published benchmarks. |
| 9B | Deepfake-Eval-2024, in the wild | Does it generalize, or did it memorize FakeAVCeleb's quirks? We expect a drop; the question is how large, per stream. Hypothesis: DINOv2 and the cross-modal streams degrade less than the artifact CNNs. |
| 9C | Held-out manipulation types | Re-split so an entire generation method never appears in training. Maps the system's blind spots. |
| 9D | Robustness | Re-encode at lower bitrates, add audio noise, reduce resolution, and plot degradation curves. Artifact CNNs are typically compression-sensitive; the claim to test is that cross-modal streams hold up better. |

Deepfake-Eval-2024 and any held-out manipulation must never leak into training or into
fusion fitting and tuning. Preprocessing will fail more often on in-the-wild footage, and
we report that failure rate honestly as a deployment caveat rather than quietly dropping
the clips.

### Stage 8 (stretch): self-supervised pretraining

In the spirit of AVFF: pretrain the cross-modal components on real-only video, so the model
first learns what genuine audio-visual correspondence looks like and "fake" is later
recognized as a departure from that norm. Then fine-tune and measure the lift, specifically
on out-of-distribution data, since in-distribution AUC can hide or fake the benefit. This
is explicitly optional and hard-timeboxed. A complete system without Stage 8 is a strong
project; a half-finished Stage 8 that eats the evaluation and write-up time is not. "It did
not help at our scale" is a legitimate reportable finding.

---

## 12. Explainability

Two threads, both in Stage 10.

**Per-stream attribution: which stream caught which fake.** Nearly free, because the feature
store already holds every stream's embedding for every clip. For representative fakes,
especially the partial fakes that fusion caught but no single stream did, we pull each
stream's standalone score and visualize the breakdown: visual streams unsure, cross-modal
stream lit up, fusion confident. Curate one case study per detection principle: a pure face
swap where a visual stream is the hero, a wav2lip clip where lip-sync is, and an
`rtvc` voice clone where only the cross-modal streams fire at all. We also show an honest
failure case, because a report that only shows wins reads as less credible.

**Grad-CAM on the surviving visual streams.** Heatmaps overlaid on example crops showing
where on the face each artifact CNN attends: blending boundaries, mouth region, and so on,
tied back to the architecture write-ups.

---

## 13. Tooling, environment, and engineering discipline

### Three tools, three jobs, never blurred

1. **Streamlit** is for small-sample inspection and preprocessing iteration only. It never
   contains or triggers a training loop. It reads results; it does not produce them.
2. **Weights and Biases** tracks every actual training run. `wandb.init(config={...})` must
   include everything that varies across experiments, including preprocessing parameters
   whenever those are being compared, because anything left out of config cannot be
   compared later. `wandb.log` every epoch with per-category accuracy. W&B Sweeps drive the
   "test all combinations" search over backbone, freeze schedule and learning rate, rather
   than a hand-built toggle UI.
3. **Training itself** is a background script or terminal process: `python train.py` or
   `wandb agent <sweep_id>`. Never a Streamlit callback, never a notebook cell that cannot
   survive a disconnect.

### Compute

- **Primary GPU:** NVIDIA RTX 5070 Ti, 15.9 GB VRAM, compute capability 12.0, CUDA 13.0.
- **Secondary:** RTX 3060 Laptop, 6 GB VRAM. This machine produced the earlier Stage 2
  results and is why batch size 2 with gradient accumulation to an effective 16, CPU-side
  MTCNN in the pre-cache workers, and building the lightest backbone first were all
  necessary.
- **Rule:** plan against 6 GB for anything the team must reproduce, and treat the 16 GB box
  as headroom for the expensive stages, not as the baseline. A batch size that only fits in
  16 GB will silently fail on the laptop.

### Environment

Python 3.13, managed by `uv`, pinned in `pyproject.toml` and locked in `uv.lock` so every
machine resolves to identical versions. Exactly one of two extras, never both:
`uv sync --extra cu130` for NVIDIA, `uv sync --extra cpu` otherwise. They install the same
torch version from different indexes. PyAV bundles its own ffmpeg, so no system ffmpeg is
required for clip extraction, audio decoding, or the dashboard's video player.

### Build principles

- **One stream first, end to end**, validated completely before scaffolding the rest. Do
  not build five unvalidated streams in parallel.
- **Validate on a small subset before scaling** to the full dataset.
- **Freeze the preprocessing interface early.** Every stream writes the same embedding
  format or the feature store and fusion cannot work.
- **Reuse the proven pipeline.** Each new stream clones the same template. If a stream needs
  the template changed, change the template, do not fork a per-stream hack.
- **Long-running jobs are scripts**, resumable and able to survive disconnects.
- **Write the report as you go**, not crammed at the end. Stage 10 is cheap only if this
  held.

### Testing

166 tests collected across `tests/`, mirroring the package layout: pure preprocessing ops,
dashboard library functions, the visual-stream module and its introspection hooks, and
Streamlit `AppTest` smoke tests for every page.

---

## 14. The inspection dashboard

A Streamlit app, `dashboard/app.py`, that is the small-sample iteration loop. It never
trains and never writes `data/processed/`.

| Page | What it does |
|---|---|
| **Overview** | Static landing page: what is being detected, the three-stage build, one line per page. |
| **Preprocessing** | The page that computes. Pick a clip, then step through the visual and audio pipelines with every step as a toggle, applied cumulatively, ending in the exact tensor a model would receive. |
| **Streams** | Takes that tensor into the model. A hub that configures all three visual streams, plus a page per stream that walks one clip through it stage by stage, showing real activations at every step. |
| **Fusion**, **Explainability** | Locked. Each says what will land there and what unlocks it, rather than showing controls that do not work. |
| **Documentation** | The in-depth reference: every step, model and design decision. |

The pictures on the stream pages are real. `models/streams/common/introspect.py` hooks the
backbone's `feature_info` stages, runs one forward pass, and returns the activations, so the
architecture described in the Documentation tab and the thing that actually runs are
visibly the same object.

Two bugs surfaced while building this and are worth a line in the report:
`grad_checkpointing` defaults to True but `legacy_xception` asserts on it, and a
checkpointed backbone runs as one flattened segment in timm so stage hooks never fire (the
trace disables checkpointing for the duration of the pass).

A constraint this puts on the trainer when it lands: **save the config as a plain dict**,
not a `StreamConfig` instance, because the dashboard loads checkpoints with
`weights_only=True` and will refuse to unpickle arbitrary objects.

---

## 15. Roadmap: the ten stages

| Stage | Content | Gate: done when | State |
|---|---|---|---|
| 1 | Data pipeline | Identity-disjoint manifests, DataLoader yields verified shapes, feature store round-trips, leakage checked independently | **Built** |
| 2 | First visual stream end to end | EfficientNet-B0 plus BiLSTM plus temporary head trains, loss drops, full metrics compute, reusable template documented | **Next** |
| 3 | Remaining visual streams | Xception and DINOv2 embeddings in the feature store with standalone AUCs recorded | Module built, not trained |
| 4 | Lip-sync cross-modal stream | Video plus audio in, one 256-dim vector out, no transcription, embeddings stored, early evidence it behaves differently on RVFA clips | Designed |
| 5 | Emotion cross-modal stream | Same cross-attention module, direction flipped by config, embeddings stored | Designed |
| 6 | Fusion | Fused test metrics computed, fit on train and tuned on val, beating or honestly not beating the best single stream | Designed |
| 7 | Ablation | Complete table, keep/drop decisions justified from data, final stream set chosen | Designed |
| 8 | Self-supervised pretraining | Controlled pretrained-versus-not comparison on generalization data | Stretch, optional |
| 9 | Full evaluation | Four evaluations complete, generalization quantified including failures | Designed |
| 10 | Explainability and write-up | Attribution figures, Grad-CAM, report assembled, presentation rehearsed | Designed |

Stage 4 does not depend on Stages 2 and 3. Visual and cross-modal streams are independent
and could be built in either order. Cross-modal is the riskier, more novel half, so it
deserves more schedule slack. If it proves very hard, a visual-only plus one-cross-modal
system is still a valid intermediate result, and must not be allowed to block everything
else indefinitely.

---

## 16. Current status and results so far

### What is built

| Component | State |
|---|---|
| Preprocessing | Built. Shared pure functions in `preprocessing/ops/`, used by both the batch pipeline and the dashboard. |
| Manifests and splits | Built. 21,544-clip manifest, identity-disjoint splits, `verify_splits.py` passing with zero identity and zero file overlap. |
| Per-clip cache | Built and versioned. Full precache of all three splits, roughly 2000 clips, has been run. |
| Feature store | Schema frozen, round-trips a dummy embedding. |
| Visual stream module | Built. One config-driven module; EfficientNet-B0, Xception and DINOv2 all wired and forward-passing. |
| Dashboard | Built. Six pages, real activations on the stream pages, 166 tests green. |
| Training | Not written. The streams are defined, but nothing has been trained since the preprocessing rebuild. |
| Lip-sync, emotion, fusion, evaluation, explainability | Designed, not built. |

### The result we are rebuilding toward

An earlier build of the visual stream, EfficientNet-B0 (`tf_efficientnet_b0.ns_jft_in1k`)
plus BiLSTM plus a temporary head, reached on the held-out test split:

| Metric | Value |
|---|---|
| Accuracy | 0.963 |
| AUC-ROC | 0.994 |
| F1 | 0.974 |
| EER | 0.018 |
| Val AUC | 0.999 |

In-distribution only, measured under the 1:3 real:fake val/test ratio. Stability required
gradient clipping and frozen BatchNorm during fine-tuning.

**Read this number carefully.** It is the bar to re-clear, not a current result. Five-point
face alignment has since changed the cached pixels, so `data/processed/` must be
re-precached and the visual stream re-validated before any comparison is meaningful. The
pre-rebuild implementation is preserved in commit `926624a`.

### The deliberate reset

On 2026-07-23 the Stage 1 to 2 implementation was removed on purpose so it could be rebuilt
step by step with each piece understood as it was added. Nothing was lost: the full prior
implementation is preserved in commit `926624a` and can be read file by file with
`git show 926624a:preprocessing/dataset.py`. The pinned environment, the documentation and
the stage plans were all kept. This is worth stating plainly in the write-up: it was a
learning-driven choice by a team starting from zero on deep learning, not a recovery from a
failure.

---

## 17. Risks, open decisions, and known contradictions

### Risks, ranked

1. **Identity leakage.** The single highest risk and the most common way deepfake projects
   produce invalid results. Mitigated by identity-grouped splits built once and verified by
   an independent script. Rule of thumb: if a later stage's AUC looks suspiciously high on
   unseen fakes, suspect leakage before celebrating.
2. **Test leakage through fusion fitting.** Same mistake one level up. Train-fit,
   validation-tune, test-report. No exceptions.
3. **The cross-modal streams are the risky, novel part.** R(2+1)D-18 and Wav2Vec2 were never
   trained together or for this task, so the cross-attention layer has to learn the
   alignment from scratch. If results are weak, try lightweight encoder fine-tuning, which
   the config already supports, before concluding the architecture does not work.
4. **Ablation cost under feature-level fusion.** A fusion MLP per subset is not free. Have a
   prioritized subset list ready.
5. **Compute mismatch.** A configuration that only fits in 16 GB fails silently on the 6 GB
   machine.
6. **Stage 8 as a time sink.** Self-supervised pretraining is finicky and slow to debug.
   Timebox hard, or skip.
7. **The write-up.** Stage 10 is cheap only if drafts accumulated along the way. Discovering
   otherwise on the last day is the classic failure.

### Open decisions

| # | Decision | Status |
|---|---|---|
| 1 | Which backbone is the template | Settled in practice: EfficientNet-B0 first, being lightest. Xception and DINOv2 clone it by supplying a different `StreamConfig.backbone_name`. |
| 2 | Val/test class distribution | **Decided 2026-07-23:** 1:3 in every split, with rationale recorded. |
| 3 | Experiment tracking | **Settled 2026-07-31:** W&B. `wandb` is a dependency, imported lazily. The old hand-maintained `experiments.csv` is gone. |
| 4 | Frozen backbones versus end-to-end fine-tuning at fusion | Open, to be decided in Stage 6 and justified against the compute budget. |
| 5 | Per-category logging every epoch | Required, and must be built in from the start this time. The previous trainer logged aggregates only. |
| 6 | Cross-modal encoder choice | **Contradiction to resolve.** See §9. PROJECT_OVERVIEW.md §2 says AV-HuBERT plus Whisper and HSEmotions; the stage plans and the pivot spec say R(2+1)D-18 plus Wav2Vec2 and `trpakov/vit-face-expression`. |

### Honest limitations to state in the report

- Results so far are in-distribution only, on a single dataset, at a 1:3 class ratio that
  is not the natural distribution.
- The evaluation splits are small, 300 clips each, which puts real error bars on every
  reported percentage.
- The mouth ROI is a 5-landmark approximation of what AV-HuBERT-style preprocessing
  expects.
- Preprocessing failure rates will be materially higher on in-the-wild footage.

---

## 18. Repository layout

```
assets/          diagrams used by the docs and the dashboard
checkpoints/     trained weights, one folder per stream (git-ignored)
dashboard/       Streamlit app: app.py, lib/ (pure, unit-tested), pages/
data/            the dataset and derived manifests (git-ignored)
docs/            main project document, glossary, math writeups, stage plans
evaluation/      metrics and ablation reporting (not built yet)
feature_store/   per-clip embedding cache that fusion reads
fusion/          the fusion MLP (not built yet)
models/streams/  one config-driven stream module, cloned per backbone
preprocessing/   manifests, splits, the ops/ functions, per-clip cache
tests/           pytest suite mirroring the package layout
training/        training scripts (not written yet)
```

Every Python directory holds an `__init__.py` so it is an importable package. Scripts run
from the repo root, for example `python -m preprocessing.audit_dataset`, and import each
other with `from preprocessing.dataset import ClipDataset`. `pyproject.toml` sets
`package = false`, so there is nothing to install or build; the repo root just needs to be
the working directory.

Pipeline scripts, in dependency order:

| # | Script | Job |
|---|---|---|
| 1 | `audit_dataset.py` | Walk the dataset, build `full_manifest.csv`, run the integrity and leading-silence audit. |
| 2 | `build_splits.py` | Identity-disjoint train/val/test splits at the 1:3 ratio. |
| 3 | `verify_splits.py` | Assert no identity and no file leaks across splits. |
| 4 | `extract_clip.py` | Per-clip aligned-face and aligned-audio extraction with a versioned disk cache. |
| - | `ops/` | The shared per-step pure functions used by 4 *and* by the dashboard. |
| 5 | `dataset.py` | The shared PyTorch `Dataset` and `DataLoader`. Every stream imports this; nobody writes their own. |
| 6 | `precache.py` | One-time parallel pre-caching so epoch 1 is not crippled by lazy extraction. |

---

## 19. Glossary

**Stream.** A dedicated path that processes one input type end to end and produces an
embedding. Visual streams see only frames; cross-modal streams see audio and video together.

**Embedding.** A fixed-length vector summarizing a clip, produced by a stream. Our streams
emit embeddings, not probabilities, which is what makes feature-level fusion possible.

**Feature-level fusion.** Concatenating stream embeddings and passing them through an MLP.
More powerful than late fusion, and less directly interpretable.

**Late (score-level) fusion.** Each stream outputs a probability; those are averaged or
combined by a small learned model. Simpler, and its weights are readable. Rejected as the
primary approach, kept as an optional comparison.

**Cross-attention.** `softmax(QK^T / sqrt(d)) V`. One modality's embedding forms the Query,
the other's forms Key and Value, so one modality attends to the other. The mechanism both
cross-modal streams are built on.

**Identity-disjoint split.** Train, validation and test share no person. The single most
important guard against inflated results in deepfake work.

**Temporal model.** LSTM, GRU or mean-pool over the 16 per-frame embeddings, producing one
clip-level vector.

**AUC-ROC.** Threshold-free measure of how well the model separates the two classes. 0.5 is
a coin flip, 1.0 is perfect. Our primary metric.

**EER.** The threshold at which the false-acceptance rate equals the false-rejection rate.

**LogLoss.** Punishes confidently wrong predictions harder than hedged ones.

**Overfitting.** The model memorizes training examples instead of the general pattern, so it
underperforms on unseen data.

**Ablation.** Removing components one at a time and measuring the effect, to establish what
each part actually contributes.

**Grad-CAM.** A heatmap over the input showing which regions most influenced a CNN's
prediction.

---

## 20. References

**Surveys**

- Hashmi, A. et al. (2024). *Understanding Audiovisual Deepfake Detection Techniques,
  Challenges, Human Factors and Perceptual Insights.*
- Khan, S., Khan, A. and Ahmad, R. (2025). *A Comprehensive Survey of DeepFake Generation
  and Detection Techniques in Audio-Visual Media.*

**Architectures**

- Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.*
  CVPR. https://arxiv.org/abs/1610.02357
- Tan, M. and Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks.* ICML. https://arxiv.org/abs/1905.11946
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
  https://arxiv.org/abs/2304.07193
- Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words* (ViT).
  https://arxiv.org/abs/2010.11929
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers*
  (DINO). https://arxiv.org/abs/2104.14294

**Methods implemented**

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Chung, J. S. and Zisserman, A. (2016). *Out of Time: Automated Lip Sync in the Wild*
  (SyncNet). ACCV.
- Mittal, T. et al. (2020). *Emotions Don't Lie: An Audio-Visual Deepfake Detection Method
  Using Affective Cues.* ACM MM.

**Read-only comparison**

- Bohacek, M. and Farid, H. (2024). *Lost in Translation: Lip-Sync Deepfake Detection from
  Audio-Video Mismatch.*
- Oorloff, T. et al. (2024). *AVFF: Audio-Visual Feature Fusion for Video Deepfake
  Detection.* CVPR.
- Zhou, Y. and Lim, S.-N. (2021). *Joint Audio-Visual Deepfake Detection.* ICCV.

**Datasets**

- Khalid, H. et al. (2022). *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset.*
- Chandra, N. et al. (2026). *Deepfake-Eval-2024.*
- Rössler, A. et al. (2019). *FaceForensics++.*
- Li, Y. et al. (2020). *Celeb-DF.*

---

## 21. Suggested slide outline

A 15 to 20 slide deck, in the order that tells the story best. Each line names the slide and
the one point it must land.

| # | Slide | The single point |
|---|---|---|
| 1 | Title | Audio-visual deepfake detection: three detection principles, five streams, one decision. |
| 2 | What a deepfake is | Three forms, with one example image each: face swap, lip-sync repaint, cloned voice. |
| 3 | One label, three problems | The §2 family table. 4,694 face swaps, 15,872 lip-sync repaints, 500 voice clones. Different evidence, different place. This slide sets up everything after it. |
| 4 | Where each one hides | Face swap: residue everywhere, tractable. Lip-sync: a few percent of the frame, compression eats it. Voice clone on real video: zero visual evidence. Show a crop of each. |
| 5 | The insight | One event, captured twice. Synthesis breaks the correspondence between the tracks, and correspondence survives compression better than pixel residue. |
| 6 | Three principles, five streams | Visual artifacts, audio-visual synchrony, audio-visual affect. The §3 principle table, then the system diagram from §6. The most important pair of slides in the deck. |
| 7 | Why embeddings, not scores | Averaging five opinions cannot express "artifacts weak, sync mismatch strong". Feature-level fusion can. |
| 8 | Preprocessing | 16 timestamps, one video path and one audio path, three tensors. Show the contract shapes. |
| 9 | Preprocessing done right | Two war stories: 5-point alignment, and the reflection-padding bug that pasted a mirrored second face into every crop. |
| 10 | The data | FakeAVCeleb: four categories on one axis, seven manipulation methods on the other, and why we report per-method accuracy rather than one number. |
| 11 | Identity-disjoint splits | The one slide that says "our numbers are honest". Include the split table. |
| 12 | The three visual streams | Table of backbone, size, character. Highlight why DINOv2 is trained frozen. |
| 13 | The two cross-modal streams | Cross-attention diagram: Q from one modality, K and V from the other. |
| 14 | Fusion | Concatenate 5 x 256, MLP, sigmoid. Train-fit, val-tune, test-report. |
| 15 | Evaluation plan | The four Stage 9 evaluations as a 2 x 2. In-distribution, in the wild, held-out method, robustness. |
| 16 | Ablation | The table format, and the promise that dropping a stream is a finding. |
| 17 | Explainability | One attribution figure per principle: a face swap the visual stream caught, a wav2lip clip lip-sync caught, a voice clone only cross-modal saw. Plus a Grad-CAM overlay. |
| 18 | Where we are | The status table, the AUC 0.994 bar being rebuilt toward, and what remains. |
| 19 | Limitations | Small eval splits, single dataset, 1:3 ratio, in-distribution only so far. |
| 20 | Roadmap and close | The ten stages with today marked, and one sentence on what success looks like. |

Figures worth preparing: `assets/flow.png` (the system diagram), a four-up of crops from
each manipulation family (real, `faceswap`, `wav2lip`, `rtvc`) to make slide 4 concrete, an
aligned-versus-unaligned crop pair, a 16-frame grid with the paired audio windows
underneath, and screenshots of the dashboard's stream page showing real activations.

---

## 22. Anticipated questions

**Why not just train one big audio-visual model end to end?**
Compute, and diagnosability. Five separate streams writing to a shared feature store can be
trained one at a time on a 6 GB GPU, and when something works or fails we can attribute it
to a specific component. A single end-to-end model gives one number and no story. The
ablation, which is a large part of the contribution, requires separable streams.

**Why is there no audio-only stream?**
Because "does this voice sound synthetic" is a different question from "do this voice and
this face belong to the same event", and only the second catches partial fakes. It is
excluded for scope and focus, not because it is useless. If the ablation shows a gap on
`RealVideo-FakeAudio`, an off-the-shelf audio spoof detector can be added as one more
fusion input.

**Is AUC 0.994 not already solved?**
No. It is in-distribution, on one dataset, with a 300-clip test split at a 1:3 ratio, and it
was measured before face alignment changed the input pixels. The interesting numbers are the
in-the-wild ones from Stage 9B and the held-out-method ones from 9C, and those are expected
to be much lower. Anyone quoting a 0.99 AUC without an out-of-distribution test is quoting a
number about their dataset, not about deepfake detection.

**What if fusion does not beat the best single stream?**
That is a reportable result, but the first thing to check is whether the cross-modal streams
are producing discriminative embeddings at all, before concluding that fusion does not help.

**What if a stream turns out to be redundant?**
Then we drop it, and say so. Xception and EfficientNet are both artifact-focused CNNs and
may well catch the same fakes. "We pruned one redundant CNN and the fused AUC did not move"
is good report material.

**Why rebuild something that already worked?**
The team is learning deep learning from zero on this project. The Stage 1 to 2 code was
removed deliberately so it could be rebuilt with each piece understood as it was added, and
the original is preserved in git at commit `926624a`. The rebuild also produced real
upgrades: 5-point alignment, the leading-silence fix, and one shared implementation of every
preprocessing step instead of two divergent ones.

**How do you know your results are not leakage?**
Splits are grouped on the `source` identity, built once, and verified by a separate script
that asserts zero identity and zero file overlap. Fusion is fit on train, tuned on
validation, and reported on test only. Both of those are stated as hard rules in the stage
gates rather than as good intentions.
