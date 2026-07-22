# Audio-Visual Deepfake Detection: Master Reference

**This is the project's single main document.** Goals, architecture, data, tooling,
constraints, and build order all live here. When the project changes, this file is
where the change gets recorded — stage plans (`stage-N-plan.md`) hold per-stage
detail, but they follow this document, they don't override it.

---

## 1. What We're Building

A deepfake detector that catches **lip-sync manipulations** — forgeries where only the
mouth is altered to match a fabricated audio track, while the rest of the face and
lighting stay untouched. Vision-only detectors miss these because most of the frame is
genuine.

**Core idea:** fuse visual artifact detection with cross-modal audio-visual consistency
checking, so even if the visual stream is fooled by clean pixels, an audio-video
mismatch still gets caught.

**The insight restated:** the fake often does not live inside the video alone or the
audio alone. It lives in the *mismatch* between them — lips that do not line up with
the sound, or a voice emotion that does not match the face.

**Correction from earlier research (kept as a warning):** we do NOT split into separate
spatial, temporal, and standalone audio models. A standalone audio model cannot tell
whether audio and video *agree*, and the disagreement is where the fake usually hides.
The system is audio-visual by construction, via cross-modal streams that compare audio
against video.

---

## 2. System Architecture

**Five streams, each producing an embedding vector (not a final score), combined by
feature-level fusion.**

### Visual-only streams (3)

Each reads a sequence of face frames and outputs one clip-level embedding. These three
never see the audio.

- **Xception** — low-level artifacts: blending edges, colour inconsistencies
- **EfficientNet** — a second artifact-focused view, different architecture
- **DINOv2 (ViT)** — a third view that is *self-supervised* rather than supervised. Its
  features are learned without fake/real labels, so they describe images generally and
  are expected to generalize better to unseen fake types than the two artifact-focused
  CNNs (see [math/dinov2.md](math/dinov2.md)).

Every visual stream runs each frame through its backbone to get per-frame embeddings,
then passes the frame sequence through a temporal model (LSTM or GRU, configurable) to
produce one clip-level embedding capturing how frames change over time. The template is
config-driven, so adding a 4th visual backbone later is mechanical if the ablation calls
for it.

### Cross-modal streams (2)

Each uses **cross-attention between audio and video embeddings** — this is what makes
the system audio-visual.

- **Lip-sync stream:** **AV-HuBERT** (video) + **Whisper** (audio) → synchronization
  mismatch. The video encoder reads mouth-region motion into an embedding (Key/Value);
  the audio encoder reads the audio track into an embedding (Query); scaled dot-product
  cross-attention (`softmax(QKᵀ/√d)V`, audio attends to video) produces the stream's
  vector. **No transcription or lip-reading-to-text anywhere** — everything is vectors.
  Closer to a SyncNet-style temporal/embedding sync check than a semantic word-level
  comparison.
- **Emotion stream:** **HSEmotions** (video) + **Wav2Vec2** (audio) → emotional-
  consistency mismatch. The face-emotion encoder reads expression into an embedding
  (Key/Value); the voice-emotion encoder reads vocal affect into an embedding (Query);
  cross-attention (voice attends to face) produces the stream's vector.

Each cross-modal stream outputs a **fixed-size mismatch feature vector that feeds
fusion — not a standalone prediction.**

### Two things to always keep straight

- Xception, EfficientNet, and DINOv2 do **not** do lip-syncing or emotion matching. They
  never see audio. That work belongs to the cross-modal streams.
- **There is no standalone audio-only stream.** An audio-only model cannot see whether
  audio and video agree, which is where partial fakes hide. That said, fusion benefits
  from independent signals, and an off-the-shelf audio spoof detector (catching
  TTS/voice-clone artifacts directly, useful for RealVideo-FakeAudio) could be added
  later as one more fusion input if the ablation shows a gap. Excluded for scope, not
  because it is useless.

---

## 3. Fusion

**Feature-level fusion, NOT score averaging.** Streams hand over internal features
(embeddings), not final scores, into one joint model — the more powerful,
harder-to-build option, chosen deliberately over late (score-level) fusion.

How it works:

1. Each stream produces a clip-level embedding vector.
2. Each embedding is projected (`Linear` + `LayerNorm`) to a shared `common_dim`
   (default 256) so streams of different native dimensionality become compatible.
3. Every stream's projected embedding is written into one shared feature store, keyed by
   clip ID.
4. Fusion reads that table, concatenates all included streams' embeddings into one long
   vector, passes it through an MLP, and applies sigmoid for the final fake-probability.
   Threshold at 0.5 for the label.

**Open decision (Stage 6):** frozen backbones with only the fusion head trained, vs.
end-to-end fine-tuning. This must be stated and justified against the actual compute
budget in §8 — on a 6 GB laptop GPU, frozen-backbone + cached embeddings is the
realistic default, and end-to-end fine-tuning of five backbones is likely out of reach
without a bigger machine.

**Trade-off, stated honestly:** feature-level fusion does not expose a single
interpretable "how much did stream X contribute" number the way per-stream scores plus a
logistic regression's learned weights would. That interpretability is recovered
differently — via the Stage 7 ablation (run subsets of streams, compare fused metrics)
rather than via fusion weights.

**Development-time check, not part of fusion:** during Stages 2–5, each stream trains a
*temporary* simple classifier head on its own embedding so its standalone discriminative
power can be confirmed before fusion exists. That head's score is a sanity check for
that stage only — it is not what gets written to the feature store, and it is discarded
once the stream folds into fusion.

**Optional comparison (stretch, do not block Stage 9):** late fusion (weighted average,
then logistic regression, over calibrated per-stream scores) as a comparison point. The
feature-level-vs-late-fusion trade-off is itself a reportable result.

---

## 4. Keep or Drop Streams (via Ablation)

All three visual streams get built, then Stage 7 tests whether each earns its place via
an ablation table (each stream alone and in combinations). Because fusion is
feature-level, running a subset means reconfiguring the fusion MLP's input dimension to
match the included streams' concatenated embeddings and re-evaluating — not simply
masking a score column the way late-fusion ablation would.

- If streams each catch different fakes, keep all of them.
- If two overlap too much (Xception and EfficientNet are both artifact-focused CNNs and
  may catch the same fakes), drop the redundant one.
- DINOv2 is expected to survive by catching different fakes than the two artifact CNNs,
  since it is self-supervised rather than trained on manipulation artifacts.

Dropping a stream because the data told you to is a genuine finding, not a failure. This
applies only to the visual half of the system.

---

## 5. Datasets

- **FakeAVCeleb v1.2** — primary training and testing set. Chosen because it has real
  *audio* manipulation. ~500 real videos vs ~19,500 fakes.
- **Deepfake-Eval-2024** — held aside for the final real-world, in-the-wild
  generalization test. Never used for training.
- **FaceForensics++ / Celeb-DF** — optional visual-only baselines. Both are visual-only
  (no manipulated audio), so they cannot test cross-modal mismatch.

### FakeAVCeleb categories

| Category | Video | Audio | Notes |
|---|---|---|---|
| RealVideo-RealAudio (RVRA) | real | real | the only genuine class |
| RealVideo-FakeAudio (RVFA) | real | fake | invisible to visual-only streams |
| FakeVideo-RealAudio (FVRA) | fake | real | |
| FakeVideo-FakeAudio (FVFA) | fake | fake | |

The manipulation *method* is a separate column (`real`, `faceswap`, `fsgan`, `wav2lip`).
The headline "mouth altered to match fake audio" case is **FakeVideo-FakeAudio produced
by wav2lip (`FVFA-WL`)**; `FVFA-FS`/GAN denotes the face-swap/GAN-generated variants.
Per-category *and* per-method accuracy both get reported, because the project's whole
point is catching lip-sync fakes specifically, not aggregate accuracy.

### Hard constraints on splitting

- **Identity-disjoint splits, always.** The same identities appear across categories; a
  real clip in train and its fake derivative in test share background, lighting, and
  framing, which a model exploits for a fake-high AUC. This is the most common way
  deepfake projects produce invalid results. Splits are built once by
  `preprocessing/build_splits.py` on the `source` identity — **never re-split randomly
  downstream.**
- **Class imbalance (~40:1 fake:real)** is handled at TRAIN time only, via class
  weighting / `WeightedRandomSampler`.

> ⚠️ **Unresolved — see §14.** The stated intent is that val/test remain at the natural
> distribution. The shipped `build_splits.py` instead undersamples fakes to a 1:3
> real:fake ratio in *every* split, including val and test, to keep
> precision/recall/F1 stable on 300-clip evaluation sets. These two positions
> contradict each other, and every reported precision/recall/F1 number depends on which
> one wins.

---

## 6. Preprocessing — ALREADY BUILT, DO NOT REDO

The data pipeline is done. Do not rebuild it. It produces:

| Artifact | What it is |
|---|---|
| `data/full_manifest.csv` | every clip: `clip_id`, `video_path`, `label`, `manipulation_type`, `method`, `source`, `target1`, `target2`, `race`, `gender` |
| `data/train.csv`, `val.csv`, `test.csv` | the **identity-based** split, one file per split (each row also carries its split name) |
| `data/processed/<clip_id>/` | per-clip cache of face crops + time-aligned audio windows, written on first access and reused every epoch |
| mono audio @ 16 kHz | extracted per clip, QC'd for FakeAVCeleb's known **leading-silence shortcut bug** (fake-audio clips carry extra silence at t=0 that models can cheat on) |
| face + mouth crops | MTCNN face crops (224×224) and landmark-derived mouth crops (96×96), with per-frame detection confidence |

Scripts in `preprocessing/`:

- `audit_dataset.py` — dataset integrity + the silence-shortcut audit
- `build_splits.py` — identity-disjoint train/val/test splits
- `crop_faces.py` — MTCNN face and mouth-region cropping
- `extract_clip.py` — per-clip frame + aligned-audio extraction and disk cache
- `dataset.py` — the shared PyTorch `Dataset`/`DataLoader`
- `precache.py` — one-time parallel pre-caching so epoch 1 isn't crippled by lazy extraction
- `verify_splits.py` — asserts no identity leaks across splits
- `download_samples.py` — small sample fetch for local iteration

**Shared dataloader contract** (`preprocessing/dataset.py`) — every stream imports this,
nobody writes their own:

```
face_crop_sequence : [16, 3, 224, 224] float32, ImageNet-normalized
audio              : [16, window_samples] float32 waveform windows
label              : scalar int, 1 = fake / 0 = real
```

Normalization is ImageNet mean/std, applied once here rather than per-stream, because
all three visual backbones are ImageNet-pretrained in `timm`.

**Label semantics matter per stream.** A visual-only stream sees the *video track's*
authenticity, not the clip's: `FakeVideo-*` → fake, `RealVideo-*` → real — including
`RealVideo-FakeAudio`, whose fakeness is audio-only and correctly invisible to a visual
stream. That is not a bug; it is the entire reason the cross-modal streams exist.

---

## 7. Tooling — DECIDED, DO NOT RE-LITIGATE

Three tools, three separate jobs. Do not blur them.

### 1. Streamlit — two uses only, neither one trains anything

- **Preprocessing dashboard:** toggle preprocessing params (silence trim on/off, frame
  sample fps, face-crop margin, detection confidence threshold) against a **small sample
  (10–20 clips)** and see the resulting crop/waveform immediately. A fast iteration loop
  for preprocessing decisions only.
- **Optional read-only viewer:** pulls finished/in-progress run data from W&B via
  `wandb.Api()` for display alongside the preprocessing view.

**Streamlit never triggers or contains a training loop.** It reads results; it does not
produce them.

### 2. Weights & Biases — tracks every actual training run

- `wandb.init(config={...})` at the start of any training script. Config must include
  **everything that varies across experiments**: backbone, `freeze_backbone`, lr, batch
  size, **and** preprocessing params (silence-trim, frame rate, crop margin) whenever
  those are being compared. Anything left out of config cannot be compared later.
- `wandb.log({...})` every epoch: train/val loss, `val_accuracy`, **and per-category
  accuracy** (`val_acc_FVFA-WL`, `val_acc_FVFA-FS`, …) — aggregate accuracy alone hides
  whether lip-sync fakes are actually being caught.
- **W&B Sweeps** (`sweep.yaml` + `wandb agent`) for the real "test all combinations"
  search across backbone / freeze / lr — not a hand-built toggle UI.

### 3. Training itself — a background script or terminal process

`python train.py`, or `wandb agent <sweep_id>`. **Never** a Streamlit callback, and never
a notebook cell that cannot survive a disconnect.

---

## 8. Compute and Environment Assumptions

**State these explicitly before writing any training loop.**

- **Local environment** — Jupyter and plain scripts. **Not Colab**: no Drive-mounted
  paths, no Colab-specific assumptions.
- **Primary GPU: NVIDIA RTX 3060 Laptop, 6 GB VRAM (CUDA 13).** This is the binding
  constraint. It is what drove batch size 2 with gradient accumulation to an effective
  16, CPU-side MTCNN in the pre-cache workers, and building the lightest visual backbone
  first.
- Python 3.13, `uv`-managed. `uv sync --extra cu130` for GPU, `--extra cpu` otherwise —
  exactly one, never both. See [../README.md](../README.md).

> Earlier drafts of this document claimed "no compute limit." That was never true of the
> machine the work actually runs on. Plan against 6 GB unless a bigger box is confirmed.

**Team:** three people, starting from zero on deep learning (comfortable with Python and
basic math). Work is labelled by broad workstream — Research (literature, math, report),
Data (extraction, preprocessing, DataLoader), ML (models, fusion, evaluation) — as a
description of the *kind* of work in each stage plan, not a fixed person assignment.

---

## 9. Build Principles

- **One stream first, end to end.** Build and validate a single visual stream completely
  — dataloader → backbone → temporal model → binary head → tracked training loop → eval
  with per-category breakdown — before scaffolding the rest. **Don't scaffold all five
  in parallel unvalidated.**
- **Validate on a small subset before scaling to the full dataset.**
- **Freeze the preprocessing interface early.** Every stream writes the same embedding
  format, or the feature store and fusion cannot work.
- **Reuse the proven pipeline.** Each new stream clones the same end-to-end template.
- **Long-running jobs are scripts, not notebook cells.** They must survive
  kernel/browser disconnects, run in the background, and be resumable.
- **Notebooks and Streamlit are for inspection and small-sample iteration only** — never
  the heavy compute.
- **Write the report as we go**, not crammed at the end.

---

## 10. Work Order

1. **Shared PyTorch Dataset/DataLoader** over the manifest + cached crops + audio,
   returning time-aligned face frames, mouth frames, and audio segments as fixed-length
   windows. Built once; all five streams import it. ✅ **done** —
   `preprocessing/dataset.py`.
2. **First visual stream end to end, as the validated template:** dataloader → backbone
   → temporal modeling (state which: mean-pool / LSTM / temporal attention, and why) →
   binary head → tracked training loop → eval on val including per-category breakdown.
   ✅ **done with EfficientNet-B0**, not Xception — see §14.
3. **Remaining visual streams** cloning the validated pattern: Xception and DINOv2.
4. **Cross-modal streams:** Lip-sync (AV-HuBERT + Whisper) and Emotion (HSEmotions +
   Wav2Vec2), each consuming aligned audio+video from the shared dataloader,
   cross-attention between modalities, each outputting a fixed-size mismatch feature
   vector.
5. **Fusion:** MLP over the concatenated features from all five streams. State and
   justify frozen backbones vs. end-to-end fine-tuning against §8.
6. **Evaluation:** per-stream ablation (drop each stream, measure the delta via W&B run
   comparison) and generalization testing (hold out a manipulation method, plus an
   external out-of-distribution dataset) to support the real-world generalization claim.

### Stage plans

The work order above maps onto the numbered stage plans, which hold per-stage detail:

| Stage | Content | Plan |
|---|---|---|
| 1 | Data pipeline | [stage-1-plan.md](stage-1-plan.md) |
| 2 | First visual stream end to end | [stage-2-plan.md](stage-2-plan.md) |
| 3 | Remaining visual streams | [stage-3-plan.md](stage-3-plan.md) |
| 4 | Lip-sync cross-modal stream | [stage-4-plan.md](stage-4-plan.md) |
| 5 | Emotion cross-modal stream | [stage-5-plan.md](stage-5-plan.md) |
| 6 | Fusion | [stage-6-plan.md](stage-6-plan.md) |
| 7 | Ablation support | [stage-7-plan.md](stage-7-plan.md) |
| 8 | Self-supervised pretraining (stretch) | [stage-8-plan.md](stage-8-plan.md) |
| 9 | Full evaluation | [stage-9-plan.md](stage-9-plan.md) |
| 10 | Explainability and write-up | [stage-10-plan.md](stage-10-plan.md) |

---

## 11. Key Papers

**Surveys:**

- Hashmi et al. (2024), *Understanding Audiovisual Deepfake Detection Techniques,
  Challenges, Human Factors and Perceptual*
- Khan, Khan & Ahmad (2025), *A Comprehensive Survey of DeepFake Generation and
  Detection Techniques in Audio-Visual Media*

**Visual architectures (math we present):** Xception (depthwise separable convolutions),
EfficientNet (compound scaling), DINOv2 (self-supervised self-distillation).

**Cross-modal lineage (design basis for the implemented streams):**

- Vaswani et al. (2017), *Attention Is All You Need* → the scaled dot-product
  cross-attention both cross-modal streams are built on
- Chung & Zisserman (2016), *Out of Time: Automated Lip Sync in the Wild* (SyncNet) →
  closest prior art for the lip-sync stream's embedding/temporal-sync approach
- Mittal et al. (2020), *Emotions Don't Lie* → conceptual basis for the emotion-mismatch
  stream, here implemented via cross-attention on learned embeddings rather than their
  original scoring method

**Read-only, for comparison (not implemented):**

- Bohacek & Farid (2024), *Lost in Translation: Lip-Sync Deepfake Detection from
  Audio-Video Mismatch* — lip-reads the video, transcribes the audio, and compares the
  *words*. Deliberately not implemented (our lip-sync stream is embeddings, never
  transcripts); kept as a benchmark and possible future stream.
- Zhou & Lim (2021), *Joint Audio-Visual Deepfake Detection*
- Oorloff et al. (2024), *AVFF: Audio-Visual Feature Fusion* — no public official code,
  heavy two-stage pretraining, too much to build from zero. Used as a benchmark to
  compare against.

---

## 12. What Makes It Novel

- No paper in our reading list combines these specific signals this way — fusing
  cross-modal mismatch cues (lip-sync + emotion, both via cross-attention on embeddings)
  at the *feature* level alongside strong visual backbones.
- We test how the whole system holds up on real-world, in-the-wild deepfakes
  (Deepfake-Eval-2024), not just clean academic data.
- A full ablation showing which streams matter and how much fusion helps.
- Explainability: which stream's embedding shifts on which fake, and where the visual
  streams look (Grad-CAM).

---

## 13. Repo Layout

Grows as each stream is earned; stream folders are added when that stream starts, not
upfront.

```
deepfake-detection/
├── data/                     # manifests + splits in git; raw media and caches local only
├── preprocessing/            # frozen face + audio extraction, splits, shared DataLoader
├── models/
│   ├── streams/
│   │   ├── common/           # the shared, config-driven visual-stream template
│   │   ├── efficientnet/     # built
│   │   ├── xception/         # planned
│   │   ├── dinov2/           # planned
│   │   ├── lipsync/          # planned — AV-HuBERT + Whisper cross-attention
│   │   └── emotion/          # planned — HSEmotions + Wav2Vec2 cross-attention
│   └── baseline/             # early practice / first slice
├── training/                 # background training scripts (never notebooks)
├── fusion/                   # planned — feature-level fusion (concat + MLP + sigmoid)
├── evaluation/               # metrics, ablation, robustness tests, experiments.csv
├── feature_store/            # shared table: clip_id -> per-stream embeddings
├── notebooks/                # exploration and inspection only
└── docs/                     # this file, glossary, math writeups, stage plans
```

---

## 14. Current Status and Open Reconciliations

### Done

- **Stage 1 — data pipeline:** manifests, identity-disjoint splits, per-clip cache, and
  the shared `ClipDataset` are complete and verified.
- **Stage 2 — first visual stream:** **EfficientNet-B0**
  (`tf_efficientnet_b0.ns_jft_in1k`) + BiLSTM + temporary head, trained and evaluated on
  the held-out test set: **accuracy 0.963, AUC 0.994, F1 0.974, EER 0.018** (val AUC
  0.999). Recorded in `evaluation/experiments.csv` and
  `models/streams/efficientnet/RESULTS.md`. Stability required gradient clipping and
  frozen BatchNorm during fine-tuning. In-distribution only; cross-dataset evaluation is
  deferred to Stage 9.

### Open reconciliations — decide these, then edit this document

Places where the stated plan and the shipped code disagree. Each needs a decision
recorded here.

1. **Which backbone is the template.** The plan names Xception as the validated
   template; the repo built **EfficientNet-B0 first**, deliberately, because it is the
   lightest of the three and suits the 6 GB GPU. The template is backbone-agnostic and
   config-driven, so this cost nothing — Xception and DINOv2 clone it in Stage 3 by
   supplying a different `StreamConfig.backbone_name`. *Recommendation: keep
   EfficientNet as the validated template, treat Xception as a Stage 3 clone.*
2. **Val/test class distribution.** Plan: val/test stay at the natural ~40:1
   distribution. Code: `build_splits.py` undersamples fakes to 1:3 in *every* split.
   This changes every reported precision/recall/F1 number and must be settled before
   Stage 9.
3. **Experiment tracking.** The decision is W&B (`wandb.init` / `wandb.log` / Sweeps),
   but nothing in the repo uses it — tracking today is a hand-maintained
   `evaluation/experiments.csv`, and `wandb` is not in `pyproject.toml`. Migrating
   `training/train_visual_stream.py` to W&B is unstarted work.
4. **Streamlit dashboard.** Decided in §7, does not exist yet; `streamlit` is not a
   dependency.
5. **Per-category logging.** §7 requires per-category val accuracy every epoch
   (`val_acc_FVFA-WL`, …). The current trainer logs aggregate metrics only.

---

*This project previously used a phase/day/team-role structure (Phase 0.5 through Phase
6), replaced by the stage sequence above. See
[superpowers/specs/2026-07-09-architecture-pivot-design.md](superpowers/specs/2026-07-09-architecture-pivot-design.md)
for the record of what changed and why.*
