# Audio-Visual Deepfake Detection: Master Reference

**This is the project's single main document.** Goals, architecture, data, tooling,
constraints, and build order all live here. When the project changes, this file is
where the change gets recorded — stage plans (`stage-N-plan.md`) hold per-stage
detail, but they follow this document, they don't override it.

---

## 1. What We're Building

An **audio-visual deepfake detector**: one that examines the video track, the audio track,
and the relationship between them. It applies **three detection principles** to every clip,
because "deepfake" covers manipulations that leave evidence in completely different places.

| Principle | Question it asks | Streams | Best against |
|---|---|---|---|
| Visual artifacts | Does this face show manufacturing residue? | Xception, EfficientNet-B0, DINOv2 | Face swaps (`faceswap`, `fsgan`) — 4,694 clips |
| Audio-visual synchrony | Do the mouth's movements and the sound belong to one event? | Lip-sync | Lip-sync repaints (`wav2lip` and its combinations) — 15,872 clips |
| Audio-visual affect | Does the emotion on the face match the emotion in the voice? | Emotion | Voice clones on genuine video (`rtvc`) — 500 clips |

Counts are FakeAVCeleb v1.2 `meta_data.csv`. None of the three families is a corner case,
and **no single principle covers all three**: a face swap has residue everywhere and is the
tractable case; a wav2lip repaint alters a few percent of the frame and compression erases
most of what it leaves; a cloned voice over untouched video alters no pixels at all.

**Core idea:** fuse visual artifact detection with cross-modal audio-visual consistency
checking, so a manipulation that defeats one principle is still caught by another.

**The insight restated:** for the harder families the fake does not live inside the video
alone or the audio alone. It lives in the *mismatch* between them — lips that do not line
up with the sound, or a voice emotion that does not match the face. The visual streams
remain essential, since a pure face swap has no cross-modal mismatch to find.

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

The manipulation *method* is a separate column, and it is the axis that maps onto §1's
three detection principles: `real` (500), `rtvc` (500, voice clone over genuine video),
`faceswap` (730) and `fsgan` (3,964) for pure swaps, and `wav2lip` (9,602),
`fsgan-wav2lip` (3,553) and `faceswap-wav2lip` (2,717) for lip-sync repaints, the last two
layered on top of a swap.

Per-category *and* per-method accuracy both get reported. Aggregate accuracy is dominated
by the 15,872 wav2lip-derived clips and can look excellent while an entire family — the
4,694 pure swaps or the 500 voice clones — is being missed. Per-method numbers are how we
show all three principles work.

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

## 6. Preprocessing — the Contract

> **Status:** this pipeline was fully built once, then removed in the 2026-07-23 reset
> (§14) to be rebuilt step by step. It is preserved in commit `926624a` — read it with
> `git show 926624a:preprocessing/<file>.py` rather than reinventing the decisions.
> **This section is the specification the rebuild must satisfy.** The output contract
> below is what every later stage depends on; the internals are yours to rewrite.

What it must produce:

> **Step-by-step reference: [preprocessing.md](preprocessing.md).** Each main
> step is an individual pure function in `preprocessing/ops/`, imported by both
> the batch pipeline and the dashboard (one implementation, not two). Enhancement/
> degradation ops ("extras") are a separate, off-by-default layer.

| Artifact | What it is |
|---|---|
| `data/full_manifest.csv` | every clip: `clip_id`, `video_path`, `label`, `manipulation_type`, `method`, `source`, `target1`, `target2`, `race`, `gender`, `leading_silence_sec` |
| `data/train.csv`, `val.csv`, `test.csv` | the **identity-based** split, one file per split (each row also carries its split name) |
| `data/processed/<clip_id>/` | per-clip cache of face crops + time-aligned audio windows + a `version.txt` (`PIPELINE_VERSION`); written on first access, reused every epoch, transparently re-extracted when the version bumps |
| mono audio @ 16 kHz | extracted per clip. FakeAVCeleb's **leading-silence shortcut bug** (fake-audio clips carry extra silence at t=0) is measured (`leading_silence_sec`) and neutralized by starting frame+audio sampling past it, keeping the two modalities aligned |
| face + mouth crops | MTCNN face crops (224×224), **5-point aligned** to a canonical template (pose-normalized), and landmark-derived mouth crops (96×96), with per-frame detection confidence |

The scripts that did this, in dependency order — a reasonable rebuild order too, one
script at a time:

| # | Script | Job |
|---|---|---|
| 1 | `audit_dataset.py` | walk `data/raw/`, build `full_manifest.csv`, integrity + leading-silence-shortcut audit |
| 2 | `build_splits.py` | identity-disjoint train/val/test splits (see reconciliation 2 first) |
| 3 | `verify_splits.py` | assert no identity leaks across splits |
| 4 | `extract_clip.py` | per-clip aligned-face + aligned-audio extraction, with versioned disk cache |
| — | `ops/` | the shared per-step functions (detect/align/crop/mouth, decode/window, extras) used by 4 **and** the dashboard |
| 5 | `dataset.py` | the shared PyTorch `Dataset`/`DataLoader` |
| 6 | `precache.py` | one-time parallel pre-caching so epoch 1 isn't crippled by lazy extraction |
| — | `download_samples.py` | small sample fetch for local iteration (optional) |

> The old standalone `crop_faces.py` CLI (and its `ffmpeg-python` dependency) was
> removed; its face-crop logic now lives in `ops/faces.py`. All decoding goes
> through PyAV, so no system ffmpeg is required.

**Shared dataloader contract** (`preprocessing/dataset.py`) — every stream imports this,
nobody writes their own:

```
face_crop_sequence : [16, 3, 224, 224] float32, ImageNet-normalized
audio              : [16, window_samples] float32 waveform windows
label              : scalar int, 1 = fake / 0 = real
```

Normalization is ImageNet mean/std, applied once here rather than per-stream, because
all three visual backbones are ImageNet-pretrained in `timm`. Face crops are now
**5-point aligned** before normalization (pose-normalized); this changes the cached
pixels, so `data/processed/` must be re-precached and the visual stream re-validated
against the AUC-0.994 bar.

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
  whether all three detection principles are working, or whether one family carries the
  number while another is missed entirely.
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
- **Primary GPU (this machine): NVIDIA RTX 5070 Ti, 15.9 GB VRAM, compute capability
  12.0, CUDA 13.0.** Verified 2026-07-23.
- **Secondary machine: RTX 3060 Laptop, 6 GB VRAM.** The Stage 2 results in §14 were
  produced here, by a teammate. Its 6 GB is what forced batch size 2 with gradient
  accumulation to an effective 16, CPU-side MTCNN in the pre-cache workers, and building
  the lightest visual backbone first.
- Python 3.13, `uv`-managed. `uv sync --extra cu130` for GPU, `--extra cpu` otherwise —
  exactly one, never both. See [../README.md](../README.md).

> **Plan against 6 GB for anything the team must reproduce**, and treat the 16 GB box as
> headroom for the expensive stages (cross-modal streams, end-to-end fusion fine-tuning)
> rather than as the baseline. A batch size that only fits in 16 GB will silently fail
> on the laptop.
>
> Earlier drafts of this document claimed "no compute limit," then "6 GB, this is the
> binding constraint." Both were wrong: the first was aspirational, the second read a
> teammate's hardware off `experiments.csv` and assumed it was universal.

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

> Items 1 and 2 were completed once and then removed in the 2026-07-23 reset (§14).
> They are being rebuilt step by step. Item 1 is the current work.

1. **Shared PyTorch Dataset/DataLoader** over the manifest + cached crops + audio,
   returning time-aligned face frames, mouth frames, and audio segments as fixed-length
   windows. Built once; all five streams import it. ← **current work**, spec in §6.
2. **First visual stream end to end, as the validated template:** dataloader → backbone
   → temporal modeling (state which: mean-pool / LSTM / temporal attention, and why) →
   binary head → tracked training loop → eval on val including per-category breakdown.
   Previously reached AUC 0.994 with EfficientNet-B0 — that is the bar (§14).
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

**The repo is at a deliberate reset** (2026-07-23) — see §14. The directory skeleton
exists as a map of where things go; **every folder is empty of logic**, and code is
written into it one script at a time as the work that needs it starts.

```
deepfake-detection/
├── data/
│   ├── raw/FakeAVCeleb_v1.2/ # the dataset (gitignored) — 21,544 clips
│   └── processed/            # per-clip frame + audio cache (gitignored)
├── preprocessing/            # face + audio extraction, splits, shared DataLoader
├── models/streams/
│   ├── common/               # the shared, config-driven visual-stream template
│   ├── xception/  efficientnet/  dinov2/
│   ├── lipsync/              # AV-HuBERT + Whisper cross-attention
│   └── emotion/              # HSEmotions + Wav2Vec2 cross-attention
├── checkpoints/<stream>/     # trained weights coming back from Kaggle/W&B (gitignored)
├── training/                 # background training scripts (never notebooks)
├── fusion/                   # feature-level fusion (concat + MLP + sigmoid)
├── evaluation/               # metrics, ablation, robustness tests
├── feature_store/            # shared table: clip_id -> per-stream embeddings
├── notebooks/                # exploration and inspection only
└── docs/                     # this file, glossary, math writeups, stage plans
```

Each Python directory holds an empty `__init__.py` so it is an importable package —
scripts run from the repo root (`python -m preprocessing.audit_dataset`) and can import
each other (`from preprocessing.dataset import ClipDataset`). `pyproject.toml` sets
`package = false`, so there is nothing to install or build; the repo root just needs to
be the working directory.

### Where the dataset goes

Unzip `FakeAVCeleb_v1.2.zip` so the layout is:

```
data/raw/FakeAVCeleb_v1.2/
├── meta_data.csv
├── RealVideo-RealAudio/
├── RealVideo-FakeAudio/
├── FakeVideo-RealAudio/
└── FakeVideo-FakeAudio/
        └── <race>/<gender>/<identity>/*.mp4
```

`data/*` is gitignored, so the media never enters git. Manifests written to `data/*.csv`
(top level) *are* tracked.

---

## 14. Current Status

### Deliberate reset — 2026-07-23

The Stage 1–2 implementation was **removed on purpose** so it can be rebuilt slowly,
step by step, with each piece understood as it is added. Nothing was lost: the full
prior implementation is preserved in commit
**`926624a` — "Snapshot Stage 1-2 pipeline and consolidate docs before restart."**

What was deleted: `preprocessing/`, `models/`, `training/`, `evaluation/`,
`feature_store/`, `notebooks/`, and the derived manifests `data/*.csv`.
What was kept: the pinned environment (`pyproject.toml`, `uv.lock`, `.venv`), this
document and the stage plans, and `README.md`.

To consult the old implementation while rebuilding:

```bash
git show 926624a --stat                          # everything that existed
git show 926624a:preprocessing/dataset.py        # read one file
git checkout 926624a -- preprocessing/dataset.py # restore one file
```

### What the previous implementation achieved (the bar to rebuild toward)

- **Stage 1:** manifests, identity-disjoint splits, per-clip cache, shared `ClipDataset`.
- **Stage 2:** EfficientNet-B0 (`tf_efficientnet_b0.ns_jft_in1k`) + BiLSTM + temporary
  head. Held-out test: **accuracy 0.963, AUC 0.994, F1 0.974, EER 0.018** (val AUC
  0.999). Stability required gradient clipping and frozen BatchNorm during fine-tuning.
  In-distribution only. Measured under the 1:3 val/test ratio — see reconciliation 2.

### Next step

**Stage 1 is rebuilt and verified (2026-07-23).** The dataset is extracted to
`data/raw/FakeAVCeleb_v1.2/` (21,544 clips, 0 corrupt in a 200-clip spot-check). The
pipeline scripts were restored from snapshot `926624a`, with `audit_dataset.py` now
routing its label through the tested `preprocessing/manifest.clip_label`, and
`crop_faces.py`'s heavy CLI-only imports (ffmpeg/librosa/pandas/torch) made lazy so the
dataset path no longer drags `ffmpeg-python`. Verified this session:

- `full_manifest.csv` (21,544 rows) + identity-disjoint `train/val/test.csv` (1400/300/300).
- `verify_splits.py` PASS — zero identity and zero file overlap across splits.
- `ClipDataset` DataLoader yields `faces [B,16,3,224,224]`, `audio [B,16,5600]`,
  `label [B]`, ImageNet-normalized — shapes printed.
- `feature_store.store` round-trips a dummy embedding.
- The multi-page dashboard (`dashboard/app.py`) boots clean under Streamlit `AppTest`
  (all four pages, 16/16 faces on a sample clip).
- Full precache of all three splits (~2000 clips) run via `preprocessing.precache`.

### Preprocessing made state-of-the-art — 2026-07-24

The main preprocessing steps were rebuilt as individual, shared, pure functions in
`preprocessing/ops/` (imported by both the batch pipeline and the dashboard — no
more parallel reimplementation), and upgraded:

- **5-point face alignment** (`ops/faces.align_face`) is now the default, pose-
  normalizing every face crop onto a canonical ArcFace-style template — the main
  SOTA gain, done with MTCNN's own landmarks (no new dependency).
- **Leading-silence shortcut** (§6) is now handled: `audit_dataset.py` records
  `leading_silence_sec` per clip and `extract_clip.py` offsets frame+audio
  sampling past it, removing the shortcut while keeping frame↔audio alignment.
- **Extras** (sharpen/denoise/CLAHE/blur/JPEG/downscale, audio noise/bandpass/
  RMS/spectral-denoise/mel) are isolated in `ops/extras_*` — off by default.
- **Redundancy removed**: deleted the legacy `crop_faces.py` CLI + `ffmpeg-python`,
  the duplicate `dashboard/lib/{visual,audio}_ops.py`, the unused `mouth_region`,
  and the 4 copied ImageNet constants (now one home in `ops/constants.py`).
- Cache is versioned (`PIPELINE_VERSION`) so stale unaligned caches re-extract.

Full reference in [preprocessing.md](preprocessing.md); design/plan under
`docs/superpowers/{specs,plans}/2026-07-24-sota-preprocessing-refactor*`. 63 tests
green. **Follow-up before training: re-precache all splits and re-validate the
visual stream against the AUC-0.994 bar** (alignment changed the cached pixels).

**Next: Stage 2** — the first visual stream (EfficientNet-B0 + BiLSTM), rebuilt toward
the AUC 0.994 bar. See [stage-2-plan.md](stage-2-plan.md).

### Streams unlocked in the dashboard — 2026-07-31

The Streams section is live: a hub that configures all three visual streams, and a page
per stream that walks one clip through the model stage by stage. The pictures on those
pages are real — `models/streams/common/introspect.py` hooks the backbone's
`feature_info` stages, runs one forward pass, and returns the activations — so the
architecture in the Documentation tab and the thing that runs are visibly the same
object. DINOv2 is wired in alongside Xception and EfficientNet-B0 (a ViT, so it is
built with an explicit `img_size=224`; its stages are token matrices, not channel maps).

**This did not change §7: the dashboard still never trains.** It went further the other
way. The Train tab that emitted a background-trainer command is deleted, because a
command builder is not a training feature and having one there implied the section was
where training lived. Training happens on Kaggle or a GPU box, tracked in W&B; what
comes back is a checkpoint, and `dashboard/lib/checkpoints.py` loads it from
`checkpoints/<stream>/` or a W&B artifact. `wandb` is now a dependency
(reconciliation 3), imported lazily so the dashboard starts without it.

A constraint this puts on the trainer when it lands: **save the config as a plain dict**,
not a `StreamConfig` instance. The dashboard reads checkpoints with
`weights_only=True`, which refuses to unpickle arbitrary objects.

Two bugs surfaced and were fixed on the way: `grad_checkpointing` defaults to True but
`legacy_xception` asserts on it (now caught and recorded rather than fatal), and a
checkpointed backbone runs as one flattened segment in timm, so stage hooks never fire
(the trace disables it for the duration of the pass).

### Open reconciliations — decide these, then edit this document

Decisions the rebuild has to make. The first two are the ones that change results; the
rest are "not built yet, build it this way."

1. **Which backbone is the template.** §10 names Xception; the previous build used
   **EfficientNet-B0 first**, deliberately, because it is the lightest of the three and
   suits the 6 GB GPU. The template is backbone-agnostic and config-driven, so the
   choice costs nothing — the other two clone it by supplying a different
   `StreamConfig.backbone_name`. *Recommendation: build EfficientNet first again, treat
   Xception as a Stage 3 clone.*
2. **Val/test class distribution — DECIDED 2026-07-23: 1:3 in every split.** §5's
   natural ~40:1 was considered and rejected for val/test. Rationale: it keeps
   precision/recall/F1 meaningful on 300-clip eval sets, and it keeps the rebuilt
   numbers directly comparable to the previous build's 0.963/0.994 bar (also measured
   under 1:3). `build_splits.py` undersamples fakes to `REAL_TO_FAKE_RATIO = 1/3` per
   split; train-time class weighting still layers on top. The rebuilt splits are
   train=1400 (350r/1050f), val=300 (75r/225f), test=300 (75r/225f), all
   identity-disjoint over 500 `source` identities (verified by `verify_splits.py`).
3. **Experiment tracking — SETTLED 2026-07-31.** §7 decides W&B (`wandb.init` /
   `wandb.log` / Sweeps). The previous build used a hand-maintained
   `evaluation/experiments.csv` instead. `wandb` is now in `pyproject.toml`, pulled in
   by the dashboard's checkpoint loader, so the first training script has it already.
4. **Streamlit dashboard — BUILT 2026-07-23, restructured to multi-page.** Entry point
   `dashboard/app.py` (`st.navigation`). Two sections: **Data Preprocessing** (Visual,
   Audio) where every preprocessing step is an independent on/off toggle shown
   **original-vs-processed** — Core (detect/crop/resample/window), Representation
   (ImageNet normalize, mouth crop, mel-spectrogram) and Enhancement (sharpen,
   denoise, CLAHE, blur, JPEG re-compress, downscale, add-noise, bandpass) — with a
   shared Dataset/Target/Clip selector — the dataset list is **discovered** by scanning
   `data/` (`dashboard/lib/datasets.py`: a raw drop is any dir with a `meta_data.csv`,
   whose manifest is built in memory before `audit_dataset.py` has run; a manifest CSV
   attaches to the dataset its `video_path` points into), with a ↻ button to rescan;
   and **Streams** (Visual, Audiovisual) read-only
   scaffolds showing planned architecture + a W&B placeholder, no compute. Pure ops in
   `dashboard/lib/` are unit-tested; pages are AppTest-smoked. Read-only — never writes
   `data/processed/`, never trains. `streamlit==1.60.0` (1.55+ needed for pillow 12).
   Run: `uv run streamlit run dashboard/app.py`. Spec + plan in
   `docs/superpowers/{specs,plans}/2026-07-23-preprocessing-experiment-dashboard*`.
   Nav sections: **Overview** (`pages/overview.py`, the landing page — short and static:
   what is detected, the three-stage build, and a one-line-per-page guide; no compute,
   loads no models),
   **Data Pre-processing** (single page, Config/Visual/Audio slider with a
   modal clip picker), **Streams** (Visual — real configurable Xception + EfficientNet
   model boxes via `models/streams/common/`; LipSync and Emotions scaffolds), **Fusion**
   and **Explainability** (locked — listed with a lock icon, no interactive controls until
   Stages 6 and 10), and **Documentation** (`pages/documentation.py`, the long-form
   reference — architecture, every preprocessing step, every model, fusion/evaluation
   design, data and splits; static, no compute). The Visual page's "Build & inspect" instantiates the
   real `VisualStream` and forward-passes a dummy clip (inference only) — training stays a
   background script per this section.
5. **Per-category logging.** §7 requires per-category val accuracy every epoch
   (`val_acc_FVFA-WL`, …). The previous trainer logged aggregate metrics only. Build it
   in from the start this time — it is the project's headline claim.

---

*This project previously used a phase/day/team-role structure (Phase 0.5 through Phase
6), replaced by the stage sequence above. See
[superpowers/specs/2026-07-09-architecture-pivot-design.md](superpowers/specs/2026-07-09-architecture-pivot-design.md)
for the record of what changed and why.*
