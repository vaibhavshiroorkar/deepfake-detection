# Audio-Visual Deepfake Detection Project: Master Reference

A complete project reference covering goals, architecture, datasets, folder structure, and stages.

---

## 1. What We're Building

An audio-visual deepfake detector that combines several independent detection signals and fuses their outputs into one final real-or-fake decision.

**The problem:** Audio-visual deepfakes are hard to catch, especially ones that only alter part of the picture. A lip-sync fake only changes the mouth to match a fake voice, so the video frames look almost real. A detector that only looks at images will often miss it.

**Our core insight:** The fake often does not live inside the video alone or the audio alone. It lives in the mismatch between them, lips that do not line up with the sound, spoken words that do not match lip movements, or a voice emotion that does not match the face.

**Important correction from earlier research:** We do NOT split into separate spatial, temporal, and standalone audio models. A standalone audio model cannot tell whether audio and video agree, and the disagreement is where the fake usually hides. Instead we treat it as audio-visual, using cross-modal streams that compare audio against video.

---

## 2. System Architecture

The system has two kinds of components, combined by fusion. **Five streams total**, each producing an embedding vector (not a final score — see Section 3).

**Visual streams (look at a sequence of face frames, output one clip-level embedding):**

- Xception, catches low-level artifacts like blending edges and colour inconsistencies
- EfficientNet, a second artifact-focused view using a different architecture
- DINOv2, a third view that is *self-supervised* rather than supervised — its features are learned without fake/real labels, so they describe images generally and are expected to generalize better to unseen fake types than the two artifact-focused CNNs (see [docs/math/dinov2.md](docs/math/dinov2.md)). The config-driven stream design makes adding a 4th visual backbone later mechanical, if the ablation ever calls for it.

Each visual stream runs every frame through its backbone to get per-frame embeddings, then passes the frame sequence through a temporal model (LSTM or GRU, configurable) to produce one clip-level embedding that captures how the frames change over time. These three streams never see the audio.

**Cross-modal streams (compare audio against video via cross-attention on embeddings — this is what makes it audio-visual):**

- **Lip-sync stream:** a video encoder reads mouth-region motion into an embedding (Key/Value); an audio encoder reads the audio track into an embedding (Query); scaled dot-product cross-attention (`softmax(QK^T/√d)V`, audio attends to video) produces the stream's vector. No transcription or lip-reading-to-text anywhere — everything is vectors. This is closer to a SyncNet-style temporal/embedding sync check than to a semantic (word-level) comparison.
- **Emotion stream:** a face-emotion encoder reads facial expression into an embedding (Key/Value); a voice-emotion encoder reads vocal affect into an embedding (Query); cross-attention (voice attends to face) produces the stream's vector.

**Key distinction to always keep clear:** the visual streams and cross-modal streams are separate components. Xception, EfficientNet, and DINOv2 do NOT do lip syncing or emotion matching — they never see the audio. That work is done by the cross-modal streams.

**Why there is no standalone audio-only stream:** an audio-only model cannot see whether audio and video agree, which is where partial fakes hide, so effort goes into cross-modal streams instead. That said, fusion benefits from independent signals, and an off-the-shelf audio spoof detector (catching TTS/voice-clone artifacts directly, useful for the RealVideo-FakeAudio category) could be added later as one more fusion input if the ablation shows a gap there. Excluded for scope, not because it is useless.

---

## 3. Fusion

Fusion combines the per-stream embeddings into one final decision. This is **feature-level fusion**: streams hand over internal features (embeddings), not final scores, into one joint model — the more powerful, harder-to-build option, chosen deliberately over late (score-level) fusion for this project.

**How it works:**

- Each stream produces a clip-level embedding vector.
- Each embedding is projected (`Linear` + `LayerNorm`) to a shared `common_dim` (default 256) so streams of different native dimensionality become compatible.
- Every stream's projected embedding is written into one shared feature store, keyed by clip ID.
- Fusion reads that table, concatenates all included streams' embeddings into one long vector, passes it through an MLP (a few fully-connected layers), and applies sigmoid to get the final fake-probability. Threshold at 0.5 for the real/fake label.

**Trade-off, stated honestly:** feature-level fusion does not expose a single interpretable "how much did stream X contribute" number the way per-stream scores + a logistic regression's learned weights would. That interpretability is recovered a different way — via the Stage 7 ablation (run subsets of streams, compare fused metrics) rather than via fusion weights.

**Development-time check, not part of fusion:** during Stages 2–5, each stream trains a *temporary* simple classifier head on top of its own embedding so its standalone discriminative power can be confirmed before fusion exists. That head's score is a sanity check for that stage only — it is not what gets written to the shared feature store, and it is discarded once the stream is folded into fusion.

**Optional comparison (later, not required):** late fusion (weighted average, then logistic regression, over a calibrated per-stream score instead of an embedding) could still be built as a comparison point — the feature-level-vs-late-fusion trade-off is itself a reportable result. Treat as stretch; do not block Stage 9 (evaluation) on it.

---

## 4. Keep or Drop Models (via Ablation)

All three visual streams get built, then Stage 7 tests whether each one earns its place using an ablation table (each stream alone and in combinations). Because fusion is feature-level, running a subset means reconfiguring the fusion MLP's input dimension to match the included streams' concatenated embeddings and re-evaluating — not simply masking a score column the way late-fusion ablation would work.

- If streams each catch different fakes, keep all of them
- If two overlap too much (Xception and EfficientNet are both artifact-focused CNNs, they may catch the same fakes), drop the redundant one
- DINOv2 is expected to survive the ablation by catching different fakes than the two artifact CNNs, since it's self-supervised rather than trained on manipulation artifacts

Dropping a stream because the data told you to is a genuine finding, not a failure. This decision applies only to the visual half of the system.

---

## 5. Datasets

- **FakeAVCeleb**, primary training and testing set. Chosen because it has real audio manipulation. Its four categories: RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio. These drive the manipulation-type splits.
- **Deepfake-Eval-2024**, kept aside for a final real-world, in-the-wild generalization test. Not used for training.
- **FaceForensics++ and Celeb-DF**, optional visual-only baselines for comparison. Note: both are visual-only (no manipulated audio), so they cannot test cross-modal mismatch.

**Caution on FakeAVCeleb:** it is heavily imbalanced (roughly 500 real videos vs ~19,500 fakes) and the same identities appear across categories. Splitting must use balanced sampling and **identity-disjoint train/test splits** (no identity appears in both), or the AUC numbers will be inflated by leakage. This is the most common way deepfake projects get invalid results.

---

## 6. Key Papers

**Surveys:**

- Hashmi et al. (2024), Understanding Audiovisual Deepfake Detection Techniques, Challenges, Human Factors and Perceptual
- Khan, Khan & Ahmad (2025), A Comprehensive Survey of DeepFake Generation and Detection Techniques in Audio-Visual Media

**Visual model architectures (math we present):**

- Xception (depthwise separable convolutions)
- EfficientNet (compound scaling)
- DINOv2 (self-supervised, self-distillation)

**Cross-modal methods (design lineage for the implemented streams):**

- Vaswani et al. (2017), Attention Is All You Need → the scaled dot-product cross-attention mechanism both cross-modal streams are built on
- Chung & Zisserman (2016), Out of Time: Automated Lip Sync in the Wild (SyncNet) → closest prior art for the lip-sync stream's embedding/temporal-sync approach (no transcription involved, same spirit as our stream)
- Mittal et al. (2020), Emotions Don't Lie → conceptual basis for the emotion mismatch stream (face vs. voice affect), now implemented via cross-attention on learned embeddings rather than their original scoring method

**Read-only, for comparison (not implemented):**

- Bohacek & Farid (2024), Lost in Translation: Lip-Sync Deepfake Detection from Audio-Video Mismatch. Their method lip-reads the video, transcribes the audio, and compares the *words* — a semantic, text-based comparison. We deliberately do not implement this (everything in our lip-sync stream is embeddings, not transcripts); kept as a benchmark/comparison reference and a possible future additional stream.
- Zhou & Lim (2021), Joint Audio-Visual Deepfake Detection
- Oorloff et al. (2024), AVFF, Audio-Visual Feature Fusion. Note: no public official code, heavy two-stage pretraining, too much to build from zero. Used as a benchmark to compare against.

---

## 7. What Makes It Novel

- No paper in our reading list combines these specific signals this way, especially fusing cross-modal mismatch cues (lip-sync + emotion, both via cross-attention on embeddings) at the feature level alongside strong visual backbones
- We test how the whole system holds up on real-world, in-the-wild deepfakes (Deepfake-Eval-2024), not just clean academic data
- Full ablation showing which streams matter and how much fusion helps, even without per-stream fusion-weight interpretability
- Explainability: we can show which stream's embedding shifts on which fake, and where the visual streams look (Grad-CAM)

---

## 8. Folder Structure

Starts lean and grows as each stream is earned. Stream folders are added only when that stream is actually started, not upfront.

**The repo currently contains only the lean early subset** (`data/`, `preprocessing/`, `models/baseline/`, `evaluation/`, `notebooks/`, `docs/`). The tree below is the target state it grows into:

```
deepfake-detection/
├── data/                     # datasets, splits, manifests
├── preprocessing/            # frozen face + audio extraction module
├── models/
│   ├── baseline/             # early practice / first slice
│   └── streams/              # added one at a time as streams are built
│       ├── xception/
│       ├── efficientnet/
│       ├── dinov2/
│       ├── lipsync/          # cross-attention on audio/video embeddings
│       └── emotion/          # cross-attention on face/voice embeddings
├── fusion/                   # feature-level fusion (concat + MLP + sigmoid)
├── evaluation/               # metrics, ablation, robustness tests
├── feature_store/            # shared table: clip_id -> per-stream embeddings
├── notebooks/                # exploration and training notebooks
└── docs/                     # glossary, math writeups, stage plans, report drafts
```

---

## 9. Build Principles

- **One stream first, end to end.** Build one visual stream (Xception) fully, with a temporary classifier head on its embedding, and confirm loss drops before scaffolding the rest. Prove the pipeline, then expand.
- **Freeze the preprocessing interface early.** Every stream must write into the same embedding format for the shared feature store and fusion to work.
- **Reuse the proven pipeline.** Each new stream reuses the same end-to-end template.
- **Write the report as we go**, not crammed at the end.

---

## 10. Stages

The build order is a straight sequence — each stage is built and verified before the next starts.

**Stage 1, Data pipeline:** Dataset + DataLoader that loads face-crop sequences, audio, and labels, yielding correct shapes. Full detail: [docs/stage-1-plan.md](docs/stage-1-plan.md).

**Stage 2, First visual stream end to end:** Xception + temporal model (LSTM/GRU) + a temporary classifier head, trained on a small sample, loss dropping and metrics computing. Full detail: [docs/stage-2-plan.md](docs/stage-2-plan.md).

**Stage 3, Remaining visual streams:** EfficientNet and DINOv2, cloning the Stage 2 template. Full detail: [docs/stage-3-plan.md](docs/stage-3-plan.md).

**Stage 4, Lip-sync cross-modal stream:** cross-attention on video/audio embeddings, verified to run and output a vector. Full detail: [docs/stage-4-plan.md](docs/stage-4-plan.md).

**Stage 5, Emotion cross-modal stream:** cross-attention on face/voice embeddings, cloning the Stage 4 template. Full detail: [docs/stage-5-plan.md](docs/stage-5-plan.md).

**Stage 6, Fusion:** feature-level fusion (concatenation + MLP + sigmoid) combining all five streams, full metrics (accuracy, AUC-ROC, LogLoss, precision, recall, F1, confusion matrix). Full detail: [docs/stage-6-plan.md](docs/stage-6-plan.md).

**Stage 7, Ablation support:** run any configured subset of streams and compare fused metrics, reconfiguring the fusion MLP's input dimension per subset. Full detail: [docs/stage-7-plan.md](docs/stage-7-plan.md).

**Stage 8, Self-supervised pretraining (stretch):** optionally boost generalization by pretraining the cross-modal components on real-only videos before fine-tuning, in the spirit of AVFF. Full detail: [docs/stage-8-plan.md](docs/stage-8-plan.md).

**Stage 9, Full evaluation:** in-distribution (FakeAVCeleb), real-world (Deepfake-Eval-2024), held-out manipulation types, and robustness under compression and noise. Full detail: [docs/stage-9-plan.md](docs/stage-9-plan.md).

**Stage 10, Explainability and write-up:** per-stream attribution, Grad-CAM on visual streams, then the final report and presentation. Full detail: [docs/stage-10-plan.md](docs/stage-10-plan.md).

---

## 11. Team Setup

- Three people. Starting from zero on deep learning (comfortable with Python and basic math).
- No compute limit.
- Broad workstreams: Research (literature, math, report), Data (extraction, preprocessing, DataLoader), ML (models, fusion, evaluation) — used as labels for the kind of work in each stage's plan, not fixed day-by-day person assignments.

---

## 12. Current Status

- Stage 1 (data pipeline) is in progress — see [docs/stage-1-plan.md](docs/stage-1-plan.md).
- Stages 2–7 are detailed in [docs/stage-2-plan.md](docs/stage-2-plan.md) … [docs/stage-7-plan.md](docs/stage-7-plan.md); each stage is built and verified before the next starts.
- Stages 8–10 (was Phases 4–6: pretraining stretch, full evaluation, explainability/write-up) are detailed in [docs/stage-8-plan.md](docs/stage-8-plan.md) … [docs/stage-10-plan.md](docs/stage-10-plan.md) and remain milestone-level until reached.

---

Note: this project previously used a phase/day/team-role structure (Phase 0.5 through Phase 6). That structure has been replaced by the stage sequence above; see [docs/superpowers/specs/2026-07-09-architecture-pivot-design.md](docs/superpowers/specs/2026-07-09-architecture-pivot-design.md) for the full record of what changed and why.
