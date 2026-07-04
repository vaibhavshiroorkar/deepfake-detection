# Audio-Visual Deepfake Detection Project: Master Reference

A complete project reference covering goals, architecture, datasets, folder structure, and phases.

---

## 1. What We're Building

An audio-visual deepfake detector that combines several independent detection signals and fuses their outputs into one final real-or-fake decision.

**The problem:** Audio-visual deepfakes are hard to catch, especially ones that only alter part of the picture. A lip-sync fake only changes the mouth to match a fake voice, so the video frames look almost real. A detector that only looks at images will often miss it.

**Our core insight:** The fake often does not live inside the video alone or the audio alone. It lives in the mismatch between them, lips that do not line up with the sound, spoken words that do not match lip movements, or a voice emotion that does not match the face.

**Important correction from earlier research:** We do NOT split into separate spatial, temporal, and standalone audio models. A standalone audio model cannot tell whether audio and video agree, and the disagreement is where the fake usually hides. Instead we treat it as audio-visual, using cross-modal streams that compare audio against video.

---

## 2. System Architecture

The system has two kinds of components, combined by fusion. **Five streams total.**

**Visual models (look at face frames only):**

- Xception, catches low-level artifacts like blending edges and colour inconsistencies
- EfficientNet, a second artifact-focused view using a different architecture
- DINOv2, a higher-level view that generalizes better to unseen fakes (fallback: a Swin Transformer, if DINOv2's self-supervised math/integration is too heavy for the deadline — but note Swin is *supervised* and won't match DINOv2's generalization to unseen fakes, so it's a lower-effort substitute, not an equal one; see [docs/math/dinov2.md](docs/math/dinov2.md))

These three are purely visual. They never see the audio. Their math is what we are presenting.

**Cross-modal streams (compare audio against video, this is what makes it audio-visual):**

- Lip-sync/semantic mismatch, do the words on the lips match the words in the audio (based on Bohacek & Farid, see Section 6 for what this stream actually measures)
- Emotion/affective mismatch, does the emotion in the voice match the emotion on the face (based on Mittal et al.)

**Key distinction to always keep clear:** The visual models and cross-modal streams are separate components. Xception, EfficientNet, and DINOv2 do NOT do lip syncing or emotion matching, they cannot, they never see the audio. That work is done by the cross-modal streams.

**Why there is no standalone audio-only stream:** an audio-only model cannot see whether audio and video agree, which is where partial fakes hide, so we spend our effort on cross-modal streams instead. That said, fusion benefits from independent signals, and an off-the-shelf audio spoof detector (catching TTS/voice-clone artifacts directly, useful for the RealVideo-FakeAudio category) could be added later as one more fusion input if the ablation shows a gap there. Excluded for scope, not because it is useless.

---

## 3. Fusion

Fusion combines the outputs of all the streams into one final decision.

**What gets fused:** the outputs (scores), not the models themselves. Each model stays separate and intact. Only their verdicts meet, at the end. This is called late fusion.

**How it works:**

- Each stream runs on its own and outputs a fake-probability (0 = real, 1 = fake) for a clip
- Every score is written into one shared feature store, keyed by clip ID
- Fusion reads that table and combines the scores into one final verdict

**Two levels, built in order:**

1. Simple weighted average of the stream scores
2. A small learned model (logistic regression) that learns how much to trust each stream

**Worked example (clip_047, an actual lip-sync fake):**

- Xception → 0.55, EfficientNet → 0.48, DINOv2 → 0.60 (visual models unsure, because the frames look almost real)
- Lip-sync/semantic mismatch → 0.94 (the cross-modal stream lights up, because that is where the fake lives)
- Affective mismatch → 0.52

Plain average = 0.62 → fake (correct). Learned weighting, which trusts the reliable streams more, pushes it to about 0.85 → confidently fake. No single stream would have caught this alone. Fusion is what caught it.

**Optional upgrade (later):** feature-level fusion, where streams hand over internal features instead of final scores, fed into one joint model. More powerful, harder to build. Comparing late vs feature-level fusion is itself a reportable result.

---

## 4. Keep or Drop Models

We build all three visual models, then test whether all three earn their place using an ablation table (each stream alone and in combinations).

- If they each catch different fakes, keep all three
- If two overlap too much (Xception and EfficientNet are both artifact-focused CNNs, they may catch the same fakes), drop the redundant one
- DINOv2 is most likely to survive, since it works differently and catches different fakes

Dropping a model because the data told you to is a genuine finding, not a failure. This decision applies only to the visual half of the system.

---

## 5. Datasets

- **FakeAVCeleb**, primary training and testing set. Chosen because it has real audio manipulation. Its four categories: RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio. These drive the manipulation-type splits.
- **Deepfake-Eval-2024**, kept aside for a final real-world, in-the-wild generalization test. Not used for training.
- **FaceForensics++ and Celeb-DF**, optional visual-only baselines for comparison. Note: both are visual-only (no manipulated audio), so they cannot test cross-modal mismatch.

**Caution on FakeAVCeleb:** it is heavily imbalanced (roughly 500 real videos vs ~19,500 fakes) and the same identities appear across categories. Phase 1 must use balanced sampling and **identity-disjoint train/test splits** (no identity appears in both), or the AUC numbers will be inflated by leakage. This is the most common way deepfake projects get invalid results.

---

## 6. Key Papers

**Surveys:**

- Hashmi et al. (2024), Understanding Audiovisual Deepfake Detection Techniques, Challenges, Human Factors and Perceptual
- Khan, Khan & Ahmad (2025), A Comprehensive Survey of DeepFake Generation and Detection Techniques in Audio-Visual Media

**Visual model architectures (math we present):**

- Xception (depthwise separable convolutions)
- EfficientNet (compound scaling)
- DINOv2 (self-supervised, self-distillation)

**Cross-modal methods (implemented as streams):**

- Bohacek & Farid (2024), Lost in Translation: Lip-Sync Deepfake Detection from Audio-Video Mismatch → the lip-sync/semantic mismatch stream. To be precise about what it measures: it lip-reads the video, transcribes the audio, and compares the *words* — a semantic comparison. This is our lip-sync stream. Classic temporal sync checking (SyncNet-style: do mouth movements align in time with the waveform) is a different, older technique; it could be added later as a sixth stream if the ablation motivates it.
- Mittal et al. (2020), Emotions Don't Lie → affective mismatch

**Read-only, for comparison (not implemented):**

- Zhou & Lim (2021), Joint Audio-Visual Deepfake Detection
- Oorloff et al. (2024), AVFF, Audio-Visual Feature Fusion. Note: no public official code, heavy two-stage pretraining, too much to build from zero. Used as a benchmark to compare against.

---

## 7. What Makes It Novel

- No paper in our reading list combines these specific signals this way, especially fusing the cross-modal mismatch cues (lip-sync + emotion) alongside strong visual backbones
- We test how the whole system holds up on real-world, in-the-wild deepfakes (Deepfake-Eval-2024), not just clean academic data
- Full ablation showing which signals matter and how much fusion helps
- Explainability: we can show which stream caught which fake

---

## 8. Folder Structure

Starts lean and grows as each stream is earned. Stream folders are added only when that stream is actually started, not upfront.

**The repo currently contains only the lean Phase 0.5 subset** (`data/`, `preprocessing/`, `models/baseline/`, `evaluation/`, `notebooks/`, `docs/`). The tree below is the target state it grows into:

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
│       ├── sync/             # lip-sync/semantic mismatch (cross-modal)
│       └── affective/        # emotion mismatch (cross-modal)
├── fusion/                   # late fusion, then feature-level
├── evaluation/               # metrics, ablation, robustness tests
├── feature_store/            # shared table: clip_id -> per-stream scores
├── notebooks/                # exploration and training notebooks
└── docs/                     # glossary, math writeups, report drafts
```

---

## 9. Build Principles

- **One stream first, end to end.** Build one visual stream (Xception) fully on real data and get a real AUC number before scaffolding the rest. Prove the pipeline, then expand.
- **Freeze the preprocessing interface early.** Every stream must write into the same format for the shared feature store and fusion to work.
- **Reuse the proven pipeline.** Each new stream reuses the same end-to-end template.
- **Write the report as we go**, not crammed at the end.

---

## 10. Phases

**Phase 0.5, Foundations (Days 1 to 5):** Learn the skills and build a tiny working prototype. By the end: all three of us can train and test a model, the preprocessing pipeline is frozen, dataset requests are in, and all three visual architecture math writeups are drafted (split one per person to hit the deadline). Building the foundation, not the real system yet. Full day-by-day breakdown: [docs/phase-0.5-plan.md](docs/phase-0.5-plan.md).

**Phase 1, First stream end to end:** Build one visual stream (Xception) fully on real FakeAVCeleb data and get a real AUC number. Proves the whole pipeline before scaling. Full day-by-day breakdown: [docs/phase-1-plan.md](docs/phase-1-plan.md).

**Phase 2, Add remaining streams:** Add the other visual streams (EfficientNet, DINOv2), then the cross-modal streams (lip-sync/semantic, emotion), each reusing the proven pipeline and writing scores into the shared feature store. Milestone breakdown: [docs/phase-2-plan.md](docs/phase-2-plan.md).

**Phase 3, Fusion:** Combine stream scores into one decision. Start with weighted average, then learned logistic regression. Run the ablation to see which streams earn their place and which combination performs best. Milestone breakdown: [docs/phase-3-plan.md](docs/phase-3-plan.md).

**Phase 4, Self-supervised pretraining (stretch):** Optionally boost generalization by pretraining the cross-modal components on real-only videos before fine-tuning, in the spirit of AVFF. Milestone breakdown: [docs/phase-4-plan.md](docs/phase-4-plan.md).

**Phase 5, Full evaluation:** Test in-distribution (FakeAVCeleb), real-world (Deepfake-Eval-2024), held-out manipulation types, and robustness under compression and noise. Milestone breakdown: [docs/phase-5-plan.md](docs/phase-5-plan.md).

**Phase 6, Explainability and write-up:** Add visualizations (which stream caught which fake, Grad-CAM on visual streams), then assemble the final report and presentation. Milestone breakdown: [docs/phase-6-plan.md](docs/phase-6-plan.md).

---

## 11. Team Setup

- Three people. Starting from zero on deep learning (comfortable with Python and basic math).
- No compute limit.
- Roles: Research (literature, math, presentation), Data (extraction, preprocessing, DataLoader), ML (models, fusion, evaluation).
- For the Phase 0.5 math deadline, the three architecture writeups are split one per person: Xception → Research lead, EfficientNet → Data lead, DINOv2 → ML lead.

---

## 12. Current Status

- Phase 0.5 is fully detailed day by day in [docs/phase-0.5-plan.md](docs/phase-0.5-plan.md).
- Phase 1 is fully detailed day by day in [docs/phase-1-plan.md](docs/phase-1-plan.md).
- Phases 2 through 6 are detailed at the milestone level (role-balanced, with "done when" gates) in [docs/phase-2-plan.md](docs/phase-2-plan.md) … [docs/phase-6-plan.md](docs/phase-6-plan.md). Each milestone gets expanded into day-by-day detail when that phase is actually started, using the Phase 0.5 / Phase 1 plans as the pattern.

---

Note: Phase 0.5 and Phase 1 have full day-by-day breakdowns; Phases 2–6 are planned at milestone level and get expanded to daily detail when begun. Later phases depend on real-world unknowns (dataset access timing, stream results), so their day-level detail is deliberately deferred until those unknowns resolve.
