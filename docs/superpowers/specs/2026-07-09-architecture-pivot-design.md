# Architecture Pivot: Embedding Streams + Feature-Level Fusion + Stage-Based Plan

**Date:** 2026-07-09
**Status:** Approved — supersedes the fusion, lip-sync, and phase-structure decisions in the pre-existing docs.

## Why this doc exists

A new build spec was handed to the assistant that conflicts with several deliberate decisions already recorded in `PROJECT_OVERVIEW.md` and the `docs/phase-*.md` plans. Rather than silently overwrite documented decisions, the conflicts were surfaced and resolved with the user before any file was touched. This doc is the record of what changed and why, so future readers don't have to reconstruct the reasoning from diffs.

## Decisions made

### 1. Fusion: feature-level, not late (score-level)

**Before:** each stream wrote a calibrated `[0,1]` fake-probability score to the feature store; fusion = weighted average, then logistic regression on those scores; the ablation table over scores was the reportable centerpiece (old Phase 3).

**Now:** each stream writes a clip-level **embedding vector** to the feature store; fusion concatenates all stream embeddings and passes them through an MLP + sigmoid. Late fusion (weighted average / logistic regression on scores) is **not** part of the primary plan — it was considered and explicitly rejected in favor of feature-level fusion for the reportable result.

**Consequence:** the feature store schema changes from `clip_id, stream_name, score, ...` to `clip_id, stream_name, embedding, ...`. Each stream still trains a *temporary* standalone classifier head during its own development stage (to confirm the embedding is discriminative before fusion exists), but that head's score is a development-time check, not what's written to the shared store.

### 2. Lip-sync stream: embedding cross-attention, not transcription

**Before:** implemented Bohacek & Farid's method — lip-read the video into text, transcribe the audio into text, compare the words (a semantic/text mismatch score).

**Now:** no transcription anywhere. A video encoder produces an embedding from mouth-region motion (Key/Value); an audio encoder produces an embedding from the audio track (Query); scaled dot-product cross-attention (`softmax(QK^T/√d)V`) produces the stream's output vector. This is architecturally closer to a SyncNet-style temporal/embedding sync check than to Bohacek & Farid's semantic approach.

**Consequence:** Bohacek & Farid moves from "implemented as stream 4" to the read-only/comparison reading list, alongside Zhou & Lim and AVFF. It is not deleted from the project's knowledge — it's just no longer what stream 4 *is*. If there's time later, their semantic-mismatch method could still be added as an additional stream, same as the project already planned to treat classic SyncNet-style sync checking as a possible 6th stream before this pivot (the two ideas have effectively swapped places).

### 3. Encoder choices (surfaced, not guessed)

All four cross-modal encoders are pretrained, off-the-shelf, and used as embedding extractors (frozen by default, per-encoder configurable):

| Encoder | Model | Output dim |
|---|---|---|
| Audio (lip-sync stream) | `facebook/wav2vec2-base-960h` (HF) | 768 |
| Video (lip-sync stream, mouth motion) | `torchvision.models.video.r2plus1d_18` (Kinetics-400) | 512 |
| Face-emotion | `trpakov/vit-face-expression` (HF, penultimate layer) | 768 |
| Voice-emotion | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (HF) | 1024 |

Chosen for being embedding-only (no ASR/lip-reading decoder attached) and usable directly via `transformers`/`torchvision` without custom pretraining — appropriate for a team learning from zero.

### 4. Frames per clip and audio alignment

Reuses the frame↔audio-window sync already built into `preprocessing/crop_faces.py` (fixed-duration window, default 0.35s, centered on each sampled frame — see `sync_audio_to_frames`). On top of that: sample a fixed **N=16 frames per clip**, uniformly across the clip, so every batch is `[batch, 16, 3, 224, 224]`. This is a new decision (the frame count) layered on an existing, already-frozen mechanism — preprocessing itself is not being redone.

### 5. Fine-tune vs. frozen

Visual backbones (Xception/EfficientNet/Swin): configurable, default staged fine-tuning (freeze → unfreeze), matching what was already established for Xception. Cross-modal encoders (table above): default **frozen**, configurable per-encoder.

### 6. Dimension compatibility

Every stream ends in a `Linear(+LayerNorm)` projection to a shared `common_dim` (default 256), configurable, before concatenation into the fusion MLP.

### 7. Third visual backbone build order

Swin Transformer is built now as the third visual stream; DINOv2 is added later as a fourth. This isn't actually a reversal — the pre-existing docs already named Swin as the documented fallback for exactly this situation (DINOv2's self-supervised training math/integration being too heavy for the current timeline). This pivot exercises that fallback rather than contradicting it. `docs/math/swin.md` and `docs/math/dinov2.md` get their framing flipped accordingly (Swin's caveat note becomes "built now"; DINOv2's note becomes "planned future addition").

### 8. Plan structure: replace phases/days/roles with a stage order

**Before:** 6 phases, each with day-by-day breakdowns and 3 named team roles (Research/Data/ML), building toward a course/capstone report.

**Now:** the phase/day/role scaffolding for what used to be Phases 1–3 is replaced by the pasted spec's Stage 1–7 build order (data pipeline → one visual stream → remaining visual streams → lip-sync → emotion → fusion → ablation support).

**Important scope boundary:** Phases 4–6 (self-supervised pretraining stretch, full evaluation suite, explainability + report/presentation) are **not** covered by the Stage 1–7 spec at all — that spec only describes building the model, not evaluating or writing it up. Per explicit confirmation, that content is preserved and renumbered as **Stage 8 (was Phase 4), Stage 9 (was Phase 5), Stage 10 (was Phase 6)**, with internal references updated (e.g. "Phase 3 complete" → "Stage 7 complete") but the substance (identity-disjoint splits, Deepfake-Eval-2024 in-the-wild testing, held-out manipulation types, robustness curves, Grad-CAM, report assembly) kept intact.

### 9. Ablation under feature-level fusion (new build note, not in the original spec)

Late-fusion ablation is simple: drop a score column. Feature-level fusion ablation is not — running a subset of streams means the concatenated input dimension changes, so the fusion MLP must be re-configured (input dim = sum of included streams' `common_dim`) and re-evaluated per subset, not just masked post-hoc. Stage 7's plan documents this explicitly so it isn't discovered as a surprise mid-implementation.

## Files affected

- `PROJECT_OVERVIEW.md` — architecture (§2), fusion (§3), keep/drop→ablation (§4), key papers (§6), folder structure (§8), phases→stages (§10), team setup (§11), status (§12).
- `README.md` — stream list, fusion description, current-stage pointer.
- `docs/glossary.md` — visual stream list, lip-sync description, fusion section, reading index.
- `docs/math/swin.md`, `docs/math/dinov2.md` — flip primary/fallback framing.
- `docs/phase-1-plan.md`, `docs/phase-2-plan.md`, `docs/phase-3-plan.md`, `docs/phase-0.5-plan.md` — superseded by new `docs/stage-1-plan.md` … `docs/stage-7-plan.md` (deleted; content folded forward where still relevant — identity-disjoint splits, labeling convention, anti-leakage discipline, calibration cautions).
- `docs/phase-4-plan.md` → `docs/stage-8-plan.md`, `docs/phase-5-plan.md` → `docs/stage-9-plan.md`, `docs/phase-6-plan.md` → `docs/stage-10-plan.md` (renamed, references updated).

## Self-review

- **Placeholders:** none left; every decision above has a concrete resolution.
- **Internal consistency:** fusion (feature-level) and ablation (Stage 7, config-driven subset with MLP re-configuration) now agree with each other; lip-sync method and the key-papers list now agree (Bohacek & Farid moved to read-only).
- **Scope:** this doc covers the architecture/plan-structure pivot only. It does not re-litigate datasets, splits, or evaluation methodology, which are unaffected and carried forward unchanged.
- **Ambiguity resolved:** "replace with 7-stage solo build order" was scoped to the Phase 1-3 equivalent only, per explicit follow-up confirmation that Phases 4-6 survive as Stage 8-10.

## Addendum (same day): Swin removed, DINOv2 only

Decision #7 above (build Swin now, add DINOv2 later) was reversed by explicit instruction. **Swin is removed from the project entirely** — not deferred, not kept as a fallback note. DINOv2 is the third visual stream directly, built in Stage 3 alongside EfficientNet.

- `docs/math/swin.md` deleted.
- `docs/math/dinov2.md` — the "deferred in favor of Swin" framing and the shared-ViT-foundation cross-reference to `swin.md` are both removed; DINOv2's write-up is now self-contained and its status is "built now."
- `docs/stage-3-plan.md` — the Swin section is replaced with a DINOv2 section, carrying forward the older Phase 2 guidance this project already had for DINOv2 (frozen backbone / lightweight probe first, since it's self-supervised; only fine-tune if the probe underperforms; cache extracted features per `clip_id` if extraction is slow) adapted to the current embedding + LSTM/GRU + `common_dim` projection template.
- `PROJECT_OVERVIEW.md`, `README.md`, `docs/glossary.md`, `docs/stage-7-plan.md` — all Swin mentions removed; DINOv2 listed as a current stream, not a planned addition.

Sections 2, 6, and 7 of this doc's main body (which describe the Swin-now/DINOv2-later decision) are left as-is for the historical record of *why* Swin was tried first — they no longer describe the current plan.
