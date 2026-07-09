# Stage 5: Emotion Cross-Modal Stream

**Goal:** the second cross-modal stream, cloning Stage 4's cross-attention pattern with different encoders and the opposite default Q/K-V assignment. Compares emotion on the face against emotion in the voice.

**Prerequisite:** Stage 4 complete — cross-attention module proven and documented.

---

## Architecture for this stage

- **Face-emotion encoder (Key/Value):** `trpakov/vit-face-expression` (HuggingFace `transformers`, FER-trained ViT) — use the penultimate layer's embedding, not the classification logits. Run per sampled frame, then combine across the 16-frame sequence (simplest: mean-pool the per-frame embeddings before projection; note this as a design choice, an LSTM/GRU is a reasonable upgrade if mean-pooling underperforms). Output: 768-dim.
- **Voice-emotion encoder (Query):** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (HuggingFace `transformers`, dimensional speech-emotion model). Output: 1024-dim.
- **Cross-attention:** same module as Stage 4, direction flipped by config — voice embedding is Query, face embedding is Key/Value (the project default for this stream).
- **Output:** projected to `common_dim` (256), same as every other stream.
- **Frozen by default**, same as Stage 4's encoders.

---

## Tasks

**Data:**
- Reuse the synced face + audio inputs from Stage 4 — no new preprocessing needed.
- Done when: a batch yields the same aligned `(face_frames, audio_waveform, label)` per clip as Stage 4.

**ML:**
- Wire up the face-emotion and voice-emotion encoders, reuse the Stage 4 cross-attention module with the direction flipped via config (not a code fork).
- Verify shapes end to end; add a temporary classifier head for a smoke test, same pattern as Stages 2 and 4.
- Done when: the stream runs end to end and outputs a `[batch, common_dim]` vector; temporary head's loss is not stuck at chance on a small sample.

**Research:**
- Note that emotion mismatch is expected to be a weaker/noisier signal than lip-sync, and to help mainly in fusion (Stage 6) rather than standalone — record standalone AUC anyway for the Stage 7 ablation table, but don't be surprised if it's modest alone.
- Done when: standalone smoke-test result recorded with that framing.

---

## Done when (stage gate)

- Emotion stream runs end to end, embeddings written to the feature store.
- Cross-attention direction (voice-as-Query) confirmed working via the same reusable module as Stage 4, not a separate implementation.

## Deliverables

- `models/streams/emotion/` — the cross-attention emotion stream.

## Risks and notes

- **Mean-pooling the face-emotion sequence** is the simplest choice and the one to start with; if standalone performance is weak, an LSTM/GRU over per-frame face-emotion embeddings (same pattern as the visual streams, Stage 2) is the natural upgrade — try the simple version first.
- **Score calibration matters less here** than it did under the old late-fusion plan, since this stream now outputs an embedding, not a probability — but the temporary head's score still needs to be a real (if rough) discriminator, not noise, or Stage 6 has nothing to fuse.
