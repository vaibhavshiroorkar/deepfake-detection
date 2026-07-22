# Stage 4: Lip-Sync Cross-Modal Stream

**Goal:** the first genuinely cross-modal stream, and the first to use cross-attention. Verify it runs end to end and outputs a single vector per clip. No transcription or lip-reading-to-text anywhere — this stream is embeddings in, embeddings out.

**Prerequisite:** Stage 1 complete (data pipeline, frame↔audio sync). Does not depend on Stages 2–3 — visual and cross-modal streams are independent and could be built in either order, but cross-modal streams are the riskier, more novel part, so budget more time here.

---

## Architecture for this stage

- **Video encoder (Key/Value):** `torchvision.models.video.r2plus1d_18`, Kinetics-400 pretrained, run over the 16-frame mouth-region sequence (a 3D CNN naturally consumes the whole clip, no separate LSTM/GRU needed here). Output: 512-dim embedding.
- **Audio encoder (Query):** `facebook/wav2vec2-base-960h` (HuggingFace `transformers`), run over the audio track aligned to the same clip window (via the frame↔audio sync from Stage 1). Output: 768-dim embedding.
- **Cross-attention:** `softmax(QK^T/√d) V` — audio embedding projects to Query, video embedding projects to Key and Value. Direction is config-driven (audio-as-Query is the default, matching the project convention), not hardcoded.
- **Output:** the attention output is the stream's vector, projected (`Linear` + `LayerNorm`) to `common_dim` (256) same as the visual streams.
- **Frozen by default:** both encoders are used as fixed feature extractors (configurable to unfreeze later if standalone performance is weak).

## Why this method, not transcription

The project previously planned this stream around Bohacek & Farid's method (lip-read video → text, transcribe audio → text, compare words). That is deliberately not implemented — see [PROJECT_OVERVIEW.md §11](PROJECT_OVERVIEW.md) for the reasoning and where that citation now lives (read-only comparison). This stream is closer in spirit to SyncNet-style embedding/temporal sync checking.

---

## Tasks

**Data:**
- Confirm mouth-region crops (or full face crops, if mouth-region-specific cropping isn't built yet — record which was used) and the audio track are both available per `clip_id`, keyed consistently with Stage 1's manifests.
- Done when: a batch yields aligned `(video_frames, audio_waveform, label)` per clip, same `clip_id` keying as Stage 1.

**ML:**
- Wire up the two frozen encoders, the cross-attention module (with configurable Q/K-V direction), and the projection layer.
- Verify shapes at every step: encoder outputs, attention output, projected output. Run a forward pass on a small batch and confirm a `[batch, common_dim]` vector comes out with no NaNs/errors.
- Add a temporary classifier head (same pattern as Stage 2) to sanity-check the embedding is at least weakly discriminative on a small sample — not a tuned model, just a smoke test.
- Done when: the stream runs end to end and outputs a vector; the temporary head's loss is not stuck at chance on a small sample.

**Research:**
- Document the cross-attention spec precisely (Q/K/V projections, direction, scaling) so Stage 5 can reuse the identical pattern with a different direction default.
- Check whether this stream's temporary-head score responds differently on `RealVideo-FakeAudio` clips than the visual streams do — that's this stream's reason to exist, and worth confirming early even informally.
- Done when: cross-attention spec documented; an early informal check on `RealVideo-FakeAudio` clips recorded (even if not a full evaluation yet).

---

## Done when (stage gate)

- Lip-sync stream runs end to end: video + audio in, one `common_dim` vector out, no transcription anywhere in the path.
- Per-clip lip-sync embeddings written to the feature store.
- Early evidence (even informal) that this stream behaves differently on audio-fake-only clips than the visual streams.

## Deliverables

- `models/streams/lipsync/` — the cross-attention lip-sync stream.
- Cross-attention module documented and reusable for Stage 5.

## Risks and notes

- **This is the risky, novel part of the project.** If it's very hard to get working well, a visual-only + one-cross-modal system is still a valid intermediate result — don't let this stage block everything else indefinitely.
- **Encoder mismatch risk:** R(2+1)D-18 and Wav2Vec2 were not trained together or for this task; the cross-attention layer has to learn the alignment from scratch. If results are weak, revisit whether the encoders need lightweight fine-tuning (config already supports unfreezing) before concluding the architecture doesn't work.
