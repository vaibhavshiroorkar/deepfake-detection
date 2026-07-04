# Phase 2: Add the Remaining Streams — Milestone Plan

**Goal:** Grow from one stream to five, each one a clone of the Phase 1 template, each writing per-clip scores into the same frozen feature store. Two more visual streams (EfficientNet, DINOv2), then the two cross-modal streams (lip-sync/semantic, affective) that make this system genuinely audio-visual.

**Why milestone-level (not day-by-day):** the two visual streams are near-mechanical clones of Phase 1 and should go fast; the two cross-modal streams are research-heavy and their exact day count depends on how the underlying models (lip-reading, ASR, emotion) behave on our data. Detail each milestone into days *when you start it*, using the Phase 1 template as the pattern.

**Prerequisite:** Phase 1 complete — frozen stream template, frozen feature-store schema, identity-disjoint splits, working preprocessing.

**Order matters:** do the visual clones first (2A, 2B) to bank easy wins and stress-test the template, then the harder cross-modal streams (2C, 2D).

---

## Milestone 2A — EfficientNet visual stream

The easiest clone. Same face crops, same DataLoader, same labels as Xception — only the backbone changes.

- **ML:** Clone the Phase 1 template, swap the backbone to EfficientNet (timm), fine-tune, evaluate clip-level AUC on the same test split.
- **Data:** Reuse the exact same crops/manifests/sampler — no new preprocessing. Confirm scores write to the feature store correctly.
- **Research:** Log results into the shared table next to Xception; note whether EfficientNet catches different fakes or overlaps heavily (this feeds the Phase 3 keep/drop decision).
- **Done when:** EfficientNet clip-level AUC recorded and per-clip scores in the feature store.

---

## Milestone 2B — DINOv2 visual stream

Different in kind: DINOv2 is self-supervised features. Two options — start with the cheaper one.

- **ML:** Extract DINOv2 features for each face crop and train a lightweight head (linear probe or small MLP) on top — cheaper and often strong. Only fine-tune the backbone if the probe underperforms. Evaluate clip-level AUC.
- **Data:** Same crops; add a feature-caching step if extraction is slow (cache per `clip_id` so it isn't recomputed).
- **Research:** Record whether DINOv2 generalizes to fake types the CNNs miss (the hypothesis from its writeup — self-supervised features should be less overfit to one generator's fingerprint).
- **Done when:** DINOv2 clip-level AUC recorded and per-clip scores in the feature store.

**Checkpoint after 2B:** all three visual streams are in the feature store. This is the first moment fusion *could* run on visual-only scores — a useful early integration test even before the cross-modal streams land.

---

## Milestone 2C — Lip-sync / semantic mismatch stream (Bohacek & Farid)

The first genuinely cross-modal stream. It compares the *words on the lips* against the *words in the audio*.

- **Data:** Extend preprocessing to also produce the audio track and mouth-region crops synced to frames (the frame↔audio-window sync built in Phase 0.5 Day 3 gets reused here). Keep everything keyed by `clip_id`.
- **ML:** Build the mismatch scorer: a visual speech-recognition (lip-reading) read of the video and an ASR transcript of the audio, then a word/semantic-alignment score. High mismatch → fake. Aggregate to a per-clip score. Calibrate the score to [0,1] fake-probability so it fuses cleanly.
- **Research:** Summarize the Bohacek & Farid method precisely (inputs, mismatch scoring, thresholds) and document how our implementation differs from theirs. Check that this stream fires on the `RealVideo-FakeAudio` clips the visual streams miss — that's its reason to exist.
- **Done when:** per-clip lip-sync/semantic scores in the feature store, and evidence it catches at least some audio-only fakes the visual streams miss.

---

## Milestone 2D — Affective / emotion mismatch stream (Mittal et al.)

Compares the emotion on the face against the emotion in the voice.

- **Data:** Reuse the synced face + audio from 2C. Provide face-emotion inputs and voice-emotion inputs per clip.
- **ML:** Extract a facial-emotion representation and a speech-emotion representation, score their disagreement, aggregate per clip, calibrate to a fake-probability.
- **Research:** Summarize Mittal et al.'s "Emotions Don't Lie" approach; note that emotion mismatch is a weaker/noisier signal than lip-sync and is expected to help mainly in fusion, not alone.
- **Done when:** per-clip affective scores in the feature store.

---

## Done when (phase gate)

- All five streams write per-clip fake-probabilities to the shared feature store, keyed by `clip_id`, over the same splits, all calibrated to [0,1].
- Each stream has a recorded standalone clip-level AUC.
- At least one cross-modal stream demonstrably catches fakes the visual streams miss.

## Deliverables

- `models/streams/{efficientnet,dinov2,sync,affective}/` — four new streams from the template.
- Feature store populated with all five streams' scores.
- Results table: standalone AUC per stream, with early notes on redundancy and complementarity (input to Phase 3 ablation).

## Risks and notes

- **Cross-modal streams are the risky, novel part.** Budget more time for 2C/2D than 2A/2B. If a cross-modal stream is very hard to get working, the visual-only + one-cross-modal system is still a valid result.
- **Score calibration matters for fusion.** A stream that outputs raw distances instead of [0,1] probabilities will dominate or vanish in fusion. Calibrate before writing to the store.
- **Keep the template honest:** if a stream needs the template changed, change the template and re-note it, don't fork per-stream hacks.
