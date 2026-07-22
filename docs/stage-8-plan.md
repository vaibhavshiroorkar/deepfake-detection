# Stage 8: Self-Supervised Pretraining — Milestone Plan (STRETCH)

**Goal (optional):** Boost generalization by pretraining the cross-modal components on real-only videos before fine-tuning, in the spirit of AVFF (Oorloff et al., 2024). The idea: let the model first learn what genuine audio-visual correspondence looks like from unlabeled real footage, so that at fine-tune time "fake" is recognized as a departure from that learned norm.

**This stage is explicitly a stretch goal.** Skip it entirely if Stages 1–7 and 9 are not comfortably done. A complete system without Stage 8 is a strong project; a half-finished Stage 8 that eats the evaluation and write-up time is not. Decide honestly at the Stage 7 gate whether there is room for this.

**Prerequisite:** Stage 7 complete, with a working fused system and time genuinely to spare before the deadline.

---

## Milestone 8A — Real-only pretraining setup

- **Data:** Assemble a real-only corpus (the `RealVideo-RealAudio` clips, optionally augmented with other real talking-head footage). No fakes at this stage — the point is to learn the genuine audio-visual joint distribution.
- **ML:** Set up a self-supervised objective for the cross-modal component — e.g. predicting/contrasting whether an audio window and a video window are genuinely aligned, or masked-reconstruction across modalities (AVFF-style). No fake/real labels used here.
- **Research:** Summarize AVFF's two-stage recipe and pin down exactly which piece we are borrowing and which we are not (we are not reproducing AVFF wholesale — no public official code, heavy pretraining; it is a benchmark we compare against, per PROJECT_OVERVIEW §11).
- **Done when:** a pretraining run completes and produces reusable cross-modal weights.

---

## Milestone 8B — Fine-tune and measure the lift

- **ML:** Initialize the cross-modal stream(s) from the pretrained weights, fine-tune on the labeled task, re-embed into the feature store (as a *variant* stream so the non-pretrained version is preserved for comparison).
- **Research:** Measure the generalization lift specifically — the honest test is on held-out manipulation types and on Deepfake-Eval-2024 (Stage 9), not just in-distribution FakeAVCeleb. Report pretrained vs not.
- **Done when:** a clean before/after comparison exists, isolating the effect of pretraining.

---

## Done when (stage gate)

- Cross-modal stream(s) pretrained on real-only data, fine-tuned, and re-embedded.
- A controlled comparison (pretrained vs from-scratch) on generalization-focused evaluation, reported honestly whether or not it helped.

## Deliverables

- Pretraining code + pretrained cross-modal weights.
- Feature-store variant embeddings for the pretrained streams.
- Report subsection: the pretraining lift (or absence of it) on out-of-distribution data.

## Risks and notes

- **Time sink risk is the main risk.** Self-supervised pretraining is finicky and slow to debug. Timebox it hard.
- **Compare on the right axis.** Pretraining is supposed to help *generalization*; measuring only in-distribution FakeAVCeleb AUC can hide or fake the benefit. Lean on Stage 9's Deepfake-Eval-2024 and held-out-manipulation results.
- **Negative result is fine.** "Pretraining did not help at our scale" is a legitimate, reportable finding — and cheaper to write than to hide.
