# Stage 9: Full Evaluation — Milestone Plan

**Goal:** Stress-test the fused system beyond the clean in-distribution number. Four evaluations: in-distribution (FakeAVCeleb), real-world in-the-wild (Deepfake-Eval-2024), held-out manipulation types, and robustness under compression and noise. This is where we find out whether the system actually generalizes or just memorized FakeAVCeleb's quirks — and it is a large part of what makes the project novel (PROJECT_OVERVIEW §12).

**Why milestone-level:** each evaluation is an independent, well-defined experiment; sequence them as you go.

**Prerequisite:** Stage 7 complete (a chosen, fused final system, ablation done). Stage 8 optional. Deepfake-Eval-2024 downloaded and **never used for training** — it is a pure test set.

---

## Milestone 9A — In-distribution benchmark (FakeAVCeleb)

- **ML:** Lock the final fused system and produce the headline in-distribution result on the identity-disjoint FakeAVCeleb test set: fused AUC, accuracy, LogLoss, confusion matrix, per-category breakdown.
- **Research:** Position this against the standalone streams and, where comparable, published numbers (e.g. AVFF as a benchmark reference).
- **Done when:** the headline in-distribution table is final.

---

## Milestone 9B — Real-world generalization (Deepfake-Eval-2024)

The honesty test. In-the-wild deepfakes, unseen generators, messy conditions.

- **Data:** Run the frozen preprocessing over Deepfake-Eval-2024 (expect more failures than on clean data — log them). Nothing here was ever trained on.
- **ML:** Evaluate every stream's embedding and the fusion on it, exactly as in-distribution. Expect a drop — the question is how large.
- **Research:** Report the in-distribution → in-the-wild gap per stream and for fusion. A key hypothesis: once added, DINOv3 and the cross-modal streams should degrade less than the artifact CNNs. Confirm or refute.
- **Done when:** Deepfake-Eval-2024 results reported, with the generalization gap quantified per stream.

---

## Milestone 9C — Held-out manipulation types

Tests whether the system catches *kinds* of fakes it never trained on.

- **Data/ML:** Re-split FakeAVCeleb so one manipulation type (or generation method) is entirely held out of training and used only at test. Retrain/evaluate. Repeat leave-one-manipulation-out as time permits.
- **Research:** Report which manipulation types transfer and which don't — this maps the system's blind spots.
- **Done when:** at least one held-out-manipulation result reported.

---

## Milestone 9D — Robustness (compression and noise)

Real videos get re-encoded and degraded; a detector that dies under JPEG/H.264 compression isn't useful.

- **Data:** Generate perturbed test variants — video compression (re-encode at lower bitrates/CRF), audio noise, resolution reduction.
- **ML:** Evaluate the fused system across perturbation strengths; plot degradation curves.
- **Research:** Report which streams are fragile (artifact CNNs are typically compression-sensitive) and whether cross-modal streams hold up better.
- **Done when:** robustness curves reported across at least compression and noise.

---

## Done when (stage gate)

- Four evaluation results complete: in-distribution, in-the-wild, held-out manipulation, robustness.
- The generalization story is quantified, not hand-waved — including where the system fails.

## Deliverables

- `evaluation/` — scripts + result tables/plots for all four evaluations.
- Report: the full evaluation section, with the in-the-wild gap and robustness curves as headline evidence of (or limits to) generalization.

## Risks and notes

- **Guard the held-out sets.** Deepfake-Eval-2024 and any held-out manipulation must never leak into training or fusion-fitting/tuning.
- **Preprocessing will fail more in the wild.** Budget time for it; report the failure rate honestly (it's a real deployment caveat).
- **Report failures, don't bury them.** The blind spots found here are among the most valuable, credible parts of the write-up.
