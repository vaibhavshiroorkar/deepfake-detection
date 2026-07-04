# Phase 3: Fusion — Milestone Plan

**Goal:** Combine the five per-stream scores into one final real-or-fake decision, and run the ablation that tells us which streams actually earn their place. This is where the project's central claim gets tested: that fusing cross-modal mismatch cues with strong visual backbones beats any single stream — especially on the partial fakes no single stream catches alone.

**Why milestone-level:** fusion is small, fast code (weighted average, then logistic regression) but the *interesting* work is analysis, which depends entirely on what the Phase 2 scores look like. Detail into days when you start, once the feature store is populated.

**Prerequisite:** Phase 2 complete — all five streams' calibrated per-clip scores in the feature store, over identity-disjoint splits.

**Critical discipline:** fit any learned fusion on the **validation** split and report on the **test** split. Fitting fusion weights on the test set is the same leakage mistake as identity leakage, one level up.

---

## Milestone 3A — Weighted-average fusion (baseline)

Start dumb and honest.

- **ML:** Read the five scores per `clip_id` from the feature store, combine by a plain average (and a hand-tuned weighted average), threshold to a verdict. Compute fused clip-level AUC/accuracy/LogLoss on test.
- **Research:** Compare fused vs best-single-stream. Reproduce the worked-example logic from PROJECT_OVERVIEW §3 on real clips: find actual partial fakes where visual streams are unsure but a cross-modal stream fires, and show fusion catches them.
- **Done when:** fused-average metrics recorded and compared against every standalone stream.

---

## Milestone 3B — Learned fusion (logistic regression)

- **ML:** Fit logistic regression on the **validation** split (features = the five stream scores, target = clip label), evaluate on **test**. Inspect the learned weights — they say how much the system trusts each stream.
- **Research:** Interpret the weights against expectation (e.g. does lip-sync get a high weight on audio-fake clips?). Report learned-fusion vs weighted-average vs best-single.
- **Done when:** learned-fusion test metrics recorded; weights interpreted in the report.

---

## Milestone 3C — Ablation and keep/drop decision

The reportable centerpiece.

- **ML:** Build the ablation table: each stream alone, and meaningful combinations (all visual; visual + one cross-modal; all five; leave-one-out). Same metric, same test split.
- **Research:** From the table, make the keep/drop calls:
  - Do Xception and EfficientNet overlap so much that one is redundant? (Both are artifact CNNs — likely candidates to prune.)
  - Does DINOv2 survive by catching different fakes? (Expected to.)
  - How much does each cross-modal stream add over visual-only?
  - Document every drop as a data-driven finding, not a failure (per PROJECT_OVERVIEW §4).
- **Done when:** ablation table complete, keep/drop decisions made and justified, final stream set chosen.

---

## Optional upgrade (later, not required) — feature-level fusion

If time allows, compare late fusion (scores) against feature-level fusion (streams hand over internal features into one joint model). The late-vs-feature-level comparison is itself a reportable result (PROJECT_OVERVIEW §3). Treat as stretch; do not block Phase 5 on it.

---

## Done when (phase gate)

- A single fused verdict per clip, with test-set AUC beating the best single stream (the core hypothesis, confirmed or honestly refuted).
- Ablation table complete; final stream set chosen with justification.
- Fusion fit on validation, reported on test — no test leakage.

## Deliverables

- `fusion/` — weighted-average and logistic-regression fusion, reading from the feature store.
- Ablation table (streams alone + combinations) in the report.
- Results section: fused vs single, learned-weight interpretation, keep/drop decisions, worked real-clip examples.

## Risks and notes

- **Test leakage via fusion fitting** is the trap here. Val-fit, test-report. No exceptions.
- **If fusion doesn't beat the best single stream,** that's still a result — but first check calibration (Phase 2) and whether the cross-modal streams are actually working, before concluding fusion doesn't help.
- **The keep/drop outcome is a genuine finding either way.** Three complementary streams and "we pruned one redundant CNN" are both good report material.
