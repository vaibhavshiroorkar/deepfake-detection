# Stage 6: Fusion

**Goal:** combine all five per-clip embeddings into one final real/fake decision via **feature-level fusion** (concatenation + MLP + sigmoid), with the full metrics suite. This is where the project's central claim gets tested: that fusing cross-modal mismatch cues with strong visual backbones beats any single stream, especially on partial fakes no single stream catches alone.

**Prerequisite:** Stages 2–5 complete — all five streams write `common_dim`-projected embeddings to the shared feature store, over identity-disjoint splits.

**Critical discipline:** fit the fusion MLP on the **train** split, tune on **validation**, and report only on the **test** split. Fitting or tuning on the test set is the same leakage mistake as identity leakage, one level up.

---

## Architecture for this stage

1. Read all five streams' embeddings for a `clip_id` from the feature store.
2. Concatenate into one vector: `5 × common_dim` (e.g. `5 × 256 = 1280`) if all streams are included.
3. Pass through an MLP (a few fully-connected layers, config-driven width/depth).
4. Sigmoid → final `[0,1]` fake-probability. Threshold at 0.5 for the real/fake label.

## Tasks

**ML:**
- Build the fusion MLP reading from the feature store, keyed by `clip_id`. Discard each stream's temporary classifier head at this point — the shared embeddings are what feed fusion now.
- Train on the **train** split, select the best checkpoint on **validation**, report final numbers on **test**.
- Compute the full metrics suite on test: accuracy, AUC-ROC, LogLoss, precision, recall, F1, and a confusion matrix.
- Done when: fused test metrics are computed and recorded, with no test leakage in fitting or model selection.

**Research:**
- Compare fused vs. best-single-stream (using each stream's Stage 2–5 standalone AUC as the baseline).
- Reproduce the worked-example logic from [PROJECT_OVERVIEW.md §3](../PROJECT_OVERVIEW.md) on real clips: find actual partial fakes (especially `RealVideo-FakeAudio`) where the visual streams' standalone signal was weak but fusion still catches them.
- Done when: fused-vs-single comparison recorded, with at least one concrete worked example.

---

## Optional comparison (later, not required) — late fusion

If time allows, also build the simpler alternative: each stream's temporary head recalibrated as a proper score, combined by weighted average and then logistic regression, evaluated the same way (val-fit, test-report). Comparing feature-level fusion against this gives a genuine reportable result — see [PROJECT_OVERVIEW.md §3](../PROJECT_OVERVIEW.md). Treat as stretch; do not block Stage 9 (evaluation) on it.

---

## Done when (stage gate)

- A single fused verdict per clip, with test-set AUC (and the full metrics suite) computed, beating or honestly not beating the best single stream.
- Fusion fit on train/tuned on validation, reported on test — no test leakage.

## Deliverables

- `fusion/` — the feature-level fusion MLP, reading from the feature store.
- Results section: fused vs. best-single-stream, at least one worked real-clip example.

## Risks and notes

- **Test leakage via fusion fitting** is the trap here. Train-fit, validation-tune, test-report. No exceptions.
- **If fusion doesn't beat the best single stream,** that's still a result — but first check whether the cross-modal streams (Stages 4–5) are actually producing discriminative embeddings, before concluding fusion doesn't help.
- **Interpretability is weaker here than late fusion would have given** — there's no per-stream fusion weight to read off. Stage 7's ablation is how per-stream contribution actually gets measured under this architecture.
