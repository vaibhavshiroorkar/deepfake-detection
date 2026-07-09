# Stage 7: Ablation Support

**Goal:** the ability to run any configured subset of the five streams and compare fused metrics — the reportable centerpiece that decides which streams actually earn their place.

**Prerequisite:** Stage 6 complete — feature-level fusion working, full metrics computed on test.

---

## The build note that matters here

Late-fusion ablation (the old plan) is simple: drop a score column and re-average or re-fit a small logistic regression — cheap, no retraining of a large model. **Feature-level fusion ablation is not that simple**, and this is worth stating explicitly so it isn't discovered as a surprise mid-implementation:

- Running a subset of streams changes the fusion MLP's **input dimension** (`k × common_dim` for `k` included streams, not always `5 × common_dim`).
- This means the fusion MLP's first layer must be reconfigured (or a separate MLP instance trained) per subset — you cannot just zero out or mask embedding slots in a fixed-size input and expect a meaningful result, because the MLP was never trained to see zeros in that pattern.
- Practical approach: make the stream subset a config value that determines the concatenation order/inclusion and the resulting MLP input size, then train (or fine-tune from the full-set checkpoint, if that's faster and doesn't bias results) a fusion MLP per subset being evaluated.

---

## Tasks

**ML:**
- Build the ablation runner: given a config listing which streams to include, assemble the correct concatenated embedding, instantiate/train a correctly-sized fusion MLP, and evaluate on the same test split as Stage 6.
- Run the ablation table: each stream alone (with a minimal single-embedding "fusion" MLP), meaningful combinations (all visual; visual + one cross-modal; all five; leave-one-out).
- Done when: the ablation table is complete, same metric (test AUC, plus the full metrics suite) and same test split throughout.

**Research:**
- From the table, make the keep/drop calls:
  - Do Xception and EfficientNet overlap so much that one is redundant?
  - Does DINOv2 survive by catching different fakes than Xception/EfficientNet? (Expected to, being self-supervised rather than trained on manipulation artifacts.)
  - How much does each cross-modal stream add over visual-only?
  - Document every drop as a data-driven finding, not a failure (per [PROJECT_OVERVIEW.md §4](../PROJECT_OVERVIEW.md)).
- Done when: ablation table complete, keep/drop decisions made and justified, final stream set chosen.

---

## Done when (stage gate)

- Ablation table complete: each stream alone and in meaningful combinations, same metric and test split.
- Keep/drop decisions made and justified from the table, not from intuition alone.
- Final stream set for the fused system is chosen.

## Deliverables

- `evaluation/` — the ablation runner and its results table.
- Report section: ablation table, keep/drop decisions, final stream-set justification.

## Risks and notes

- **The reconfiguration cost is the main risk.** If retraining a fusion MLP per subset is too slow, prioritize the combinations that answer real questions (all-visual vs. all-five; leave-one-cross-modal-out) over an exhaustive power set.
- **The keep/drop outcome is a genuine finding either way.** Three complementary streams and "we pruned one redundant CNN" are both good report material.
