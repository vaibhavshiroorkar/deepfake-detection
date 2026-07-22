# Stage 10: Explainability and Write-Up — Milestone Plan

**Goal:** Turn a working, evaluated system into something a reader can *understand and trust*, then assemble the final report and presentation. Two threads: explainability (show which stream caught which fake, and where the visual streams look) and the write-up (which, if the "write the report as we go" principle held, is assembly and polish rather than a from-scratch scramble).

**Why milestone-level:** these are parallelizable, well-scoped chunks that map cleanly onto Research/Data/ML workstreams.

**Prerequisite:** Stage 9 complete — final system with full evaluation results. Report drafts accumulated across all prior stages.

---

## Milestone 10A — Per-stream attribution ("which stream caught which fake")

The system's most compelling explainability story, and nearly free — the feature store already holds per-stream embeddings per clip.

- **ML/Data:** For representative fakes (especially the partial fakes fusion caught that no single stream did), pull each stream's development-time standalone score (from its Stage 2–5 temporary head, or a small probe on its stored embedding) and visualize the breakdown (e.g. the worked-example style from PROJECT_OVERVIEW §3 — visual streams unsure, cross-modal stream lit up, fusion confident).
- **Research:** Curate a handful of clean case studies, including at least one lip-sync fake where the cross-modal stream is the hero. Also show a failure case honestly.
- **Done when:** a set of per-clip attribution figures exists, including the flagship "fusion caught what nothing else did" example.

---

## Milestone 10B — Visual explainability (Grad-CAM)

- **ML:** Run Grad-CAM (or similar) on the surviving visual stream(s) to show *where* on the face they attend — blending boundaries, mouth region, etc. Overlay heatmaps on example crops.
- **Research:** Connect the heatmaps back to the architecture writeups (why an artifact CNN lights up on blend edges; how DINOv2, once added, differs).
- **Done when:** Grad-CAM overlays for real and fake examples are produced and explained.

---

## Milestone 10C — Assemble the report

- **Research (lead):** Merge the drafts written across stages into one coherent report: Intro/Related Work, Methods (splits, streams, feature-level fusion), the combined architecture math document, Results (Stages 2–9), Explainability (10A/10B), Limitations, Conclusion.
- **Data + ML:** Supply final numbers, figures, and reproducibility details (configs, hyperparameters, split definitions) so the methods are reproducible.
- **Done when:** a complete, internally consistent report draft exists, ready for review.

---

## Milestone 10D — Presentation

- **All three:** Build the presentation from the report's spine — problem and motivation (the partial-fake insight), system diagram (five streams → feature-level fusion), headline results (in-distribution + in-the-wild), the flagship attribution example, and honest limitations.
- **Research (lead):** Own narrative flow; Data owns the pipeline/system slides; ML owns the results/ablation slides.
- **Done when:** presentation delivered/rehearsed, backed by the report.

---

## Done when (stage gate)

- Explainability figures produced (per-stream attribution + Grad-CAM), including a flagship example and an honest failure case.
- Final report assembled, consistent, reproducible.
- Presentation built and rehearsed.

## Deliverables

- Explainability figures/notebooks (attribution + Grad-CAM) under `evaluation/` or `notebooks/`.
- Final report (in `docs/`).
- Final presentation.

## Risks and notes

- **This stage is cheap only if the report was written as you went** (PROJECT_OVERVIEW §9). If drafts were skipped, budget real time here — do not discover that on the last day.
- **Explainability is a differentiator, not a footnote** (PROJECT_OVERVIEW §12). The "which stream caught which fake" story is a large part of what makes the project stand out — give it real space.
- **Show a failure case.** A report that only shows wins reads as less credible than one that shows where the system breaks and why.
