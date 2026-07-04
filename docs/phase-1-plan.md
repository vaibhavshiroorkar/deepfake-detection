# Phase 1: First Stream End to End — Day-by-Day Plan

**Goal:** Build one visual stream (Xception) fully on real FakeAVCeleb data and get a real, trustworthy AUC number on an identity-disjoint test set. This phase proves the entire pipeline — preprocessing → dataset → model → evaluation → score written to the shared feature store — before we scale to five streams. If this works cleanly, every later stream is a clone of this template.

**This is the most important phase in the project.** A wrong split here (identity leakage) produces a beautiful, meaningless AUC and poisons every downstream result. Get it right slowly rather than fast.

---

## Prerequisites (must be true before Day 1)

- FakeAVCeleb access granted and the dataset downloaded locally (requested in Phase 0.5 Day 2).
- Preprocessing module frozen from Phase 0.5 (`preprocessing/`, face crops + interface locked).
- Environment working with GPU confirmed (done: RTX 5070 Ti, 16 GB, cu130 wheels).

If FakeAVCeleb access is still pending, do not block the team — Days 1–2 data-audit work can start on a small sample, but the real AUC (Days 4–5) needs the full dataset.

---

## A note on labels for a visual-only stream (read before starting)

Xception sees only face frames. It never hears the audio. So its label is **the authenticity of the video track**, not the clip overall:

- `FakeVideo-RealAudio` and `FakeVideo-FakeAudio` → **fake** (video manipulated, Xception should catch it)
- `RealVideo-RealAudio` → **real**
- `RealVideo-FakeAudio` → **real** to Xception (the frames are genuine), even though the *clip* is a fake

That last row is the whole reason cross-modal streams exist. Xception *should* miss `RealVideo-FakeAudio` — the video really is real. Do not treat that miss as a bug. Record it; it is the motivation for Phases 2 and 3. Report the visual AUC against the video-track label, and separately report per-category recall so the `RealVideo-FakeAudio` blind spot is visible.

---

## Day 1 — Splits and data audit (anti-leakage day)

**Person 2 (Data): Build identity-disjoint, balanced splits**
- Audit FakeAVCeleb: count videos per category, list unique identities, check for corrupt/unreadable files.
- Build train/val/test manifests where **no identity appears in more than one split**. Group by identity first, then assign whole identity-groups to splits (e.g. 70/15/15 by identity, not by clip).
- Balance: FakeAVCeleb is ~500 real vs ~19,500 fake. Decide the strategy now (undersample fakes, or balanced sampling at train time — see Day 3). Record real/fake counts per split.
- Write splits as CSV manifests in `data/` (columns: `clip_id`, `video_path`, `label`, `manipulation_type`, `identity`, `split`).
- Done when: three manifest files exist, verified to have zero identity overlap and known class balance.

**Person 3 (ML): Feature-store schema + stream scaffold**
- Define and stub the shared feature store: schema `clip_id, stream_name, score, split, label, manipulation_type`. Write the reader/writer helpers (start with a CSV/Parquet in `feature_store/`).
- Create `models/streams/xception/` from the Phase 0.5 template (config, train, eval, score-writer skeletons).
- Done when: feature-store read/write round-trips a dummy row; xception stream folder scaffolded.

**Person 1 (Research): Labeling convention + report kickoff**
- Write the visual-only labeling convention (the section above) into the report's Methods.
- Draft the Methods subsections: dataset, identity-disjoint splitting rationale, and the evaluation metric (AUC-ROC primary, plus accuracy/LogLoss/confusion matrix).
- Done when: labeling + splits + metric are documented in the report draft.

---

## Day 2 — Preprocessing at scale + DataLoader

**Person 2 (Data): Run preprocessing over the splits**
- Run the frozen preprocessing across all train/val/test videos → face crops on disk, organized by `clip_id`.
- Log failures (missed faces, multiple faces, corrupt files) to a file; decide how each is handled (drop clip, keep largest face, etc.). Verify crop counts against expected.
- Done when: face crops exist for the split set, with a failure log and final usable-clip counts.

**Person 3 (ML): Dataset / DataLoader**
- Build the `Dataset`/`DataLoader` that reads face crops + labels from the manifests.
- Sanity-check a batch: tensor shapes (`N×3×224×224`), label balance in a batch, normalization, and that augmentation (if any) looks right on a few decoded images.
- Done when: a batch loads with correct shapes and labels traceable back to `clip_id`.

**Person 1 (Research): Leakage spot-check + reading**
- Independently spot-check the split manifests for identity leakage (pick random identities, confirm they live in one split only).
- Read ahead on Xception fine-tuning practice (transfer learning, freezing/unfreezing) to support Day 3.
- Done when: leakage check signed off; fine-tuning notes ready.

---

## Day 3 — Model and training

**Person 3 (ML): Fine-tune Xception**
- Load pretrained backbone from timm — **note: the model is `legacy_xception`, not `xception`** (the old name is deprecated in our pinned timm). Swap the head to a single logit (`BCEWithLogitsLoss`) or 2-class.
- Train in two stages: (1) freeze backbone, train head; (2) unfreeze, fine-tune end-to-end with a lower learning rate. Train on GPU.
- Log train/val loss and val AUC per epoch; save the best checkpoint by val AUC (not by loss).
- Done when: a checkpoint exists with a sane val AUC and no obvious overfitting (train/val curves tracked).

**Person 2 (Data): Balanced sampling + augmentation**
- Add balanced sampling (e.g. `WeightedRandomSampler`) so the model isn't overwhelmed by the fake-heavy data; add light train-time augmentation (flip, small color jitter) — avoid augmentations that destroy the artifacts we're trying to detect.
- Done when: training batches are class-balanced; augmentation validated visually.

**Person 1 (Research): Record the experiment**
- Capture hyperparameters, backbone, freezing schedule, and sampling strategy for the report. Start the results table skeleton (per-category and overall).
- Done when: experiment config is written down reproducibly.

---

## Day 4 — Evaluate and write scores

**Person 3 (ML): Clip-level evaluation**
- Xception predicts per *frame*, but labels and the feature store are per *clip*. Aggregate frame scores → clip score (start with mean pooling; note max pooling as an alternative to try).
- On the identity-disjoint **test** split, compute clip-level AUC-ROC, accuracy, LogLoss, and a confusion matrix. This is the "real AUC number."
- Done when: clip-level test metrics are computed and saved.

**Person 2 (Data): Write scores to the feature store**
- Verify the frame→clip aggregation maps correctly to `clip_id`, then write Xception's per-clip fake-probabilities (for all splits) into the feature store under the frozen schema.
- Done when: feature store contains one Xception score per clip, keyed by `clip_id`, for val and test.

**Person 1 (Research): Interpret and break down**
- Break the AUC down by manipulation category. Confirm the expected `RealVideo-FakeAudio` blind spot appears and explain it (motivates cross-modal streams).
- Done when: results section has overall AUC + per-category table + the blind-spot narrative.

---

## Day 5 — Freeze the template and the interfaces

**Person 3 (ML): Freeze the reusable stream template**
- Refactor the Xception stream into the clean, reusable end-to-end template every Phase 2 stream will clone: `config → train → eval → write scores`. Document how to point it at a new backbone.
- Freeze the feature-store schema.
- Done when: the template is documented and a second person could clone it for a new stream.

**Person 2 (Data): Freeze split + score interfaces**
- Freeze the split manifests and the score-writing interface; write a short spec so every future stream writes identical rows.
- Done when: manifest format + score-writing spec are documented and locked.

**Person 1 (Research): Finalize Phase 1 write-up**
- Finalize the Phase 1 results section; note explicitly what fusion (Phase 3) will consume from the feature store.
- Done when: Phase 1 section is review-ready.

---

## Done when (phase gate)

- A real clip-level **AUC-ROC** on an identity-disjoint FakeAVCeleb test set, with per-category breakdown.
- Xception per-clip scores written to the shared feature store under a frozen schema.
- A reusable, documented stream template that Phase 2 can clone.
- Confirmed: no identity leakage across splits.

## Deliverables

- `data/` split manifests (train/val/test, identity-disjoint).
- `feature_store/` populated with Xception scores.
- `models/streams/xception/` — the frozen reusable template.
- Report: Phase 1 Methods + Results (overall AUC, per-category table, blind-spot narrative).

## Risks and notes

- **Identity leakage** is the #1 risk. If the AUC looks suspiciously high (>0.98 on unseen fakes), suspect leakage before celebrating.
- **Dataset access delay** can push Days 4–5. Do the leakage-proof split design early regardless.
- **Frame→clip aggregation** choice (mean vs max) is a real design decision — record which you used; it may differ per stream.
