# Stage 2: First Visual Stream End to End

**Goal:** Xception + a temporal model (LSTM or GRU) + a temporary simple classifier head, trained on a small sample, with loss dropping and metrics computing. This proves the whole per-stream template — backbone → temporal model → embedding → (temporary) classifier — before the remaining streams clone it in Stage 3.

**Prerequisite:** Stage 1 complete — DataLoader yielding correct shapes, identity-disjoint splits, feature-store schema frozen.

---

## Architecture for this stage

1. Each of the 16 sampled frames goes through Xception (`timm`, `legacy_xception` — the un-prefixed `"xception"` name is deprecated in current `timm`) to get a per-frame embedding.
2. The sequence of 16 per-frame embeddings goes through an LSTM/GRU (config-selectable) to produce **one clip-level embedding**.
3. That clip-level embedding is projected (`Linear` + `LayerNorm`) to the shared `common_dim` (default 256) — this is what will eventually go into the feature store.
4. **For this stage only:** a temporary classifier head (`Linear` → sigmoid) sits on top of the clip embedding so standalone loss/metrics can be verified. This head is discarded once fusion (Stage 6) exists — it is a development-time check, not part of the final system.

## A note on labels for a visual-only stream

Xception sees only face frames, never audio. Its label is **the authenticity of the video track**, not the clip overall:

- `FakeVideo-RealAudio` and `FakeVideo-FakeAudio` → **fake**
- `RealVideo-RealAudio` → **real**
- `RealVideo-FakeAudio` → **real** to this stream, even though the *clip* is a fake

Do not treat the last row as a bug — it's the whole reason the cross-modal streams (Stages 4–5) exist.

---

## Tasks

**Model (ML workstream):**
- Load pretrained Xception from `timm`, wire in the LSTM/GRU temporal model, add the temporary classifier head.
- Train in two stages: (1) freeze the Xception backbone, train the temporal model + head; (2) unfreeze, fine-tune end-to-end at a lower learning rate. Configurable, not hardcoded.
- Log train/val loss and val AUC per epoch (using the temporary head); save the best checkpoint by val AUC, not loss.
- Done when: a checkpoint exists with a sane val AUC and no obvious overfitting.

**Data (Data workstream):**
- Add balanced sampling (`WeightedRandomSampler`) so training isn't overwhelmed by the fake-heavy data; add light train-time augmentation (flip, small color jitter) — avoid augmentations that destroy the artifacts being detected.
- Done when: training batches are class-balanced; augmentation validated visually.

**Verification (Research workstream):**
- Confirm loss drops over a small sample and that accuracy, AUC-ROC, LogLoss, precision, recall, F1, and a confusion matrix all compute correctly on that sample.
- Record hyperparameters, backbone, freezing schedule, and sampling strategy.
- Done when: a small-sample training run is documented with all metrics computing and loss visibly dropping.

**Freeze the reusable template (ML workstream):**
- Refactor into the clean, reusable pattern every later visual stream (Stage 3) clones: `config → backbone → temporal model → projection → (temporary head for dev) → embedding written to feature store`.
- Done when: the template is documented well enough that swapping the backbone is the only change needed for Stage 3.

---

## Done when (stage gate)

- Xception + LSTM/GRU + temporary head trains on a small sample; loss drops.
- All required metrics (accuracy, AUC-ROC, LogLoss, precision, recall, F1, confusion matrix) compute correctly.
- A documented, reusable stream template exists for Stage 3 to clone.

## Deliverables

- `models/streams/xception/` — the reusable end-to-end template.
- A small-sample training run with metrics recorded.

## Risks and notes

- **Frame→clip aggregation** is now done by the temporal model (LSTM/GRU), not by pooling frame-level scores — this is a deliberate change from simple mean/max pooling of per-frame predictions, made to support feature-level fusion (Stage 6), which needs one embedding per clip, not a pooled score.
- **Don't over-invest in the temporary head.** It exists only to confirm the embedding is discriminative; tuning it heavily is wasted effort once Stage 6's fusion MLP replaces it.
