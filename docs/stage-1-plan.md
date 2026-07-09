# Stage 1: Data Pipeline

**Goal:** a `Dataset` + `DataLoader` that yields `(face_crop_sequence, audio, label)` per clip with correct, verified shapes. Nothing trains yet — this stage only proves the data is loadable, aligned, and leakage-free. Every later stage depends on this being right.

**This is the most important stage in the project.** A wrong split here (identity leakage) produces a beautiful, meaningless AUC and poisons every downstream result. Get it right slowly rather than fast.

---

## Prerequisites

- Environment working, GPU confirmed.
- FakeAVCeleb access granted and downloaded locally (request early if not already in progress — it can take time to arrive).
- `preprocessing/crop_faces.py` produces face crops **and** the frame↔audio-window sync (`audio_manifest.csv`, a fixed-duration window — default 0.35s — centered on each sampled frame). This mechanism is already built; Stage 1 consumes it rather than redoing it.

---

## Tasks

**Splits (Data workstream):**
- Audit FakeAVCeleb: count videos per category, list unique identities, check for corrupt/unreadable files.
- Build train/val/test manifests where **no identity appears in more than one split** — group by identity first, then assign whole identity-groups to splits (e.g. 70/15/15 by identity, not by clip).
- Balance: FakeAVCeleb is ~500 real vs ~19,500 fake. Decide the strategy now (undersample fakes, or balanced sampling at train time). Record real/fake counts per split.
- Write splits as CSV manifests (columns: `clip_id`, `video_path`, `label`, `manipulation_type`, `identity`, `split`).
- Done when: three manifest files exist, verified zero identity overlap, known class balance.

**Feature-store schema (ML workstream):**
- Define and stub the shared feature store: schema `clip_id, stream_name, embedding, split, label, manipulation_type` (embeddings, not scores — see [PROJECT_OVERVIEW.md §3](../PROJECT_OVERVIEW.md)). Write reader/writer helpers (start with Parquet in `feature_store/`).
- Done when: feature-store read/write round-trips a dummy embedding row.

**Dataset / DataLoader (ML workstream):**
- Build the `Dataset`/`DataLoader` reading face-crop sequences + audio + labels from the manifests.
- **Frame sampling:** uniformly sample a fixed **N=16 frames per clip** so every batch has shape `[batch, 16, 3, 224, 224]`.
- **Audio alignment:** reuse the existing frame↔audio-window sync (`audio_manifest.csv`) to pull the corresponding audio window for each of the 16 sampled frames — do not build a new alignment mechanism.
- Sanity-check a batch: tensor shapes, label balance, normalization, and that a few decoded frames/audio windows look right. **Print the shapes** — this is the concrete stage-1 verification artifact.
- Done when: a batch loads with correct shapes (`[B, 16, 3, 224, 224]` for frames, matching audio tensors, `[B]` labels) traceable back to `clip_id`.

**Leakage spot-check + labeling convention (Research workstream):**
- Independently spot-check the split manifests for identity leakage (pick random identities, confirm they live in one split only).
- Write down the visual-only labeling convention for Stage 2: `FakeVideo-RealAudio` and `FakeVideo-FakeAudio` → fake; `RealVideo-RealAudio` → real; `RealVideo-FakeAudio` → **real to a visual stream**, even though the clip overall is a fake. This is the motivation for the cross-modal streams (Stages 4–5); a visual stream missing it is expected, not a bug.
- Done when: leakage check signed off; labeling convention documented.

---

## Done when (stage gate)

- Three identity-disjoint, balanced split manifests exist.
- A DataLoader yields correctly-shaped `(face_crop_sequence, audio, label)` batches, shapes printed and verified.
- Feature-store schema frozen (embeddings, not scores) and round-trip tested.
- No identity leakage across splits, confirmed independently.

## Deliverables

- `data/` split manifests (train/val/test, identity-disjoint).
- `models/` or a shared module: the `Dataset`/`DataLoader` implementation.
- `feature_store/` — schema defined, reader/writer helpers working.

## Risks and notes

- **Identity leakage** is the #1 risk. If a later stage's AUC looks suspiciously high (>0.98 on unseen fakes), suspect leakage before celebrating.
- **Frame count (N=16)** is a real design decision — if GPU memory is tight, this is the first knob to revisit; record if it changes.
- **Dataset access delay** can push this stage. Do the leakage-proof split design early regardless, on whatever sample is available.
