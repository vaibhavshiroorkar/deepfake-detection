# Stage 3: Remaining Visual Streams (EfficientNet, DINOv2)

**Goal:** grow from one visual stream to three, each cloning the Stage 2 template — same face-crop sequences, same temporal model pattern, same labels, same projection to `common_dim`. EfficientNet is a near-mechanical clone; DINOv2 differs in kind (self-supervised features) and needs a slightly different training approach.

**Prerequisite:** Stage 2 complete — frozen stream template, frozen feature-store schema.

---

## EfficientNet

The easiest clone — a second artifact-focused CNN view.

- **ML:** clone the Stage 2 template, swap the backbone to EfficientNet (`timm`), keep the same LSTM/GRU temporal model and temporary classifier head, fine-tune, evaluate standalone clip-level AUC on the same test split.
- **Data:** reuse the exact same crops/manifests/sampler — no new preprocessing.
- **Research:** log results next to Xception; note whether EfficientNet catches different fakes or overlaps heavily (feeds the Stage 7 ablation).
- Done when: EfficientNet clip-level AUC recorded (via its temporary head), embedding written to the feature store.

## DINOv2

Different in kind: DINOv2's features are self-supervised (see [docs/math/dinov2.md](math/dinov2.md)) — no fake/real labels were involved in learning them, so they describe images generally rather than latching onto one generator's fingerprint. This is expected to matter most on unseen fake types (Stage 9), where Xception/EfficientNet — both trained end-to-end on manipulation artifacts — are more likely to overfit.

- **ML:** extract per-frame DINOv2 embeddings (`timm`, `vit_large_patch14_dinov2`), pass the sequence through the same LSTM/GRU temporal model as the other visual streams to get one clip embedding, then project to `common_dim`. **Start with the backbone frozen** (train only the temporal model + temporary head — a lightweight probe on top of frozen features is cheap and often strong for self-supervised backbones); only unfreeze and fine-tune the backbone if the frozen probe underperforms Xception/EfficientNet by a wide margin. This is the opposite default from Xception/EfficientNet's staged fine-tuning, and deliberately so — fine-tuning risks destroying exactly the generalizable features DINOv2 was chosen for.
- **Data:** same crops as the other visual streams. DINOv2 extraction can be slow; add a feature-caching step (cache per `clip_id`) if it becomes a bottleneck, so it isn't recomputed every epoch while the backbone is frozen.
- **Research:** record whether DINOv2 generalizes to fake types the CNNs miss — this is the hypothesis from its writeup, and the Stage 7 ablation is where it actually gets tested at fusion level, but a standalone signal here is worth noting early.
- Done when: DINOv2 clip-level AUC recorded (via its temporary head), embedding written to the feature store.

**Checkpoint after both clones:** all three visual streams are in the feature store as embeddings. This is the first moment fusion (Stage 6) *could* run on visual-only embeddings — a useful early integration test even before the cross-modal streams (Stages 4–5) land.

---

## Done when (stage gate)

- EfficientNet and DINOv2 embeddings written to the feature store alongside Xception's, over the same identity-disjoint splits.
- Each has a recorded standalone clip-level AUC (via its temporary head).

## Deliverables

- `models/streams/efficientnet/`, `models/streams/dinov2/` — cloned from the Stage 2 template (DINOv2 with the frozen-first training variant noted above).
- Feature store populated with all three visual streams' embeddings.
- Results table: standalone AUC per visual stream, with early redundancy/complementarity notes (input to Stage 7).

## Risks and notes

- **Keep the template honest.** If a stream needs the template changed, change the template and re-note it — don't fork per-stream hacks. DINOv2's frozen-first default is a documented, deliberate variant, not a fork.
- **Redundancy risk:** Xception and EfficientNet are both artifact-focused CNNs and may catch the same fakes. DINOv2 is expected to differ from both by design (self-supervised vs. supervised) — the Stage 7 ablation is where this actually gets decided, not here — just record observations.
