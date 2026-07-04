# DINOv2: Architecture Math

**Owner:** Person 3 (ML lead)
**Paper:** Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* <https://arxiv.org/abs/2304.07193> (background: DINO, Caron et al. 2021, <https://arxiv.org/abs/2104.14294>)
**Deadline:** solid first draft Day 3, polished Day 4 (see [phase-0.5-plan.md](../phase-0.5-plan.md))

> Draft here, keep notation consistent with the other two writeups — Person 1 merges all three into one document on Days 4–5.

> **Fallback option — Swin Transformer (full note: [swin.md](swin.md)).** If integrating DINOv2 or writing up its self-supervised math (self-distillation, EMA teacher, multi-crop, centering/sharpening, iBOT, KoLeo) proves too heavy for the deadline, a **Swin Transformer** (`timm.create_model("swin_base_patch4_window7_224")`) is a lower-effort substitute: its math (windowed + shifted-window attention, hierarchical patch merging) is cleaner to explain, and in-distribution performance is comparable.
>
> **The trade-off is real, so treat this as a fallback, not an equal swap.** Swin is *supervised* (ImageNet-pretrained), so it does not fill DINOv2's actual role here — the *self-supervised* stream that generalizes to unseen fakes (§4). Replacing DINOv2 with Swin makes all three visual streams supervised, which (a) likely costs out-of-distribution generalization on the Phase 5 tests (Deepfake-Eval-2024, held-out manipulations) that were DINOv2's whole reason for selection, and (b) raises the redundancy risk the Phase 3 ablation is meant to catch. Prefer keeping DINOv2 primary; use Swin only if forced, and record the generalization hit if so.

> **Shared foundation.** DINOv2 and Swin are both ViT-based. The patch-embedding and self-attention basics are written up once in [swin.md](swin.md) §1–2 — reuse them here rather than re-deriving. What is unique to DINOv2 is the *training* (self-distillation), covered below.

## 1. Self-distillation: student and teacher

<!-- A student network learns to match a teacher network's output distribution
     on different views of the same image — no labels. The teacher is an
     exponential moving average (EMA) of the student. Write the loss
     (cross-entropy between softened output distributions) and the EMA update. -->

## 2. Multi-crop strategy

<!-- Global crops go through the teacher, global + local crops through the
     student; matching local views to global views forces the model to learn
     part-to-whole correspondence. -->

## 3. Avoiding collapse

<!-- Why the trivial solution (constant output) doesn't happen:
     centering + sharpening of the teacher outputs (DINO),
     and what DINOv2 adds (iBOT-style masked patch prediction, KoLeo, etc. —
     summarize, don't derive everything). -->

## 4. Why self-supervised features generalize to unseen fakes

<!-- Supervised deepfake detectors overfit to the artifacts of the fakes they
     trained on. DINOv2 features were learned without any fake/real labels,
     so they describe images generally — a linear probe on top has fewer ways
     to latch onto one generator's fingerprint. One or two paragraphs. -->
