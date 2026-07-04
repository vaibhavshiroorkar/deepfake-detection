# DINOv2: Architecture Math

**Owner:** Person 3 (ML lead)
**Paper:** Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* <https://arxiv.org/abs/2304.07193> (background: DINO, Caron et al. 2021, <https://arxiv.org/abs/2104.14294>)
**Deadline:** solid first draft Day 3, polished Day 4 (see [phase-0.5-plan.md](../phase-0.5-plan.md))

> Draft here, keep notation consistent with the other two writeups — Person 1 merges all three into one document on Days 4–5.

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
