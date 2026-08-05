# DINOv3: Architecture Math

**Owner:** ML workstream
**Paper:** Siméoni, O. et al. (2025). *DINOv3.* <https://arxiv.org/abs/2508.10104> (background: DINOv2, Oquab et al. 2023, <https://arxiv.org/abs/2304.07193>; DINO, Caron et al. 2021, <https://arxiv.org/abs/2104.14294>; ViT, Dosovitskiy et al. 2020, <https://arxiv.org/abs/2010.11929>; registers, Darcet et al. 2023, <https://arxiv.org/abs/2309.16588>)
**Status:** built now, the third visual stream (Stage 3).

DINOv3 is the third visual backbone, chosen over a supervised alternative specifically because it is *self-supervised* — see §5 for why that matters for generalizing to unseen fakes. It is ViT-based: patch embedding splits the image into fixed-size patches, each linearly projected to a token, then processed by standard self-attention layers. What's unique to DINOv3 is the *training*: self-distillation as in DINO and DINOv2, plus Gram anchoring (§3), which keeps the per-patch features from degrading over a long schedule.

**The variant this project uses:** `vit_small_patch16_dinov3.lvd1689m`, ~22M params, 384-dim output. It is distilled from the 7B model rather than trained from scratch, so it inherits that model's representation at a size that fits the compute budget in [PROJECT_OVERVIEW.md §8](../PROJECT_OVERVIEW.md#8-compute-and-environment-assumptions). Patch 16 against this pipeline's 224-pixel crop gives a 14x14 = 196-patch grid, and the token matrix is 201 rows: 196 patches behind 1 CLS token and 4 register tokens (§4).

## 1. Self-distillation: student and teacher

<!-- A student network learns to match a teacher network's output distribution
     on different views of the same image — no labels. The teacher is an
     exponential moving average (EMA) of the student. Write the loss
     (cross-entropy between softened output distributions) and the EMA update.
     Unchanged from DINO and DINOv2: state it once and move on. -->

## 2. Multi-crop strategy

<!-- Global crops go through the teacher, global + local crops through the
     student; matching local views to global views forces the model to learn
     part-to-whole correspondence. -->

## 3. Avoiding collapse, and Gram anchoring

<!-- Two parts.

     (a) The trivial solution (constant output) doesn't happen because the
     teacher's outputs are centered (a running mean, so no one dimension
     dominates) and sharpened (low temperature, so the distribution doesn't
     flatten). That is DINO's mechanism and it carries forward.

     (b) DINOv3's own contribution. Over a long schedule the *global* summary
     keeps improving while the *dense* per-patch features degrade: patches
     drift toward each other and the patch-to-patch similarity structure
     blurs. Gram anchoring adds a loss on the Gram matrix of the patch tokens
     (the matrix of patch-to-patch dot products) against an earlier teacher
     held as reference, constraining the *relative* geometry of the patches
     without pinning their absolute values.

     Worth spelling out rather than filing as trivia: the dashboard's stage
     viewer (models/streams/common/introspect.py) reads per-patch tokens, and
     cls_similarity_map / patch_token_map are exactly the dense structure Gram
     anchoring exists to protect. -->

## 4. Register tokens

<!-- The prefix is 5 rows, not 1: CLS plus 4 registers. Registers are extra
     learned tokens with no patch attached, added so the model has somewhere to
     keep global information; without them a ViT commandeers a few high-norm
     patch tokens for the same job, which corrupts those patches' local meaning.

     Practical consequence for this repo: any code slicing patches off the token
     matrix has to skip the prefix, and the count is not 1. introspect.token_grid
     infers it (the smallest prefix that leaves a square: 201 = 5 + 14^2), so
     nothing hardcodes 5 — but record why that inference is there. -->

## 5. Why self-supervised features generalize to unseen fakes

<!-- Supervised deepfake detectors overfit to the artifacts of the fakes they
     trained on. DINOv3's features were learned without any fake/real labels,
     so they describe images generally — a linear probe on top has fewer ways
     to latch onto one generator's fingerprint. One or two paragraphs.

     This is a hypothesis, not a result. Stage 7's ablation and Stage 9's
     in-the-wild test are where it gets decided. -->
