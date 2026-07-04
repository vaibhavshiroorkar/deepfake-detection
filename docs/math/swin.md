# Swin Transformer: Architecture Math

**Owner:** Person 3 (ML lead) — same owner as DINOv2, since Swin is its fallback and its transformer on-ramp.
**Paper:** Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. <https://arxiv.org/abs/2103.14030> (foundation: ViT — Dosovitskiy et al. 2020, <https://arxiv.org/abs/2010.11929>)
**timm id:** `swin_base_patch4_window7_224`
**Deadline:** align with the other architecture writeups (solid first draft Day 3, polished Day 4 — see [phase-0.5-plan.md](../phase-0.5-plan.md)).

> **Why this note exists.** Swin is the documented **fallback** for the DINOv2 stream (see [dinov2.md](dinov2.md)): a lower-effort substitute if DINOv2's self-supervised math or integration is too heavy for the deadline — with the caveat that Swin is *supervised* and won't match DINOv2's generalization to unseen fakes. It also doubles as a **learning on-ramp** to DINOv2 (see the map below), so the reading isn't wasted either way.

---

## How this fits the ViT family (read first)

The four transformer-family models aren't a single chain — they're one trunk with two branches:

```
              ViT  (patch embedding + self-attention)   ← the shared foundation
             /   \
          Swin    DINO → DINOv2
      (architecture   (training method:
       branch:         self-supervised
       windowed +      self-distillation
       shifted attn)   on a ViT backbone)
```

- **ViT** is the foundation everything needs.
- **Swin** changes the *architecture* (how attention is computed).
- **DINO → DINOv2** change the *training* (how the network learns, without labels) — on a plain ViT, **not** on Swin.

So Swin is not a prerequisite for DINO. But doing Swin right after ViT is a sensible order: you climb the transformer basics (patch embedding, self-attention) on a purely-architectural model, and then DINOv2 later is *only* the self-distillation story on top of foundations you already know — not the whole chain again.

**Why Swin's math is easier to present than DINOv2's:** it isn't that Swin's internals are simpler — it's that Swin has no exotic *training* story to unpack. DINOv2 forces you to explain the architecture **and** the self-supervised self-distillation (student matching an EMA teacher, no labels). Swin is used supervised, so you mostly just explain the architecture, and the parts you do cover are visual and intuitive.

---

## 1. Patch embedding (shared with ViT)

<!-- How an image is chopped into small non-overlapping patches (Swin starts at 4x4)
     and each patch is flattened + linearly projected into a vector (token).
     Give the dimensions: HxWx3 image -> (H/4)x(W/4) tokens of dimension C. -->

## 2. Self-attention (shared with ViT) — the one genuinely mathematical piece

<!-- The core transformer operation: each token forms Query/Key/Value vectors,
     attends to other tokens via softmax(QK^T / sqrt(d)) V. Write this formula —
     it's the piece you need for ANY ViT-based model (Swin or DINOv2), so it's
     never wasted. Multi-head attention = several of these in parallel. -->

## 3. Windowed attention — Swin's actual contribution

<!-- Full self-attention is O(N^2) in the number of tokens — expensive for images.
     Swin partitions tokens into small fixed windows (e.g. 7x7) and does attention
     ONLY within each window. Intuition: "attention, but locally, to save computation."
     State the complexity win: global O(N^2) -> windowed O(N * window_size), i.e.
     linear in image size instead of quadratic. -->

## 4. Shifted windows — the clever bit (the paper's "aha")

<!-- Pure windowed attention never lets tokens in different windows talk.
     Between consecutive layers Swin SHIFTS the window boundaries by half a window,
     so tokens that were split apart now share a window — information crosses
     boundaries. Alternating regular / shifted windows layer by layer gives global
     receptive field without global-attention cost. Easy to show with a
     window-grid-sliding description. -->

## 5. Hierarchical structure

<!-- Going deeper, Swin MERGES neighbouring patches (patch merging) to halve the
     resolution and grow the channel dim — building fine -> coarse, like a CNN's
     stages. This is what makes it "hierarchical" and gives multi-scale features. -->

## 6. Why this matters for deepfake detection (and the DINOv2 trade-off)

<!-- A strong supervised transformer backbone; multi-scale features can help catch
     both fine artifacts and coarse structure. BUT: being supervised, it tends to
     overfit to the training set's fake artifacts more than a self-supervised model,
     so as the third visual stream it risks REDUNDANCY with Xception/EfficientNet
     and a likely drop on the out-of-distribution generalization tests (Phase 5)
     that were DINOv2's whole reason for selection. Keep DINOv2 primary; use Swin
     only as the fallback, and record the generalization hit if you do. -->
