# EfficientNet: Architecture Math

**Owner:** Person 2 (Data lead)
**Paper:** Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. <https://arxiv.org/abs/1905.11946>
**Deadline:** solid first draft Day 3, polished Day 4 (see [phase-0.5-plan.md](../phase-0.5-plan.md))

> Draft here, keep notation consistent with the other two writeups — Person 1 merges all three into one document on Days 4–5.

## 1. The scaling problem

<!-- Three ways to grow a CNN: deeper (more layers), wider (more channels),
     higher resolution (bigger input). Each alone hits diminishing returns. -->

## 2. Compound scaling: the formula

<!-- depth d = alpha^phi, width w = beta^phi, resolution r = gamma^phi,
     with alpha * beta^2 * gamma^2 ≈ 2. One coefficient phi balances all three.
     Explain why beta and gamma are squared (FLOPs scale with width^2 and resolution^2). -->

## 3. Accuracy-per-parameter tradeoff

<!-- Why compound scaling gave better accuracy at the same parameter/FLOP budget
     than scaling any single dimension. The B0-B7 family as one curve. -->

## 4. Why this matters for deepfake detection

<!-- A second artifact-focused view with a different architecture than Xception;
     the ablation later tests whether both earn their place. One short paragraph. -->
