# EfficientNet stream — Stage 2 results

First visual stream (the reusable template instance). Backbone
`tf_efficientnet_b0.ns_jft_in1k` → Bi-LSTM (hidden 256) → Linear+LayerNorm to
256-dim → temporary classifier head. Visual-only labels (FakeVideo-* = fake,
RealVideo-* = real). Trained on the 6 GB RTX 3060 Laptop GPU.

## Setup
- Splits: identity-disjoint by `source` (train 1400 / val 300 / test 300), 1:3 real:fake, leakage-verified.
- Balanced training via `WeightedRandomSampler`. Augmentation: random horizontal flip only.
- 6 GB-VRAM knobs: batch 2 × grad-accum 8 (eff. 16), AMP, gradient checkpointing, frame chunking.
- Two-phase schedule: 2 epochs backbone frozen, then fine-tune end-to-end.
- **Stability fixes (required):** gradient clipping (norm 1.0) + BatchNorm kept in
  eval() during fine-tuning + backbone LR 5e-6. Without these, the backbone
  unfreeze spiked the loss (0.44 → 2.28) and collapsed val AUC to 0.21.

## Training curve (val, per epoch)
| epoch | phase    | train_loss | val AUC | val acc | val F1 |
|-------|----------|-----------:|--------:|--------:|-------:|
| 1     | frozen   | 0.61       | 0.886   | 0.827   | 0.883  |
| 2     | frozen   | 0.37       | 0.925   | 0.863   | 0.909  |
| 3     | unfreeze | 0.20       | 0.999   | 0.987   | 0.991  |
| 4     | finetune | 0.12       | 0.998   | 0.980   | 0.986  |
| 5     | finetune | 0.07       | **0.999** (best) | 0.993 | 0.995 |

(Epochs 6–8 were cut short by an external process kill during epoch 6; the best
checkpoint was already saved at epoch 5 and epoch 6 was deep into overfitting —
train loss ~0.004 — so nothing was lost.)

## Held-out test set (never seen in training)
`python -m evaluation.eval_checkpoint --split test`

- **AUC 0.994**, acc 0.963, P 0.995, R 0.955, F1 0.974, LogLoss 0.112
- Confusion: reals 79/80 correct (1 FP); fakes 210/220 caught.

Per-manipulation-type (visual label):
| type                | n   | behavior                    |
|---------------------|-----|-----------------------------|
| FakeVideo-FakeAudio | 125 | recall 0.952 (catches)      |
| FakeVideo-RealAudio | 95  | recall 0.958 (catches)      |
| RealVideo-FakeAudio | 5   | flagged-fake 0.000 — correctly REAL to a visual stream |
| RealVideo-RealAudio | 75  | false-positive rate 0.013   |

The `RealVideo-FakeAudio` row confirms intended behavior: audio-only fakes are
invisible to a visual stream, motivating the cross-modal streams (Stages 4–5).

## Caveat
AUC >0.98 is in the "suspect leakage" band from docs/stage-1-plan.md. Checked:
splits are identity- and file-disjoint, and the per-type behavior is sensible
(no audio leakage). FakeAVCeleb visual detection is known to be easy, so the
number is plausibly legitimate **in-distribution**. Real-world generalization is
NOT established here — that is the Stage 9 job (unseen methods, Deepfake-Eval-2024).

## Artifacts
- Checkpoint: `models/streams/efficientnet/best.pt` (val AUC 0.999)
- Training log: `runs/efficientnet_train.log`
- Reusable template: `models/streams/common/visual_stream.py` — Stage 3 clones it by changing `StreamConfig.backbone_name`.
