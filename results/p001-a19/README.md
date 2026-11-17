# P-001 amendment A19: AVH-Align's method on our protocol

**Exploratory.** Fixed in [amendment A19](../preregistration/P-001-shortcut-neutral-av-study.md) before any A19 model existed.

AVH-Align (Smeu et al., CVPR 2025; `bit-ml/AVH-Align` at `55401de`, CC BY-NC-SA 4.0) retrained with its own head, loss and score, changed only in input width, on exactly the real FakeAVCeleb windows and frozen AV-HuBERT Base features the P-001 sync path used (`scripts/run_a19.py`). **The published system is not evaluated:** its weights need features from a fine-tuned AV-HuBERT Large checkpoint that is not available here.

| Set | Clips | AVH-Align, mean of 3 seeds [95%] | P-001 sync path (same clips) | AVH-Align minus sync [95%] | Reading |
|---|---|---|---|---|---|
| FakeAVCeleb test | 2,993 | 0.778 [0.720, 0.830] | 0.775 to 0.809 | -0.018 [-0.038, +0.001] | not different |
| FakeAVCeleb identity-strict | 495 | 0.765 [0.705, 0.817] | 0.780 to 0.810 | -0.035 [-0.053, -0.016] | worse |
| LAV-DF, file label | 3,693 | 0.604 [0.589, 0.621] | 0.597 to 0.622 | -0.009 [-0.018, -0.001] | worse |
| LAV-DF, window label | 3,693 | 0.655 [0.629, 0.682] | 0.634 to 0.657 | +0.007 [-0.008, +0.023] | not different |
| DFDC | 2,908 | 0.519 [0.488, 0.549] | 0.553 to 0.575 | -0.046 [-0.066, -0.025] | worse |
| HiFi-AVDF | 3,739 | 0.536 [0.521, 0.554] | 0.559 to 0.572 | -0.030 [-0.041, -0.020] | worse |

Intervals: 1,000 bootstrap resamples of identities (seed 17), one draw shared by the three seeds for the difference. Per-seed AUCs are in `a19.json`, per-clip scores in `avh-align-*-predictions.csv`, and training curves in `training.json` (each seed kept epoch 28 of 38).

## Reading, by the rule fixed in A19

- **AVH-Align's method does not beat the P-001 sync path on any set.** It is worse on four and not different on two.
- **Both fall the same way out of domain.** On DFDC and HiFi-AVDF it is near chance (0.52, 0.54), below even the sync path; its in-domain level (0.77 to 0.78) matches the sync path's. A different real-only alignment head does not fix transfer.
- **The comparison is of methods, not systems.** The published model uses fine-tuned AV-HuBERT Large features and was trained on 45,000 real AV-Deepfake1M videos; here both heads share our features and our 697 real training windows.
