# P-001 amendment A11: stronger frozen backbones on shortcut-neutral data

**Exploratory.** Fixed in [amendment A11](../preregistration/P-001-shortcut-neutral-av-study.md) before any upgraded model existed. No P-001 verdict changes. Finished on 2026-11-17 at 14:57, inside its stop rule (23:59).

Published work found that two frozen foundation backbones transfer far better than P-001's: CLIP ViT-L/14 for faces, XLS-R for speech. That evidence came from raw files, which carry the shortcuts P-001 measured. A11 asks whether the advantage survives on cleaned data.

- **CLIP:** `vit_large_patch14_clip_224.openai`, frozen, a GRU head over 16 frames, trained on the video label.
- **XLS-R:** `facebook/wav2vec2-xls-r-300m`, frozen for all six epochs, the P-001 attentive-pooling head, trained on the audio label.
- Everything else as P-001: the same splits, caches, preprocessing hash, three seeds (17, 29, 43), thresholds on validation, and the same locked targets scored by frozen models.

| Comparison | Set | Clips | New | P-001 | Difference [95%] |
|---|---|---|---|---|---|
| CLIP vs visual | FakeAVCeleb, identity-strict | 495 | 0.864 | 0.989 | -0.125 [-0.158, -0.091] |
| CLIP vs visual | DFDC | 2,858 | 0.704 | 0.697 | +0.008 [-0.030, +0.048] |
| CLIP vs visual | LAV-DF test, file label | 3,815 | 0.536 | 0.377 | +0.159 [+0.149, +0.169] |
| CLIP vs visual | HiFi-AVDF | 3,537 | 0.508 | 0.547 | -0.039 [-0.054, -0.024] |
| XLS-R vs audio | FakeAVCeleb, identity-strict | 495 | 0.976 | 0.984 | -0.008 [-0.020, +0.003] |
| XLS-R vs audio | DFDC | 3,032 | one class | one class | no AUC |
| XLS-R vs audio | LAV-DF test, file label | 3,823 | 0.605 | 0.626 | -0.020 [-0.031, -0.010] |
| XLS-R vs audio | HiFi-AVDF | 3,784 | 0.693 | 0.742 | -0.049 [-0.065, -0.033] |

ROC-AUC on each path's own label, mean of three seeds. The difference is paired by clip, averaged over seeds, with one identity bootstrap (1,000 resamples) shared by the seeds (`scripts/compare_a11.py`, reusing the A8 bootstrap). A reading aid: the deliverables take every number from `compare.json` through `results/numbers.json`.

## Reading

- **On cleaned data, neither backbone transfers better.** XLS-R is no better in-domain and worse on LAV-DF and HiFi-AVDF. CLIP is much worse in-domain, no better on DFDC, and at chance on HiFi-AVDF.
- **CLIP's one gain is not detection.** On LAV-DF it scores 0.536, near chance, where the P-001 visual path is inverted (0.377). So CLIP does not read the pixel difference between LAV-DF's real and fake files that inverts the P-001 path (amendment A13 measures that difference). It does not detect LAV-DF's fakes either.
- **Deviations from the published methods,** fixed in A11: no LayerNorm tuning for CLIP, no weighted sum of XLS-R layers. A published recipe might do better; this one does not.

## Files

- `{clip,xlsr}-seed{S}-{part}-predictions.csv` and `-metrics.json`: per-clip scores and metrics with identity-bootstrap intervals; `-threshold.json`: the validation threshold. XLS-R also has `-branch-report.json` from `ddf evaluate branch`.
- `compare.json`: every paired comparison (per seed, mean, interval, one-sided p).
- `histories/`: the training history of each model.
- `receipts/a11-compare.json`: the receipt of the comparison. The training and scoring runs had no receipts, unlike the P-001 runs; the predictions and histories here are their record.
- XLS-R first failed on a wrong flag and was rerun the same afternoon (handbook mistake 73). Nothing about the design changed.
