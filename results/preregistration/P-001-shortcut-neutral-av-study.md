# P-001: Shortcut-neutral audio-visual detection, attribution and false-alarm control

Registered 2026-11-13, before any model in this study is trained or scored on neutral data.
Results go in `results/<experiment-id>/` in later commits. Changes are new dated amendments
at the end of this file, committed before the run they affect. Nothing above an amendment is edited.

## Questions

- **Q1 (protocol).** How much of the real-or-fake signal in FakeAVCeleb, LAV-DF and DFDC can a model get from file and audio-edge cues alone? Does neutralization remove it?
- **Q2 (detection).** After neutralization, does fusing the visual, audio and AV-alignment paths beat the visual path alone on held-out corpora?
- **Q3 (modality).** Can the model say which stream was manipulated, and does that transfer between FakeAVCeleb and LAV-DF?
- **Q4 (false alarms).** Does a threshold calibrated on real clips from the target domain hold the false-positive rate where a source-calibrated threshold fails?

## Data and splits

- **Training corpora:**
  - FakeAVCeleb v1.2. Neutral copy from `scripts/neutralize_dataset.py`.
  - The LAV-DF train partition. Neutral copy.
- **Split:**
  - 70/15/15 by source identity, seed 17, `build_source_split(group_by="source")`.
  - FakeAVCeleb's identity graph is fully connected, so we also report the identity-strict subset beside every in-domain number.
  - Manifest targets that are file names are ignored (fixed 2026-11-13).
- **Locked targets.** Never used for training, selection or thresholds (`data/guards.py`):
  - DFDC
  - Celeb-DF v2 (visual only)
  - MNW (read once, last)
  - DeepSpeak v2 and AV-Deepfake1M test, if access arrives before the freeze.
- **Seeds:** 17, 29, 43 for every trained model.

## Primary tests

All intervals are 95%, from 1,000 bootstrap resamples of source identity. "Holds" means the interval excludes the stated bound.

| ID | Test | Success | Failure |
|---|---|---|---|
| H1 | Shortcut ceiling (`scripts/shortcut_audit.py`, all cues, grouped 5-fold) on neutral copies, clip label | AUC <= 0.60 on each training corpus | AUC > 0.60: neutralization incomplete; report which cue remains |
| H2 | Fusion minus visual-only, ROC-AUC, paired, mean of 3 seeds | Holds above 0 on at least 2 of {DFDC, LAV-DF test, FakeAVCeleb test-strict} | Otherwise reported as a negative result; no further tuning |
| H3 | Manipulated-modality macro-F1 (audio-fake, video-fake as two binary heads), train FakeAVCeleb test LAV-DF and the reverse | Both directions above the clip-label-only baseline, interval excludes 0 | Report as not transferring |
| H4 | Realized FPR with the real-only conformal threshold, alpha = 0.05, n_cal = 200 real target clips | FPR <= 0.07 on every target | Report the targets where it fails |

Holm correction across H2 to H4. H1 is a precondition: if it fails, H2 to H4 still run and are reported with that caveat.

## Fixed choices

These are fixed now so no test set can choose them:

- **Visual path:** EfficientNet-B0 + GRU, live BatchNorm. The earlier choice of live BatchNorm was made on DFDC. It is kept to avoid a new selection, and the history is disclosed.
- **Audio path:** Wav2Vec2 Base, attentive pooling.
- **Sync path:** frozen AV-HuBERT Base, AlignmentHead trained on real clips only (`avsync/heads.py`).
- **Fusion:** logistic regression on out-of-fold branch logits (primary). The StreamFusion MLP with per-modality heads is secondary.
- **Thresholds:** balanced accuracy on validation (`ddf threshold --split val`), except H4.
- **No ensembles across seeds or architectures** in primary results.

## Reported for every run

ROC-AUC, PR-AUC, balanced accuracy, EER (interpolated), FPR and FNR at the frozen threshold, FPR at 95% TPR, Brier score, ECE, coverage. Per-clip predictions are committed under `results/`.
