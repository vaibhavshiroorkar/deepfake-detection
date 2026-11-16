# P-002 (draft): multi-corpus, temporally localized audio-visual deepfake detection

**Status: draft, not frozen.** Written on 2026-11-16, during the P-001 submission. It becomes binding only when a commit renames it to `P-002.md` and states "frozen". That must happen before any P-002 model is trained, and before any P-002 result is read. Changes before freezing need no amendment. Changes after freezing follow the P-001 rule: a dated amendment, committed before the result it could affect.

Design and reasons: [P-002 design](../../docs/research/P-002-design.md). Backbones: [P-002 backbones](../../docs/research/P-002-backbones.md).

## Questions

1. **Does training on several corpora improve transfer to a corpus never seen?** Leave-one-corpus-out, against P-001's single-corpus models.
2. **Does window-level, temporal scoring catch short edits that a clip-level score misses?** Most AV-Deepfake1M++ fake segments last about 0.33 s.
3. **Do stronger frozen backbones transfer better once the shortcuts are gone?** The candidates are CLIP ViT-L/14 with LayerNorm-only tuning for faces and XLS-R for audio. Their published scores come from raw files.
4. **Does a path that does not rely on lip mismatch help on generators that make sound and picture together,** such as Sora 2 and Veo 3.1?
5. **Can a model say which stream was faked, across corpora?** This is H3, carried over from P-001.
6. **Does real-only calibration still control false alarms** when real clips mix conditions (Mondrian, one threshold per condition)?

## Data

- **Training pool**, used only when each corpus has passed the ingestion checks:
  - FakeAVCeleb, LAV-DF and the full DFDC;
  - AV-Deepfake1M or 1M++, as a stratified subset sized to the disk (about 519 GB free, against 1.46 TB for the full set);
  - KoDF and DeepSpeak v2, if granted.
- **Real-only data** for the one-class parts: VoxCeleb2 and LRS3, with speaker overlap against AV-HuBERT's pretraining measured and reported.
- **Locked external sets:**
  - HiFi-AVDF;
  - one in-the-wild benchmark, such as Deepfake-Eval-2024 or MNW;
  - phone clips recorded with consent, used only to measure the false-alarm rate on real video.

  These never train, select or calibrate anything, except a real-only calibration split defined in advance.
- **Splits:** within each corpus, grouped by identity or by source recording. Across corpora, leave-one-corpus-out.

## Ingestion checks (must pass before any model reads a corpus)

Implemented in `src/deepfake_detection/data/ingestion_checks.py` and run by `scripts/check_ingestion.py`.

| Check | Bar |
|---|---|
| Face coverage: share of windows with a face in at least 80% of frames | at least 0.90 |
| Full span: share of clips whose video and audio both last the fixed span | 1.00 |
| Label in window: share of fake clips with at least one frame of edit in a scored window | 1.00 with temporal scoring |
| Shortcut ceiling on the neutral copies | at most 0.60 |

If a corpus fails, it is fixed or left out before training. Either way, the failure and the decision are recorded.

## Hypotheses (bars to be fixed at freezing, after the baselines are reproduced)

| ID | Test | Draft bar |
|---|---|---|
| Q1 | Leave-one-corpus-out AUC of the multi-corpus model minus the single-corpus P-001 model, per held-out corpus, paired identity bootstrap | Interval above 0 on at least half the held-out corpora |
| Q2 | Window-level temporal AUC (AV-Deepfake1M++ protocol) of temporal scoring against the mean-of-windows clip score | Interval above 0 |
| Q3 | The full system against the strongest reproduced baseline (AVH-Align, HAVIC, a CLIP linear probe) on the same shortcut-free splits | Mean gain at least 0.03 AUC, interval above 0, Holm across held-out corpora |
| Q4 | AUC on the joint-generator subset of HiFi-AVDF (Sora 2, Veo 3.1), with the fourth path against without it | Interval above 0 |
| Q5 | Modality-attribution macro-F1 across corpora (H3) | Above the clip-label baseline in both directions |
| Q6 | Realized false-positive rate with Mondrian real-only calibration on mixed-condition real clips | At most alpha + 0.02 in every condition |

## Fixed choices (draft)

- **Visual:** CLIP ViT-L/14 with LayerNorm-only tuning, on paired real and fake clips where a corpus provides pairs.
- **Audio:** frozen XLS-R with a weighted sum of layers and a light AASIST-style head.
- **Lip-sync:** AV-HuBERT Large, trained on real clips only, at the window level.
- **Fourth path:** an audio-visual model fine-tuned on segment labels.
- **Fusion:** late fusion trained out-of-fold on corpus-held-out predictions, with masks for a missing modality.
- **Scoring:** window-level scores; a clip's score is its strongest window (the top-k mean, with k fixed at freezing).
- **Seeds:** three. 95% intervals from 1,000 bootstrap resamples of identities. Thresholds on validation, and real-only calibration for deployment.
- **Training-time randomization** of codec, bitrate, resolution, loudness and window offset. The neutralizer stays for evaluation copies.

## Open before freezing

- The training subset of AV-Deepfake1M++ and the disk plan.
- The licences marked "assumed" in the backbones record.
- The bars of Q1 to Q6, set after the baselines are reproduced on the P-002 splits.
- Compute: the stages that need a rented GPU, with a budget approved first.
