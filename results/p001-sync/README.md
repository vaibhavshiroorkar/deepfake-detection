# P-001 sync path on neutral FakeAVCeleb

Frozen AV-HuBERT Base features. An AlignmentHead is trained and epoch-selected on real clips only (`scripts/train_sync_path.py`), at seeds 17, 29 and 43. Two 2-second windows per 3-second neutral clip, at 0 and 1 s. A clip's score is its mean window misfit.

| Set | Clips | Seed 17 | Seed 29 | Seed 43 |
|---|---|---|---|---|
| validation | 3,104 | 0.791 | 0.830 | 0.814 |
| test | 2,993 | 0.775 [0.713, 0.832] | 0.809 [0.751, 0.861] | 0.806 [0.748, 0.859] |
| identity-strict test | 495 | 0.781 [0.721, 0.835] | 0.809 [0.754, 0.859] | 0.810 [0.754, 0.862] |

ROC-AUC with 95% identity-bootstrap intervals (1,000 resamples). Coverage is 1.000 everywhere.

- **Per method (test, seed 29):** voice cloning (rtvc) 0.942; Wav2Lip variants 0.84 to 0.86; face swap alone (faceswap, fsgan) 0.63 to 0.68. A face swap keeps the original lips and voice, so there is little misalignment to find.
- **False alarms:** at the threshold chosen on validation (balanced accuracy), the false-positive rate on real test clips is 0.27 to 0.39.
- **Training size:** 697 real training windows and 150 real validation windows, so the head trains on about 350 real speakers.
- **Shortcut ceiling** on the same neutral copies: 0.632 (clip label), `results/shortcut-audit/fakeavceleb-neutral`.

Thresholds were chosen on validation (`*-threshold.json`, with the clip ids they used). Each evaluation has a content-bound receipt in `receipts/`. Feature extraction and training ran before receipts were added, so they have none.
