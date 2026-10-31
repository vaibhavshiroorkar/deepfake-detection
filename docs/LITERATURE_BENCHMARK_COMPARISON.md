# LITERATURE BENCHMARK COMPARISON

Written 2026-10-30. Research report: how the project's face-video detector (SigLIP base, fully fine-tuned on FaceForensics++ c23) compares with published detectors on the same datasets, and whether a ranking can be defended. No model was run, no threshold or split changed, nothing trained, no dataset or weight downloaded. Our numbers come from stored score files and logged experiments; published numbers were read in the papers' own tables.

Companion files: docs/LITERATURE_BENCHMARK_SUMMARY.md (viva summary), docs/LITERATURE_BENCHMARK_EVIDENCE.md (one row per external number with table and page), docs/LITERATURE_BENCHMARK_REFERENCES.bib (verified entries). This file extends docs/PERFORMANCE_BENCHMARK.md (2026-10-27), which remains valid; where the two differ, this file records the newer reading.

**Citation rule.** CLAUDE.md rule 6 allows only papers in docs/RESEARCH.md (READ or PARTIAL) in the paper and report. Several papers below (FTCN, RealForensics, AltFreezing, TALL, LAA-Net, UCF, DFD-FCG, GenD, DF40) are not yet in RESEARCH.md. They must be added there, with the parts read, before any of them is cited in a deliverable. They may be used in the viva as context now.

## 1. Executive summary

- **Our primary result:** video-level AUROC 0.8218 [0.8020, 0.8393] on the sealed Celeb-DF family test (Celeb-DF v2 test_x plus Celeb-DF++ fakes; 7,650 videos, 744 real and 6,906 fake; E-014, confirmatory test T1).
- **No published paper evaluates this test population.** The Celeb-DF family pooling, the removal of 15 development identities and the use of Celeb-DF v2 reals for Celeb-DF++ fakes are this project's own protocol. Therefore the primary result has exactly one directly comparable (Tier A) competitor: the Xception baseline we re-ran under the same pipeline.
- **Tier A ranking (2 models, our protocol):** first of two by point estimate (0.8218 against 0.8042). The paired difference is +0.0176 [+0.0025, +0.0327] at 95 percent and [-0.0011, +0.0355] at the Bonferroni level of 98.33 percent. The defensible statement is "non-inferior (margin 0.0100), superiority not shown after correction". On two sub-blocks (Celeb-DF++ fakes, DFD) the re-run Xception is higher in point estimate.
- **Against the literature (Tier B, Celeb-DF v2, video level, FF++ training):** 12 published video-level results were verified, from 73.7 (Xception as re-trained by LipForensics) to 96.0 (GenD, CLIP ViT-L/14, WACV 2026). Our closest number is the post hoc 0.8478 [0.7998, 0.8884] on 399 of the 518 official test videos (the other 119 are in our development slice). Placed descriptively, that value sits above LipForensics (82.4) and Xception (73.7) and below every other verified video-level method, including all 2023 to 2026 foundation-model detectors (95.0 to 96.0). This is a placement, not a rank: the test populations, frames per video and training compression differ, and no published value has an interval.
- **Conclusion:** within its own protocol the model is non-inferior to a re-run Xception; against published methods it is not directly rankable, and the closest evidence places it below the current state of the art by roughly 0.10 AUROC on Celeb-DF v2.
- **Accuracy warning:** accuracy is 0.6188 on a test set that is 90.3 percent fake (always answering "fake" scores 0.9027), so accuracy must never be quoted alone.

## 2. Methodology and search record

| Item | Detail |
|---|---|
| Search date | 2026-10-30 (all searches, downloads and Crossref queries) |
| Workstreams | (1) Celeb-DF v2 cross-dataset literature; (2) DFD, Celeb-DF++, FF++ within-dataset, multi-dataset and still-image literature; (3) independent verification of every number against the arXiv PDF and of every DOI against Crossref |
| Queries | 21 WebSearch queries (listed in docs/LITERATURE_BENCHMARK_EVIDENCE.md section 4) plus direct fetches of arXiv abstract, HTML and PDF pages |
| Reading method | arXiv PDFs extracted with pdftotext (layout and raw modes, `-table` for scrambled tables); arXiv HTML where the PDF failed. CVF open-access PDFs returned HTTP 403, so arXiv versions were read; the version is named in the evidence file |
| Verification | 17 claim groups re-read from the PDFs by a separate workstream: all printed values matched; corrections were made to labels, arXiv ids and DOIs (section 9) |
| Inclusion | Published detectors trained on FaceForensics++ and tested on another dataset, or tested within FF++; values from the paper's own runs preferred, quoted values labelled |
| Exclusion | Values read only in search snippets; values whose table could not be opened (marked "not verified") |

Comparability tiers (applied to every row):

- **Tier A, directly comparable:** same test items, same metric and aggregation, same training data and pipeline.
- **Tier B, partially comparable:** same dataset and metric family, but a material difference in test subset, frames per video, training compression or training paradigm.
- **Tier C, context only:** different unit (frame against video), different dataset version, different aggregation (per-method averaged against pooled), different task, or numbers the paper does not define.

## 3. Our verified baselines

Source: docs/EXPERIMENTS.md E-014 result table (generated by scripts/freeze/summarise_final.py), stored scores `data/_final/scores/I-4.csv` (SHA-256 e0c78a8c...f383f) and `BASE.csv` (045230cb...e6a0), thresholds `data/_final/eval/I-4.json` and `BASE.json`. Threshold metrics recomputed on 2026-10-30 with scripts/eval/benchmark_audit_2026-10-27/frozen_rates.py (read-only; reproduces docs/test_evidence/2026-10-28_phase_b/frozen_rates_I4_replay.txt and E-032). AUROC intervals: 95 percent group bootstrap, 2,000 resamples, 3 seeds. Threshold metrics: seed means at each seed's validation threshold (FF++ validation, maximum balanced accuracy), post hoc, no interval.

### 3.1 Final system (I-4 visual = fine-tuned SigLIP base) and the re-run Xception (BASE)

| Test block | n (real / fake) | Ours AUROC | BASE AUROC | Ours accuracy | Ours balanced acc. | BASE balanced acc. | Ours FPR | Ours sensitivity |
|---|---|---|---|---|---|---|---|---|
| Celeb-DF family test_x (primary, T1, T3) | 744 / 6,906 | 0.8218 [0.8020, 0.8393] | 0.8042 [0.7841, 0.8223] | 0.6188 | 0.7399 | 0.6727 | 0.1098 | 0.5896 |
| Celeb-DF v2 test_x | 744 / 4,057 | 0.8999 [0.8759, 0.9210] | 0.8527 [0.8237, 0.8777] | 0.7440 | 0.8037 | 0.6889 | 0.1098 | 0.7172 |
| Celeb-DF++ fakes vs v2 reals | 744 / 2,849 | 0.7105 [0.6877, 0.7331] | 0.7351 [0.7083, 0.7599] | 0.5077 | 0.6490 | 0.6495 | 0.1098 | 0.4079 |
| DFD (all c23 videos) | 363 / 3,068 | 0.8797 [0.8566, 0.9041] | 0.8847 [0.8655, 0.9052] | 0.7457 | 0.7959 | 0.7828 | 0.1405 | 0.7323 |
| FF++ c23 test_in (same dataset) | 140 / 560 | 0.9899 [0.9826, 0.9959] | 0.9895 [0.9823, 0.9952] | 0.9614 | 0.9705 | 0.9676 | 0.0143 | 0.9554 |
| Celeb-DF v2 official list, the 399 videos in test_x (post hoc) | 150 / 249 | 0.8478 [0.7998, 0.8884] | 0.7615 [0.7096, 0.8087] | not computed | | | | |

Primary-block full metrics (ours): precision 0.9804, recall 0.5896, specificity 0.8902, FPR 0.1098, FNR 0.4104, F1 0.7360 (seed mean of per-seed F1). The majority-class rate is 0.9027.

### 3.2 Other evaluations (not face-swap cross-dataset video, kept separate)

| Evaluation | AUROC | Accuracy | Note |
|---|---|---|---|
| FakeAVCeleb sealed identity half (E-015) | 0.9552 [0.9442, 0.9657] | 0.5318 (TP 2,168, FN 2,345, FP 2, TN 498; stored counts, scripts/eval/benchmark_audit_2026-10-27/e015_rates_output.txt) | sensitivity 0.480; codec metadata alone gives 0.9985 (E-021) |
| LAV-DF test_in, 2,000-video subset (E-014, I-5) | 0.3226 | not recorded | below chance |
| Community Forensics-Eval face images, still, E-028 F3 | 0.6020 [0.5836, 0.6202] | 0.5563 (computed exactly from cached scores, 2026-10-30) | 1,524 real, 1,595 fully generated; a different task |
| Hourglass fakes vs FFHQ portraits, E-028 F1 | 0.5141 [0.4648, 0.5642] | not recorded | near chance |

Discrepancies found in Phase 1: none in the numbers. Two earlier readings are corrected: (a) the FakeAVCeleb accuracy is exactly 0.5318 from stored counts (previously given as "about 53 percent"); (b) the figure 0.6786 mentioned earlier is not a project result: it is Face X-ray's frame-level AUROC on Celeb-DF v2 in DeepfakeBench Table 3.

## 4. Published results by setting

All published values are single runs without intervals unless stated. "Own" means the paper's own experiment; "quoted" means copied from another paper.

### 4.1 FF++ within-dataset (train and test FF++)

| Model | Source | Unit | Value | Tier | Reason |
|---|---|---|---|---|---|
| Xception | LipForensics (CVPR 2021), Table 4, HQ (c23) | video AUC | 99.3 (raw 99.8, LQ 92.0) | B | same test split family, video level; their frames and face detector differ |
| LipForensics | same, Table 4 | video AUC | 99.7 | B | as above |
| Face X-ray (re-trained) | same, Table 4 | video AUC | 97.8 | B | as above |
| AltFreezing | AltFreezing (CVPR 2023), Tables 4 and 7 (ablation) | video AUC | 99.7 | B | c23; ablation table |
| SBI | SBI (CVPR 2022), Table 9 | video AUC | 99.64 (140 real, 560 fake) | B | same 700 test videos; training on reals only, compression of evaluation stated as raw in Section 4.3 |
| Xception, EfficientNet-B4, F3Net, SPSL, UCF | DeepfakeBench (NeurIPS 2023), Table 3, FF-c23 | frame AUC | 0.9637, 0.9567, 0.9635, 0.9610, 0.9705 | C | frame level; models selected on the test set (Appendix) |
| Xception | FaceForensics++ (ICCV 2019), Table 1, HQ | frame accuracy | 95.73 | C | accuracy, frame level |
| **Ours / BASE** | E-014 | video AUC | **0.9899 / 0.9895** | | |

Reading: everything is at the ceiling (0.98 to 0.998). No ordering is meaningful.

### 4.2 FF++ to Celeb-DF v2 (the main comparison)

Video-level AUROC, trained on FF++ (verified values, sorted by value):

| # | Model (venue) | Value | Backbone | Training | Test set | Frames | Tier vs our 0.8478 (official-list part) | Key differences |
|---|---|---|---|---|---|---|---|---|
| 1 | GenD, CLIP (WACV 2026), Table 2 | 96.0 (5-seed mean 94.6 plus or minus 0.9, Table S3) | CLIP ViT-L/14, LayerNorm tuning | FF++ c23 | official 518 | 32 | B | 77 percent overlap of test videos; 32 frames; larger encoder |
| 2 | Forensics Adapter (CVPR 2025), Table 2 | 95.7 | CLIP ViT-L/14 frozen + adapter | FF++ c23 | not stated | 32 | B | test list not stated |
| 3 | Effort (ICML 2025), Table 1 | 95.6 (GenD reproduction: 93.2) | CLIP ViT-L/14, orthogonal subspace | FF++ c23 | not stated | 32 | B | reproduction 2.4 points lower |
| 4 | LAA-Net with SBI (CVPR 2024), Table 1 | 95.40 (DFD-FCG re-run: 93.6) | EfficientNet-B4, 384 px | FF++ reals only, pseudo-fakes | not stated | | B | different training paradigm; compression not stated |
| 5 | DFD-FCG (CVPR 2025), Table 1 | 95.0 | CLIP ViT-L/14 + side network | FF++ HQ | not stated | 10-frame clips | B | temporal input |
| 6 | SBI (CVPR 2022), Tables 1 and 9 | 93.18 (AltFreezing c23 re-run with ResNet-50: 85.7) | EfficientNet-B4 | FF++ reals only | 178 real + 340 fake | 32 | B | trained on the raw version according to AltFreezing (not stated in SBI's text); reals only |
| 7 | TALL-Swin (ICCV 2023), Table 3 | 90.79 | Swin-B | FF++ HQ | not stated | 8 clips x 4 | B | thumbnail layout, temporal |
| 8 | AltFreezing (CVPR 2023), Table 1 | 89.5 | 3D ResNet-50 | FF++ c23 | 518 | clips | B | spatio-temporal model |
| 9 | FTCN (ICCV 2021), Table 4 | 86.9 | temporal convolution | FF++ HQ | 518 | 32-frame clips | B | temporal model |
| 10 | RealForensics (CVPR 2022), Table 2 | 86.9 (DFD-FCG re-run: 88.9) | CSN + self-supervised real data | FF++ c23 + LRS3 reals | 518 | clips | B | extra real training data |
| 11 | LipForensics (CVPR 2021), Table 2 | 82.4 | lip-reading pretrained 3D CNN | FF++ (four methods) | 518 | whole video | B | mouth region only |
| 12 | Xception, re-trained by LipForensics, Table 2 | 73.7 | Xception | FF++ | 518 | whole video | B | same architecture as our BASE |

Frame-level results on Celeb-DF v2 (Tier C against our video-level values): DeepfakeBench Table 3, 15 detectors trained on FF++ c23, frame AUROC from 0.6091 (Meso4) to 0.7650 (SPSL): Xception 0.7365, EfficientNet-B4 0.7487, UCF 0.7527, SRM 0.7552, F3Net 0.7352, Face X-ray 0.6786, Capsule 0.7472, FFD 0.7435, CORE 0.7428, RECCE 0.7319, MesoIncep 0.6966, CNN-Aug 0.7027, DSP-FWA 0.6680. Effort 0.901 and Forensics Adapter 0.900 at frame level. Celeb-DF paper Table 4 (public checkpoints, key frames): Xception-c23 65.3.

Values with undefined unit or version (Tier C): UCF 0.824 (ICCV 2023, "CelebDF", unit and version not stated); SPSL 76.88 (CVPR 2021, Table 3; the same paper gives 72.39 in Table 5).

### 4.3 FF++ to Celeb-DF++ and newer manipulation methods

Celeb-DF++ (arXiv 2507.18015, preprint, not peer-reviewed): GFD-eval protocol, train FF++ HQ, test every Celeb-DF++ method, 32 frames, AUROC computed per method and averaged over 22 columns (the original Celeb-DF column plus 21 new methods), Table XIII (video) and Table XII (frame):

| Model | Video avg AUROC (XIII) | Frame avg (XII) | Tier vs our 0.7105 |
|---|---|---|---|
| Effort | 85.0 | 80.4 | C |
| CLIP | 79.1 | 74.9 | C |
| Xception | 79.0 | 73.5 | C |
| UCF | 76.1 | 72.2 | C |
| Forensics Adapter | 75.4 | 71.9 | C |
| SBI | 73.8 | 71.2 | C |
| EfficientNet-B4 | 73.2 | 70.0 | C |
| Average of 24 detectors | 73.7 | 69.4 | C |
| **Ours (pooled, 2,849 fakes vs 744 v2 reals)** | **0.7105** | | |
| **BASE Xception, same pooling** | **0.7351** | | |

Tier C because: per-method averaging against our pooled AUROC; different reals (their 178 Celeb-DF test reals against our 744 v2 test_x reals); 15 identities excluded on our side; 16 against 32 frames. The same-architecture check is informative: their Xception averages 79.0 while our Xception under our pooling gives 0.7351, so the protocols differ by several points for one architecture. The hardest methods in their Table XIII are SimSwap 58.4, SadTalker 58.6 and GHOST 59.0, consistent with our observation that the Celeb-DF++ part is the hardest part of our test.

### 4.4 FF++ to DFD

| Model | Source | Test set | Unit | Value | Tier vs our 0.8797 |
|---|---|---|---|---|---|
| AltFreezing | AltFreezing Table 1 | not stated | video | 98.5 | B |
| SBI | SBI Table 9 | all DFD, 363 real + 3,068 fake (same population as ours) | video, 32 frames | 97.56 | B (closest: same videos, but reals-only training, compression not stated) |
| Effort | Effort Table 1 | not stated | video | 0.965 | B |
| FTCN (run by SBI authors) | SBI Table 1 | all DFD | video | 94.40 | B |
| Forensics Adapter | Forensics Adapter Table 1 | not stated | frame | 0.933 | C |
| Xception, EfficientNet-B4, SPSL, UCF, F3Net | DeepfakeBench Table 3 | DeepfakeBench's own DFD split | frame | 0.8163, 0.8148, 0.8122, 0.8074, 0.7975 | C |

Our 0.8797 (and BASE 0.8847) are below every verified video-level value on DFD; the only same-population row (SBI) uses a different training paradigm.

### 4.5 Multi-dataset generalisation (context)

DeepfakeBench Table 3 cross-domain averages over 8 test sets (frame): Xception 0.7718, UCF 0.7801, SPSL 0.7875. Effort Table 1 average over 7 sets (video): Effort 0.917, SBI 0.821. Forensics Adapter average over 5 sets (frame): 0.896. Our project has no equivalent average (different test sets); Tier C.

### 4.6 Still-image generated-face detection (context only)

DF40 (NeurIPS 2024) Table 3: detectors trained on DF40's face-swap data tested on fully generated faces in the FF++ domain: Xception 0.810, CLIP 0.837; Table 4, Celeb-DF domain: Xception 0.642, CLIP 0.730. Unit (frame or image) and CLIP size not stated in the parts read; "FF" and "CDF" mean DF40's regenerated data. No paper tests FF++-trained detectors on Community Forensics. Our E-028 F3 value (0.6020) agrees in direction (face-swap training transfers poorly to fully generated faces) but is Tier C.

## 5. Comparability assessment

| Check | Our primary result | Typical published Celeb-DF v2 result | Consequence |
|---|---|---|---|
| Dataset version | Celeb-DF v2 plus Celeb-DF++ fakes | Celeb-DF v2 only | primary result has no external counterpart |
| Test items | test_x: v2 minus 15 identities (4,801 videos) | official list, 518 videos | at most 399 shared videos |
| Manipulation methods | v2 method plus 21 Celeb-DF++ methods | v2 method | ours is harder by construction |
| Real/fake population | 744 real (all v2 test_x reals) | 178 real, 340 fake | different class ratio (AUROC is ratio-insensitive, accuracy is not) |
| Evaluation unit | video, mean of up to 16 frames | video, 32 frames or clips; some frame level | frame-level numbers are 4 to 6 points lower for the same model (Celeb-DF++ Tables IV and V; Effort 0.901 vs 0.956) |
| Metric | ROC AUC (Mann-Whitney, ties half) | ROC AUC | same metric family |
| Preprocessing | YuNet, largest face, 1.3x, 224 px | RetinaFace or Dlib, various sizes | not equivalent |
| Training data | FF++ c23, four methods, official split | c23 or HQ mostly; SBI and LAA-Net reals only (likely raw); RealForensics adds LRS3 | several rows differ by design |
| Identity overlap | dev identities excluded and checked by script; FF++ to Celeb-DF identity overlap not verifiable | rarely stated | unknown on both sides |
| Threshold | per-seed validation threshold (FF++) | not used for AUROC | threshold metrics not comparable |
| Model selection | on a development slice of 15 Celeb-DF identities (not the test) | DeepfakeBench: selected on the test set (Appendix p.21) | DeepfakeBench values are optimistic by design |
| Seeds | 3 seeds, group bootstrap intervals | single runs (GenD: 5 seeds, std about 0.9) | published values have no intervals |
| Reproducibility | stored scores, ledger | reproductions differ by up to 2.4 points (Effort 95.6 vs 93.2; LAA-Net 95.40 vs 93.6) | published point values carry several points of hidden spread |

## 6. Ranking

**Rules applied.** A numerical rank is given only within a group that shares test items, unit, metric and pipeline (Tier A). AUROC is never ranked against accuracy or EER. Frame-level and video-level values are never ranked together. Ties: two models whose paired interval contains zero are reported as not distinguished.

### 6.1 Answers to the ranking questions

1. **Published models on Celeb-DF v2 under the most comparable protocol:** the 12 video-level, FF++-trained results of section 4.2 (all Tier B). None uses our test items.
2. **FF++ c23 to Celeb-DF:** of those, FF++ c23 or HQ training is stated for GenD, Forensics Adapter, Effort, DFD-FCG, TALL, AltFreezing, FTCN and RealForensics (plus extra reals); SBI and LAA-Net train on reals only; LipForensics does not restate the compression for its cross-dataset table.
3. **Where 0.8218 sits:** it has no published counterpart (Celeb-DF family). For Celeb-DF v2, our values are 0.8999 on test_x and 0.8478 on the 399 official-list videos in test_x.
4. **Higher under the closest conditions:** GenD (96.0), Forensics Adapter (95.7), Effort (95.6), LAA-Net (95.40), DFD-FCG (95.0), SBI (93.18), TALL (90.79), AltFreezing (89.5), FTCN (86.9), RealForensics (86.9), all above 0.8478.
5. **Lower:** LipForensics (82.4) and the LipForensics re-training of Xception (73.7).
6. **Fairly comparable models:** one (Tier A): our re-run Xception. Twelve partially comparable (Tier B).
7. **Rank among directly comparable models:** first of two on the Celeb-DF family by point estimate; not distinguished after the Bonferroni correction (non-inferior, margin 0.0100). On Celeb-DF v2 test_x first of two (0.8999 against 0.8527, no paired interval for this block alone); on the Celeb-DF++ block and on DFD second of two in point estimate.
8. **Against Xception and EfficientNet:** our re-run Xception is the only exact comparison (above). Published Xception values on Celeb-DF v2 range from 65.3 (frame, public checkpoint) to 73.7 (video, LipForensics) and 0.7365 (frame, DeepfakeBench); our Xception on the 399 official-list videos gives 0.7615 at video level, in the same range, which suggests our pipeline does not inflate a standard baseline. EfficientNet-B4 is published only at frame level (0.7487, DeepfakeBench): Tier C.
9. **Including partially comparable results:** a descriptive placement of 0.8478 among the 12 Tier B values puts it 11th of 13 (above LipForensics and Xception); 0.8999 (test_x, a different subset) would sit 8th of 13, between AltFreezing (89.5) and TALL (90.79). The two placements differ by three positions only because of the test subset, which is itself the reason a rank is not defensible.
10. **What prevents a definitive ranking:** different test items (at most 77 percent overlap with the official list), 16 against 32 frames, different face detectors and crop sizes, different training compression and paradigms, published single runs without intervals, reproductions that differ by up to 2.4 points, DeepfakeBench's test-set model selection, and the fact that our primary set (Celeb-DF family) does not exist in the literature.

### 6.2 Outcome

- Tier A: rank 1 of 2 by point estimate on the primary set; statistically, not distinguished from Xception after correction; non-inferior.
- Tier B: no defensible rank. Descriptively, our best-matched value (0.8478) is below the ten strongest published video-level methods and above two.
- The gap to the 2023 to 2026 foundation-model detectors on Celeb-DF v2 is roughly 0.10 to 0.11 AUROC in point estimate (0.8478 against 0.950 to 0.960); its size cannot be stated precisely because the test sets differ.

## 7. Performance-gap analysis

### 7.1 Demonstrated by our own experiments

| Finding | Evidence |
|---|---|
| Threshold transfer fails: FPR 0.0143 on FF++ but 0.1098 on the Celeb-DF family and 0.1405 on DFD; sensitivity 0.9554 against 0.5896 | E-032, section 3.1 |
| Newer manipulation methods are much harder: 0.8999 on Celeb-DF v2 against 0.7105 on Celeb-DF++ fakes | E-014 |
| The re-run Xception is higher than ours on Celeb-DF++ fakes (0.7351 against 0.7105) and DFD (0.8847 against 0.8797), so our gain over Xception comes from the Celeb-DF v2 part | E-014, D-091 |
| Codec shortcuts inflate AUROC: FakeAVCeleb codec metadata alone 0.9985; our visual AUROC drops from 0.9732 to 0.8772 after a common re-encode (development half) | E-021 |
| Face-swap training does not transfer to fully generated faces (0.6020, near chance on portraits 0.5141) | E-028 |
| Accuracy is dominated by class imbalance (0.6188 against a majority rate of 0.9027) | section 3.1 |

### 7.2 Supported by published research

| Explanation | Source |
|---|---|
| Video-level aggregation raises AUROC by about 4 to 6 points over frame level for the same model | Celeb-DF++ Tables IV and V (23 of 24 detectors); Effort Tables 1 and 7; Forensics Adapter Tables 1 and 2 |
| The same method's published value moves by several points with the pipeline (SBI 93.18 own, 85.7 under c23 with ResNet-50; Xception 65.3 to 81.6) | SBI, AltFreezing Table 1, Celeb-DF Table 4, Celeb-DF++ Table XIII |
| Training compression matters within FF++ (Xception 99.8 raw, 99.3 HQ, 92.0 LQ) | LipForensics Table 4 |
| Larger frozen or lightly adapted vision-language encoders (CLIP ViT-L/14) currently give the highest cross-dataset values | GenD, Effort, Forensics Adapter, DFD-FCG |
| Detectors learn identity information, which hurts cross-dataset generalisation | Dong et al., CVPR 2023 (Implicit Identity Leakage) |
| Training on synthetic pseudo-fakes from real faces (self-blending) generalises well to Celeb-DF and DFD | SBI, LAA-Net |
| Seed spread is about 0.8 to 0.9 AUROC points even for strong models | GenD Table S3 |

### 7.3 Plausible hypotheses not verified here

- Full fine-tuning of SigLIP base on four FF++ methods may overwrite general pretrained features with FF++-specific cues; adapter or LayerNorm-only tuning (Effort, GenD) is designed to avoid this. Not tested in this project.
- SigLIP base (a ViT-B/16 vision tower) is smaller than the CLIP ViT-L/14 used by the leaders; capacity may explain part of the gap. Our M-003 measurement shows ViT-L/14 could not be fine-tuned at batch 32 on the 12 GB GPU.
- 16 frames instead of 32 may cost a little AUROC; no published test-frame ablation was found.
- Our YuNet face crops at 224 px may lose detail compared with RetinaFace crops at 256 to 384 px.
- The 15 excluded development identities may make our test_x slightly harder or easier than the official list; the 399-video slice (0.8478) is lower than the full test_x (0.8999), which suggests the official list is the harder part, but this was not tested.

## 8. Limitations of this comparison

- No Tier A published result exists; the only exact competitor is our own re-run baseline.
- Published values are single runs without intervals (except GenD's seed table); reproductions differ by up to 2.4 points.
- Our closest value on the official list (0.8478) is post hoc, covers 399 of 518 videos and was not registered before the sealed scores were read.
- Celeb-DF++ is a preprint; its averages are per method, not pooled.
- DeepfakeBench selects models on the test set; its values are optimistic and frame level.
- Several papers do not state the test list, the training compression or the evaluation unit (UCF, SPSL, LAA-Net, Effort, Forensics Adapter, TALL); these are marked in the evidence file.
- The literature search was broad but not exhaustive; in particular no SigLIP-based face-forgery detector with an FF++ to Celeb-DF v2 result was found.
- Many of the papers here are not yet logged in docs/RESEARCH.md (citation rule above).

## 9. Corrections made during verification

| Item | Correction |
|---|---|
| TALL arXiv id | 2307.07494 (2308.10233 is an unrelated paper) |
| Implicit Identity Leakage arXiv id | 2210.14457 (2303.14457 is an unrelated paper) |
| "IID" in Effort, Forensics Adapter and Celeb-DF++ tables | refers to Huang et al., CVPR 2023, "Implicit Identity Driven Deepfake Face Swapping Detection", not Dong et al. |
| Celeb-DF++ "Average" column | covers 22 columns (original Celeb-DF plus 21 new methods), not 21 |
| Celeb-DF++ Xception values | 81.6 = Celeb-DF v2 column; 79.1 = Table V Celeb-DF++ column; 79.0 = Table XIII GFD-eval average; this report uses Table XIII |
| DOIs | DeepfakeBench 10.52202/075280-0201 and DF40 10.52202/079017-0925 exist (one workstream had reported none) |
| FTCN and UCF pages | Crossref pagination 15024-15034 and 22355-22366 |
| IID (Dong et al.) Table 3 Xception 49.03 on Celeb-DF | printed as such, not an extraction error; origin not explained in the paper |

## 10. Bibliography

Full verified entries: docs/LITERATURE_BENCHMARK_REFERENCES.bib (21 entries). Peer-reviewed unless marked.

| Key | Paper | Venue, year | DOI or URL |
|---|---|---|---|
| rossler2019faceforensics | Rossler et al., FaceForensics++: Learning to Detect Manipulated Facial Images | ICCV 2019 | 10.1109/ICCV.2019.00009 |
| li2020celebdf | Li et al., Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics | CVPR 2020 | 10.1109/CVPR42600.2020.00327 |
| li2025celebdfpp | Li et al., Celeb-DF++: A Large-scale Challenging Video DeepFake Benchmark for Generalizable Forensics | arXiv 2025, **preprint** | https://arxiv.org/abs/2507.18015 (no DOI) |
| yan2023deepfakebench | Yan et al., DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection | NeurIPS 2023 D&B | 10.52202/075280-0201 |
| yan2024df40 | Yan et al., DF40: Toward Next-Generation Deepfake Detection | NeurIPS 2024 D&B | 10.52202/079017-0925 |
| haliassos2021lips | Haliassos et al., Lips Don't Lie | CVPR 2021 | 10.1109/CVPR46437.2021.00500 |
| zheng2021ftcn | Zheng et al., Exploring Temporal Coherence for More General Video Face Forgery Detection | ICCV 2021 | 10.1109/ICCV48922.2021.01477 |
| haliassos2022realforensics | Haliassos et al., Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection | CVPR 2022 | 10.1109/CVPR52688.2022.01453 |
| shiohara2022sbi | Shiohara and Yamasaki, Detecting Deepfakes with Self-Blended Images | CVPR 2022 | 10.1109/CVPR52688.2022.01816 |
| wang2023altfreezing | Wang et al., AltFreezing for More General Video Face Forgery Detection | CVPR 2023 | 10.1109/CVPR52729.2023.00402 |
| xu2023tall | Xu et al., TALL: Thumbnail Layout for Deepfake Video Detection | ICCV 2023 | 10.1109/ICCV51070.2023.02071 |
| yan2023ucf | Yan et al., UCF: Uncovering Common Features for Generalizable Deepfake Detection | ICCV 2023 | 10.1109/ICCV51070.2023.02048 |
| dong2023implicit | Dong et al., Implicit Identity Leakage | CVPR 2023 | 10.1109/CVPR52729.2023.00389 |
| nguyen2024laanet | Nguyen et al., LAA-Net | CVPR 2024 | 10.1109/CVPR52733.2024.01647 |
| yan2025effort | Yan et al., Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection (Effort) | ICML 2025, PMLR 267 | https://proceedings.mlr.press/v267/yan25b.html (no DOI) |
| cui2025forensicsadapter | Cui et al., Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection | CVPR 2025 | 10.1109/CVPR52734.2025.01789 |
| han2025dfdfcg | Han et al., Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model | CVPR 2025 | 10.1109/CVPR52734.2025.02141 |
| yermakov2026gend | Yermakov et al., Deepfake Detection that Generalizes Across Benchmarks | WACV 2026 | 10.1109/WACV61042.2026.00082 |
| chollet2017xception | Chollet, Xception | CVPR 2017 | 10.1109/CVPR.2017.195 |
| tan2019efficientnet | Tan and Le, EfficientNet | ICML 2019, PMLR 97 | https://proceedings.mlr.press/v97/tan19a.html (no DOI) |
| zhai2023siglip | Zhai et al., Sigmoid Loss for Language Image Pre-Training (SigLIP) | ICCV 2023 | 10.1109/ICCV51070.2023.01100 |
