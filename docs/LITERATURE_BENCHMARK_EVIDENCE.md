# LITERATURE BENCHMARK EVIDENCE

Written 2026-10-30. Audit trail for docs/LITERATURE_BENCHMARK_COMPARISON.md: every external number used there, with its source, table, metric, protocol and comparability tier. Each value was read in the paper itself (arXiv PDF extracted with pdftotext, or arXiv HTML where stated) by one workstream and re-read from the PDF by a second, independent workstream on 2026-10-30. "VERIFIED" means both readings agree with the printed value. Tiers: A directly comparable, B partially comparable, C context only (definitions in the comparison report, section 2).

Our comparison values (for orientation): Celeb-DF family 0.8218 (E-014); Celeb-DF v2 test_x 0.8999; official-list part of test_x 0.8478 (post hoc); DFD 0.8797; Celeb-DF++ fakes vs v2 reals 0.7105; FF++ c23 0.9899; re-run Xception 0.8042, 0.8527, 0.7615, 0.8847, 0.7351, 0.9895 on the same blocks.

## 1. External metrics

| # | Paper (venue) | Source (DOI or URL; version read) | Location | Metric and unit | Exact value | Dataset and protocol | Tier | Qualification | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeepfakeBench (NeurIPS 2023 D&B) | 10.52202/075280-0201; arXiv 2307.01426v2 | Table 3, p.6 | frame AUC, 32 frames | Celeb-DF v2: Meso4 0.6091, MesoIncep 0.6966, CNN-Aug 0.7027, Xception 0.7365, EfficientB4 0.7487, Capsule 0.7472, DSP-FWA 0.6680, Face X-ray 0.6786, FFD 0.7435, CORE 0.7428, RECCE 0.7319, UCF 0.7527, F3Net 0.7352, SPSL 0.7650, SRM 0.7552 | train FF++ c23; Celeb-DF v2 test from the authors' list file; Dlib crops | C | frame level; appendix: validation set equals test set and the model with the highest test performance is kept | VERIFIED (column mapping checked against the printed averages for all 15 rows) |
| 2 | DeepfakeBench | as 1 | Table 3 | frame AUC | FF-c23: Xception 0.9637, EfficientB4 0.9567, Face X-ray 0.9592, FFD 0.9624, CORE 0.9638, RECCE 0.9621, UCF 0.9705, F3Net 0.9635, SPSL 0.9610, SRM 0.9576; Xception within-domain average 0.9450, FF-c40 0.8261 | within FF++ | C | as 1 | VERIFIED |
| 3 | DeepfakeBench | as 1 | Table 3 | frame AUC | DFD: Xception 0.8163, EfficientB4 0.8148, SPSL 0.8122, SRM 0.8120, RECCE 0.8119, UCF 0.8074, FFD 0.8024, CORE 0.8018, F3Net 0.7975, Face X-ray 0.7655 | DFD duplicated and split by DeepfakeBench itself (proportions not given; dataset table lists 363 real, 3,000 fake) | C | own DFD split; frame level | VERIFIED |
| 4 | DeepfakeBench | as 1 | Table 3 | frame AUC, cross-domain average of 8 sets | Xception 0.7718, UCF 0.7801, SPSL 0.7875 | 8 test sets | C | different set of datasets from ours | VERIFIED |
| 5 | LipForensics (CVPR 2021) | 10.1109/CVPR46437.2021.00500; arXiv 2012.07657 | Table 2; Appendix A.2 | video AUC (%) | LipForensics 82.4; Xception 73.7; CNN-aug 75.6; Face X-ray 79.5; Multi-task 75.7; DSP-FWA 69.5 (public model); Two-branch 76.7 (cited) | train FF++ four methods; Celeb-DF v2 test set of 518 videos; mean over the whole video; RetinaFace largest face | B | compression of the cross-dataset training not restated; no seeds | VERIFIED |
| 6 | LipForensics | as 5 | Table 4 | video AUC (%) within FF++ | Xception raw/HQ/LQ 99.8/99.3/92.0; LipForensics 99.9/99.7/98.1; Face X-ray HQ 97.8 | trained and tested at each quality | B | | VERIFIED (row order confirmed from raw text) |
| 7 | FTCN (ICCV 2021) | 10.1109/ICCV48922.2021.01477; arXiv 2108.06693 | Table 4 | video AUC (%) | 86.9 on Celeb-DF v2 | train FF++ HQ; 518 test videos | B | temporal model; other rows copied from LipForensics | VERIFIED |
| 8 | RealForensics (CVPR 2022) | 10.1109/CVPR52688.2022.01453; arXiv 2201.07131 | Table 2 | video AUC (%) | 86.9 | train FF++ c23 plus LRS3 real faces (self-supervised stage); 518 test videos | B | extra real training data | VERIFIED |
| 9 | SBI (CVPR 2022) | 10.1109/CVPR52688.2022.01816; arXiv 2204.08376 | Table 9 (Appendix B); Table 1 | video AUC (%), 32 frames, AP | Celeb-DF v2 93.18 (AP 96.35), 178 real + 340 fake; DFD 97.56 (AP 99.70), 363 real + 3,068 fake; FF++ 99.64 (AP 99.92), 140 real + 560 fake | train on real FF++ only with self-blended pseudo-fakes | B | training compression not stated (AltFreezing says raw); DFD row uses the same videos as ours | VERIFIED |
| 10 | SBI | as 9 | Table 1 | video AUC (%) | FTCN on DFD 94.40 (SBI authors' run with official code) | all DFD | B | | VERIFIED |
| 11 | AltFreezing (CVPR 2023) | 10.1109/CVPR52729.2023.00402; arXiv 2307.08317 | Table 1 | video AUC (%) | AltFreezing Celeb-DF v2 89.5, DFD 98.5; SBI-R50 reproduced on c23 85.7 (DFD 94.0) | train FF++ c23; 518 Celeb-DF test videos | B | DFD test set not stated | VERIFIED |
| 12 | AltFreezing | as 11 | Tables 4 and 7 | video AUC (%) | FF++ c23 99.7 | within FF++ | B | ablation tables | read once (workstream 2), not re-read |
| 13 | TALL (ICCV 2023) | 10.1109/ICCV51070.2023.02071; arXiv 2307.07494 | Table 3 | video AUC (%) | TALL-Swin 90.79 | train FF++ HQ; 8 clips x 4 frames | B | test-list size not stated | VERIFIED |
| 14 | UCF (ICCV 2023) | 10.1109/ICCV51070.2023.02048; arXiv 2304.13949 | Tables 1 and 3 | AUC, unit not stated | "CelebDF" 0.824 (own Xception 0.672); DFD 0.945 (own Xception 0.727) | train FF++ HQ | C | Celeb-DF version and evaluation unit not stated | VERIFIED |
| 15 | SPSL (CVPR 2021) | 10.1109/CVPR46437.2021.00083 | Tables 3 and 5 | AUC (%), unit not stated | 76.88 (Table 3) and 72.39 (Table 5) on Celeb-DF | FF++, compression not stated in Table 3 | C | internally inconsistent; read by workstream 1 only | read once, not re-read |
| 16 | Implicit Identity Leakage (CVPR 2023) | 10.1109/CVPR52729.2023.00389; arXiv 2210.14457 | Table 3; Table 1 and Section 3.2 | AUC (%) | ID-unaware detector (EfficientNet-B4) Celeb-DF 93.88 (0.70); Xception as printed 49.03 | Celeb-DF version and unit not stated | C | used as evidence that detectors learn identity; 49.03 is printed, origin unexplained | VERIFIED |
| 17 | LAA-Net (CVPR 2024) | 10.1109/CVPR52733.2024.01647; arXiv 2401.13856v2 | Table 1 | video AUC (%), AP | with SBI: 95.40 (AP 97.64); with BI: 86.28 (91.93) | train on real FF++ only with pseudo-fakes; 384 px EfficientNet-B4 | B | compression not stated; test list not stated | VERIFIED |
| 18 | Effort (ICML 2025) | https://proceedings.mlr.press/v267/yan25b.html (no DOI); arXiv 2411.15633v4 | Table 1 (Protocol-1); Table 7 | video AUC (mean of frame probabilities, 32 test frames); frame AUC | Celeb-DF v2 0.956 (frame 0.901); DFD 0.965; SBI row 0.886 and 0.827 (cited); average of 7 sets 0.917 | train FF++ c23 (DeepfakeBench code) | B | rows without a mark are cited from other papers | VERIFIED |
| 19 | Forensics Adapter (CVPR 2025) | 10.1109/CVPR52734.2025.01789; arXiv 2411.19715v2 | Table 2 (video, cited from originals); Table 1 (frame) | AUC | Celeb-DF v2 video 0.957; frame 0.900; DFD frame 0.933; frame average of 5 sets 0.896 | train FF++ c23; 32 frames | B (video) / C (frame) | test list not stated; arXiv v4 is the extended TPAMI 2026 version (10.1109/TPAMI.2026.3706492) | VERIFIED |
| 20 | DFD-FCG (CVPR 2025) | 10.1109/CVPR52734.2025.02141; arXiv 2404.05583v3 | Table 1 | video AUC (%) | 95.0; re-evaluated RealForensics 88.9, LAA-Net 93.6 | train FF++ HQ; 10-frame clips | B | | VERIFIED |
| 21 | GenD (WACV 2026) | 10.1109/WACV61042.2026.00082; arXiv 2508.06248v4 | Table 2; Table 4; Table S3; Table 5 | video AUROC (%) | CLIP 96.0 (Table 2); 5-seed mean 94.6 plus or minus 0.9 (Tables 4 and S3; PE 95.0 plus or minus 0.8, DINOv3 92.5 plus or minus 0.9); Effort reported 95.6, reproduced 93.2 (Table 5) | train FF++ c23; 518 videos (178 real, 340 fake); 32 frames | B | Table 2 single value higher than the seed mean; paper does not explain | VERIFIED |
| 22 | Celeb-DF++ (arXiv 2025, preprint) | https://arxiv.org/abs/2507.18015 (v1, no DOI) | Tables XII (frame) and XIII (video), GFD-eval | AUC (%) averaged over 22 columns (original Celeb-DF plus 21 methods) | video/frame: Effort 85.0/80.4; CLIP 79.1/74.9; Xception 79.0/73.5; UCF 76.1/72.2; Forensics Adapter 75.4/71.9; SBI 73.8/71.2; EfficientNet-B4 73.2/70.0; average row 73.7/69.4 | train FF++ HQ; 32 random frames; 178 test reals; 200 or 300 fakes per method | C | per-method average, not pooled; preprint | VERIFIED |
| 23 | Celeb-DF++ | as 22 | Tables IV, V, XIII | AUC (%) | Xception: 81.6 on Celeb-DF v2 (video); 79.1 Table V Celeb-DF++ column; 79.0 Table XIII average | as 22 | C | three different numbers for one model, explained by table and column | VERIFIED |
| 24 | Celeb-DF++ | as 22 | Tables IV and V | AUC (%) | video above frame for 23 of 24 detectors; Xception 73.7 to 79.1; average 69.6 to 73.8 | as 22 | C | evidence for the frame-to-video gap | read by workstream 2; values in rows 22 and 23 re-read |
| 25 | Celeb-DF++ | as 22 | Table XIII | video AUC (%) | hardest methods: SimSwap 58.4, SadTalker 58.6, GHOST 59.0 | as 22 | C | | read by workstream 2 |
| 26 | Celeb-DF (CVPR 2020) | 10.1109/CVPR42600.2020.00327; arXiv 1909.12962v4 | Table 4 | frame AUC (%), key frames | Xception-c23 65.3, Xception-c40 65.5, Xception-raw 48.2, DSP-FWA 64.6 | public pretrained models, not retrained | C | which videos were evaluated not stated | read by workstream 1 |
| 27 | FaceForensics++ (ICCV 2019) | 10.1109/ICCV.2019.00009; arXiv 1901.08971v3 | Table 1 | frame accuracy (%) | XceptionNet raw 99.26, HQ 95.73, LQ 81.00 | within FF++, four methods, 1.3x face crop | C | accuracy, not AUC | VERIFIED |
| 28 | DF40 (NeurIPS 2024 D&B) | 10.52202/079017-0925; arXiv 2406.13495v2 | Tables 3 and 4 | AUC, unit not stated | train DF40 face swap (FF domain): fully generated faces FF domain Xception 0.810, CLIP 0.837; Celeb-DF domain Xception 0.642, CLIP 0.730 | DF40's own regenerated data | C | CLIP size and frame vs image unit not stated | VERIFIED |
| 29 | Face X-ray (CVPR 2020) | 10.1109/CVPR42600.2020.00505 | Table 2 | AUC | 74.76 (BI only), 80.58 (FF++ and BI) | Celeb-DF **v1** (408 real, 795 fake) | not used | different dataset version | read by workstream 1 |

## 2. Our values used in the comparison

| Value | Source | Status |
|---|---|---|
| 0.8218 [0.8020, 0.8393], 0.8042 [0.7841, 0.8223] and the other E-014 AUROCs | docs/EXPERIMENTS.md E-014 result table | logged, sealed |
| T3 +0.0176 [+0.0025, +0.0327]; 98.33 percent [-0.0011, +0.0355] | E-014, D-091 | logged, confirmatory |
| 0.8478 [0.7998, 0.8884] and BASE 0.7615 [0.7096, 0.8087] on 399 official-list videos | docs/PERFORMANCE_BENCHMARK.md section 12.4; scripts/eval/benchmark_audit_2026-10-27/celebdf_official_subset_output.txt | post hoc, exploratory |
| Accuracy, balanced accuracy, FPR, sensitivity, precision for I-4 and BASE | frozen_rates.py on data/_final/scores/I-4.csv and BASE.csv, replayed 2026-10-30; I-4 equals docs/test_evidence/2026-10-28_phase_b/frozen_rates_I4_replay.txt | post hoc (E-032 for the FPRs) |
| F1 0.7360 and per-seed confusion counts (Celeb-DF family) | computed 2026-10-30 from data/_final/scores/I-4.csv, test_x rows of celebdf_v2 and celebdf_pp, per-seed thresholds of data/_final/eval/I-4.json | post hoc |
| FakeAVCeleb accuracy 0.5318 | stored counts in scripts/eval/benchmark_audit_2026-10-27/e015_rates_output.txt (TP 2,168, FN 2,345, FP 2, TN 498) | post hoc |
| E-028 F3 accuracy 0.5563 | computed 2026-10-30 from data/_results/modern/e028/cache/*.npz and data/cf_eval/_meta/index.csv; AUROC, FPR, FNR equal e028_results.json | post hoc for accuracy |

## 3. Interpretations made in the report (not printed in any paper)

| Interpretation | Basis |
|---|---|
| Descriptive placement 11th of 13 (0.8478) and 8th of 13 (0.8999) among video-level Celeb-DF v2 results | sorting rows 5, 7, 8, 9, 11, 13, 17, 18, 19, 20, 21 and LipForensics' Xception; not a rank (Tier B) |
| Gap to foundation-model detectors about 0.10 to 0.11 AUROC | 0.8478 against 0.950 to 0.960 (rows 18 to 21); different test items |
| Our pipeline does not inflate Xception | our Xception 0.7615 on the 399 official-list videos against 73.7 (row 5, video) and 0.7365 (row 1, frame); same architecture, different pipelines |
| The Celeb-DF++ comparison is Tier C | per-method averaging, different reals and 15 excluded identities on our side |

## 4. Search record (2026-10-30)

WebSearch queries:
1. Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain arXiv
2. Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection arXiv ICML 2025 Celeb-DF
3. Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection arXiv CVPR 2025
4. TALL: Thumbnail Layout for Deepfake Video Detection ICCV 2023 arXiv
5. Facial Component Guided Adaptation for Foundation Model video deepfake detection CLIP Celeb-DF video-level AUC CVPR 2025
6. Deepfake detection that generalizes across benchmarks CLIP layer norm tuning Yermakov 2025 arXiv
7. "Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection" proceedings.mlr.press ICML 2025
8. DeepfakeBench NeurIPS 2023 Datasets and Benchmarks proceedings Yan Zhang Yuan Lyu Wu
9. "Deepfake Detection that Generalizes Across Benchmarks" WACV 2026 GenD
10. SigLIP face forgery deepfake detection FaceForensics++ Celeb-DF cross-dataset AUC
11. "Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection" ICML 2025 proceedings PMLR
12. "Forensics Adapter" Cui CVPR 2025 openaccess.thecvf.com
13. "DF40: Toward Next-Generation Deepfake Detection" NeurIPS 2024 Datasets and Benchmarks proceedings
14. "Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization" CVPR 2023 openaccess
15. openaccess.thecvf.com "AltFreezing for More General Video Face Forgery Detection" CVPR 2023
16. openaccess.thecvf.com "Exploring Temporal Coherence for More General Video Face Forgery Detection" ICCV 2021; "Lips Don't Lie" CVPR 2021 Haliassos
17. Haliassos "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection" CVPR 2021 openaccess
18. "UCF: Uncovering Common Features for Generalizable Deepfake Detection" ICCV 2023 openaccess
19. FaceForensics++ trained deepfake detector evaluated on StyleGAN generated faces generalization AUC face-swap detector fails entire face synthesis
20. "Celeb-DF++" Large-scale Challenging Video DeepFake Benchmark Li Zhu Cui Lyu accepted journal 2026
21. Crossref bibliographic queries for every paper in docs/LITERATURE_BENCHMARK_REFERENCES.bib (api.crossref.org/works?query.bibliographic=...)

PDFs read (arXiv): 2204.08376, 2012.07657, 2108.06693, 2201.07131, 2307.08317, 2304.13949, 2401.13856v2, 1909.12962v4, 1912.13458, 2103.01856, 2307.07494, 1901.08971, 2007.09355, 2307.01426v2, 2411.15633v4, 2411.19715v2, 2404.05583v3, 2508.06248v4, 2507.18015v1, 2210.14457, 2406.13495v2. CVF open-access PDFs returned HTTP 403 and were not used.

## 5. Not verified or not found

- DeepfakeBench's DFD split size; Celeb-DF++ total counts (13,646 plus 20,279 does not reach the stated 53,196) and the test-fake count of its original Celeb-DF column.
- Training compression of SBI and LAA-Net; test list of Effort, Forensics Adapter, TALL, LAA-Net and DFD-FCG; Celeb-DF version and unit of UCF and of Implicit Identity Leakage Table 3; DF40 unit and CLIP size.
- How GenD's Table 2 single value (96.0) relates to its 5-seed mean (94.6).
- No SigLIP-based face-forgery detector with an FF++ to Celeb-DF v2 result was found; no published test-frame-count ablation was found.

## 6. Additional external metrics, 2026-10-31 (generated image, speech, audio-visual)

Added for docs/POST_FREEZE_BENCHMARK_COMPARISON.md. Same rule as section 1: each value was extracted from the primary PDF by one agent and re-read from the primary source by a second agent on 2026-10-31 ("VERIFIED"). Venue and DOI sources and the parts read are recorded in docs/RESEARCH.md, "Benchmark verification of 2026-10-31". Tiers: A same data, split, labels and metric; B same dataset, different protocol, subset or metric; C contextual.

| # | Paper (venue, register id) | Location | Metric and unit | Exact value | Dataset and protocol | Our nearest value | Tier | Qualification | Status |
|---|---|---|---|---|---|---|---|---|---|
| 30 | Community Forensics (CVPR 2025, R-038) | Table 2, Comp. column | mean of per-generator AP; mean of per-generator accuracy (fractions) | Ojha et al. (frozen CLIP ViT-L/14 + linear, ProGAN) mAP 0.592, Acc 0.540; Wang et al. 0.537, 0.513; ViT-S 224 0.971, 0.861; high res. 0.987, 0.892 | CommunityForensics-Eval Comprehensive set, 21 generators, each paired with its own reals | E-034 mean per-generator AP and accuracy (secondary analysis); E-028 macro AUROC 0.7039 | B | metric AP, not AUROC; our reals pooled per group, 12 of 21 generators in E-034; accuracy threshold not stated by the paper | VERIFIED |
| 31 | UnivFD, Ojha et al. (CVPR 2023, R-011) | Tables 2 and 3, LC row | AP and accuracy (percent) | total mAP 93.38; total accuracy 81.38; Guided 87.77, LDM 200 99.14, DALL-E 97.15 (AP) | train ProGAN; 19 test generators (CNN family, Guided, LDM, Glide, DALL-E) | our T-I (same recipe): diffusion test_x pooled AUROC 0.9354; CNN-synthesis test_x 0.9316 (E-014) | B | AP vs AUROC; per-generator mean vs pooled; our CNN test set is the 8-generator public release | VERIFIED |
| 32 | CNNDetection, Wang et al. (CVPR 2020, R-047) | Table 2 | AP (percent), mean over 11 generators | Blur+JPEG (0.1) 92.6; Blur+JPEG (0.5) 90.8 | ResNet-50 trained on ProGAN | T-I CNN-synthesis test_x 0.9316 AUROC | C | different architecture, metric and generator list | VERIFIED |
| 33 | Muller et al. (Interspeech 2022, R-033) | Table 1 (ISCA PDF) | EER (percent), mean plus or minus std of 3 runs | RawNet2 4 s: LA eval 4.351, In-the-Wild 33.943 (best In-the-Wild value in the table) | train ASVspoof 2019 LA; test LA eval and In-the-Wild (whole set) | T-A: LA eval EER 6.94, In-the-Wild EER 15.78 (E-030, post hoc), AUROC 0.9835 and 0.9148 (E-014) | B | same datasets and metric; our In-the-Wild test excludes the 12 dev_x speakers; models trained from scratch vs our frozen SSL heads | VERIFIED |
| 34 | Pascu et al. (Interspeech 2024, R-083) | Tables 2 and 3 | EER (percent) | frozen XLS-R 2B + logistic regression: ASVspoof 2019 0.5, In-the-Wild 7.2; frozen WavLM base plus + LR: 3.6, 38.6 | train ASVspoof 2019 LA train and dev; test eval and In-the-Wild | T-A as row 33 | B | training also uses LA dev (ours uses dev for threshold and temperature only); Table 3 alignment reconstructed from the layout (R-083 note) | read 2026-10-27 (R-083), not re-read on 2026-10-31 |
| 35 | ASVspoof 2019 database (CSL 2020, R-013) | Table 8 | min t-DCF; EER (percent) | CQCC-GMM 0.2366, 9.57; LFCC-GMM 0.2116, 8.09 | LA evaluation set, official baselines | T-A LA eval EER 6.94 (E-030) | B | t-DCF of the integrated system; we report no t-DCF | VERIFIED |
| 36 | AASIST (ICASSP 2022, R-012) | Table 1 | EER (percent); min t-DCF | mean of 3 seeds 1.13, 0.0347 (best seed 0.83, 0.0275) | LA evaluation set | T-A 6.94 | B | trained end to end vs our frozen linear heads | VERIFIED |
| 37 | Wang and Yamagishi (Odyssey 2022, R-087) | Table 2 | EER (percent), 3 rounds | fixed W2V-XLSR + GF (pooling + linear) 15.96, 16.72, 16.98; fine-tuned XLSR + LGF 1.28, 1.28, 1.50 | train LA train only; test LA test set | T-A 6.94 | B | closest published shallow head on a frozen front end; different encoders | VERIFIED |
| 38 | LAV-DF (DICTA 2022, R-024) | v1 Table 5 | video-level AUC | BA-TFD 0.990, EfficientViT 0.965, MDS 0.828 | full LAV-DF test split (26,100), label any modality modified; models trained on LAV-DF train | E-033 secondary: each product head's AUROC on the unread pool against the same label | C | their models are trained on LAV-DF; ours zero-shot; our pool excludes 7,717 test clips; single head, not fusion | VERIFIED (F3Net 0.520 is arXiv v2 only, not used) |
| 39 | AVFF (CVPR 2024, R-023) | Table 1 | accuracy, AUC (percent) | AVFF 98.6, 99.1; Xception 67.9, 70.5 | FakeAVCeleb, 70/30 split of all samples (identity separation not stated), trained on FakeAVCeleb | T-V on the sealed identity-disjoint FakeAVCeleb half: AUROC 0.9552, sensitivity 0.480 (E-015) | C | in-dataset training vs our zero-shot; split not identity-disjoint as far as stated; AV label vs our visual label | VERIFIED |
| 40 | FakeAVCeleb (NeurIPS 2021 D&B, R-014) | Table 3 | frame-level AUC (percent) | Xception-comp 72.5, Capsule 70.9, Meso4 60.9 | detectors trained elsewhere, frames of FakeAVCeleb v1.2 | T-V 0.9552 (video level, E-015) | C | frame vs video level; which videos were tested not stated | VERIFIED |
