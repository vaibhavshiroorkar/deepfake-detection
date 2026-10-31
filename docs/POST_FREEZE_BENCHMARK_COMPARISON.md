# POST-FREEZE BENCHMARK COMPARISON: published detectors against our frozen checkpoint

Written 2026-10-31. It sets our measured numbers next to published numbers on overlapping datasets, with the comparability tier of every row.

- Published values are traced, with page and table, in docs/LITERATURE_BENCHMARK_EVIDENCE.md: section 1 for face-swap video, section 6 for generated image, speech and audio-visual. Each value was read in the primary source by two independent agents, and the venues and DOIs are in docs/RESEARCH.md.
- Our values come from the logged experiments named in each row. E-033 and E-034 are new post-freeze evaluations on previously unread data (D-094). All other rows come from the frozen records of the freeze (E-014, E-015, E-030, E-032) and were not recomputed.
- The face-swap video comparison of 2026-10-30, with 13 published methods on Celeb-DF v2, is in docs/LITERATURE_BENCHMARK_COMPARISON.md and is only summarised here.

**How to read the table**

- NR means the cited source does not report that metric. It is never estimated.
- Tier A means the same dataset version, test items, labels, preprocessing and metric. Tier B means the same dataset with a different subset, split, metric or label. Tier C is context only.
- **No row is Tier A.** A Tier B or C row places two numbers side by side. It does not rank the two models.
- Our threshold metrics use the threshold frozen in the checkpoint. That threshold was chosen on each head's own validation data, never on the test data.
- Our scores are ranking scores. They were calibrated in-domain only and are not probabilities on these datasets.

## 1. Comparison table

| # | Dataset (version) | Published model (source, location) | Our model (checkpoint head; experiment) | Published AUROC | Our AUROC [95% CI] | Published F1 | Our F1 | Published precision | Our precision | Published recall | Our recall | Published specificity | Our specificity | Our sample counts (real / fake / total) | Protocol (ours; theirs) | Tier | Caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Celeb-DF v2 | GenD, CLIP (WACV 2026; Table 2) and LipForensics (CVPR 2021; Table 2) | T-V (I-4 / I-1 SigLIP base); E-014 | 96.0 (GenD); 82.4 (LipForensics); Xception as retrained by LipForensics 73.7 | 0.8999 [0.8759, 0.9210] on our test_x; 0.8478 on the 399 official-list videos inside it (post hoc) | NR | 0.8256 (derived, post hoc) | NR | 0.9727 | NR | 0.7172 | NR | 0.8902 | 744 / 4,057 / 4,801 (344 identity groups) | ours: trained FF++ c23, identity-disjoint test_x excluding 15 dev identities, 16 frames, video level; theirs: the 518-video official list, 32 frames | B | different test items (at most 399 of 518 shared), frames, face crop; published values are single runs without intervals |
| 2 | DFD (c23) | SBI (CVPR 2022; Table 9) | T-V; E-014 | 97.56 (video AUC, 363 real + 3,068 fake) | 0.8797 [0.8566, 0.9041] | NR | 0.8374 (derived, post hoc) | NR | 0.9778 | NR | 0.7323 | NR | 0.8595 | 363 / 3,068 / 3,431 (28 groups) | ours: the whole DFD c23 set as test_x; theirs: trained on real FF++ with self-blended fakes | B | the closest row: SBI's DFD set is the same 363 real and 3,068 fake videos as ours (evidence row 9), but frames (32 vs 16), face crop and training paradigm differ, so it is not Tier A; SBI's training compression is not stated |
| 3 | FakeAVCeleb v1.2 | AVFF (CVPR 2024; Table 1) | T-V (visual label); E-015 | 99.1 (AUC), accuracy 98.6 | 0.9552 [0.9442, 0.9657] | NR | 0.6488 (derived, post hoc) | NR | 0.9991 | NR | 0.4804 | NR | 0.9960 | 500 / 4,513 / 5,013 (250 identity groups) | ours: zero-shot (never trained on FakeAVCeleb), sealed identity-disjoint half, visual label; theirs: trained on 70 percent of all samples, audio-visual label | C | in-dataset training and a split not stated to be identity-disjoint against our zero-shot identity-disjoint test |
| 4 | FakeAVCeleb v1.2 | Xception-comp (FakeAVCeleb paper, NeurIPS 2021 D&B; Table 3) | T-V; E-015 | 72.5 (frame-level AUC) | 0.9552 [0.9442, 0.9657] (video level) | NR | 0.6488 | NR | 0.9991 | NR | 0.4804 | NR | 0.9960 | as row 3 | theirs: public pretrained detectors on frames; tested videos not stated | C | frame level vs video level; E-021 found codec metadata alone gives AUC 0.9985 on FakeAVCeleb, so a high AUROC here is not evidence of manipulation detection |
| 5 | LAV-DF, official test split | BA-TFD and MDS (LAV-DF paper, DICTA 2022; v1 Table 5) | T-V and T-A, product checkpoint, single heads; **E-033** (new, unread clips) | 0.990 (BA-TFD), 0.965 (EfficientViT), 0.828 (MDS); label: any modality modified | against the same label on the unread pool: T-V 0.2954 [0.2890, 0.3019]; T-A 0.6448 [0.6383, 0.6514] | NR | NR for this label (see row 6) | NR | NR | NR | NR | NR | NR | 5,170 / 13,213 / 18,383 (5,170 source groups) | ours: zero-shot single heads (no fusion), unread 18,383 of 26,100 test clips; theirs: models trained on LAV-DF train, full test split | C | their detectors are trained on LAV-DF; our pool excludes 7,717 test clips; we have no fused score by design |
| 6 | LAV-DF, unread test clips (own modality labels) | none published for single-modality zero-shot heads | T-V against modify_video; T-A against modify_audio; **E-033**, group G4 | NR | T-V 0.3803 [0.3748, 0.3860]; T-A 0.6447 [0.6372, 0.6526] | NR | T-V 0.0764 [0.0647, 0.0893]; T-A 0.5729 [0.5615, 0.5852] | NR | T-V 0.2109; T-A 0.5990 | NR | T-V 0.0467; T-A 0.5490 | NR | T-V 0.8393; T-A 0.6665 | T-V 5,521 / 5,078 / 10,599; T-A 5,557 / 5,042 / 10,599 (3,000 source groups) | see docs/EXPERIMENTS.md E-033 | n/a | our own measurement, no published counterpart |
| 7 | ASVspoof 2019 LA, evaluation partition | AASIST (ICASSP 2022; Table 1); LFCC-GMM baseline (ASVspoof 2019 database paper, CSL 2020; Table 8); frozen W2V-XLSR + shallow GF back end (Wang and Yamagishi, Odyssey 2022; Table 2) | T-A (I-4: frozen HuBERT, WavLM, Whisper with linear heads); E-014, E-030 | NR (EER only): AASIST EER 1.13 percent; LFCC-GMM 8.09; XLSR + GF 15.96 to 16.98 | 0.9835 [0.9798, 0.9872]; EER 6.94 percent [5.95, 7.82] (E-030, post hoc) | NR | 0.8936 (derived, post hoc) | NR | 0.9977 | NR | 0.8092 | NR | 0.9835 | 7,355 / 63,882 / 71,237 (67 speakers) | same partition and EER definition; ours: frozen encoders, linear heads, about 4 s window | B | architectures differ; AASIST is trained end to end; our EER is post hoc |
| 8 | In-the-Wild (Muller et al. release) | RawNet2 4 s (Muller et al., Interspeech 2022; Table 1); frozen WavLM base plus + logistic regression and frozen XLS-R 2B + LR (Pascu et al., Interspeech 2024; Tables 2 and 3) | T-A; E-014, E-030 | NR (EER only): RawNet2 33.943 percent; WavLM base plus + LR 38.6; XLS-R 2B + LR 7.2 | 0.9148 [0.8896, 0.9383]; EER 15.78 percent [12.69, 18.83] (E-030, post hoc) | NR | 0.7864 (derived, post hoc) | NR | 0.7272 | NR | 0.8560 | NR | 0.8284 | 16,159 / 8,629 / 24,788 (42 speakers) | same training corpus (ASVspoof 2019 LA); ours: test_x excludes the 12 dev_x speakers; theirs: whole set | B | different subset; Pascu et al. also train on LA dev; the Pascu Table 3 alignment was reconstructed from the layout |
| 9 | Community Forensics-Eval (Comprehensive) | Ojha et al. frozen CLIP ViT-L/14 + linear head on ProGAN, as evaluated in Community Forensics (CVPR 2025; Table 2) | T-I (same recipe: frozen OpenAI CLIP ViT-L/14 + linear head on ProGAN); **E-034** (new, unread shards), group G4 | NR (mean AP 0.592, mean accuracy 0.540, per generator) | 0.7909 [0.7803, 0.8014] (pooled); mean per-generator AUROC 0.7640 (post hoc, paired cells) | NR | 0.3494 [0.3336, 0.3653] | NR | 0.9167 | NR | 0.2158 | NR | 0.9667 | 2,673 / 4,536 / 7,209 (12 of 21 generators) | ours: unread shards, RAISE-paired generators excluded, duplicates removed; theirs: all 21 generators with their own reals | B (dataset) / C (metric) | AP moves with the class ratio of each cell: our mean AP is 0.342 (preregistered, against all reals) or 0.768 (post hoc, own reals) for the same scores, so neither can be set against 0.592; their accuracy threshold is not stated |
| 10 | Diffusion test sets of Ojha et al. (Guided, LDM, Glide, DALL-E) | UnivFD linear probe (Ojha et al., CVPR 2023; Table 2) | T-I; E-014 | NR (AP: Guided 87.77, LDM 200 99.14, DALL-E 97.15; mAP over 19 generators 93.38) | 0.9354 [0.8967, 0.9665] (pooled) | NR | 0.8063 (derived, post hoc) | NR | 0.9876 | NR | 0.6813 | NR | 0.9653 | 2,000 / 8,000 / 10,000 (10 groups) | ours: pooled AUROC, threshold from ProGAN validation; theirs: per-generator AP | B | AP vs AUROC; per-generator mean vs pooled |
| 11 | Community Forensics-Eval, PNG only (format-matched) | none published | T-I; **E-034**, group G5 | NR | 0.5693 [0.5082, 0.6279] | NR | 0.3108 | NR | 0.5821 | NR | 0.2120 | NR | 0.8478 | 184 / 184 / 368 | our diagnostic | n/a | shows how AUROC changes when file format cannot separate the classes (an association, not proof of cause; the group is small and its reals are mostly FFHQ and LAION PNGs) |

Experiment and manifest references for our values:

- **E-014:** data/_final/eval/I-4.json and the ledger.
- **E-015:** data/_results/e015/predictions_test_x.csv, SHA-256 ac58b9f8...c9e5.
- **E-030:** scripts/eval/e030_eer.py.
- **E-032 rates:** docs/test_evidence/2026-10-28_phase_b/frozen_rates_I4_replay.txt (seed means over the three I-4 seeds). The "derived" F1 values are computed from the stored seed-mean precision and recall, so they are post hoc and not a logged metric.
- **E-033:** data/_manifests/posteval/e033_lavdf.groups.json (SHA-256 7a4c6614...7544) and data/_results/posteval/E-033/results.json (SHA-256 dea2f83f...a7).
- **E-034:** data/_manifests/posteval/e034_cfeval.groups.json (6b1ba80b...f21c) and data/_results/posteval/E-034/results.json (8412bccf...2d71).

## 2. Effect of sample size (new evaluations, nested groups)

AUROC [95% CI] per group. G1 is in G2, G2 is in G3, and so on. The groups are not independent of each other.

| Head and data | G1 Small | G2 Medium | G3 Large | G4 Very large | FULL | G5 diagnostic |
|---|---|---|---|---|---|---|
| T-V, LAV-DF (modify_video) | 0.3567 [0.3113, 0.3971] (190) | 0.3715 [0.3529, 0.3901] (1,055) | 0.3795 [0.3707, 0.3887] (4,265) | 0.3803 [0.3748, 0.3860] (10,599) | 0.3800 [0.3759, 0.3841] (18,383) | produced only: 0.4864 [0.4793, 0.4930] (3,065) |
| T-A, LAV-DF (modify_audio) | 0.6924 [0.6334, 0.7672] (190) | 0.6545 [0.6275, 0.6839] (1,055) | 0.6450 [0.6320, 0.6585] (4,265) | 0.6447 [0.6372, 0.6526] (10,599) | 0.6458 [0.6399, 0.6519] (18,383) | edit inside the scored window: 0.7331 [0.7161, 0.7492] (3,618) |
| T-I, Community Forensics-Eval | 0.7886 [0.7253, 0.8480] (198) | 0.7676 [0.7345, 0.7993] (791) | 0.7863 [0.7683, 0.8047] (2,372) | 0.7909 [0.7803, 0.8014] (7,209) | not offered | PNG only: 0.5693 [0.5082, 0.6279] (368) |

In every row the point estimate settles from G2 onwards. The G1 interval is three to six times as wide as the G4 interval. The T-A G1 estimate (0.69) is the largest departure from the others, and its interval still overlaps G2's. This illustrates why a small evaluation sample can mislead.

## 3. Interpretation

**Where our model does better.**

- On In-the-Wild (row 8), our frozen-encoder speech head has an EER of 15.78 percent. That is below the published from-scratch models trained on the same corpus (RawNet2 33.9 percent) and below a frozen WavLM base plus with a linear head (38.6 percent).
- It is above a frozen 2-billion-parameter XLS-R with a linear head (7.2 percent).
- On ASVspoof 2019 LA (row 7), our EER of 6.94 percent is lower than the official LFCC-GMM baseline (8.09) and lower than a frozen XLSR with a shallow back end (about 16).
- These are Tier B placements: the same datasets and the same metric, but different subsets and architectures. They are not head-to-head results.

**Where it does worse.**

- On Celeb-DF v2 and DFD (rows 1 and 2), our AUROC is about 0.10 below current published face-forgery detectors.
- On ASVspoof 2019 LA it is far from fully trained systems such as AASIST (1.13 percent).
- On Community Forensics-Eval (row 9), it misses about 78 percent of modern generated images at its frozen threshold.
- On LAV-DF (rows 5 and 6), the face-video head ranks edited videos below untouched ones (AUROC 0.38). Against the any-modality label it scores 0.30, where detectors trained on LAV-DF reach 0.83 to 0.99.

**Whether the comparison is direct.** No row is direct. Rows 1, 2, 7, 8, 9 and 10 use the same dataset but differ in test items, metric or training data (Tier B). Rows 3, 4 and 5 compare a zero-shot model with models trained on the test dataset, or compare a frame-level result with a video-level one (Tier C). Rows 6 and 11 have no published counterpart.

**What the results establish.**

1. The frozen checkpoint's behaviour on two unread pools: the AUROCs above, with group intervals.
2. That the face-video head's below-chance ranking on LAV-DF disappears within produced clips (AUROC 0.486, about chance). This points to the dataset's production pipeline rather than the edit as the source, as in E-010 and E-011.
3. That the speech head ranks LAV-DF clips better when the edit starts inside its roughly 4 s window (0.733 against 0.645 on the group that contains them). This is a descriptive comparison: the subset also changes the composition of the clips, so it is consistent with the window explanation but does not prove it.
4. That on Community Forensics-Eval T-I's AUROC falls from 0.79 to 0.57 when only PNG images are compared. This is an observed association with file format, consistent with E-028's format-only AUC of 0.93. It does not prove that the head uses format: the PNG-only group also differs in content, real sources and size.
5. How the estimates and intervals change with sample size.

**What the results do not establish.**

- Any global ranking of our model above or below a published model.
- Any performance on content outside these datasets.
- Any approval of a head for deployment: the Evaluation Lab marks these as experimental outputs, and APPROVED_HEADS stays empty.
- Any audio-visual (fused) detection result, because the product has none.

**Causes of the differences, where the evidence supports them.**

- **Dataset shift and task mismatch** (LAV-DF): T-V was trained on FF++ face swaps and reenactment. LAV-DF edits a few words and the lip region inside clips re-produced by one pipeline. The G5 diagnostic is consistent with the pipeline explanation.
- **Class imbalance:** the accuracy on Community Forensics G4 (0.494) and on FakeAVCeleb (0.532) is dominated by the larger class, so balanced accuracy and specificity are given alongside. The AP comparison in row 9 is not interpretable for the same reason.
- **Protocol differences:** different test items and subsets (Celeb-DF, In-the-Wild), metrics (AP against AUROC, frame against video level) and training regimes (rows 3 and 5).
- **Shortcut cues in the data:** file format on Community Forensics (E-028, E-034 G5) and codec metadata on FakeAVCeleb (E-021) separate the classes on their own. A high AUROC on such data can therefore come partly from a non-forensic cue; these are observed associations, and how much a given head relies on them is not measured directly.

## 4. Sources (full citations in docs/RESEARCH.md)

| Row | Paper | Register id | Status there |
|---|---|---|---|
| 1 | Yermakov, Cech, Matas, Fritz, "Deepfake Detection that Generalizes Across Benchmarks" (GenD), WACV 2026 | R-088 | PARTIAL (Tables 2, 4, S3, 5) |
| 1 | Haliassos, Vougioukas, Petridis, Pantic, "Lips Don't Lie" (LipForensics), CVPR 2021 | R-021 | PARTIAL (abstract, introduction; Table 2 from the 2026-10-30 verification) |
| 2 | Shiohara, Yamasaki, "Detecting Deepfakes with Self-Blended Images" (SBI), CVPR 2022 | R-034 | READ |
| 3 | Oorloff et al., "AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection", CVPR 2024 | R-023 | PARTIAL (Table 1 verified 2026-10-31) |
| 4 | Khalid, Tariq, Kim, Woo, "FakeAVCeleb", NeurIPS 2021 Datasets and Benchmarks | R-014 | PARTIAL (Tables 1 and 3 verified 2026-10-31) |
| 5 | Cai, Stefanov, Dhall, Hayat, "Do You Really Mean That?" (LAV-DF), DICTA 2022 | R-024 | PARTIAL (v1 Table 5 verified 2026-10-31) |
| 7 | Jung et al., "AASIST", ICASSP 2022 | R-012 | PARTIAL |
| 7 | Wang, Yamagishi et al., "ASVspoof 2019: A large-scale public database ...", Computer Speech and Language 2020 | R-013 | PARTIAL (Table 8 verified 2026-10-31) |
| 7 | Wang, Yamagishi, "Investigating self-supervised front ends for speech spoofing countermeasures", Odyssey 2022 | R-087 | PARTIAL |
| 8 | Muller, Czempin, Dieckmann, Froghyar, Bottinger, "Does Audio Deepfake Detection Generalize?", Interspeech 2022 | R-033 | PARTIAL (Table 1 of the ISCA PDF verified 2026-10-31) |
| 8 | Pascu, Stan, Oneata, Oneata, Cucu, "Towards generalisable and calibrated audio deepfake detection with self-supervised representations", Interspeech 2024 | R-083 | PARTIAL (read 2026-10-27; Table 3 alignment reconstructed) |
| 9 | Park, Owens, "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors", CVPR 2025 | R-038 | PARTIAL (Table 2 verified 2026-10-31) |
| 10 | Ojha, Li, Lee, "Towards Universal Fake Image Detectors that Generalize Across Generative Models", CVPR 2023 | R-011 | PARTIAL (Tables 2 and 3 verified 2026-10-31) |

None of these papers is cited yet in the submitted paper, report or slides (submission/final/ is unchanged). Proposed insertions for the team's decision: docs/PROPOSED_DELIVERABLE_INSERTIONS.md.
