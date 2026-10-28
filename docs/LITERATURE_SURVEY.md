# LITERATURE SURVEY

Written 2026-10-27 by the literature agent (W2, docs/LIVE_STATUS.md). Scope: the primary papers behind each task the project actually implements, with the protocol each reported number was measured under, and what each paper means for our module. The comparison of these numbers with our own measured results is in docs/PERFORMANCE_BENCHMARK.md.

## 0. Rules followed in this file

- Every paper below is in docs/RESEARCH.md with status READ or PARTIAL. For PARTIAL papers only the sections listed in its RESEARCH.md status cell were read, and only statements resting on those sections are made here. The R-number is given with every paper.
- Every number quoted from a paper names its table or section. Numbers that were re-checked on 2026-10-27 against the paper's own PDF text (pdftotext of the arXiv or proceedings file) are marked "(re-checked 2026-10-27)". Numbers taken from the RESEARCH.md row without a new check are marked "(RESEARCH.md row)"; those rows were filled from the paper's own text by earlier sessions.
- Venues come from the publisher, proceedings or Crossref record (paper/NEW_REFERENCES.md, report/NEW_REFERENCES.md, and the checks of 2026-10-27 listed in the bibliography of docs/PERFORMANCE_BENCHMARK.md). Where no venue was found the paper is cited as an arXiv preprint.
- A number from a paper is never our result. In-dataset numbers (train and test from the same dataset) and cross-dataset numbers are never put in one cell.
- Comparability letters (added 2026-10-27): docs/PERFORMANCE_BENCHMARK.md labels every comparison A (directly comparable), B (same dataset, different protocol), C (cross-dataset or out-of-distribution) or D (contextual only); the master tables per modality are its section 10 and the register of the published values used (with URL or DOI and how each was verified) its section 11. Where this survey says "partially comparable" it means B, and "context only" means C or D.
- Our own results are not repeated here except where needed to say why a paper matters; they are in docs/PERFORMANCE_BENCHMARK.md with their experiment IDs.

## 1. Paper-to-module map

| Our module (task) | What our module does | Primary papers (R-ID) | What we took from them | What we did differently |
|---|---|---|---|---|
| T-V face_video (face manipulation on video) | Face crops of 16 frames (YuNet), fine-tuned SigLIP base image encoder, clip score = mean of frame probabilities; trained on FF++ c23; tested cross-dataset on Celeb-DF v2, Celeb-DF++, DFD, FakeAVCeleb, LAV-DF | FaceForensics++ (R-001), Celeb-DF (R-002), DeepfakeBench (R-017), SBI (R-034), Effort (R-035), Forensics Adapter (R-048), LipForensics (R-021), Implicit identity leakage (R-044), Celeb-DF++ (R-053), Face X-ray (R-004) | Training set and split (R-001); the "train FF++ c23, test elsewhere" protocol and the evidence that plain backbones are close to elaborate methods under one recipe (R-017); video-level AUC as the mean of frame scores (R-034, R-017); identity-disjoint evaluation (R-044) | Group-disjoint split of Celeb-DF v2 into a 15-identity development slice and a sealed rest, so the official Celeb-DF test list is not used (D-035); three seeds and group-bootstrap intervals; a re-run Xception baseline under our own recipe instead of quoted numbers |
| T-I image_detector (generated images) | Frozen OpenAI CLIP ViT-L/14 with a linear head, trained on the 8,000-image ProGAN validation set of ForenSynths | CNNDetection (R-047), UnivFD (R-011), Cozzolino et al. (R-045), NPR (R-039), DIRE (R-040), Effort (R-035) | The ProGAN-train, many-generators-test protocol (R-047); the frozen CLIP feature plus linear probe idea (R-011) | Trained on 8,000 instead of 720,000 ProGAN images; AUC with group-bootstrap interval instead of AP and accuracy; only 8 of the 13 CNN-synthesis generators and the released 1,000-image-per-domain diffusion subset |
| T-I modern generators (experimental; E-028 and E-029 completed 2026-10-27) | Community Forensics evaluation and training subsets | Community Forensics (R-038), GenImage (R-037), Fake or JPEG (R-046), Deepfake-Eval-2024 (R-049) | Generator diversity as the lever (R-038); format and size shortcuts to control (R-046) | E-028: frozen T-I macro AUC 0.7039 [0.6855, 0.7229], inadequate; E-029: retrained head 0.8863 [0.8756, 0.8965] but 52 percent of real images flagged at its threshold, kept out of the product |
| T-A speech (synthetic speech) | Mean logit of three frozen speech encoders (HuBERT base, WavLM base plus, Whisper small encoder) with linear heads; silence trim and peak normalisation; nuisance augmentation; trained on ASVspoof 2019 LA | ASVspoof 2019 (R-013), In-the-Wild (R-033), Pascu et al. (R-083), AASIST (R-012), RawNet2 (R-028), Tak et al. wav2vec 2.0 (R-029), ASVspoof 2021 (R-030), ASVspoof 5 (R-031), WaveFake (R-032) | Training and in-corpus test data (R-013); the out-of-corpus test (R-033); the frozen self-supervised encoder plus logistic regression recipe and its scaling with encoder size (R-083); seed variance (R-012) | AUC as the headline in docs/EXPERIMENTS.md E-014 (EER is stored with intervals in the sealed reports data/_final/eval/*.json and was recomputed in E-030; corrected 2026-10-27); a 42-speaker sealed part of In-the-Wild instead of the full set; base-size encoders only (12 GB card) |
| lip_sync (audio-visual offset) | SyncNet v2 port: offset search over plus or minus 15 frames, confidence, empirical band of -6 to +2 frames | SyncNet (R-084), Wav2Lip (R-085), AVAD (R-026), Zhou and Lim (R-020), Chugh et al. (R-015) | The offset and confidence definitions (R-084); SyncNet as the de facto lip-sync measure used to train and score lip-sync generators (R-085) | Used as weak evidence only, with an offset band calibrated on real clips; no detector claim |
| generated_video (experimental, gate closed) | Logistic regression on the mean DINOv2-small class token of 8 canonical frames (C2); ReStraV-style trajectory MLP (C1) as a candidate; DF26 open-generator heads under E-027 | ReStraV (R-078), STALL (R-079), D3 (R-080), DF26 (R-081), DeepAction (R-082), DINOv2 (R-068) | The trajectory features and the DINOv2-S encoder (R-078); the held-out benchmark and its zero-shot protocol (R-081); the development corpus (R-082) | Canonical re-encode of every clip to remove container shortcuts (E-019); source-confound control (E-026); leave-one-family-out training on DF26 open generators (E-027, running) |
| audio-visual mode (no fused verdict) | Separate visual and audio scores; late fusion tested only on LAV-DF (E-011) | FakeAVCeleb (R-014), LAV-DF (R-024), AV-Deepfake1M (R-025), Zhou and Lim (R-020), MRDF (R-022), AVFF (R-023), AVAD (R-026), MIS-AVoiDD (R-027) | Per-modality labels (R-014, R-024); the finding that naive fusion is weak (R-014, R-020) | No fused score shown for video; the LAV-DF fusion result is reported as driven by a production cue (E-011, E-012) |
| media_metadata and content_credentials (provenance) | File and stream metadata report; C2PA manifest verification against an operator trust anchor | C2PA specification 2.2 (R-086), Fake or JPEG (R-046) | What a C2PA validation establishes (R-086); that format and compression metadata are shortcuts in detector datasets (R-046), which our audits E-018 and E-021 confirmed on DeepAction and FakeAVCeleb | Metadata is never used as detection evidence; only as information and as an audit tool |


## 2. Task 1: face manipulation and face-swap detection on video (our T-V)

Our setting: train on FaceForensics++ c23 (official train split, 720 videos per class; validation 140), face crops of 16 uniformly spaced frames, fine-tuned SigLIP base (I-1), clip score = mean of the 16 frame probabilities. Sealed tests: FF++ c23 test (test_in, 700 clips), Celeb-DF v2 (test_x, 4,801 videos of 44 identity groups plus the 300 YouTube real videos; the 15 development identities are excluded, so the official Celeb-DF test list is not used, D-035), Celeb-DF++ fakes against the Celeb-DF v2 real videos, and the full DFD at c23 (363 real, 3,068 fake). Metric: video-level AUC with a group-bootstrap interval.

The classic published protocol is "train on FF++ c23, test on Celeb-DF v2 and DFD, video-level or frame-level AUC". The aggregation level matters: the same model can differ by up to about 0.06 AUC between frame level and video level (R-048), so every number below names its level.

### 2.1 FaceForensics++ (R-001)

- Citation: Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Niessner. "FaceForensics++: Learning to Detect Manipulated Facial Images". ICCV 2019, pp. 1-11, doi 10.1109/iccv.2019.00009. arXiv:1901.08971.
- Data: 1,000 real YouTube videos manipulated with Deepfakes, Face2Face, FaceSwap and NeuralTextures (4,000 fake videos) at raw, c23 and c40; fixed split 720 / 140 / 140 videos.
- Reported (RESEARCH.md row): in-dataset per-frame accuracy of XceptionNet on face crops, all four methods pooled: 99.26 (raw), 95.73 (c23), 81.00 (c40) percent.
- Limitations: four 2019 methods; in-dataset evaluation; the identity overlap between the fixed train and test videos was not established in the sections read.
- Relevance: our training set and in-dataset test. PARTIAL.

### 2.2 Celeb-DF (R-002) and Celeb-DF++ (R-053)

- Celeb-DF: Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, Siwei Lyu. CVPR 2020, pp. 3207-3216, doi 10.1109/cvpr42600.2020.00327. 590 real videos of 59 celebrities and 5,639 face-swap videos. Nine detectors run with their released models on key frames, frame-level AUC on Celeb-DF: Xception-c23 65.3, Xception-c40 65.5, DSP-FWA 64.6 (RESEARCH.md row). Released models trained on different data, so the numbers are not like for like. PARTIAL.
- Celeb-DF++: Yuezun Li, Delong Zhu, Xinjie Cui, Siwei Lyu. arXiv:2507.18015 (no venue found). Same 590 real videos; 53,196 fakes from 22 methods in three scenarios (face swap, face reenactment, talking face). Reported (RESEARCH.md row, READ): trained on Celeb-DF face swap only, eight retrained detectors average about 71.7 (frame) and 72.1 (video) AUC over the 22 methods; the best, Effort, 83.0 and 84.4; trained on FF++ HQ, average AUC drops by 5.2 (frame) and 7.1 (video) points on Celeb-DF++ against Celeb-DF. Real videos come from 59 people only.
- Relevance: our main cross-dataset test family. Our Celeb-DF++ test pairs its fakes with the Celeb-DF v2 real videos, a pooling the paper does not use.

### 2.3 DeepfakeBench (R-017)

- Citation: Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu. "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection". NeurIPS 2023 Datasets and Benchmarks Track, pp. 4534-4565. arXiv:2307.01426v2.
- Method: one codebase, 15 detectors, 9 datasets, one preprocessing pipeline; all detectors trained on FF++ c23.
- Reported (re-checked 2026-10-25 by the paper stream, paper/NEW_REFERENCES.md part D; Table 3, frame-level AUC): plain Xception within-domain average 0.9450 (FF-c23 column 0.9637) and cross-domain average 0.7718 over eight test sets (Celeb-DF v1, Celeb-DF v2, DF-1.0, DFD, DFDC, DFDCP, FaceShifter, UADFV). Per-column values (re-checked 2026-10-27 in the ar5iv rendering; the flattened row order was mapped to the eight cross-domain columns and the mapping confirmed because the eight values average to the printed 0.7718, 0.7718 and 0.7801): Celeb-DF v2 / DFD for Xception 0.7365 / 0.8163, EfficientNet-B4 0.7487 / 0.8148, UCF 0.7527 / 0.8074.
- Test sets: Celeb-DF uses the authors' test list; DFD has no official split, so DeepfakeBench makes its own train, validation and test split (Appendix A.3), so a DeepfakeBench-pipeline DFD number is not on the full DFD.
- Limitations: frame-level only ("Our benchmark currently adopts the frame level evaluation"); the number of seeds is not stated in the text read.
- Relevance: the evidence that a plain backbone is close to elaborate methods under one recipe, which is why our baseline is a re-run Xception rather than a quoted number. PARTIAL.

### 2.4 Self-Blended Images (SBI) (R-034)

- Citation: Kaede Shiohara, Toshihiko Yamasaki. "Detecting Deepfakes with Self-Blended Images". CVPR 2022 (arXiv comments: "CVPR 2022 Oral"), doi 10.1109/cvpr52688.2022.01816. arXiv:2204.08376.
- Method: EfficientNet-B4 (ImageNet) trained only on real FF++ frames and their self-blended versions (no manipulated video), SAM optimiser, 100 epochs, 8 frames per video for training and 32 for inference; video score = mean of frame scores (Sections 4.1 and 4.2, re-checked 2026-10-27).
- Data and split: official splits; the training compression level is not stated in the main text read (the "raw version" statement in Section 4.3 concerns the FF++ evaluation).
- Reported (re-checked 2026-10-27, Table 9, video-level AUC, with test counts): FF++ (140 real, 560 fake) 99.64; DFD (363 real, 3,068 fake) 97.56; Celeb-DF v2 (178 real, 340 fake, the official test list) 93.18; DFDC 72.42; DFDCP 86.15; FFIW 84.83.
- Elsewhere: the Effort paper lists SBI at 0.886 on Celeb-DF v2 and 0.827 on DFD (Table 1, re-checked 2026-10-27; values cited from DeepfakeBench-style re-runs, aggregation level of cited rows not stated), so the published value of one method differs by pipeline.
- Limitations stated by the authors: frame-level model; fails on whole-image synthesis (AUC 69.11 on FFHQ against StyleGAN). No seeds or intervals.
- Relevance: the DFD test set of SBI is the full DFD, the same videos as our DFD test. READ.

### 2.5 Effort: Orthogonal Subspace Decomposition (R-035)

- Citation: Zhiyuan Yan, Jiangming Wang, Peng Jin, Ke-Yue Zhang, Chengchun Liu, Shen Chen, Taiping Yao, Shouhong Ding, Baoyuan Wu, Li Yuan. "Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection". Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267:70268-70288 (PMLR index, 2026-10-27). arXiv:2411.15633v4.
- Method: SVD of the pretrained CLIP ViT-L/14 weights; the principal subspace is frozen and only the residual (minor) subspace is trained with orthogonality and singular-value constraints; 0.19M trainable parameters.
- Deepfake protocol (re-checked 2026-10-27, implementation details): DeepfakeBench codebase, trained on FF++ c23, 8 frames per video for training and 32 for inference, Adam with fixed learning rate 2e-4, batch 32; "video-level Area Under the Curve (AUC)", computed from the average output probability of each video.
- Reported (re-checked 2026-10-27, Table 1, cross-dataset video-level AUC): Celeb-DF v2 0.956, DFD 0.965, DFDC 0.843, DFDCP 0.909, DFo 0.977, WDF 0.848, FFIW 0.921, mean 0.917. In the same table (baselines cited from DeepfakeBench and two later papers, or run from released checkpoints; their aggregation level is not restated): SBI 0.886 and 0.827, UCF 0.837 and 0.867, CDFA 0.938 and 0.954 on Celeb-DF v2 and DFD.
- Generated-image protocol: trained on ProGAN (UniversalFakeDetect), 19 test subsets, mAP 99.41 and mean accuracy 95.19 at threshold 0.5 (Section 5 text and Tables 2 and 3).
- Limitations: the DFD and Celeb-DF test sets are those of the DeepfakeBench pipeline (its own DFD split, the authors' Celeb-DF list), not the full DFD; no seeds, intervals or VRAM in the text (RESEARCH.md row); the chosen rank was justified on the same test sets as the headline (RESEARCH.md row).
- Relevance: the strongest published FF++ c23 to Celeb-DF v2 and DFD numbers read, from a CLIP-based model with very few trainable parameters. PARTIAL.

### 2.6 Forensics Adapter (R-048)

- Citation: Xinjie Cui, Yuezun Li, Delong Zhu, Jiaran Zhou, Junyu Dong, Siwei Lyu. "Forensics Adapter: Unleashing CLIP for Generalizable Face Forgery Detection". IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, 2026, pp. 13354-13369, doi 10.1109/tpami.2026.3706492 (Crossref record, 2026-10-27; the arXiv v4 read is this journal version; the conference version is CVPR 2025, pp. 19207-19217, doi 10.1109/cvpr52734.2025.01789).
- Reported (RESEARCH.md row): trained on FF++ c23, frame-level AUC Celeb-DF v2 0.900, DFD 0.933 (ForAda); video-level Celeb-DF v2 0.957 and DFDC 0.872.
- Relevance: a CLIP adapter (5.7M trainable parameters); shows the frame-versus-video gap. PARTIAL.

### 2.7 LipForensics (R-021)

- Citation: Alexandros Haliassos, Konstantinos Vougioukas, Stavros Petridis, Maja Pantic. "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection". CVPR 2021, doi 10.1109/cvpr46437.2021.00500. arXiv:2012.07657.
- Reported (RESEARCH.md row): trained on all four FF++ methods, video-level AUC Celeb-DF v2 82.4, DFDC 73.5, FaceShifter 97.1, DeeperForensics 97.6; Xception 75.3 and Face X-ray 81.2 on the same mean. Strong under corruptions.
- Relevance: a frozen lip-reading feature as the cue; the visual counterpart of our lip-sync idea. PARTIAL.

### 2.8 Face X-ray (R-004) and Implicit Identity Leakage (R-044)

- Face X-ray (Li et al., CVPR 2020, pp. 5001-5010): Celeb-DF AUC 36.19 for an FF++-trained Xception, 74.76 for Face X-ray trained on blended images only, 80.58 with FF++ plus blended images (RESEARCH.md row). PARTIAL.
- Implicit Identity Leakage (Dong et al., CVPR 2023, pp. 3994-4004): identity is linearly decodable from detector features; trained on FF++, video AUC 91.15 to 93.88 on Celeb-DF (RESEARCH.md row). It motivates our identity-disjoint splits. READ.

## 3. Task 2: generated-image detection (our T-I)

Our setting: frozen OpenAI CLIP ViT-L/14 (IMG-13) with a linear head (UnivFD-style linear probe), trained on the ForenSynths ProGAN validation set (8,000 images; 14 classes for training, 6 for validation). Sealed tests: ProGAN test set (test_in, 8,000 images, 20 classes), 5 of the 7 CNN-synthesis generator units (test_x, 48,805 images: BigGAN, GauGAN, StyleGAN with StyleGAN2, Deepfake, CycleGAN; StarGAN and whichfaceisreal form the development slice), and the released diffusion test subset of UnivFD (test_x, 10,000 images: 1,000 each of LDM 200, LDM 200 with guidance, LDM 100, Glide 100-27, Glide 50-27, Glide 100-10, Guided and DALL-E (DATA_LOG I2 names the DALL-E set "DALL-E-mini"), with 1,000 LAION and 1,000 ImageNet and ImageNet reals). Metric: AUC pooled over the units with a unit-group bootstrap interval.

### 3.1 CNN-generated images are surprisingly easy to spot... for now (CNNDetection, ForenSynths) (R-047)

- Citation: Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, Alexei A. Efros. CVPR 2020, pp. 8695-8704, doi 10.1109/cvpr42600.2020.00872. arXiv:1912.11035.
- Method: ResNet-50 (ImageNet) trained on 720K images from 20 ProGAN LSUN models (360K real, 360K fake) with blur and JPEG augmentation; tested on 11 other CNN generators; AP (threshold-free) and accuracy.
- Reported (RESEARCH.md row; the extracted Table 2 had scrambled row labels): mean AP over the test generators about 90 to 93 depending on augmentation. The UnivFD paper lists this detector (Blur+JPEG 0.1) at total mAP 83.58 and mean accuracy 69.58 over its 19-set benchmark (re-checked 2026-10-27, see 3.2), and the Effort paper Table 2 lists the same 83.58 (re-checked 2026-10-27).
- Threshold effect (RESEARCH.md row): accuracy with the 0.5 threshold against an oracle threshold, e.g. DeepFake 53.5 against 80.7 percent.
- Relevance: the data source of our T-I training and CNN-synthesis test sets. PARTIAL.

### 3.2 Towards Universal Fake Image Detectors that Generalize Across Generative Models (UnivFD) (R-011)

- Citation: Utkarsh Ojha, Yuheng Li, Yong Jae Lee. CVPR 2023, pp. 24480-24489, doi 10.1109/cvpr52729.2023.02345 (Crossref, 2026-10-27). arXiv:2302.10174v2.
- Method: frozen CLIP ViT-L/14 image features; nearest-neighbour or a linear classifier (logistic) on top; no fine-tuning.
- Data: trained on the 720K ProGAN images of R-047 (20 LSUN categories); tested on 19 sets: ProGAN, CycleGAN, BigGAN, StyleGAN, GauGAN, StarGAN, Deepfakes, SITD, SAN, CRN, IMLE, Guided, three LDM variants, three Glide variants and DALL-E; for LDM, Glide and DALL-E the real class is LAION (re-checked 2026-10-27, arXiv HTML).
- Reported (re-checked 2026-10-27, Tables 2 and 3 totals): linear probe total mAP 93.38 and mean accuracy 81.38; nearest neighbour (k = 1) 90.32 and 82.30; trained deep network (Blur+JPEG 0.1) 83.58 and 69.58. By group (RESEARCH.md row): linear probe 99.31 mAP on GAN-family sets and 95.00 on unseen diffusion and autoregressive sets, against 94.19 and 75.51 for trained baselines.
- Limitations: AP and accuracy, not AUC; the released diffusion data hold 1,000 images per domain (repository README, DATA_LOG I2), and the per-domain count used in the paper was not stated in the text read.
- Relevance: our T-I recipe is this linear probe on the same encoder, trained on 90 times fewer ProGAN images. PARTIAL.

### 3.3 Raising the Bar of AI-generated Image Detection with CLIP (R-045)

- Citation: Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Niessner, Luisa Verdoliva. CVPR Workshops 2024, pp. 4356-4366, doi 10.1109/cvprw63382.2024.00439.
- Method: CLIP features with a linear SVM trained on as few as 1,000 to 10,000 real and fake pairs (Latent Diffusion against COCO).
- Reported (RESEARCH.md row): AUC on 18 generators including DALL-E 2 and 3, Midjourney v5 and Firefly; the 10k version beat the best competitor by 6.8 average AUC points (the absolute average was not cleanly extractable). Accuracy at 0.5 does not always rise with N while AUC does.
- Relevance: the evidence that a few thousand training images suffice for a CLIP linear detector, which supports our 8,000-image training set; it uses AUC, as we do. PARTIAL.

### 3.4 Other generated-image detectors (RESEARCH.md rows)

- NPR, Rethinking the Up-Sampling Operations (R-039; Tan et al., arXiv:2312.10461): trained on a 4-class ProGAN subset, mean accuracy / AP 92.5 / 96.1 on 8 ForenSynths GANs and 95.3 / 99.8 on DiffusionForensics. PARTIAL.
- DIRE (R-040; Wang et al., arXiv:2303.09295): reconstruction error of a diffusion model; strong within LSUN-Bedroom diffusion data. PARTIAL.
- Effort (R-035, section 2.5): UniversalFakeDetect protocol, mAP 99.41 and mean accuracy 95.19 over 19 subsets (re-checked 2026-10-27); its ablation reports a linear probe at 81.02 mean accuracy (RESEARCH.md row).

## 4. Task 3: modern AI-generated image detection (Community Forensics, GenImage; our E-028 and E-029)

Our setting: E-028 scores the frozen T-I detector on a test-only subset of Community Forensics-Eval (21 modern generators); E-029 trains a new linear head on the frozen T-I encoder with Community Forensics-Small and evaluates it once on the E-028 subset. Results (docs/EXPERIMENTS.md): E-028, the frozen detector misses about three quarters of the generated images (pooled miss rate 0.7395 [0.7300, 0.7500]; macro AUC 0.7039 [0.6855, 0.7229]); E-029, the retrained head ranks them clearly better (macro AUC seed mean 0.8839, +0.1800 [0.1643, 0.1959] over the frozen head) but its validation threshold flags about half of the real images, so it is not eligible for the product. This agrees in direction with the published point that generator diversity drives generalisation, at about 1 percent of the published training scale; it is not a replication of the published numbers (different metric, subset and image size).

### 4.1 Community Forensics: Using Thousands of Generators to Train Fake Image Detectors (R-038)

- Citation: Jeongsoo Park, Andrew Owens. CVPR 2025, pp. 8245-8257, doi 10.1109/cvpr52734.2025.00772 (Crossref record, 2026-10-27). arXiv:2411.04125v2.
- Method: ViT-S (CLIP pretrained) and ConvNeXt-S classifiers trained on images from thousands of diffusion models (mostly Stable Diffusion derivatives from public model hubs); AdamW, learning rate 2e-5, batch 512, 52K iterations (RESEARCH.md row).
- Evaluation set: 26K images from 21 generators not used in training; metrics mean AP and accuracy at a fixed threshold, averaged per generator (RESEARCH.md row).
- Reported (RESEARCH.md row): on its own evaluation set, the 384 px model reaches mAP 0.987 and accuracy 0.892; on other benchmarks mAP 0.967 (Wang et al. set), 0.996 (Ojha et al. set), 0.974 (Synthbuster), 0.998 (GenImage). Performance rose with the number of distinct generators at a fixed image count (Figures 1 and 4, mean and standard error over random subsets). Freezing the backbone lowered performance (figure, no number read).
- Limitations: a single training run for the headline; training budget (batch 512) far above a 12 GB card; mostly diffusion generators; the authors state the dataset is not meant for in-the-wild use.
- Relevance: the evaluation data of E-028 and the training data of E-029, and the published claim (diversity of generators) that E-029 tests at small scale. Its numbers are AP and accuracy, so an AUC from E-028 or E-029 is at most partially comparable. PARTIAL.

### 4.2 GenImage (R-037)

- Citation: Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang. "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image". Advances in Neural Information Processing Systems 36 (NeurIPS 2023), pp. 77771-77782, doi 10.52202/075280-3398 (Crossref record, 2026-10-27). arXiv:2306.08571.
- Data: ImageNet-class prompts, 8 generator subsets (Midjourney, SD 1.4, SD 1.5, ADM, GLIDE, Wukong, VQDM, BigGAN).
- Reported (RESEARCH.md row): trained on SD 1.4, mean cross-generator accuracy ResNet-50 72.1, Swin-T 74.8, CNNSpot 64.2; same-generator accuracy at least 98.8 percent.
- Limitations: R-046 shows JPEG and image-size shortcuts in this benchmark. PARTIAL.

### 4.3 Bias and in-the-wild evidence

- Fake or JPEG (R-046, section 9.2): format and size shortcuts in GenImage.
- Deepfake-Eval-2024 (R-049; Chandra et al., arXiv): on 2024 in-the-wild content the open-source image detectors lose on average 45 percent AUC against their original test data; the CLIP linear-probe model (UFD) collapsed to predicting everything fake after fine-tuning. PARTIAL.

## 5. Task 4: synthetic speech detection (our T-A)

Our setting: frozen speech encoders with linear heads, trained on ASVspoof 2019 LA (train; dev as validation), development slice of In-the-Wild (12 of 54 speakers), sealed tests on the ASVspoof 2019 LA evaluation set (test_in, 71,237 utterances) and on the other 42 In-the-Wild speakers (test_x, 24,788 clips). Headline metric AUC with a speaker-group bootstrap interval.

### 5.1 ASVspoof 2019 (R-013)

- Citation: Xin Wang, Junichi Yamagishi, Massimiliano Todisco, Hector Delgado, Andreas Nautsch, Nicholas Evans and others. "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech". Computer Speech and Language, vol. 64, art. 101114, 2020, doi 10.1016/j.csl.2020.101114. arXiv:1911.01601.
- What it is: the database, not a detector. Logical access (LA) part: text-to-speech and voice-conversion attacks built from a multi-speaker English corpus (VCTK based, 107 speakers). Training and development use 6 attacks (A01 to A06); evaluation uses 13 attacks (A07 to A19).
- Counts (RESEARCH.md row, as printed in the paper): LA training 2,580 bona fide, development 2,548 bona fide, evaluation 7,355 bona fide and 63,882 spoofed. The evaluation total, 71,237, equals our test_in count, so our in-corpus test is the full official evaluation set.
- Protocol and metric: EER and the tandem detection cost function (t-DCF).
- Limitations: read speech in studio conditions, one source corpus for every partition; later work (R-033, R-083) shows that in-corpus results do not transfer to real-world audio.
- Relevance: our training data and in-corpus test. PARTIAL (abstract, database design, partitions, attack list).

### 5.2 Does Audio Deepfake Detection Generalize? In-the-Wild (R-033)

- Citation: Nicolas M. Muller, Pavel Czempin, Franziska Dieckmann (Crossref prints Diekmann), Adam Froghyar, Konstantin Bottinger. Interspeech 2022, pp. 2783-2787, doi 10.21437/interspeech.2022-108. arXiv:2203.16263 (v5 read).
- Method: re-implements twelve architectures (LCNN variants, LSTM, MesoNet, MesoInception, ResNet18, Transformer, CRNNSpoof, RawNet2, RawPC, RawGAT-ST) with cqtspec, logspec, melspec or raw input and 4 s or full-length inputs; Adam, learning rate 1e-4, up to 100 epochs, early stopping (Section 4.1.1, re-checked 2026-10-27).
- Data and split: trained on the ASVspoof 2019 LA train and dev parts; tested on ASVspoof 2019 LA eval and on the new In-the-Wild set (Section 4.1.2). In-the-Wild: 58 celebrities and politicians, 20.8 hours bona fide and 17.2 hours spoofed (Figure 1 caption; the contribution list says 20.7 hours bona fide and 37.9 hours in total, a small inconsistency inside the paper), clips of about 4.3 s on average, 16 kHz (Section 3, re-checked 2026-10-27). The project's copy lists 54 speakers in its meta file (D-035 Amendment 1).
- Generators: unknown; fakes are segments of public videos that advertise themselves as audio deepfakes.
- Metric: EER on In-the-Wild (no t-DCF, because the costs exist only for ASVspoof), EER and t-DCF on ASVspoof (Section 4.1.3). Three random initialisations, mean and standard deviation (Table 1 caption).
- Reported (re-checked 2026-10-27, Section 5.2 text): "up to 1.2% EER on ASVspoof and 33.9% EER on the in-the-wild dataset (RawGAT-ST and RawNet2)", so the best in-corpus and the best out-of-corpus model are different models. Section 5.3: EER deteriorates "by about 200 to 1000 percent" from ASVspoof eval to In-the-Wild, and models often do no better than chance. Retraining RawNet2 (4 s input) on all ASVspoof 2019 splits including eval gave 33.1 plus or minus 0.2 percent EER on In-the-Wild, no improvement. Mean over models by input length: ASVspoof eval EER 9.85 (full) and 18.89 (4 s); In-the-Wild 60.10 and 67.25 (the running text says 19.89 for the 4 s ASVspoof mean, the table 18.89).
- Limitations: fake clips are self-declared publicity fakes (a biased sample of fakes); genuine clips were matched by hand; no hyper-parameter tuning; the Table 1 layout is scrambled in text extraction, so per-model cells are not quoted here, only numbers stated in the running text.
- Relevance: the source of our out-of-corpus test and the reason our audio iterations (I-2 to I-4) target the In-the-Wild gap. READ.

### 5.3 Towards generalisable and calibrated audio deepfake detection with self-supervised representations (R-083, added 2026-10-27)

- Citation: Octavian Pascu, Adriana Stan, Dan Oneata, Elisabeta Oneata, Horia Cucu. Interspeech 2024, pp. 4828-4832, doi 10.21437/interspeech.2024-1302 (Crossref record). arXiv:2309.05384v2.
- Method: the closest published recipe to our T-A working version: frozen self-supervised speech representations (mean of the last hidden layer) with scikit-learn logistic regression (C = 10^6, at most 1,000 iterations); 11 encoder variants of wav2vec2 and WavLM; an MLP and a self-attention head on the 2B model as alternatives (Sections 3 and 4, re-checked 2026-10-27).
- Data: trained on ASVspoof 2019 LA train and dev (Table 4 caption; about 50k samples); tested on the test splits of eight datasets including ASVspoof 2019 (ASV), In-the-Wild (ITW), TIMIT-TTS, FakeOrReal, partially spoofed and multilingual sets (Table 1). Section 3 (re-read in the arXiv PDF text 2026-10-27): the train and dev splits have 50k samples, "test has 70k samples (63k fake and 7k real)" (the LA evaluation set, equal to our 71,237 in count), and "we use both train and dev for training".
- Metric: EER and ECE (15 bins); intervals are two standard deviations over ten bootstrap runs on the test data (Table 2 caption).
- Reported (re-checked 2026-10-27): Table 2, wav2vec2 XLS-R 2B frozen plus logistic regression: ASV EER 0.5 plus or minus 0.1, ITW EER 7.2 plus or minus 0.3, mean over the eight datasets 8.8; RawNet2 trained the same way: ASV 5.9, ITW 46.7, mean 30.9. Table 3 (frozen encoder plus logistic regression, ASV / ITW EER): wav2vec2 base (94M, LibriSpeech 1k h) 4.0 / 45.9; WavLM base (94M) 4.8 / 37.6; WavLM base plus (94M, 84k h) 3.6 / 38.6. The text states that mean EER over four datasets falls with encoder size (wav2vec2 36.3 percent at 94M to 26.9 percent at 300M; WavLM 32.8 to 24.7). The row-to-value alignment of Table 3 was reconstructed from the extracted layout and checked against the means quoted in the text (36.3 wav2vec2 base, 34.3 WavLM base, 32.8 WavLM base plus); other Table 3 cells are not quoted.
- Limitations: EER only (no AUC); the 2B encoder needs about 9.3 GB of memory at prediction (Table 2), outside our base-size choice; ITW is evaluated in full, not split by speaker; the authors' own HuBERT and Whisper encoders are not in the table.
- Relevance: under the same training data, frozen base-size encoders with a linear head reach only about 38 to 46 percent EER on In-the-Wild in this paper, while a 2B-parameter frozen encoder reaches 7.2 percent, so encoder scale is the main lever for this recipe. PARTIAL (abstract, Sections 1, 3, 4.1 to 4.5 with Tables 2 to 4; the reliability figure discussion only skimmed; references not read).

### 5.4 AASIST (R-012)

- Citation: Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, Hye-jin Shim, Joon Son Chung, Bong-Jin Lee, Ha-Jin Yu, Nicholas Evans. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks". ICASSP 2022, pp. 6367-6371, doi 10.1109/icassp43922.2022.9747766. arXiv:2110.01200.
- Method: raw-waveform encoder with heterogeneous graph attention over spectral and temporal nodes; AASIST-L has 85K parameters.
- Data: ASVspoof 2019 LA; train and development hold 6 attacks, evaluation 13 (A07 to A19); 4 s inputs.
- Reported (re-checked 2026-10-27, Table 1 caption and Section 4 text): pooled EER on the 2019 LA evaluation set 1.13 percent and min t-DCF 0.0347 as the mean of three seeds; best seed 0.83 percent and 0.0275. The RawGAT-ST baseline varied between 1.19 and 2.06 percent EER across seeds.
- Cross-dataset: none in the sections read.
- Limitations: in-corpus only; the headline is the best of three seeds.
- Relevance: the reference in-corpus number for ASVspoof 2019 LA and a record of seed variance. PARTIAL (abstract, method, experiments, Tables 1 to 3).

### 5.5 Other audio papers used for context (RESEARCH.md rows, not re-checked)

- RawNet2 for anti-spoofing (R-028; Tak et al., ICASSP 2021, doi 10.1109/icassp39728.2021.9414234): ASVspoof 2019 LA eval pooled EER 4.66 to 5.64 percent for RawNet2 variants against 3.50 for the LFCC-GMM baseline; score fusion 1.12 to 1.35 percent. READ.
- wav2vec 2.0 front end with data augmentation (R-029; Tak et al., Odyssey 2022, pp. 112-119): trained on 2019 LA, tested on 2021 LA, EER 0.82 best seed (1.00 mean) with augmentation; 2021 DF 2.85 (3.69). Cross-condition, not In-the-Wild. READ.
- ASVspoof 2021 (R-030): evaluation-phase best LA EER 1.32 percent, DF 15.64 percent; the DF best fell to 0.10 percent in the progress phase and rose again in evaluation. READ (sections listed in RESEARCH.md).
- ASVspoof 5 (R-031): baselines RawNet2 and AASIST reach 36.04 and 29.12 percent pooled EER on the evaluation set. PARTIAL.
- WaveFake (R-032; NeurIPS 2021 Datasets and Benchmarks): single-generator training transfers poorly across vocoders and languages. READ.
- Deepfake-Eval-2024 (R-049): open-source audio detectors lose on average 48 percent AUC on 2024 in-the-wild content against their original test data. PARTIAL.

## 6. Task 5: audio-visual synchronisation and lip sync (our lip_sync module)

Our setting: a port of SyncNet (version syncnet-v2-port-2), offset search over plus or minus 15 frames at 25 fps, a confidence value, applicability gates (face, speech-likeness heuristic, clip length, one speaker), and an empirical offset band of -6 to +2 frames calibrated on real clips (E-025). The module reports `consistent`, `potential_mismatch` or `inconclusive`; it is weak evidence, not a detector.

### 6.1 Out of time: automated lip sync in the wild (SyncNet) (R-084, added 2026-10-27)

- Citation: Joon Son Chung, Andrew Zisserman. "Out of Time: Automated Lip Sync in the Wild". Computer Vision, ACCV 2016 Workshops, Lecture Notes in Computer Science, Springer, 2017, pp. 251-263, doi 10.1007/978-3-319-54427-4_19 (Crossref record). Author PDF: robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf.
- Method (re-checked 2026-10-27, Section 2): two-stream ConvNet; the audio stream takes 13 MFCC bands at 100 Hz for a 0.2 s window (20 time steps), the video stream takes 5 grayscale mouth crops of 111 x 111 (0.2 s at 25 Hz); trained with a contrastive loss on genuine and shifted audio-video pairs from BBC television, assuming broadcast audio is usually in sync. The offset of a clip is the shift with minimum mean distance between the video feature and the audio features within plus or minus 1 second, averaged over samples of the clip (Section 4). The confidence of an offset is "the difference between the minimum and the median of the Euclidean distances" (Section 5).
- Data: several hundred hours of BBC News programmes (2013 to 2016) split by time into train, validation and test (Section 3, Table 1); clips with more than one face are discarded.
- Protocol and metric: no ground-truth offset exists, so a random sample of several hundred test clips was judged manually: "successful if the lip-sync error is not detectable to a human" (Section 4).
- Reported (re-checked 2026-10-27, Table 2): accuracy within the human-detectable range 81 percent for a single 0.2 s sample and above 99 percent when averaged over a clip. Also active speaker detection F1 on the Columbia dataset (Table 3) and lip reading on OuluVS2 (Table 4).
- Context it states: the detectability threshold for an average viewer is about -125 ms (audio lags) to +45 ms (audio leads), citing a broadcast recommendation (Section 1).
- Limitations: the evaluation is a manual judgement on a few hundred clips of one broadcaster; no deepfake data; no false-alarm rate on unmodified clips is reported.
- Relevance: the method and the offset and confidence definitions our module implements. It is not a forgery detector, and our E-022 and E-024 results show that unmodified clips of LAV-DF and FakeAVCeleb measure off-sync by 3 or more frames in about 12 to 15 percent of cases, so a tolerance of 2 frames produces false alarms. PARTIAL (abstract, 1, 2, 3, 4 with Table 2, 5 summary with Table 3 caption; Section 6 lip reading skimmed).

### 6.2 A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild (Wav2Lip) (R-085, added 2026-10-27)

- Citation: K R Prajwal, Rudrabha Mukhopadhyay, Vinay P. Namboodiri, C. V. Jawahar. Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 484-492, doi 10.1145/3394171.3413532 (Crossref record). arXiv:2008.10010.
- Method (re-checked 2026-10-27 against the ar5iv rendering, Sections 3.2, 3.3 and 4.2.1): a lip-sync generator trained against "a pre-trained expert lip-sync discriminator that is accurate in detecting sync in real videos", which is "a modified version of SyncNet". The paper's evaluation metrics use the pre-trained SyncNet: LSE-D, "the average error measure calculated in terms of the distance between the lip and audio representations" (lower is better), and LSE-C, "the average confidence score" (higher is better).
- Reported (re-checked 2026-10-27, Table 1, LRS2): Wav2Lip LSE-D 6.386 and LSE-C 7.789; real videos 6.736 and 7.838. By the SyncNet distance, Wav2Lip output is at least as well synchronised as real video on LRS2.
- Relevance: Wav2Lip is a generator, not a detector. It is cited because (a) the FakeAVCeleb and LAV-DF fakes we test include Wav2Lip output, and (b) a SyncNet offset or confidence is the quantity this generator was optimised against. Our E-025 measurement agrees: Wav2Lip clips of FakeAVCeleb sit at offsets of -3 and -2 frames, inside the natural spread of real clips, and none is flagged under the calibrated band. PARTIAL (abstract, Sections 3.2, 3.3, 4.2.1 and Table 1 via the ar5iv HTML rendering; the rest not read).

### 6.3 Self-Supervised Video Forensics by Audio-Visual Anomaly Detection (AVAD) (R-026)

- Citation: Chao Feng, Ziyang Chen, Andrew Owens. CVPR 2023, pp. 10491-10503, doi 10.1109/cvpr52729.2023.01011 (Crossref record, 2026-10-27). arXiv:2301.01767.
- Method: an audio-visual synchronisation model trained on real videos only (LRS2 or LRS3), then an autoregressive anomaly model over its time-delay features; no fake training data.
- Data and protocol: FakeAVCeleb test split of 600 videos (100 real, 500 fake) and KoDF (100 real, 100 fake); AP and AUC at video level (RESEARCH.md row).
- Reported (RESEARCH.md row): on FakeAVCeleb, the weak case is fake audio with real video (RVFA): AP 62.4 and AUC 71.6 (trained on LRS2); mean AUC over the fake-visual categories 94.5 (LRS2) and 92.8 (LRS3). Cross-dataset to KoDF: AP 87.6, AUC 86.9.
- Limitations: cannot flag fakes that remain lip-synchronised; small test sets.
- Relevance: the strongest published evidence that a synchronisation signal can detect visual manipulations zero-shot; it uses learned time-delay features and an anomaly model, not a fixed offset rule like our module, so its AUC is not a target for our module. PARTIAL.

### 6.4 Other synchronisation and dissonance papers (RESEARCH.md rows)

- Not made for each other (R-015; Chugh et al., ACM MM 2020): a modality dissonance score; DFDC subset video AUC 91.54 in-dataset. PARTIAL.
- Joint Audio-Visual Deepfake Detection (R-020; Zhou and Lim, ICCV 2021): a sync stream with its own label improved accuracy over independent streams in-dataset and under held-out categories on FF++ with vocoder-made fake audio. READ.

## 7. Task 6: AI-generated video detection (our experimental generated_video module)

Our setting: the installed head (E-020 C2) is a logistic regression on the mean DINOv2-small class token of 8 frames of a canonical clip (first 2.0 s, 7 fps, 224 px centre crop, fixed codec settings, E-019 v3); it is experimental, its gate is closed and its finding is always inconclusive. A ReStraV-style trajectory MLP (C1) and a pixel-statistics control (C3) were measured beside it. E-027 trained heads on DF26's open generators.

### 7.1 AI-Generated Video Detection via Perceptual Straightening (ReStraV) (R-078)

- Citation: Christian Interno, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, David Klindt. Advances in Neural Information Processing Systems 38 (NeurIPS 2025), pp. 23468-23501, doi 10.52202/085713-0697 (Crossref record, 2026-10-27). arXiv:2507.00583.
- Method: per video, 21 values from a frozen DINOv2 ViT-S/14 over 24 frames in a 2 s window (7 stepwise distances and 6 curvature angles, with mean, variance, minimum and maximum), fed to a two-layer MLP (64, 32).
- Data: VidProM, 50,000 generated videos (Pika, VideoCraft2, Text2Video-Zero, ModelScope, Sora) against 50,000 DVSC2023 natural videos, stratified 50/50 split; also GenVidBench, DeMamba, a Veo3 zero-shot set and Physics-IQ (RESEARCH.md row).
- Reported (RESEARCH.md row, PARTIAL): Table 1 (MLP, VidProM test) accuracy 94.17 and AUROC 98.63 (the abstract's 97.17 percent does not match Table 1); unseen-generator accuracy 89.45 and future (Sora) 80.05; Veo3 zero-shot accuracy 83.2 percent; about 48 ms per video on an RTX 2080.
- Limitations stated by the authors: a generator could be trained to fool the measure; the metric is invariant to time reversal; encoder biases.
- Relevance: the method our candidate C1 adapts (8 of 14 frames, our own data and classifier). In our E-020 it was at chance on DeepAction leave-one-generator-out and on DF26, so our adaptation did not reproduce the paper's result on our data; this says nothing about the paper's own protocol. PARTIAL.

### 7.2 Training-free detection via spatial-temporal likelihoods (STALL) (R-079) and second-order features (D3) (R-080)

- STALL: Omer Ben Hayun, Roy Betser, Meir Yossef Levi, Levi Kassel, Guy Gilboa; accepted to CVPR 2026 per the arXiv comments field (proceedings page not checked); arXiv:2603.15026. Whitened DINOv3 frame embeddings with a Gaussian likelihood and a temporal-difference likelihood, calibrated on real videos only; reported average AUC 0.82 over three benchmarks against 0.64 for D3 (RESEARCH.md row). DINOv3 is excluded by our decision D-038.
- D3: Chende Zheng and 8 co-authors; arXiv:2508.00701 (no venue). Standard deviation of second-order feature differences, training-free; mAP 98.46 on GenVideo, 88.46 on VidProM, with a failure on Text2Video-Zero (AP 45.11) (RESEARCH.md row).
- Relevance: training-free alternatives that need only real reference videos; not implemented. PARTIAL (both).

### 7.3 DF26: We Cannot Tell Fake From Real Anymore (R-081)

- Citation: Severyn Shykula, Andrii Yermakov, Ivan Samarskyi, Dmytro Mishkin, Jan Cech, Anastasiia Mishchuk. arXiv:2609.07369 (v1, 7 September 2026), no venue listed.
- Data: 2,691 five-second 1280 x 720 single-person speaking videos: 271 real (OpenVid-1M 197, TalkingCelebs 62, MAVOS-DD 12) and 2,420 generated by three open families (Wan 2.2 A14B, HunyuanVideo 1.5, LTX 2.3 distilled; image-to-video and text-to-video) and four closed generators (Wan 2.6, Veo 3.1, Grok Imagine 1.0, Kling 3.0) (RESEARCH.md row).
- Protocol: pretrained detectors evaluated without training on DF26; frame detectors average "the scores of 32 evenly sampled frames" (Section 6.1); AUROC primary, EER secondary.
- Reported (re-checked 2026-10-27 against the arXiv HTML, Table 5, caption "Cross-dataset video-level AUROC (%) pooled across all samples and EER (%) of state-of-the-art deepfake detectors. All detectors were trained on FF++"): DF26 AUROC / EER: DFD-FCG 48.2 / 51.9; PwTF-DVD 61.6 / 40.8; ForAda 56.6 / 45.8; Effort 44.0 / 55.7; FSFM 52.6 / 48.1; GenD-CLIP 54.2 / 47.7; GenD-PE 69.7 / 35.9; GenD-DINO 54.7 / 47.3; DFD-HR 63.5 / 40.7. Table 3 (temporal detectors, open-source / closed-source generators, AUROC / EER): DFD-FCG 57.4 / 45.2 and 34.5 / 61.7; PwTF-DVD 70.5 / 33.8 and 48.3 / 51.3. Supplementary Table S2 gives different pooled values for the two temporal detectors (DFD-FCG 49.8, PwTF-DVD 63.1), not explained in the text read. The paper says "pooled across all samples"; the arXiv abstract states the benchmark size, "271 real and 2,420 synthetic videos" (abstract page opened 2026-10-27), which equals the 2,691 clips of our E-020 evaluation. Human accuracy 76.0 percent on real and 52.6 percent on fake clips (RESEARCH.md row).
- Limitations stated by the authors: small scale; visual only; generator type may correlate with model family, post-processing and quality; the real pool is mostly OpenVid.
- Relevance: our held-out evaluation set (E-020) and, under the licence's open-generator exception, the development pool of E-027. PARTIAL.

### 7.4 Human Action CLIPs: Detecting AI-generated Human Motion (DeepAction) (R-082)

- Citation: Matyas Bohacek, Hany Farid. Workshop on Deepfake Detection, Localization and Interpretability at IJCAI 2025 (arXiv journal reference). arXiv:2412.00526.
- Data: 3,100 generated clips from 7 text-to-video models and 100 real Pexels clips matched by action; 100 actions; 80/20 split by action prompt, 5 repetitions (RESEARCH.md row).
- Method: CLIP-family frame embeddings with an SVM; a video is fake if most frames are.
- Reported (RESEARCH.md row): video-level accuracy, CLIP linear SVM 97.0 percent, fine-tuned CLIP RBF 99.1 percent; leave-one-model-out average 95.2 percent; 55.8 percent macro accuracy on DeepSpeak talking-head deepfakes.
- Limitations stated by the authors: real clips from one source (Pexels); SVM biased towards fake; fails on talking-head deepfakes. Our audits add: container metadata separates real from generated perfectly in the files as distributed (E-018), and the frame rates differ by source.
- Relevance: our development corpus (E-018 to E-020, E-026). PARTIAL.

### 7.5 What the literature does not provide

No paper read reports a detector trained on one generated-video corpus and tested on DF26 with a source-confound control like E-026; the DF26 numbers above come from face-forgery detectors trained on FF++.

## 8. Task 7: multimodal and audio-visual deepfake detection (our audio-visual mode)

Our setting: no fused verdict for video. The frozen system shows a visual and an audio score separately. Late fusion was tested once, on a LAV-DF subset (E-010 to E-012), and the frozen system was scored zero-shot on the sealed identity half of FakeAVCeleb (E-015) with per-modality labels.

### 8.1 FakeAVCeleb (R-014)

- Citation: Hasam Khalid, Shahroz Tariq, Minha Kim, Simon S. Woo. "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset". NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2108.05080 (v4).
- Data: 500 real videos (VoxCeleb2 celebrities of four ethnic groups) and 19,500 fake videos; four label combinations (RVRA, RVFA, FVRA, FVFA); fake video from face swap (FaceSwap, FSGAN) and Wav2Lip lip sync; fake audio from SV2TTS voice cloning (RTVC). Access by request form.
- Reported (RESEARCH.md row, appendix tables): AUC on its own test set containing real and all three fake categories: audio-only best about 97.8 (VGG), video-only best 93.3 (EfficientNet-B0), ensembles of unimodal detectors at most 82.8 and often near 50, three published multimodal models 53.3 to 68.8. All trained in-dataset.
- Limitations: 500 real videos only; the splits used by later papers differ; our audits E-021 and E-023 found that codec alone (mpeg4 for fake video, h264 for real video) separates real from fake video at AUC about 0.99 in this dataset, so its AUCs mix manipulation cues and encoding cues.
- Relevance: our second sealed set (E-015). PARTIAL (abstract, dataset construction, appendix results).

### 8.2 LAV-DF and BA-TFD (R-024)

- Citation: Zhixi Cai, Kalin Stefanov, Abhinav Dhall, Munawar Hayat. "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization". DICTA 2022, pp. 1-10, doi 10.1109/dicta56598.2022.10034605. arXiv:2204.06228.
- Data: 136,304 videos (36,431 real, 99,873 fake) of 153 VoxCeleb2 identities; transcript words replaced by antonyms, audio regenerated with SV2TTS and video with Wav2Lip for the changed segments; identity-independent split: train 78,703 videos (91 identities), validation 31,501 (31), test 26,100 (31).
- Reported (RESEARCH.md row, READ): classification AUC on the full LAV-DF test: BA-TFD 0.990, EfficientViT 0.965, MDS 0.828, F3Net 0.520. Localisation AP@0.5 76.90 (multimodal) against 58.55 (visual-only variant) on the full test set, 85.20 and 83.55 once audio-only fakes are removed.
- Limitations: one audio generator and one video generator; one split, no seeds.
- Relevance: our fusion iteration (I-5, I-6) used a balanced 2,000-video subset per split of the official splits; the visual-only gap on audio-only fakes explains why a visual detector cannot see audio-only fakes. READ.

### 8.3 AV-Deepfake1M (R-025)

- Citation: Zhixi Cai, Shreya Ghosh, Aman Pankaj Adatia, Munawar Hayat, Abhinav Dhall, Tom Gedeon, Kalin Stefanov. ACM MM 2024, doi 10.1145/3664647.3680795. arXiv:2311.15308.
- Reported (RESEARCH.md row, READ): in-dataset detection AUC, Xception 68.68 on AV-Deepfake1M against 83.58 on LAV-DF (each trained and tested within its own dataset); aggregation matters (Xception mean of frames 61.44, mean of top five 68.81, max 68.68).
- Relevance: shows that in-dataset numbers from different datasets are not comparable and that frame-to-video aggregation is a protocol choice. READ.

### 8.4 Fusion methods trained on FakeAVCeleb (RESEARCH.md rows)

- MRDF (R-022; Zou et al., ICASSP 2024, pp. 4900-4904, doi 10.1109/icassp48485.2024.10447248, Crossref 2026-10-27): 5-fold identity-independent cross-validation on a balanced FakeAVCeleb set, ACC 94.05 and AUC 92.43; the plain fused baseline is weakest on real audio with fake video (70.0 percent accuracy). READ.
- AVFF (R-023; Oorloff et al., CVPR 2024, pp. 27092-27102, doi 10.1109/cvpr52733.2024.02559, Crossref 2026-10-27): FakeAVCeleb 70/30 split, ACC 98.6 and AUC 99.1; trained on FakeAVCeleb and tested on a KoDF subset, AUC 95.5. Requires both modalities. PARTIAL.
- MIS-AVoiDD (R-027; Katamneni and Rattani, ICMLA 2023, pp. 1371-1378, doi 10.1109/icmla58977.2023.00207, Crossref 2026-10-27): FakeAVCeleb AUC 0.973; KoDF 0.962. PARTIAL, table reconstructed from a garbled extraction.
- Joint Audio-Visual Deepfake Detection (R-020; Zhou and Lim, ICCV 2021): joint late fusion was no better than independent streams combined by an OR rule (whole-sequence accuracy 94.25 against 95.83 on FF++). READ.
- Common limitation: every one of these is trained and tested on FakeAVCeleb (or its relatives) with its own split; none controls for the codec shortcut that E-021 found.

## 9. Task 8: provenance, C2PA and metadata (our content_credentials and media_metadata modules)

Our setting: `content_credentials` verifies an embedded C2PA manifest against an operator-configured trust anchor and reports credentials_valid, credentials_invalid, no_credentials_found or inconclusive; `media_metadata` reports container and stream facts for information only. Neither is used as detection evidence.

### 9.1 C2PA Technical Specification, version 2.2 (R-086, added 2026-10-27)

- Citation: Coalition for Content Provenance and Authenticity (C2PA). "Content Credentials: C2PA Technical Specification", version 2.2, May 2025 (version history, Section 5.2.1). https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html (c2pa.org redirects there). A standard, not a peer-reviewed paper.
- What it defines (quoted from the HTML, 2026-10-27): a C2PA Manifest is "The set of information about the provenance of an asset based on the combination of one or more assertions" (Section 2.3.4); "A claim gathers together all the assertions about an asset at a given time" (Section 10.1); authenticity is "A property of digital content comprising a set of facts (such as the provenance data and hard bindings) that can be cryptographically verified as not having been tampered with" (Section 2.3.10).
- What validation does not establish (Section 1.2, guiding principles): "C2PA specifications SHOULD NOT provide value judgments about whether a given set of provenance data is 'good' or 'bad,' merely whether the assertions included within can be validated as associated with the underlying asset, correctly formed, and free from tampering."
- Trust: signer identity is tied to the signing certificate, and trust is anchored in a list of X.509 trust anchors (Sections 1.3.1, 2.3.15).
- Limitations: only the first 100,000 characters (scope, definitions, overview) were read; the trust model and validation sections (14 and 15) were not read in full.
- Relevance: supports our wording that a valid credential shows integrity and signer identity relative to a trust anchor, not that the content is true, and that an untrusted signer is inconclusive. PARTIAL (Sections 1.2, 1.3, 2.3 definitions, 5.2.1 version history, 10.1).

### 9.2 Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets (R-046)

- Citation: Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, Janis Keuper. ECCV 2024 Workshops, LNCS, doi 10.1007/978-3-031-92089-9_6. arXiv:2403.17608.
- Reported (RESEARCH.md row, READ): a ResNet-50 trained on the GenImage Midjourney subset classifies natural FFHQ images as natural with 80.45 percent accuracy for PNG and 94.84 to 100.0 percent for JPEG quality 95 to 60, so JPEG compression alone acts as a "real" label; removing JPEG and size biases raised mean cross-generator accuracy by about 11 points.
- Relevance: the published precedent for our metadata and format audits: E-018 (container metadata separates DeepAction real from generated clips at AUC 1.0000), E-019 (canonical re-encode protocol), E-021 and E-023 (codec shortcut in FakeAVCeleb). READ.

### 9.3 Gaps

No peer-reviewed paper that measures C2PA adoption, stripping rates or verification failure rates was read, so no number about C2PA prevalence is stated anywhere in the project.

## 10. What this survey does not cover, and reading status

- Papers cited in this file: 46 (R-001, R-002, R-004, R-011 to R-015, R-017, R-020 to R-035, R-037 to R-040, R-044 to R-049, R-053, R-068, R-078 to R-086; the rPPG IDs named in the next item are pointers, not citations). By status in docs/RESEARCH.md: READ 13 (R-020, R-022, R-024, R-025, R-028, R-029, R-030, R-032, R-033, R-034, R-044, R-046, R-053), PARTIAL 33. The four added on 2026-10-27 (R-083 to R-086) are all PARTIAL. Every PARTIAL paper is cited only for the sections listed in its RESEARCH.md status cell.
- Not retrieved and therefore not cited: AVoiD-DF, DF40, the rPPG papers listed as NR in docs/RESEARCH.md. rPPG detection (our I-7, not adopted) is covered in docs/RESEARCH.md rows R-005 to R-010 and R-054 to R-067 and is outside the eight tasks of this survey.
- No paper read reports, for the same test items as ours: Celeb-DF v2 without the 15 development identities, In-the-Wild restricted to 42 speakers, or AUC (rather than EER) on In-the-Wild. That is why most external comparisons in docs/PERFORMANCE_BENCHMARK.md are partially comparable.
- Claims dropped during writing because they could not be verified in the sections read: per-generator AP values of the UnivFD table (the HTML rendering scrambles the columns; only totals are quoted); per-model In-the-Wild EER cells of Muller et al. Table 1 (scrambled; only the running-text values are quoted); the training compression of SBI and of LipForensics' cross-dataset table (not stated in the text read); the number of clips in DF26's pooled AUROC (not stated in the text read).
