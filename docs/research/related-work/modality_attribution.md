# Modality attribution in audio-visual deepfake detection (novelty check, Nov 2026)

Proposed contribution under test: a cue-attributing fusion model that predicts which modality was manipulated. It trains on per-modality labels from FakeAVCeleb (RVRA/RVFA/FVRA/FVFA) and LAV-DF (modify_audio/modify_video). It is evaluated cross-corpus (train on one dataset, test attribution on the other) under a shortcut-controlled protocol.

Closeness scale used below: same = already does the proposed thing; partial = does one or two parts; different = adjacent only.

## Which papers already train multi-label or per-modality heads on FakeAVCeleb, LAV-DF, AV-Deepfake1M or similar?

### Takeaway
Per-modality heads are not new. At least three lines of work already train them: Multimodaltrace (multi-label head, CVPRW 2023), Yu et al.'s dual-label classifier (FakeAVCeleb, DFDC and LAV-DF, ACM IH&MMSec 2025), and MRDF (per-modality targets as regularizers, ICASSP 2024 track, arXiv 2401.05746). The big benchmarks (1M-Deepfakes challenge, AV-Deepfake1M++) score only binary detection and temporal localization, not attribution.

### Cited Findings

Per-paper list (title, venue and year, URL, what it does, closeness):

- **Multimodaltrace: Deepfake Detection Using Audiovisual Representation Learning.** Raza and Malik, CVPR Workshops (WMF) 2023. Frames AV deepfake detection as multi-label classification with one sigmoid output for audio and one for video. It is trained on FakeAVCeleb. It reports per-class precision, recall and F1 for the four classes ARVR / AFVR / ARVF / AFVF (F1 0.872 / 0.908 / 0.945 / 0.917). Overall accuracy is 0.929 for the multi-label head and 0.903 for a 4-way multi-class head. Its cross-dataset tests go to World Leaders (FaceSwap 75.84, LipSync 83.33, Imposter 91.66) and the Presidential Deepfakes Dataset (70.00). Those numbers are binary accuracy, not attribution, and the multi-label head beats the multi-class head on every subset. Closeness: **partial**. It uses per-modality labels on FakeAVCeleb, but it does no LAV-DF and no cross-corpus attribution. -- [NSF PAR full text](https://par.nsf.gov/servlets/purl/10427043); [CVF open access](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Raza_Multimodaltrace_Deepfake_Detection_Using_Audiovisual_Representation_Learning_CVPRW_2023_paper.pdf); [mlanthology entry](https://mlanthology.org/cvprw/2023/raza2023cvprw-multimodaltrace)
- **A Unified Framework for Modality-Agnostic Deepfakes Detection** (published as "Modality-Agnostic Deepfakes Detection"). Cai Yu, Peng Chen, ..., Shan Jia, Siwei Lyu, Jizhong Han. arXiv July 2023, then ACM IH&MMSec 2025. It uses a "dual-label" classifier: each clip gets y_m in {0,1} for audio and for video (for example [0,1] = real audio, fake video). An AVSR-pretrained backbone plus a modality compensation adapter handle missing modalities. Datasets are DFDC (they build audio labels by hashing audio against the originals), FakeAVCeleb and LAV-DF. Metrics are overall F1 (OF1), per-class F1 (CF1), weighted F1 (WF1), plus per-modality accuracy and F1 (VACC, AACC, VF1, AF1). Table X ("Generalization study") trains on DFDC and tests on FakeAVCeleb, and the reverse, and reports per-modality accuracy. Full model: Cross-FakeAVCeleb OF1 68.57, VACC 80.00, AACC 48.89; Cross-DFDC OF1 58.57, VACC 61.42, AACC 94.07. It argues that binary methods "cannot specifically predict which modality has been manipulated", but it runs no experiment that isolates label confusion. Closeness: **partial, and the closest prior work**. It already does per-modality heads, FakeAVCeleb plus LAV-DF, and cross-dataset per-modality accuracy. The cross-dataset pair is DFDC<->FakeAVCeleb, not FakeAVCeleb<->LAV-DF, and there is no shortcut control. -- [arXiv abs](https://arxiv.org/abs/2307.14491); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2307.14491); [ACM DL](https://dl.acm.org/doi/10.1145/3733102.3733133)
- **Cross-Modality and Within-Modality Regularization for Audio-Visual DeepFake Detection (MRDF).** Zou, Shen, Hu, Chen, Chng, Rajan (NTU). arXiv Jan 2024 (ICASSP 2024). It keeps a single binary output. Per-modality labels y_a and y_v supervise within-modality regularizers on the audio and video branches before fusion. It is evaluated on FakeAVCeleb only, with a balanced 1:1:1:1 split over FAFV/FARV/RAFV/RARV and 5-fold CV. Its key claim: "joint training with fused information may lead to inaccurate mappings of unimodal data to multimodal targets." Unimodal features that carry a label different from the clip label become "inconsistent" through "backpropagation from opposing multimodal targets." Evidence: Table 4 per-category accuracy. The binary-label baseline AVDF gets 70.0 on RAFV (real audio, fake video), versus 86.0/87.4 with modality-specific regularization; RARV goes from 86.8 to 89.4/89.2. It also re-runs Multimodaltrace: multi-label 92.25 ACC / 89.83 AUC, multi-class 92.60 / 90.93. Closeness: **partial**. It is the clearest prior claim that clip-level labels hurt fusion, but it is in-dataset only and does not output attribution. -- [arXiv PDF](https://arxiv.org/pdf/2401.05746)
- **Shared-Private Cross-Modal Discrepancy Learning for Imbalanced Multi-Source Audio-Visual Deepfake Detection.** Qiwu Wen, Jahangir Alam. arXiv Oct 2026. Trains jointly on FakeAVCeleb and LAV-DF as two known sources, with clip-level binary labels only ("a video is classified as fake if either its audio stream, visual stream, or both are manipulated"). It states that unknown-source inference is out of scope. Metrics: per-source AUC, macro AUC, worst-source AUC, delta AUC. Compares to MRDF. Closeness: **different** (same dataset pair, but no attribution and no cross-corpus test). -- [arXiv HTML](https://arxiv.org/html/2610.25301v1)
- **Attribution-Guided Multimodal Deepfake Detection via Cross-Modal Forensic Fingerprints (AMDD).** Ahmad, Zhang, Mao (BIT). arXiv Apr 2026. Here "attribution" means generator attribution (real / FaceSwap / Wav2Lip / FSGAN on FakeAVCeleb), not modality. Attribution accuracy is 95.9% in-dataset and falls to 11.0% without the attribution loss. Cross-dataset tests (DF-TIMIT, DFDM, LAV-DF) cover detection only and are near chance (LAV-DF video-only fake: balanced acc 52.2, AUC 51.6). Closeness: **different**, but a name clash risk: reviewers searching "attribution FakeAVCeleb LAV-DF" will find it. -- [arXiv abs](https://arxiv.org/abs/2604.26453); [arXiv HTML](https://arxiv.org/html/2604.26453)
- **Joint Audio-Visual Deepfake Detection.** Yipin Zhou, Ser-Nam Lim, ICCV 2021. Separate video and audio streams plus a sync stream. Built on video deepfake data with original audio, plus converted DFDC speech for audio fakes. The search snippets say joint training beats independently trained models and generalizes better to unseen deepfake types. I could not confirm from the text I opened whether it reports per-modality output accuracy. Closeness: **partial** (early per-stream design, no FakeAVCeleb/LAV-DF attribution). -- [CVF open access](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Joint_Audio-Visual_Deepfake_Detection_ICCV_2021_paper.html); [supplement](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Zhou_Joint_Audio-Visual_Deepfake_ICCV_2021_supplemental.pdf)
- **AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection.** Oorloff et al., CVPR 2024. Labels a video fake "if either or both audio and visual modalities have been manipulated". It counts a video as fake for unimodal baselines only if the visual stream is fake. This is the standard clip-level collapse the proposal argues against. Closeness: **different** (useful as the baseline protocol to contrast). -- [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/papers/Oorloff_AVFF_Audio-Visual_Feature_Fusion_for_Video_Deepfake_Detection_CVPR_2024_paper.pdf)
- **From Single-modal to Multi-modal Facial Deepfake Detection: Progress and Challenges** (survey). arXiv 2406.06965 v4. Describes "fine-grained classification" where the model "separately predicts the authenticity of the visual and audio components" (labels Y^v, Y^a). It lists multi-label metrics (mAP) for this. In the part I could read, it names no cross-dataset evaluation of per-modality prediction. Closeness: **different** (confirms the task is a known framing). -- [arXiv HTML](https://arxiv.org/html/2406.06965v4)
- **1M-Deepfakes Detection Challenge** (ACM MM 2024) and **AV-Deepfake1M++** (ACM MM 2025 challenge). There are two tasks: video-level binary classification (AUC) and temporal localization. The top 2025 team reached 0.9783 AUC, versus 0.5509 for the Xception baseline. None of the search results showed a modality-attribution track. Closeness: **different**. -- [1M-Deepfakes challenge](https://arxiv.org/pdf/2409.06991); [AV-Deepfake1M++](https://arxiv.org/abs/2507.20579)
- **Contextual Cross-Modal Attention for AV Deepfake Detection and Localization** (arXiv 2408.01532) trains on FakeAVCeleb and tests on AV-Deepfake1M, LAV-DF and TVIL. Cross-dataset is binary only. **FauForensics** (arXiv 2505.08294) reports both directions of FakeAVCeleb<->LAV-DF transfer, also binary (average cross-dataset gain of 4.83%). Closeness: **different**. They show the cross-corpus pair is standard for detection, not for attribution. -- [2408.01532](https://arxiv.org/html/2408.01532v2); [FauForensics](https://arxiv.org/html/2505.08294v2)
- **Intra-modal and Cross-modal Synchronization for AV Deepfake Detection and Temporal Localization** (ICCV 2025) uses FakeAVCeleb-style scenario categories (real visual + fake audio, and so on) to analyse detectors. It does this per category, but as detection accuracy, not as predicted attribution. Closeness: **different**. -- [CVF open access](https://openaccess.thecvf.com/content/ICCV2025/papers/Anshul_Intra-modal_and_Cross-modal_Synchronization_for_Audio-visual_Deepfake_Detection_and_Temporal_ICCV_2025_paper.pdf)

### Inferences
- "Train a per-modality / multi-label head on FakeAVCeleb" is already done (Multimodaltrace 2023, Yu et al. 2023/2025, MRDF 2024). It cannot be the headline contribution.
- Using both FakeAVCeleb and LAV-DF per-modality labels in one model is done by Yu et al. (they evaluate on both), though they do not transfer between them.

### Gaps
- I did not open the AVoiD-DF, BA-TFD / BA-TFD+, UMMAFormer, MMDet or DiMoDif papers in this pass. From what I know of them, they output a binary label and/or temporal segments, not a modality attribution, but I did not verify this here.
- I did not confirm whether AV-Deepfake1M's metadata ships per-modality flags in the same form as LAV-DF. I believe it does, since it comes from the same group as LAV-DF, but I did not check the README.
- I could not see the full Yu et al. table to check whether they also ran a LAV-DF cross-dataset row. The ar5iv text I read describes only DFDC<->FakeAVCeleb in Table X.

## Does any of them report attribution accuracy cross-dataset? Does any report that clip-level labels hide or harm fusion gains?

### Takeaway
Yes on the first point, but only partly. Yu et al. report per-modality accuracy cross-dataset for DFDC<->FakeAVCeleb, and audio attribution collapses (AACC 48.89 when tested on FakeAVCeleb). I found no paper that reports modality attribution for FakeAVCeleb<->LAV-DF transfer, and none that does it under a shortcut-controlled protocol. MRDF already claims that clip-level labels harm fused representations, with in-dataset evidence only. I found no paper showing that clip-level evaluation hides fusion gains.

### Cited Findings
- Yu et al. Table X: Cross-FakeAVCeleb VACC 80.00 / AACC 48.89; Cross-DFDC VACC 61.42 / AACC 94.07; OF1 68.57 and 58.57. This is much lower than in-dataset. -- [ar5iv](https://ar5iv.labs.arxiv.org/html/2307.14491)
- Multimodaltrace's cross-dataset tests (World Leaders, Presidential Deepfakes) report binary accuracy per test subset, not per-modality attribution. -- [NSF PAR](https://par.nsf.gov/servlets/purl/10427043)
- MRDF: binary-label fusion "may lead to inaccurate mappings of unimodal data to multimodal targets." The AVDF baseline misclassifies 30.0% of real-audio/fake-video clips, versus 14.0% and 12.6% with per-modality regularization. -- [arXiv 2401.05746](https://arxiv.org/pdf/2401.05746)
- AMDD's cross-dataset LAV-DF result is detection only, near chance (52.2 balanced acc). Its "attribution" is generator ID. -- [arXiv 2604.26453](https://arxiv.org/html/2604.26453)
- FakeAVCeleb has a known "silence shortcut". Klemt, Segna and Rohrbach (arXiv June 2025) analyse it and propose a better evaluation scheme for FakeAVCeleb plus an audio-shortcut mitigation. The abstract page gives no effect size and does not mention per-modality labels or LAV-DF. -- [DeepFake Doctor, arXiv 2506.05851](https://arxiv.org/abs/2506.05851)
- The Oct 2026 multi-source paper treats FakeAVCeleb and LAV-DF as known sources and rules out unseen-source inference. So the newest work on this exact pair does not do cross-corpus attribution. -- [arXiv 2610.25301](https://arxiv.org/html/2610.25301v1)

### Inferences
- Nobody I found has done FakeAVCeleb->LAV-DF and LAV-DF->FakeAVCeleb attribution transfer. That is the open cell. The pair is interesting because the label semantics differ. FakeAVCeleb fakes whole clips with face-swap/reenactment and voice cloning. LAV-DF makes local, word-level, content-driven edits with modify_audio/modify_video flags. So attribution transfer tests whether the head learned "which stream is fake" or "which generator family is present".
- No paper ties attribution to shortcut control. The silence shortcut in FakeAVCeleb is exactly the kind of cue that would make an audio-fake head look good in-dataset and fail cross-corpus. Yu et al.'s AACC collapse (48.89) fits that story, but they do not test it.
- The "clip-level labels hide or harm fusion gains" claim is partly covered by MRDF ("harm", in-dataset, one dataset). The "hide" side, where clip-level AUC masks a fusion gain that only shows up in per-modality attribution, appears untested.

### Gaps
- I did not find the DeepFake Doctor full text, so I cannot quote its shortcut magnitude or exact protocol.
- I could not check every 2026 ACM MM / ICASSP paper. A Nov 2026 search of arXiv and CVF turned up nothing closer, but proceedings indexing lags.

## What metric do they use for attribution?

### Takeaway
There is no shared metric. Papers use per-class P/R/F1 over the 4 FakeAVCeleb categories (Multimodaltrace), multi-label F1 variants plus per-modality accuracy (Yu et al.), or per-category detection accuracy (MRDF). Nobody reports per-modality AUC as a cross-corpus attribution metric.

### Cited Findings
- Multimodaltrace: per-class precision, recall and F1 over ARVR/AFVR/ARVF/AFVF, plus overall accuracy. -- [NSF PAR](https://par.nsf.gov/servlets/purl/10427043)
- Yu et al.: OF1, CF1, WF1 for the dual-label task; VACC, AACC, VF1, AF1 per modality. Per-modality AUC appears only in unimodal tables (video-only 99.9 on FakeAVCeleb, audio-only 99.9). -- [ar5iv](https://ar5iv.labs.arxiv.org/html/2307.14491)
- MRDF: per-category accuracy (FAFV/FARV/RAFV/RARV) plus binary ACC/AUC. -- [arXiv 2401.05746](https://arxiv.org/pdf/2401.05746)
- The survey lists mAP for multi-label fine-grained settings. -- [arXiv 2406.06965](https://arxiv.org/html/2406.06965v4)
- AMDD: attribution accuracy (generator ID). -- [arXiv 2604.26453](https://arxiv.org/html/2604.26453)

### Inferences
- A threshold-free per-modality AUC (audio-fake AUC, video-fake AUC) plus 4-class macro-F1 would make results comparable across corpora with different class balances. FakeAVCeleb is heavily skewed (500 real vs 19,500 fake). Accuracy-based numbers from earlier papers cannot be compared directly.

### Gaps
- No paper I found reports calibration or chance-corrected attribution metrics. Whether reviewers expect them is unknown.

## Verdict

**Partially covered.** The parts that are already done:
- per-modality / multi-label heads (Multimodaltrace 2023, Yu et al. 2023/2025);
- per-modality labels as a training signal for fusion (MRDF 2024);
- cross-dataset per-modality accuracy on DFDC<->FakeAVCeleb (Yu et al.).

The parts I found nobody doing as of Nov 2026:
1. Attribution transfer between FakeAVCeleb and LAV-DF in both directions, where the two corpora's label schemes have to be mapped onto one shared (audio-fake, video-fake) target.
2. Attribution measured under a shortcut-controlled protocol (silence or leading-silence trimmed, codec and length matched). The protocol would show how much in-dataset attribution comes from shortcuts.
3. Evidence that clip-level evaluation hides fusion gains. A fusion model that ties a strong unimodal model on clip-level AUC could still beat it on cross-corpus attribution.

Sharpest framing that stays novel: "Which stream is fake does not transfer: a shortcut-controlled, cross-corpus benchmark for modality attribution between FakeAVCeleb and LAV-DF. It shows that per-modality labels reveal (or fail to reveal) fusion gains that clip-level AUC hides." Cite Multimodaltrace, Yu et al. and MRDF as the in-dataset precedent. Present the contribution as the cross-corpus, shortcut-controlled evaluation and the clip-level vs per-modality finding, not as the attribution head itself. Do not call the head "attribution" without qualifying it, because AMDD (2026) uses that word for generator ID on the same datasets.
