# Conformal and FPR-controlled thresholds for deepfake detection: novelty check (Nov 2026)

Proposed contribution: set a deepfake detector's threshold with split-conformal quantiles on REAL-ONLY target-domain clips, show that source-calibrated thresholds break the FPR target under codec, JPEG and cross-corpus shift, and report the TPR cost.

Searched: arXiv, Zenodo, ISCA archive, Project Euclid, general web. About 15 tool calls. Several hits are 2026 preprints or Zenodo records that I could only read at abstract or record-page level; this is flagged per item.

## Q1. Which papers apply conformal prediction or conformal risk control to deepfake, forensic or spoof detection?

### Takeaway
Only two items apply conformal methods to deepfake detection, and neither does exactly the proposed thing. One uses conformal p-values as an uncertainty feature (no FPR control). The other (PGA-G, speech, Oct 2026) does "conformal calibration of operating points" from a genuine-speech reference, which is the nearest prior work, but its public record does not say what error it controls or which shifts it tests.

### Cited Findings
- **PGA-G: Selective Adaptation and Conformal Calibration for Speech Deepfake Detection.** Saed Alqaraleh, Zenodo software and evidence archive, published 26 Oct 2026. Decides when a frozen speech deepfake detector should adapt, using unlabelled deployment data plus a genuine-speech reference. Has three parts: a prevalence-based adaptation gate, consistency screening for reference-dependent shortcuts, and "a genuine-reference procedure [that] sets operating points under stated assumptions." Datasets: ASVspoof 2021 DF and ASVspoof 5; detectors SSL-AASIST, XLSR-Mamba, AASIST. The record page does not say which error is controlled, whether the reference comes from the target domain, or which shifts are tested. The journal manuscript is not in the archive. Closeness: **partial, possibly close** (real-only reference plus conformal operating point, speech only).: [Zenodo 23652611](https://zenodo.org/records/23652611); [Zenodo 23660842](https://zenodo.org/records/23660842)
- **Architecture-Adaptive Uncertainty Fusion for Deepfake Detection.** Sharma, Ghasemigol, Motai, arXiv 2606.06666, June 2026. Uses split-conformal p-values (nonconformity = 1 minus predicted-class probability) as one of five uncertainty sources fused to predict errors. Calibration set is a held-out validation split (in-domain, labelled, both classes). Conformal correlation with errors drops from 0.405 (FF++) to 0.083 (CelebDF) and -0.040 (DFDC). On DFDC all eleven architectures call 100% of samples fake. No FPR control. Closeness: **different**.: [arXiv 2606.06666](https://arxiv.org/html/2606.06666v1)
- **Robust and Calibrated Detection of Authentic Multimedia Content.** Hashmi, Elsayed, Alam, Poppi, Lukas (MBZUAI), arXiv 2512.15182, Dec 2025. Resynthesis-based check that certifies authenticity in a high-precision, low-recall setting; claims "low and controllable" FPR. The abstract does not say how the threshold is set.: [arXiv 2512.15182](https://arxiv.org/abs/2512.15182)
- **Certification of Real Images through Calibrated Content Authentication.** Same MBZUAI group, arXiv 2610.05870, 5 Oct 2026. Sets a per-generator threshold at the (1 - alpha) quantile (alpha = 1%) of scores on held-out GENERATED samples, so "at most 1% of generated calibration samples are certified", with a confidence bound. "Calibration happens entirely offline, before deployment." Tests 24 transforms including JPEG (JPEG q10 nearly reaches the threshold), PGD attacks, unseen generators, a Reddit corpus and Deepfake-Eval-2024 video. Reports coverage cost: on Reddit the SD2.1 threshold certifies 37.2% of images while 2024-era thresholds certify 1.83% to 2.63%. Closeness: **partial**. It is quantile calibration with an FPR bound, but calibrated on fakes (to bound fakes passed as real), on the source side, offline; not real-only target calibration.: [arXiv 2610.05870](https://arxiv.org/html/2610.05870v1)
- **Estimating the confidence of speech spoofing countermeasures** (Xin Wang, Junichi Yamagishi, ICASSP 2022 per my recall; title and venue not confirmed on the page I fetched). Adds energy-based and NN confidence estimators so a CM "can defer a decision when it is not confident." Selective prediction, not conformal, no FPR guarantee. Closeness: **different** (abstention).: [arXiv 2110.04775](https://arxiv.org/abs/2110.04775)
- No conformal face anti-spoofing paper turned up. One-class FAS work trains on live faces only (Hyp-OC; OC-SCMNet, CVPR 2024) but does not give a conformal or formal FAR bound.: [Hyp-OC arXiv 2404.14406](https://arxiv.org/pdf/2404.14406); [OC-SCMNet CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_One-Class_Face_Anti-spoofing_via_Spoof_Cue_Map-Guided_Feature_Learning_CVPR_2024_paper.pdf)

### Inferences
- The conformal-for-deepfake literature is thin and almost all of it is from 2026. Expect concurrent work.
- PGA-G is the item a reviewer would cite against the speech version of the claim. Its record suggests it calibrates on genuine speech, but without the manuscript it is unknown whether it targets FPR (bona fide rejection) or spoof acceptance.

### Gaps
- PGA-G manuscript, exact guarantee and shift types were not visible. Worth reading its repository before claiming novelty for audio.
- I did not run a dedicated search for "conformal presentation attack detection" or "conformal risk control forensics". A residual chance remains of a missed workshop paper.

## Q2. Does any work use real-only (one-class) target-domain data to calibrate the threshold?

### Takeaway
Yes, in two very recent places. A preregistration (Nov 2026) plans exactly "recalibrate an AI-image detector's NP threshold on n real photos per processing chain and measure the detection cost", but has no results yet. PGA-G uses a genuine-speech reference. In general ML the method itself (conformal outlier p-values, NP umbrella order statistic) is textbook.

### Cited Findings
- **How Many Real Photos Does It Take? Finite-Sample False-Alarm Control of Frozen AI-Image Detectors under Paired Camera-Processing Shift (Preregistration v1.0).** Mohammed Mahdaoui, Zenodo, 1 Nov 2026 (a newer version exists). Six frozen AI-image detectors. H1: an NP threshold calibrated on a reference processing chain fails to hold FPR on other chains (denoising, learned restoration, JPEG, resizing, demosaicing). H2: recalibration needs n >= 59 real photos for alpha = delta = 0.05, and the study measures the detection-rate cost. H3: blind image-computed groups vs a single worst-group threshold. Data: 300 RAW photos (MIT-Adobe FiveK), 306 images from 9 generators (Community Forensics). No results; analysis script tested on simulated data only. Closeness: **same idea, still images, protocol only**.: [Zenodo 23863746](https://zenodo.org/records/23863746)
- **ConformalGuard: False-Positive-Controlled Action Gating for LM Classifiers**, arXiv 2610.14365, Oct 2026. Calibrates on confirmed-legitimate emails only and gives finite-sample control of how often legitimate inputs get flagged; calls this "a standard conformal guarantee". Shows the recipe is now routine in adjacent security tasks. Closeness: **same recipe, different domain**.: [arXiv 2610.14365](https://arxiv.org/html/2610.14365v1)
- **Testing for outliers with conformal p-values.** Bates, Candes, Lei, Romano, Sesia, Annals of Statistics 51(1), 2023 (arXiv 2021). Conformal p-values against an inlier-only reference set; marginal FPR control; calibration-conditional p-values for stronger guarantees; a uniform confidence bound on FPR of any detector as a function of its threshold. Notes the marginal guarantee only holds on average over calibration sets.: [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-1/Testing-for-outliers-with-conformal-p-values/10.1214/22-AOS2244.full); [arXiv 2104.08279](https://arxiv.org/pdf/2104.08279)
- **Neyman-Pearson classification algorithms and NP receiver operating characteristics.** Tong, Feng, Li, Science Advances 4(2), 2018. Umbrella algorithm: picks the threshold as an order statistic of class-0 scores so type I error <= alpha with probability >= 1 - delta, for any scoring classifier. Shows that capping empirical type I error at alpha gives true type I error well above alpha. R package nproc. (2018 is just before the 2019 window, included as foundation.): [PMC5804623](https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623)
- Conformal p-value thresholds that keep false alarm <= alpha with probability >= 1 - delta are also used in OOD detection.: [arXiv 2501.12204](https://arxiv.org/pdf/2501.12204)
- A text AI-use detection framework shows that flagging at a fixed watermark p-value can miss the FPR target, while conformal p-values on rule-following essays came out roughly uniform (4.69% below alpha = 0.05).: [arXiv 2507.23113](https://arxiv.org/html/2507.23113v1)

### Inferences
- The math is not new. Split-conformal on inliers equals the Bates et al. marginal guarantee; the NP umbrella gives the high-probability version. Any novelty must be empirical and domain-specific.
- The Mahdaoui preregistration is the most direct threat for the image/JPEG part. If its results appear before the user's work, the JPEG and processing-chain story for still images is covered. It does not cover video, codecs (H.264/H.265), audio-visual detectors or cross-corpus deepfake video.

### Gaps
- Could not see whether the newer version of the Mahdaoui record contains results.

## Q3. Do deepfake papers show thresholds fail to transfer even when AUC holds?

### Takeaway
Yes, this is now well documented in 2026, so "source thresholds break under shift" alone is not a contribution. It is a motivating result to reproduce on the user's own setting.

### Cited Findings
- **When AUC Misleads: Polarization-Aware Evaluation of Deepfake Detectors under Domain Shift.** Nguyen, Radoi, Hermary, Astrid, Mejri, Ghorbel, Aouada (Univ. of Luxembourg), arXiv 2606.19184, June 2026. Per-dataset optimal thresholds vary widely: Xception from 0.24 to 0.70 (std 0.16); LAA-Net from 0.03 (DFDC) to 0.96 (DFD) (std 0.31) even with 98.47 AUC on DFD. Says the threshold discrepancy "will cause lower performance than expected after deployment." Proposes Cross-AUC (mean AUC plus a Wasserstein polarity term). No calibration fix, no real-only or conformal method, and main text reports no FPR at a fixed threshold. Closeness: **partial (motivation only)**.: [arXiv 2606.19184](https://arxiv.org/html/2606.19184v1)
- **LAION-Mobile: Evaluating Deepfake Detectors On One Million Smartphone Photos.** von Stryk, Keuper, arXiv 2609.11134, 10 Sep 2026. Twelve detectors. Thresholds fitted on older GAN data give real-photo FPR under 11%; refitting the same criterion on modern content flags 17% to 91% of real photos. No detector both beats chance on modern AI content and keeps a deployable real-photo FPR. No conformal method. Closeness: **partial (shows FPR drift, images)**.: [arXiv 2609.11134](https://arxiv.org/abs/2609.11134)
- **Continuously Evolving Deepfake Detection** (arXiv 2607.13234, 2026) reports TPR at 1% FPR and ECE alongside AUC and finds calibration strongly dataset-dependent (from search snippet only; not opened).: [arXiv 2607.13234](https://arxiv.org/pdf/2607.13234)
- Outside deepfakes, a backdoor-detection paper finds the signal survives shift but the threshold drifts, and recommends recalibrating the operating point on deployment traffic (search snippet only).: [arXiv 2609.11799](https://arxiv.org/pdf/2609.11799)
- **What Do Deepfake Speech Detectors Actually Hear?** (arXiv 2606.10912, 2026) reports bona fide FAR/FRR at a fixed threshold taken from the baseline EER (search snippet only).: [arXiv 2606.10912](https://arxiv.org/html/2606.10912v1)

### Inferences
- The "AUC holds but threshold fails" claim is established for image and face-swap benchmarks. The user should cite these and not frame it as a finding.

### Gaps
- No paper found that quantifies FPR breakage specifically under video codec re-encoding (H.264/H.265 CRF sweep) or for audio-visual detectors. That looks open.

## Q4. ASVspoof fixed-threshold or a-priori threshold evaluation

### Takeaway
ASVspoof has long scored systems at a threshold fixed before seeing evaluation data (actDCF at the Bayes threshold from calibrated LLRs), and ASVspoof 5 shows large minDCF to actDCF gaps. This is calibration of LLRs with known priors and costs, using labelled spoof and bona fide data, not real-only distribution-free FPR control.

### Cited Findings
- **ASVspoof 5 evaluation paper** (arXiv 2601.03944, Jan 2026). actDCF uses a threshold fixed in advance, tau_Bayes = -log(beta), beta about 1.90; "the actDCF is a measure of realised cost when the threshold is set before observation of either evaluation data or labels." Some top open-condition systems had actDCF near 1.0 despite low minDCF; teams with actDCF 1.0 submitted probability-like scores. Calibration-aware metrics (actDCF, Cllr) appear for the first time in the series.: [arXiv 2601.03944](https://arxiv.org/html/2601.03944)
- **ASVspoof 5 database paper** (arXiv 2408.08739, 2024) defines minDCF, actDCF and Cllr.: [arXiv 2408.08739](https://arxiv.org/pdf/2408.08739)
- **t-DCF** (Kinnunen et al., Odyssey 2018; tandem fundamentals, IEEE/ACM TASLP 2020) defines tandem CM plus ASV cost with spoof false alarm rate; **a-DCF** (arXiv 2403.01355, Interspeech 2024) is an architecture-agnostic cost with a spoof class.: [arXiv 1804.09618](https://arxiv.org/pdf/1804.09618); [arXiv 2007.05979](https://arxiv.org/pdf/2007.05979); [arXiv 2403.01355](https://arxiv.org/pdf/2403.01355)

### Inferences
- The ASVspoof actDCF result is the speech community's version of "the threshold does not transfer." The proposed method is a different fix: instead of calibrating LLRs on labelled source data, it re-sets the threshold from target bona fide data alone and gives a finite-sample FPR bound. That contrast is worth stating in the paper.

### Gaps
- Did not verify the Wang et al. 2026 ASVspoof 5 calibration analysis (only seen on an aggregator).

## Verdict

**Partially covered.** The statistics (split-conformal on inliers, NP order statistic) are standard since 2018 to 2023. The empirical claim that thresholds break under shift while AUC holds is published for deepfake images (2606.19184, 2609.11134). Real-only target recalibration with a finite-sample FPR bound is planned for AI-generated still images under JPEG and camera-processing shift (Mahdaoui preregistration, Nov 2026, no results yet), and something close exists for speech deepfakes (PGA-G, Oct 2026, details not public on the record page). Calibration on fakes with a quantile bound exists for image authenticity certification (2610.05870).

**Sharpest framing that stays novel:**
"For audio-visual deepfake VIDEO detectors, real-only conformal recalibration on target-domain clips restores a stated false-alarm rate under video codec re-encoding (H.264/H.265 bitrate or CRF), JPEG frame compression and cross-corpus shift (for example FakeAVCeleb -> other AV corpora). We measure how many real target clips are needed (finite-sample bound, n for given alpha and delta), report the TPR cost per shift, and show per-modality (audio, video, fused) where the cost lands."

Points that keep it distinct:
- Video and audio-visual, not still images or speech-only.
- Codec shift as a named axis; no found paper tests FPR breakage under video codecs.
- A clip-level real-only budget (how many real clips per domain), plus the PAC-style (alpha, delta) version, not only marginal coverage.
- TPR cost reported as a per-shift table, ideally split by fake type, so the trade-off is concrete.
- Cite Bates et al. and Tong et al. for the method, 2606.19184 and 2609.11134 for the motivation, the Mahdaoui preregistration and PGA-G as concurrent work, and ASVspoof actDCF as the labelled-calibration alternative.

Do not claim novelty for: the conformal method itself; the observation that source thresholds fail under shift; real-only recalibration as a general idea.
