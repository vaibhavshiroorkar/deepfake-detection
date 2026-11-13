# Dataset shortcuts and spurious cues in AV and audio-only deepfake datasets

Scope: 2019 to Nov 2026. Every claim has a source link. Numbers are as the paper reports them. Where I could only see an abstract or a secondary summary, I say so.

## 1. Leading silence in FakeAVCeleb and AV-Deepfake1M (Smeu et al., CVPR 2025)

### Takeaway
Fake clips in FakeAVCeleb and AV-Deepfake1M start with about 25 to 30 ms of digital silence; real clips start with noise. A one-feature silence classifier gets about 98% AUC on both. Trimming the silence at test time collapses several supervised AV detectors, and a follow-up (DeepFake Doctor, 2025) confirms it and shows temporal jittering mostly removes the reliance.

### Cited Findings
- Fake samples start with roughly 25 to 30 ms of silence; real videos start with noise. Silence is audio magnitude below tau = 5e-4, and results are robust to the threshold as long as it is small. [Smeu et al., arXiv 2412.00175v3](https://arxiv.org/html/2412.00175v3)
- Leading-silence-only classifier (real-video-real-audio vs fake-video-fake-audio): 98.4% AUC on FakeAVCeleb, 98.2% AUC on AV-Deepfake1M. On the official AV-Deepfake1M test set the silence classifier scores 98.44 AUC. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- Per FakeAVCeleb split: leading silence gives 100% AUC on RVFA (real video, fake audio) and 62.4% on FVRA (fake video, real audio). So the cue rides on the fake audio track. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- Other audio cues in the same paper: trailing silence gives over 99% AUC on FakeAVCeleb and near 50% on AV-Deepfake1M; max volume gives 67.6% AUC on FakeAVCeleb; LAV-DF shows a trailing-silence cue at 65% AUC; AVLips and DFDC showed none of the three cues. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- Fix used for evaluation: cut the start of each clip up to the first non-silent sample, rounded up to a multiple of 1/FPS so audio and video stay in sync. Trimming is applied only to test data. The paper does not give prevalence percentages or the mean trimmed length. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- AUC untrimmed -> trimmed (Table 1). Silence classifier: FakeAVCeleb 98.4 -> 54.8, AV-Deepfake1M 98.2 -> 50.6. RawNet2 (audio): 99.9 -> 97.3 and 99.9 -> 88.1. MDS (AV): 90.4 -> 73.8 and 99.2 -> 54.9. AVAD (unsupervised): unchanged (95.2 and 52.9). [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- Their own supervised variant AVH-Align/sup trained on AV-Deepfake1M drops 100.0 -> 83.1 on AV-Deepfake1M after trimming. The real-only AVH-Align stays at 94.6 -> 94.6 on FakeAVCeleb and 85.9 -> 83.5 on AV-Deepfake1M. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- Smeu et al. release code only ([github.com/bit-ml/AVH-Align](https://github.com/bit-ml/AVH-Align)); the paper does not say trimmed datasets are released. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- DeepFake Doctor (Klemt, Segna, Rohrbach, 2025): leading silence appears in every FakeAVCeleb manipulation that involves fake audio and is absent from real and FVRA samples. Models on consecutive frames without jitter lose on average 10.89 AUC (binary) and 15.76 (multiclass) on trimmed test clips, up to 46.29 on the RVFA split. [DeepFake Doctor, arXiv 2506.05851](https://arxiv.org/html/2506.05851v1)
- DeepSpeak v1 also has the silence cue, concentrated in the Wav2Lip and Retalking splits. Average drops are 5.27 (binary) and 3.00 (multiclass). [DeepFake Doctor](https://arxiv.org/html/2506.05851v1)
- Temporal jittering of the input window shrinks the FakeAVCeleb untrimmed-vs-trimmed gap to between -0.17 and -0.81 AUC, and to near zero or positive on DeepSpeak v1. [DeepFake Doctor](https://arxiv.org/html/2506.05851v1)

### Inferences
- The cue is an audio pipeline artifact (TTS/voice cloning output padded with zeros), not a visual one. Any fusion model that gets the first 30 ms of audio can learn it.
- Trimming only at test time measures reliance. It does not train a model free of the cue. A protocol should trim (or jitter, or random-crop) at both train and test time.
- Trailing silence on FakeAVCeleb (over 99% AUC) is as strong as leading silence and gets less attention. Trimming only the start does not neutralize it.

### Gaps
- No paper reports the share of real vs fake clips with leading silence, or the mean silence length per class.
- No released trimmed copy of FakeAVCeleb or AV-Deepfake1M found; only scripts.
- Neither paper checked LAV-DF fully beyond the 65% trailing-silence AUC.

## 2. Codec, container and bitrate differences between real and fake

### Takeaway
I found no paper that shows codec or container metadata alone separates real from fake in FakeAVCeleb, LAV-DF, DFDC or DeepSpeak. The image-domain analog (JPEG vs PNG) is well documented, and some new benchmarks re-encode both classes to block it. For AV datasets this is an open, testable gap.

### Cited Findings
- Image domain: many AI-generated image datasets have JPEG-compression and image-size biases; detectors on GenImage learn them, and removing them gives over 11 percentage points more cross-generator accuracy for ResNet50 and Swin-T. [Grommelt et al., "Fake or JPEG?", arXiv 2403.17608](https://arxiv.org/abs/2403.17608v1)
- A 2026 survey of AI-generated video detection repeats that early image detectors learned PNG vs JPEG compression differences instead of real-vs-generated content. [arXiv 2601.11035](https://arxiv.org/pdf/2601.11035)
- Music deepfakes: Deezer note that all Riffusion songs are exported as mp3 at 192 kbps, so a detector could learn file format; they re-encode real tracks to match. [Afchar et al., "Detecting music deepfakes is easy but actually hard", arXiv 2405.04181](https://ar5iv.labs.arxiv.org/html/2405.04181)
- Smeu et al. cite prior work that ASVspoof 2019 leaks labels through silence and bitrate information. I could not locate the primary bitrate paper in this pass. [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- CCDF (2026 surveillance benchmark) re-encodes real and fake clips to the same H.264 codec and fixed frame rate to prevent shortcut learning; ten recent detectors fail on it. [CCDF, arXiv 2610.07939](https://arxiv.org/html/2610.07939v1)
- FaceForensics++ compresses real and fake videos with the same H.264 settings (c23, c40), so it avoids a between-class codec gap. [FF++, arXiv 1901.08971](https://arxiv.org/pdf/1901.08971) (as summarized in search; settings are in the paper)
- Local project observation (not a publication): in this repo's FakeAVCeleb copy the Wav2Lip fakes are stored as mpeg4 (Part 2) video, while other clips are not; see the project memory note `fakeavceleb-mixed-codecs`. This needs an ffprobe census before it can be cited.

### Inferences
- If FakeAVCeleb's Wav2Lip fakes use a different codec than its real clips, a detector can separate them from bitstream statistics (block artifacts, GOP structure) without ever looking at the face. This would be a second, visual-side shortcut next to the audio silence cue.
- An ffprobe census (codec, profile, bitrate, frame rate, sample rate, audio codec, duration) per class across FakeAVCeleb, LAV-DF, AV-Deepfake1M, DeepSpeak and DFDC, plus a metadata-only classifier AUC, would be a cheap and new result.

### Gaps
- No published per-class codec/container/bitrate census for any AV deepfake dataset found.
- No paper found that trains a classifier on container metadata alone for AV deepfakes.
- The ASVspoof 2019 bitrate leak is cited secondhand by Smeu et al.; primary source not confirmed here.
- One search snippet (DiMoDif, arXiv 2411.10193) says "these differences are significant on FakeAVCeleb, smaller on LAV-DF, much smaller on AV-Deepfake1M"; I could not confirm what "differences" it measures, so I do not rely on it. [DiMoDif](https://arxiv.org/html/2411.10193v2)

## 3. Silence and duration shortcuts in ASVspoof 2019 LA, and In-the-Wild

### Takeaway
Müller et al. (2021) showed bonafide ASVspoof 2019 LA clips have longer leading and trailing silence than spoofs. Leading-silence duration alone gives 15.1% EER, and trimming silence pushes RawNet2 EER from 3.6% to 15.5%. Their 2022 In-the-Wild paper then showed ASVspoof-trained models fall to 34 to 66% EER on real-world data.

### Cited Findings
- Bonafide instances have significantly longer leading and trailing silences than spoofs. TTS systems usually emit no leading or trailing silence. [Müller et al., arXiv 2106.12914](https://arxiv.org/html/2106.12914)
- A small network fed only leading-silence duration: 15.12% eval EER (31.09% on dev), 19.93% on LA21 and 17.42% on DF21 progress sets; abstract says up to 85% accuracy. [Müller et al.](https://arxiv.org/html/2106.12914)
- With vs without silence trimming (ASVspoof 2019 eval EER): ResNet 5.87 -> 27.23, CNN-GRU 7.48 -> 26.27, LSTM 8.33 -> 27.28, RawNet2 3.61 -> 15.50. Trimming only eval data raises LSTM EER from 7.35 to 35.32. [Müller et al.](https://arxiv.org/html/2106.12914)
- Their recommendations: release a version with silence removed (librosa or Montreal Forced Aligner) and balance silence duration across classes and attack types. [Müller et al.](https://arxiv.org/html/2106.12914); venue: [ASVspoof 2021 workshop, ISCA](https://www.isca-archive.org/asvspoof_2021/muller21_asvspoof.html)
- In-the-Wild: 37.9 hours (17.2 h fake, 20.7 h real) of celebrities and politicians. [Müller et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.html)
- Best EER on ASVspoof19 eval vs In-the-Wild: RawGAT-ST 1.23 -> 37.15, RawNet2 3.15 -> 33.94, RawPC 3.07 -> 45.72, LCNN 6.35 -> 65.56. Motivation: all ASVspoof splits come from VCTK and share properties with training data; they cite the silence paper. [Müller et al., arXiv 2203.16263v4](https://arxiv.org/html/2203.16263v4)
- cqtspec or logspec features beat melspec by 37% EER on average; full-length input beats fixed 4 s. [Müller et al. 2022](https://arxiv.org/abs/2203.16263v4)

### Inferences
- The AV leading-silence cue (section 1) is the same failure mode as ASVspoof 2019 four years later. The AV community did not import the audio fix. That history is a strong motivating line for a protocol paper.
- Duration itself (clip length per class) is untested on AV datasets as a standalone cue.

### Gaps
- Exact mean silence durations per class are only in a figure, not in text.
- No AV paper reports a clip-duration-only classifier.

## 4. Compression robustness and identity leakage

### Takeaway
Compression robustness is well studied for video detectors (FF++ c23/c40). Identity leakage is documented for FF++ (Dong et al., CVPR 2023) but I found no study measuring identity leakage in FakeAVCeleb or other AV datasets.

### Cited Findings
- FF++ XceptionNet accuracy: 99.41% raw, 97.53% c23, 85.49% c40. [Rössler et al., FF++, arXiv 1901.08971](https://arxiv.org/pdf/1901.08971)
- Celeb-DF: FWA and DSP-FWA degrade sharply on recompressed video (e.g. 56.7% at c23, 54.0% at c40 for a raw-trained model), while Xception trained on compressed data holds up. [Li et al., Celeb-DF, arXiv 1909.12962](https://arxiv.org/pdf/1909.12962)
- Preprint: training Xception on real H.264 CRF variants reaches 0.963 AUC at c40 vs 0.890 with synthetic JPEG augmentation (+7.3 points). Single unreviewed preprint. [arXiv 2608.28685](https://arxiv.org/html/2608.28685)
- Social-network compression emulation needs fewer than 50 uploaded videos per resolution to fit a platform's re-encoding. [arXiv 2508.08765](https://arxiv.org/html/2508.08765v2)
- Audio: compression hurts audio deepfake detectors even at high perceptual quality, especially neural codecs and low-bitrate MP3, even for foundation models. [arXiv 2503.17577](https://arxiv.org/pdf/2503.17577)
- Implicit Identity Leakage: binary-supervised detectors are sensitive to identity, and this learned identity representation hurts generalization; CADDM (ID-unaware, artifact detection + multi-scale facial swap) beats SOTA in-dataset and cross-dataset. Trained on FF++. I could only read the abstract; specific numbers not extracted. [Dong et al., CVPR 2023, arXiv 2210.14457](https://arxiv.org/abs/2210.14457); code [megvii-research/CADDM](https://github.com/megvii-research/CADDM)
- Counterpoint: identity can help. Identity-Driven DeepFake Detection frames detection as identity verification. [arXiv 2012.03930](https://ar5iv.labs.arxiv.org/html/2012.03930)
- DeepFake Doctor: FakeAVCeleb split by identity 60/10/30. The original FakeAVCeleb leave-one-out protocol leaked Wav2Lip content (real and fake Wav2Lip audio separated) and leaked artifacts through multi-manipulation fakes (e.g. Wav2Lip in both FSGAN+Wav2Lip and Wav2Lip). They redesign it as method and family splits. [DeepFake Doctor](https://arxiv.org/html/2506.05851v1)

### Inferences
- In FakeAVCeleb every fake is derived from a real source video of the same person. Without identity-disjoint splits, a model can memorize "this face appears in real" and get credit. DeepFake Doctor's identity split guards against this, but no paper measures how much AUC identity overlap inflates on AV data.
- Smeu et al. checked VoxCeleb2 pretraining leakage by testing on AVLips and attribute the gain to features, not leakage. [Smeu et al.](https://arxiv.org/html/2412.00175v3) Pretraining overlap (AV-HuBERT on LRS3/VoxCeleb2, which share sources with FakeAVCeleb and AV-Deepfake1M) remains a live concern.

### Gaps
- No identity-probe numbers (linear ID accuracy on detector features) for AV detectors.
- Whether DeepSpeak v1 splits are identity-disjoint was not stated in the text I read.
- Compression robustness of AV fusion detectors (re-encode both streams, AAC bitrate sweep) is not systematically benchmarked in what I found.

## 5. Benchmarks that release shortcut-neutralized versions

### Takeaway
None of the AV datasets has a released, shortcut-neutralized copy. What exists: scripts (AVH-Align), revised protocols (DeepFake Doctor), recommendations (Müller 2021), and new datasets that re-encode both classes from the start (CCDF).

### Cited Findings
- AVH-Align: code only; trimming done on the fly at evaluation. [bit-ml/AVH-Align](https://github.com/bit-ml/AVH-Align); [Smeu et al.](https://arxiv.org/html/2412.00175v3)
- DeepFake Doctor: new DeepSpeak v1 protocol (7,306 / 1,798 / 2,435 train/val/test, method and family holdouts) and FakeAVCeleb redesign; I found no code URL in the text. [DeepFake Doctor](https://arxiv.org/html/2506.05851v1)
- Müller et al. asked the ASVspoof organizers for a silence-removed release; I found no evidence one shipped for 2019 LA. [Müller et al.](https://arxiv.org/html/2106.12914)
- CCDF re-encodes all clips to one H.264 setting and frame rate. [CCDF](https://arxiv.org/html/2610.07939v1)
- Grommelt et al. de-bias GenImage by matching JPEG and size (image only). [arXiv 2403.17608](https://arxiv.org/abs/2403.17608v1)

### Inferences
- A "shortcut-neutral" AV release (or deterministic script plus manifest) would be the first of its kind for FakeAVCeleb / AV-Deepfake1M / LAV-DF / DeepSpeak.

### Gaps
- Did not check ASVspoof 5 or newer AV datasets (e.g. 2026 releases) for built-in shortcut controls.
- Did not verify whether AV-Deepfake1M++ or later challenges fixed the silence cue.

## Synthesis

Well documented:
- Leading silence in FakeAVCeleb and AV-Deepfake1M (98% AUC alone; Smeu 2025, DeepFake Doctor 2025).
- Leading and trailing silence in ASVspoof 2019 LA (15.1% EER alone; Müller 2021).
- Benchmark-to-wild collapse for audio detectors (In-the-Wild, 34 to 66% EER).
- Compression degradation for video detectors (FF++ 99.4 -> 85.5% at c40).
- Format bias (JPEG vs PNG) in generated-image datasets (Grommelt 2024).
- Identity leakage in FF++ face-swap detectors (Dong 2023, numbers not extracted here).
- Protocol leakage in FakeAVCeleb's original leave-one-out splits (DeepFake Doctor).

Undocumented or untested for AV data:
- Per-class codec, container, bitrate, frame rate, audio sample rate and duration census, and a metadata-only classifier AUC. Nobody has published this for FakeAVCeleb, LAV-DF, AV-Deepfake1M, DeepSpeak or DFDC.
- Trailing silence and loudness cues, which Smeu flagged (over 99% AUC trailing on FakeAVCeleb, 65% on LAV-DF) but nobody neutralized.
- Identity leakage and pretraining-corpus overlap for AV detectors, measured with an identity probe.
- Visual-side shortcuts (face crop size, resolution, compression of the fake region) in AV sets.
- Whether removing all shortcuts at once changes the ranking of published AV detectors.

Sharpest framing for a "shortcut-neutral protocol" contribution:
- One protocol that (a) audits each dataset with cheap "cue-only" classifiers (leading/trailing silence, loudness, duration, codec metadata, bitstream statistics, identity) and reports each one's AUC, (b) neutralizes them for both train and test (silence trim plus jitter, re-encode both classes to one codec, loudness normalize, identity-disjoint splits, method/family holdouts), and (c) re-ranks existing AV detectors under it. The claim to test: "how much of reported AV deepfake AUC is explained by non-content cues?" The headline result is a "shortcut ceiling" table: the best AUC any cue-only classifier gets per dataset, next to the detector's AUC before and after neutralization. That extends Smeu (one cue, test-time only) and DeepFake Doctor (one cue plus splits) to all cues, both splits, and a released manifest.
