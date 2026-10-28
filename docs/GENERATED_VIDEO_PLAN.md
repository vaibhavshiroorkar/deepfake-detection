# AI-generated video detection: next-stage plan (WS5)

Written 2026-10-26. Status: PROPOSED. No detector exists for this capability, no model was trained or run, and no result is claimed. Candidate descriptions are summaries of what the papers and repositories state, taken from the literature pass of docs/FEASIBILITY_AND_TARGET_ARCHITECTURE.md (untracked draft); ReStraV, STALL, D3, DF26 and DeepAction were entered in docs/RESEARCH.md on 2026-10-26 as R-078 to R-082, all PARTIAL; they may be cited only for the sections listed there.

## 1. The gap

- The frozen face branch (T-V) scores face crops from FF++-style manipulation. It is not designed for fully generated video, and no measurement of it on such video exists in this repository. The platform wrapper therefore reports `not_applicable` when no face is found and carries the limitation in every face-video result.
- No module covers generated video. In the dashboard it appears as a planned, unimplemented capability, not as a module that returns "nothing found".
- The DF26 authors report (as recorded in the feasibility draft, not checked against the full paper here) that detectors trained on older face-swap data do not transfer to modern generated talking-head video. H-G4 below would measure this on our own system.

## 2. Candidates (verified 2026-10-26 from the sources listed at the end of this file)

Tags: [doc] documented by the source (a URL below); [repo] verified in this repo; [inf] inference, not tested. Papers are entered in docs/RESEARCH.md as PARTIAL (R-078 to R-082); citation is limited to the sections listed there. Reported numbers are the authors' under their own protocol and say nothing about our data.

### 2.1 Summary

| Candidate | Task and scope | Needs training? | Released code and licence | Released weights | Verdict |
|---|---|---|---|---|---|
| ReStraV (R-078) | Whole-video real vs AI-generated, visual only, any content (not face-specific) [doc] | Yes, a two-layer MLP on 21 features; the encoder is frozen [doc] | github.com/ChristianInterno/ReStraV, Apache-2.0 (LICENSE read), last commit 2026-10-13 [doc] | No pretrained classifier mentioned; README says `train.py` produces `model.pt`, `mean.npy`, `std.npy`, `best_tau.npy` [doc] | First candidate; see section 7.3 |
| STALL (R-079) | Whole-video, visual only, training-free likelihood score [doc] | No training; needs a large real-video reference pool (main 33k VATEX; results drop below about 5k) [doc] | github.com/OmerBenHayun/STALL, LICENSE.txt is Creative Commons Attribution-NonCommercial 4.0 International (read), last commit 2026-10-25 [doc] | Whitening and calibration file `precomputed/stall_params_vatex_dino_v3.npz` for DINOv3; DINOv3 weights not included [doc] | Not implementable as released (DINOv3 excluded by D-038, no licence-clean real pool); DINOv2 re-implementation is a different method |
| D3 (R-080) | Whole-video, visual only, training-free second-order feature score [doc] | No; but a decision threshold rule is not specified in the text read [doc] | github.com/Zig-HS/D3, MIT (LICENSE read), last commit 2026-06-23 [doc] | None; uses pretrained encoders (default XCLIP-B/16) [doc] | Second candidate, blocked on an XCLIP download and licence check |
| UNITE | No code or weights found in the earlier literature pass; not re-checked here and not in RESEARCH.md | n/a | n/a | n/a | Not usable |
| Frozen frame encoder with temporal pooling (our own) | Mandatory control baseline | Yes (linear head) | Our code | Encoders on disk [repo] | Control |
| Metadata-only classifier (container, codec, bit rate, frame rate, duration) | Mandatory shortcut audit (M-006) | Yes | n/a | n/a | Control |

### 2.2 ReStraV (arXiv 2507.00583)

- Source as printed: Internò, Geirhos, Olhofer, Liu, Hammer, Klindt; arXiv v1 1 July 2025, v4 3 February 2026; comments "NeurIPS 2025" [doc].
- Features [doc]: 24 frames over a 2 s window, resized to 224 x 224, DINOv2 ViT-S/14 final block (CLS plus patch tokens flattened per frame); 7 stepwise L2 distances, 6 curvature angles (arccos of cosine similarity of consecutive differences, in degrees), and mean, variance, min, max of each signal: a 21-value vector. Classifier: two-layer MLP (64, 32); threshold tuned on the training set to maximise F1. Optimiser, learning rate and epochs not given in the text read.
- Internal inconsistency in the text read [doc]: the per-frame vector is stated as (196+1) x 384 = 75,648 values while the patch grid is described as 16 x 16; 224/14 gives 16 x 16 = 256 patches, which would be 257 x 384 = 98,688. Which is used must be settled from the authors' `dinov2_features.py` (not opened here) before any implementation.
- Training data [doc]: 50,000 VidProM AI videos (Pika, VideoCraft2, Text2Video-Zero, ModelScope, Sora) and 50,000 DVSC2023 natural videos, stratified 50/50 split. Further protocols: VidProM seen/unseen/future, GenVidBench, DeMamba one-to-many, 200 Veo3 vs 200 natural, Physics-IQ pairs.
- Cost [doc]: one RTX 2080; DINOv2 forward 43.6 ms for an 8-frame batch; about 48 ms per video end to end. No VRAM figure given. Our own measurement of DINOv2-S at 0.38 GB at batch 64 is recorded in this repo's earlier work (carried over from the previous version of this file; not re-run here).
- Limitations stated [doc]: can be gamed by a generator trained against it; future generators may close the gap; invariant to time reversal; encoder biases. Abstract says 97.17% accuracy, Table 1 prints 94.17 for the MLP.
- Dependencies [doc, requirements.txt read]: gradio==6.2.0, h5py==3.7.0, huggingface_hub==0.29.1, matplotlib==3.7.1, numpy==1.24.0, pandas==1.5.3, Requests==2.32.5, scikit_learn==1.8.0, scipy==1.10.1, torch==2.5.1, torchcodec==0.9.1, torchvision==0.20.1, tqdm==4.66.4, yt_dlp==2025.12.8. No Python or CUDA version, and no Windows statement, in the README text read.
- Compatibility with Python 3.13 / torch 2.14 / transformers 5.20.2 / timm 1.0.30 [inf]: the exact pins (numpy 1.24.0, scipy 1.10.1, pandas 1.5.3, h5py 3.7.0, torch 2.5.1) plausibly have no wheels for Python 3.13 and would conflict with torch 2.14 in one environment. The method itself needs only a DINOv2 forward pass, differences, arccos and an MLP, so re-implementing it without installing the repo's requirements is plausible. Not tested.

### 2.3 STALL (arXiv 2603.15026)

- Source as printed: Ben Hayun, Betser, Levi, Kassel, Gilboa; arXiv v1 16 March 2026; comments "Accepted to CVPR 2026" [doc].
- Method [doc]: DINOv3 frame embeddings, PCA-whitened Gaussian log-likelihood (max over frames) plus whitened normalised frame differences (min over transitions); percentile fusion; reference statistics from real videos only. Model size of DINOv3 not stated in the text read; the README points to the ViT-L/16 distilled weights from Meta's download page.
- Data [doc]: ComGenVid (about 3.5k Sora and Veo3 vs about 1.7k MSVD real), VideoFeedback, GenVideo test; 8 FPS for 2 s. Dataset licences and formal splits not stated in the text read. Precomputed DINOv3 embeddings for ComGenVid on huggingface.co/datasets/OmerXYZ/comgenvid (licence not read).
- Cost [doc]: 0.49 s per video for 16 frames; memory figures in a supplement not read.
- Limitations [doc]: no dedicated limitations section in the text read; performance drops below about 5k calibration videos.
- Dependencies [doc]: `environment.yml` pins Python 3.10 (conda-forge), pip: numpy, opencv-python, Pillow, scikit-learn, pandas, tabulate, tqdm, datasets, huggingface_hub, pyarrow, matplotlib (no version pins seen in the summary read); PyTorch installed separately (README example CUDA 12.4); DINOv3 repository cloned separately. No Windows statement seen. Compatibility [inf]: the Python 3.10 pin differs from our 3.13, but the pins are loose; the real blocker is DINOv3 access (D-038), not packages.

### 2.4 D3 (arXiv 2508.00701)

- Source as printed: Zheng, Suo, Lin, Zhao, Yang, Liu, Yang, Wang, Shen; arXiv v1 1 August 2025; no venue listed (comments: 8 pages, 4 figures) [doc].
- Method [doc]: frozen XCLIP-B/16 features, up to 2 s at 8 fps, 224 x 224; score is the standard deviation of second-order differences of the feature sequence. The decision threshold rule is not specified in the text read.
- Data [doc]: GenVideo, EvalCrafter, VideoPhy, VidProM: 40 test subsets; baselines trained on Youku-mPLUG real and Pika videos.
- Cost [doc]: 1,000 samples at batch size 1 took 56 s (XCLIP-B/16) on an RTX 4090 with an AMD EPYC 7763; no memory statement.
- Limitations stated [doc]: weak on the VidProM Text2Video-Zero subset (AP 45.11); robustness depends on the encoder. No limitations section.
- Dependencies [doc, requirements.txt read]: albumentations==2.0.8, moviepy==1.0.3, numpy==2.3.3, opencv_python==4.12.0.88, opencv_python_headless==4.12.0.88, pandas==2.3.3, scikit_learn==1.7.2, timm==1.0.20, torch==2.8.0, torchvision==0.23.0, tqdm==4.67.1, transformers==4.57.0. README: Python >= 3.11.13; no CUDA version; no Windows statement.
- Compatibility [inf]: Python 3.13 satisfies ">= 3.11.13". The pins transformers==4.57.0, timm==1.0.20 and torch==2.8.0 differ from our transformers 5.20.2, timm 1.0.30 and torch 2.14.1, so installing the requirements file as is would downgrade the stack; the method is a few lines on top of any encoder, so re-implementation avoids this. Not tested.

### 2.5 Mandatory controls

A frozen-encoder pooled baseline and a metadata-only classifier (container, codec, bit rate, frame rate, duration, resolution) run under the same data, split, budget and metric. If no candidate beats the metadata-only baseline under generator-disjoint testing, the correct conclusion is a documented negative result and no module is added.

## 3. Data (acquired today, status in DATA_LOG)

| Dataset | Role | Known limits |
|---|---|---|
| DF26 open-generator subset | Development under leave-one-generator-out | Talking heads only; licence allows detector training only on the open subset and only under that protocol |
| DF26 closed-generator subset (Veo 3.1, Kling 3.0, Wan 2.6, Grok Imagine) | **Sealed final test**, touched once with `--final` | Licence forbids calibration, threshold tuning, augmentation and model selection on it; 271 real videos in total, so real-class intervals will be wide; no frames may appear in our documents without the maintainers' permission |
| DeepAction v1 | Development, shortcut audit, face-independence test | Action videos, no talking heads; 512 x 512 normalisation differs from DF26; real class is a Pexels folder of about 100 videos; Veo is a pre-release version |

Neither dataset contains a documented audio track, so audio produced by video generators is not covered; lip-sync work on generated video cannot use these.

## 4. Split, metrics, confounders

- Groups: real videos by source file family or identity where known; generated videos by prompt and, for image-to-video, by the conditioning image. Never split a prompt family or a source image across partitions.
- Development: DF26 open subset (leave-one-generator-out), DeepAction (leave-one-generator-out), remaining reals. Sealed: DF26 closed subset with a matched share of reals, evaluated once.
- Metrics: AUC with group bootstrap intervals, plus sensitivity at a threshold fixed on development data only, reported per generator, never pooled across generators.
- Confounders to control before any detector result is read: container and codec differences between real and generated files (metadata baseline), resolution and aspect ratio, duration, scene type, compression history (canonical re-encoding of every video with one fixed recipe before feature extraction, and a re-encoding robustness check), static versus moving content.

## 5. Preregistration outline (to be written into docs/EXPERIMENTS.md before any run; E-numbers to be assigned then)

1. **Shortcut audit (M-006 / E-018).** Hypothesis: a classifier on file metadata alone separates real from generated clips with AUC well above 0.5 on DF26 open and DeepAction. Run before any content-based model; if confirmed, canonical re-encoding becomes mandatory and results are reported before and after it.
2. **Scope probe (H-G4).** The frozen face branch scored once on DF26 development videos is expected near chance. Development data only.
3. **Candidate comparison (H-G1, H-G2).** Control baseline, ReStraV-style and a metadata-only baseline under identical data, split, budget and metric, three seeds, bootstrap intervals; the cheaper candidate wins ties (CLAUDE.md rule 4). Selection rule written before the run.
4. **Face-independence (H-G3).** On DeepAction, lower AUC bound above 0.6 where the face branch is `not_applicable`.
5. **One sealed evaluation** of the chosen candidate on the DF26 closed subset, only after explicit approval. Report the result whatever it is.

Thresholds in the feasibility draft (0.70, 0.60, 0.10) are proposals, not agreed criteria; the user must confirm them at preregistration.

## 6. Prerequisites and open items

- Recorded in RESEARCH.md as PARTIAL (R-078 to R-082); the unread parts (supplements, appendices) remain unread.
- DATA_LOG entries (done for acquisition once the download is verified); licence texts re-read in full before any use.
- A licence-clean real-video pool is not available; a pool is the binding constraint for training-free methods.
- Hardware: all candidates fit the 12 GB card with the frozen web server running; decoding, not the GPU, is the bottleneck.
- Not approved and not done: any training, any scoring, any sealed read.

## 7. Local assets, datasets and compatibility (checked 2026-10-26, nothing run)

Tags as in section 2: [doc] documented by the source, [repo] verified in this repo, [inf] inference.

### 7.1 Local assets

| Asset | Status | Use for this plan |
|---|---|---|
| DINOv2-small (IMG-07, model.safetensors, 88,249,960 bytes, sha256 ae1e99fc...14be1), DINOv2-base (IMG-08) | On disk, hash recorded in docs/DATA_LOG.md and data/_logs/encoder_weights.json [repo] | Frame encoder for a ReStraV-style feature and for the pooled-frame control |
| SigLIP-base (IMG-17), CLIP ViT-L/14 (IMG-13) | On disk [repo] | Alternative control encoders |
| Frame decoding, encoder wrappers, linear heads, bootstrap and ranking, score-file and ledger machinery | `dfd.prep.video`, `dfd.sweep.encoders`, `dfd.sweep.rank`, `dfd.eval` [repo] | Reused as they are |
| XCLIP (needed by D3) | Not on disk, licence not verified [repo] | D3 blocked until a download and licence check are approved |
| DINOv3 (needed by STALL as released) | Not on disk, excluded by D-038 [repo] | STALL only as a DINOv2 re-implementation, not as released |
| DeepAction v1 | data/deepaction_v1 present with folders BDAnimateDiffLightning, CogVideoX5B, Pexels, RunwayML, StableDiffusion, Veo, VideoPoet, plus captions.csv and README.md [repo, directory listing only; file counts and integrity are in DATA_LOG G1] | Development data |
| DF26 | Not available: data files returned HTTP 403 for the Hugging Face account on this machine (DATA_LOG G2) [repo]; the card says access is automatic after accepting the conditions [doc] | Sealed test cannot be prepared until access is granted |

### 7.2 Datasets as documented

**DeepAction v1** (arXiv 2412.00526; card huggingface.co/datasets/faridlab/deepaction_v1).
- Provenance [doc]: Bohacek and Farid, 2024; 3,100 generated clips from seven text-to-video models per the paper (BD AnimateDiff, CogVideoX-5B, Lumiere early version, RunwayML Gen3, Stable Diffusion plus image-to-video, pre-release Veo, VideoPoet), default parameters, prompts written by ChatGPT for 100 human actions; real class is 100 Pexels clips matched to prompts by general action. The card lists six generated folders and the Pexels folder; the paper's Lumiere clips are not among the card's folders and not in the local folder listing [doc and repo, difference noted as read, cause not investigated].
- Licence [doc, card]: generated folders "released under the CC BY 4.0 license"; real videos "released under the Pexels license" (pexels.com/license). The card has no separate use-restriction section. The paper's abstract calls it "an academic license"; the card and abstract wording differ and both are recorded here as printed.
- Confounds the authors mention [doc]: real videos from one source; matched by general action not exact content; clip length 2 to 10.7 s; resolution from 512 x 512 to 2048 x 1152 in the original (the card says videos are normalised to 512 x 512). Fps and frame sampling are not reported; the totals they give (254,632 frames in 297 min for AI, 44,475 frames in 28 min for real) imply about 14 and 26 fps respectively [inf, my arithmetic; a possible shortcut, to be tested by the metadata baseline]. Content type: human actions only; the authors report failure on talking-head deepfakes and CGI.
- Class balance [doc]: 100 real vs 3,100 generated clips, so any real-class interval will be wide.

**DF26** (arXiv 2609.07369; card huggingface.co/datasets/DF26/DF26; LICENSE.md).
- Provenance [doc]: 2,691 five-second 1280 x 720 H.264 videos (card: durations 4.01 to 6.04 s, 3.57 GB): 271 real from OpenVid-1M (197), TalkingCelebs (62), MAVOS-DD (12); 2,420 generated. Open-weight generators: Wan 2.2 A14B, HunyuanVideo 1.5, LTX 2.3 distilled, each I2V and T2V, 271 clips per mode. Commercial (I2V per the card, T2V per the paper's text as read; the two sources disagree): Grok Imagine (199 clips), Kling 3.0 (198), Veo 3.1 (200), Wan 2.6 (197), not run on the official-statement scenario. Commercial clips obtained through Higgsfield in March 2026; the authors removed visible watermarks and logos. Prompts from Gemini 2.5; I2V starts from the first frame of the real clip.
- Splits [doc]: no train/test split; evaluation set; scenarios Direct_to_Camera_Casual (100 reals), Studio_Interview (100), Official_Statement (71). Paper's main protocol is direct evaluation of pretrained detectors.
- Licence and restrictions [doc, card and LICENSE.md]: licence tag `df26-evaluation-only-license` ("DF26 Dataset License"). "DF26 may be used only for non-commercial research, benchmarking, reproducibility studies, and academic evaluation." Training, fine-tuning and other model-update procedures are "prohibited except as expressly allowed", i.e. only on the open-source-generator subset under the leave-one-generator-out protocol; "The closed-source-generator subset is strictly evaluation-only"; use to develop generative models is prohibited; redistribution, publication, mirroring or upload of the dataset is not allowed, small illustrative excerpts only when expressly permitted by the maintainers. Source licences remain: OpenVid CC BY 4.0, MAVOS-DD CC BY-NC-SA 4.0, TalkingCelebs possibly non-commercial, attribution, publicity or privacy restrictions. Commercial-generator clips are for evaluating detection methods only. The licence does not explicitly address publishing frames; written permission from the maintainers is advisable before any frame appears in a document. The summary of LICENSE.md was produced by a page-reading tool; the full file must be re-read by a person before use.
- Confounds the authors mention [doc]: scale is small; visual only (no audio, speech or lip-sync); generator source type may correlate with model family, post-processing, visual quality and modality (diagnostic, not causal); real pool mostly OpenVid; text overlays removed from reals. Frame rate and audio presence are not stated in the sources read; the real clips were trimmed and downscaled to 720p by the authors while the generated clips come from different pipelines, so a codec or encoding-history difference between classes cannot be excluded from the text read [inf].
- Reported failure modes [doc]: FF++-trained detectors score near chance on DF26 (AUROC 44.0 to 69.7 across nine detectors); temporal detectors drop on closed generators; humans reach 52.6% on fake clips.

### 7.3 Implementable now with local assets: verdict

**Verdict (inference, not tested):** a ReStraV-style detector can be implemented with no new download. Inputs on disk: DINOv2-S weights (IMG-07), the existing frame decoder and encoder wrapper, and DeepAction for MLP training. The 21 features are differences, arccos and four summary statistics per signal; a two-layer MLP needs no new package (torch and scikit-learn are already used). This does not say that the method will work on our data, and no model is claimed to suit it. It would be a re-implementation from the paper, not the released code (which pins an incompatible stack, section 2.2) and not the authors' trained classifier (none released).

What a first preregistered experiment would need before any run (to be written into docs/EXPERIMENTS.md with an E-number, hypothesis, success criterion and failure criterion):
1. Resolve the feature definition: patch-grid ambiguity (196 vs 256 patches) and whether the final layer norm is included, by reading `dinov2_features.py` in the authors' repository (a text read only, no code run, no install). Record the choice as a decision.
2. Canonical preprocessing of every clip with one fixed recipe, with frames sampled by time (24 frames over 2 s, as in the paper) rather than by frame index, because DeepAction fps may differ between classes [inf].
3. Metadata-only baseline (M-006) and pooled-frame control run first, same split, same metric.
4. Split by generator and by action prompt: leave-one-generator-out over the six local generators with prompt families kept on one side; the 100 real Pexels clips cannot be split by identity from the generated clips by prompt alone, so real clips must be assigned to partitions by their action; report the effective number of real clips per fold.
5. Three seeds, mean, standard deviation and a bootstrap interval grouped by prompt; AUC per held-out generator, not pooled.
6. Primary scope limits to state: real class is 100 clips from one source; DeepAction has no talking heads; no DF26 data used (inaccessible, and its licence permits training only on the open subset under leave-one-generator-out, and forbids calibration or selection on the closed subset).
7. A DF26 stage is possible only after access is granted and the licence is re-read in full; the open subset may then serve as development data under leave-one-generator-out; the closed subset stays sealed for one final evaluation.
8. No new download is needed for steps 1 to 6 other than a text read of the authors' repository files; D3 (XCLIP) and STALL (DINOv3) would each need a new approved download.

### 7.4 Sources opened (all fetched 2026-10-26)

- https://arxiv.org/abs/2507.00583 and https://arxiv.org/html/2507.00583 (ReStraV)
- https://arxiv.org/abs/2603.15026, https://arxiv.org/html/2603.15026 and https://omerbenhayun.github.io/stall-video (STALL)
- https://arxiv.org/abs/2508.00701 and https://arxiv.org/html/2508.00701 (D3)
- https://arxiv.org/abs/2609.07369 and https://arxiv.org/html/2609.07369 (DF26)
- https://arxiv.org/abs/2412.00526 and https://arxiv.org/html/2412.00526 (DeepAction)
- https://huggingface.co/datasets/DF26/DF26 and https://huggingface.co/datasets/DF26/DF26/blob/main/LICENSE.md
- https://huggingface.co/datasets/faridlab/deepaction_v1
- https://github.com/ChristianInterno/ReStraV, https://raw.githubusercontent.com/ChristianInterno/ReStraV/main/requirements.txt, https://raw.githubusercontent.com/ChristianInterno/ReStraV/main/LICENSE, https://api.github.com/repos/ChristianInterno/ReStraV/commits?per_page=1
- https://github.com/Zig-HS/D3, https://raw.githubusercontent.com/Zig-HS/D3/main/requirements.txt, https://raw.githubusercontent.com/Zig-HS/D3/main/LICENSE, https://api.github.com/repos/Zig-HS/D3/commits?per_page=1
- https://github.com/OmerBenHayun/STALL, https://raw.githubusercontent.com/OmerBenHayun/STALL/main/environment.yml, https://raw.githubusercontent.com/OmerBenHayun/STALL/main/LICENSE.txt, https://api.github.com/repos/OmerBenHayun/STALL/commits?per_page=1

Method note: web pages were read through a page-summarising tool, not a browser. Quotations and numbers are as that tool returned them from the page; the arXiv HTML reads covered only the first 100,000 characters of each paper (see RESEARCH.md R-078 to R-082). The ReStraV and D3 README texts did not show their requirements contents; the requirements files were fetched separately.

## 8. Measured so far (E-018, 2026-10-26)

The metadata-only audit of DeepAction (docs/EXPERIMENTS.md E-018, scripts/audit/e018_metadata_audit.py) found that container and stream metadata alone separate the 100 real Pexels clips from all 2,505 generated clips perfectly (grouped AUC 1.000, including for a generator held out of training) and identify the generator perfectly. Cues: creation-time tag, the encoder tag "AVC Coding", colour tags (all on the real clips only), file size, resolution, bit rate. Consequences for this plan, as findings requiring interpretation: (1) no content-based detector may be trained or compared on DeepAction as distributed; every clip must first be passed through one fixed re-encoding recipe that removes these cues, and the audit must be repeated on the re-encoded copies; (2) the same audit must be run on DF26's open subset before use; (3) DeepAction cannot show that a detector finds generation traces, only that a pipeline separates the two corpora. This is not evidence that any existing detector is invalid.

## 9. Outcome of E-019 and E-020 and the state of the module (2026-10-26, appended)

- Canonical preprocessing: v1 failed (colour tags leaked the label), v2 failed its preregistered GOP rule, v3 passed it (docs/EXPERIMENTS.md E-019 v1, v2 and v3 blocks). The v3 pass means the audited header, stream and GOP features no longer separate DeepAction beyond the preregistered bar; it does not mean the clips are free of every source shortcut: the pixel-statistics baseline C3 still reached 0.7613 macro AUC on DeepAction, and the v3 pass is conditional on three attempts on the same data.
- E-020 (preregistered before extraction): the ReStraV-style trajectory classifier C1 was at chance on unseen DeepAction generators (0.5073 [0.4608, 0.5542]); the appearance control C2 reached 0.8281 [0.7908, 0.8662]; C3 0.7613. The preregistered rule chose C2. On the held-out DF26 benchmark C2 scored 0.4957 [0.4755, 0.5168] (closed generators 0.4880 [0.4534, 0.5204]), so H-G1 is not supported. The trajectory method as adapted here is therefore NOT reproduced as a working detector; only the feature pipeline is reproduced, and the classifier is ours.
- What is integrated: the module `generated_video` loads `checkpoints/genvideo-E020/head.npz` (C2 weights, corrected metadata, D-072). It reports the score and quotes the measured chance-level result, always with finding `inconclusive` and the warning EXPERIMENTAL_UNVALIDATED. The integration was exercised through the live API and dashboard (scripts/platform/live_full_e2e.py); that shows the plumbing works, not that anything is detected.
- Gaps: no head with demonstrated transfer; the real-class data is 100 stock clips from one library; only the first 2 s are used; no evaluation on generators outside DeepAction and DF26; whether the C2 DeepAction result is a source/content confound is untested.
