# DOC_FACTS

Binding digest for the paper, the report and the slides. Written 2026-10-25 from DECISIONS.md, REQUIREMENTS.md, PROTOCOL.md, ITERATIONS.md, O007_SWEEP.md, DATA_LOG.md, MATH_NOTES.md and PLAN.md. Where this digest is silent or you need a number, read the source file; invent nothing. The three documents use the wording below, so that they agree. IDs (D-, O-, E-, R-) and the machine label never appear in visible text; they go in LaTeX comments, Word comments or footnotes of an appendix index, and slide notes.

## 1. Conventions

- Tense: past for what was done, present for what a method or protocol is, future only for planned work. No em-dashes or en-dashes anywhere; write "10 to 15", commas, colons.
- Results: no measured result is written unless it is logged in EXPERIMENTS.md. Today nothing is final: the encoder sweep ranking (E-001 to E-003) is exploratory and still being re-run, so every result cell, table value and sentence that needs a number shows the single placeholder `[pending]` and carries the experiment ID in a comment (LaTeX `% E-001`, Word comment, slide note). One style only: `[pending]`.
- Claims: no "real-time", "state of the art" or "better than X" without a head-to-head run. Differences are stated as differences.
- Wording to use everywhere: group-disjoint (the group is the source video together with the person identity); development slice and sealed test sets; "non-inferior" for the comparison with a published detector; "tie set" and "cheapest wins" for the encoder choice; "frozen encoder with a linear head"; "calibrated probability that the content is fake"; "staircase".
- Never mention an earlier version, a rebuild, a previous project or an old draft. Never write reviewer or tool notes in visible text.
- Identity fields: exactly as in IDENTITY.md.
- Only cite sources listed in RESEARCH.md (status READ or PARTIAL for the parts read); new sources you verify are listed in a file in your own folder for logging.

## 2. Problem and scope (D-001, D-002, D-011, D-018, D-019)

Given an image, a video with its audio, or a standalone audio clip, the system returns the probability that the content is fake and says which modality carries the evidence: a visual score, an audio score and, for a video, both shown separately (no fused claim for video until the combining rule is decided). Fake means a swapped, re-enacted or audio-driven face, a fully generated image (GAN or diffusion), synthesised or converted speech, or audio and video manipulated separately. No temporal localisation. The system is built from zero as a documented path: a simple working version first, then seven experimental iterations that each follow from a written hypothesis, then integration and freeze. The aim is results comparable with published detectors on a protocol the project defines and runs itself, not the best model.

## 3. Data (D-013, D-020, D-021, D-026, D-035, D-039, D-045; counts and licences in DATA_LOG.md)

- Visual training: FaceForensics++ (c23, official train and validation split). Visual development slice: Celeb-DF v2 (15 of 59 identity groups). Visual sealed tests: the rest of Celeb-DF v2 (test_x), Celeb-DF++ (test videos of 21 new generators), DFD.
- Audio training and validation: ASVspoof 2019 LA (train, dev); development slice: In-the-Wild (12 of 54 speakers); sealed test: the other In-the-Wild speakers; ASVspoof 2019 LA eval is test_in.
- Generated images: ProGAN validation set split by class (14 classes train, 6 validation); development slice: two of the seven CNN-synthesis generator units; sealed test: the other units and the diffusion test set.
- Audio-visual: LAV-DF (136,304 videos, all with an audio stream; per-file labels modify_video and modify_audio; official train, dev, test) for the fusion iteration.
- Licences: all datasets are research-only. Celeb-DF, Celeb-DF++, FakeAVCeleb forbid publishing derived data, so no frame, crop or per-video score of them appears in any document. The conditional licences of two encoder weights (OpenAI CLIP, WavLM) are accepted as research-only and stated as a limitation.
- Celeb-DF++ is one family with Celeb-DF v2. Its samples of the 15 development identities are excluded from the sealed test set (2,079 samples), and 150 real videos that are byte-identical to Celeb-DF v2 videos are counted once, so Celeb-DF++ alone has 2,849 fake videos and no real video in its sealed test set (44 identity groups); the pooled family test holds 744 real and 6,906 fake videos. Intervals are governed by about 44 identity groups.
- Splits: train, validation, development slice (dev_x), in-dataset test (test_in), cross-dataset test (test_x). The development slice is used to rank candidates and may be evaluated repeatedly; the two test sets are sealed and touched once per frozen iteration by a guarded command that writes a ledger. Group-disjoint everywhere; a script proves it, including content-hash de-duplication across splits.

## 4. Preprocessing (D-028, D-033, D-034, D-044)

One shared package prepares inputs for training, evaluation and inference; a configuration hash (preprocess identifier) is stored in every checkpoint and checked at load. Video: 16 uniformly spaced frames, the YuNet face detector (MIT licence), the largest face enlarged by a factor of 1.3, 224 by 224 crops, JPEG quality 95; frames without a face are left out and reported. Images: a face crop if a face is found, otherwise the whole image processed by the encoder's own rule. Audio: the first 64,600 samples at 16 kHz, repeat-padded; the Whisper encoder gets its fixed 30 second window with the 4 seconds first and only those frames pooled.

## 5. Method, working version (D-032, D-040)

A frozen pretrained encoder per task and a linear head (class-balanced cross-entropy, AdamW, three seeds), a clip score equal to the mean of frame probabilities, temperature scaling and a validation-chosen threshold, all fixed on validation data only. The working version counts as finished when a file goes in through the local web page, the winning encoders score it, the score is calibrated, metrics with intervals exist on the validation and development data, and the page shows a real result.

## 6. Encoder sweep (D-036, D-038, D-042; O007_SWEEP.md)

Three tasks: face video frames (T-V), generated images (T-I), speech audio (T-A). Frozen encoders, features extracted once, linear head, three seeds, ranking by the mean development-slice AUC with a paired 95 percent group bootstrap. The tie set is every candidate whose paired interval against the leader includes zero; the cheapest candidate (lowest end-to-end extraction time per 1,000 inputs, then fewer parameters) wins the tie set; the from-scratch reference is reported and excluded. A permutation test over groups is a sensitivity check. Shortlist: images EfficientNet-B4, Xception, DINOv2 small and base, OpenAI CLIP ViT-L/14, LAION CLIP ViT-L/14, SigLIP base, a small network from scratch; audio wav2vec 2.0 base, WavLM base plus, Whisper small encoder, HuBERT base, a small log-mel network from scratch. DINOv3 is excluded on purpose (manual access gate). Cost record per candidate: parameters, FLOPs, peak memory, model-only and end-to-end time per 1,000 inputs.

## 7. Iteration ladder (D-041, D-042; ITERATIONS.md)

I-0 is the working version. I-1 to I-6 take their lever from these families, chosen after the diagnostic battery of the previous iteration: fine-tuning depth (video task first: three encoders at three learning rates, one seed, a per-run time cap, extra seeds for the winner only; the two large CLIP models do not fit the 12 GB card), frame aggregation, loss and class weighting, augmentation and regularisation, combining encoders, late fusion with per-modality calibration. I-7 is the rPPG experiment with success and failure thresholds written beforehand and a signal-sanity check run earlier. I-8 is integration and freeze with no experiments. The diagnostic battery after every iteration: per-dataset and per-generator AUC with intervals, train versus validation gap, validation versus development gap, calibration curve with ECE and Brier before and after temperature scaling, score histograms, splits by compression, face size and number of faces, the 20 worst errors, seed variance. Verdicts on the development slice are exploratory (sign, size, interval). At the freeze every frozen iteration is evaluated once on the sealed test sets (staircase); the confirmatory claims are two pre-declared paired comparisons on sealed data: the final system against the working version, and the final system against a re-run published baseline.

## 8. Evaluation and statistics (D-012, D-024, D-025; MATH_NOTES.md, PROTOCOL.md)

Headline metric: ROC-AUC on a dataset never used in training, with a 95 percent group-bootstrap interval (2,000 resamples). Also EER, balanced accuracy at the validation-chosen threshold, log loss, expected calibration error (15 bins), Brier score, temperature scaling. "Comparable with a published detector" means non-inferior: the lower end of the paired 95 percent interval of (ours minus the re-run baseline), resampled over groups, lies above minus delta, with delta stated before the final comparison and chosen from the measured seed-to-seed spread of the baseline; an interval that only contains zero is reported as not distinguished; superior only when the whole interval is above zero. Seeds: three for the head of every candidate in the sweep; the fine-tuning budget follows the fine-tuning decision (one seed per learning rate, then two further seeds for the winner).

## 9. rPPG (D-006, O-010; ITERATIONS.md I-7)

Candidate improvement, not an adopted component. Signal check first on contiguous frames with a time-shuffled control pooled across clips (at least ten shuffles per clip); thresholds are written in ITERATIONS.md; if the pulse is not recoverable the slot reports exactly that. The adopt or do-not-adopt decision is made on the development slice.

## 10. Software (D-027, D-029, D-031, D-043, D-044)

A local web page (Flask, 127.0.0.1 only, 512 MB upload limit, files deleted after each request) calls an inference library with a fixed response structure (score per modality, status, warnings, errors, model block, a fixed sentence that says what the score means). A development stand-in always reports its source as mock and the page then shows a visible MOCK banner; the real adapter refuses to run without a checkpoint whose metadata and preprocessing identifier match. Screenshots of the page in documents are taken against the stand-in and are labelled as such.

## 11. Environment (measured, EXPERIMENTS.md)

Windows 10 Pro; one NVIDIA GeForce RTX 5070 with 12 GB (11 GiB usable); AMD Ryzen 9 7900X (12 cores, 24 threads); 31 GB RAM; Python 3.13.15 under uv; PyTorch 2.14.1 with CUDA 13.0 wheels; driver 595.79. Read the other versions (timm, transformers, Flask, OpenCV) from the environment, do not guess.

## 12. Timeline (PLAN.md)

Day 1 Sat 24 Oct: decisions, data, evaluation harness. Day 2 Sun 25 Oct: sweep, working version code path. Day 3 Mon 26 Oct: working version, diagnostics. Day 4 Tue 27 Oct: I-1 to I-3. Day 5 Wed 28 Oct: I-4 to I-7. Day 6 Thu 29 Oct: integration and freeze, sealed evaluations. Day 7 Fri 30 Oct: assembly and submission. An iteration not finished by the end of Wednesday is reported as not run.

## 13. Limitations to state

Small development slices (15 identities, 12 speakers, 2 generator units) make intervals wide and ties likely; Celeb-DF family tests have about 44 independent identity groups; frozen features may be inadequate for face forgery; research-only weights restrict reuse; cross-dataset numbers from papers use other protocols and are context only; rPPG may not be recoverable from compressed video; no result of the project is claimed beyond what its interval supports.
