# AI-generated video detection: next-stage plan (WS5)

Written 2026-10-26. Status: PROPOSED. No detector exists for this capability, no model was trained or run, and no result is claimed. Candidate descriptions are summaries of what the papers and repositories state, taken from the literature pass of docs/FEASIBILITY_AND_TARGET_ARCHITECTURE.md (untracked draft); none of these papers is in docs/RESEARCH.md as READ or PARTIAL, so none may be cited in the paper, report or slides until it has been read and entered there (status NR for now).

## 1. The gap

- The frozen face branch (T-V) scores face crops from FF++-style manipulation. It is not designed for fully generated video, and no measurement of it on such video exists in this repository. The platform wrapper therefore reports `not_applicable` when no face is found and carries the limitation in every face-video result.
- No module covers generated video. In the dashboard it appears as a planned, unimplemented capability, not as a module that returns "nothing found".
- The DF26 authors report (as recorded in the feasibility draft, not checked against the full paper here) that detectors trained on older face-swap data do not transfer to modern generated talking-head video. H-G4 below would measure this on our own system.

## 2. Candidates

| Candidate | Reusable weights? | Licence status | Inference cost | Verdict |
|---|---|---|---|---|
| ReStraV (DINOv2-S curvature features plus a small MLP) | No trained classifier released; the MLP must be trained (21 input features) | Code Apache-2.0 (documented); DINOv2-S Apache-2.0 and already on disk | Very low (DINOv2-S measured at 0.38 GB at batch 64; the paper reports tens of milliseconds per video, documented) | **First candidate** |
| STALL-style likelihood (training-free) | Whitening statistics released for DINOv3 only | DINOv3 is gated and excluded by D-038; code licence not verified | Low | Re-implement on DINOv2 only if the first candidate shows promise; needs a licence-clean real-video pool (thousands of clips) that does not exist yet |
| D3 (second-order temporal features) | Training-free | Code MIT; XCLIP encoder licence not verified | Low | Second candidate after the XCLIP licence check |
| UNITE | No code or weights found | n/a | n/a | Not usable; re-implementation not justified |
| Frozen frame encoder with temporal pooling (our own baseline) | Encoders on disk | Recorded | Low | **Mandatory control baseline** |
| Metadata-only classifier (container, codec, bit rate, frame rate, duration) | n/a | n/a | Trivial | **Mandatory shortcut audit (M-006)** |

If no candidate beats the metadata-only baseline under generator-disjoint testing, the correct conclusion is a documented negative result and no module is added.

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

- Read and record in RESEARCH.md: ReStraV, STALL, D3 and the DF26 paper (currently NR).
- DATA_LOG entries (done for acquisition once the download is verified); licence texts re-read in full before any use.
- A licence-clean real-video pool is not available; a pool is the binding constraint for training-free methods.
- Hardware: all candidates fit the 12 GB card with the frozen web server running; decoding, not the GPU, is the bottleneck.
- Not approved and not done: any training, any scoring, any sealed read.

## 7. Local assets that a first candidate could reuse (checked 2026-10-26, nothing run)

| Asset | Where recorded | Use for this plan |
|---|---|---|
| DINOv2-small (IMG-07) and DINOv2-base (IMG-08) weights | docs/DATA_LOG.md lines 286 and 287 (hashes); local copy hash-checked in data/_logs/encoder_weights.json | Frame encoder for a ReStraV-style trajectory feature and for the pooled-frame control |
| SigLIP-base (IMG-17), CLIP ViT-L/14 (IMG-13) | DATA_LOG lines 288 to 290 | Alternative frame encoders for the control baseline |
| Frame decoding, encoder wrappers, linear heads, bootstrap and ranking, score-file and ledger machinery | `dfd.prep.video`, `dfd.sweep.encoders`, `dfd.sweep.rank`, `dfd.eval` | Reused as they are |
| XCLIP (needed by D3) | Not on disk, licence not verified | D3 is blocked until a download and licence check are approved |
| DINOv3 (needed by STALL as released) | Not on disk, excluded by D-038 | STALL only as a DINOv2 re-implementation, not as released |
| DeepAction (6.06 GB) | data/deepaction_v1, download in progress at the time of writing (DATA_LOG G1) | Development data |
| DF26 | **Not available**: every data file returned HTTP 403 for the Hugging Face account on this machine (DATA_LOG G2); the user must accept the licence on the dataset page | Sealed test cannot be prepared until access is granted |

Compatibility: the existing stack (Python 3.13, torch 2.14.1+cu130, transformers 5.20.2, timm 1.0.30) already runs DINOv2 and SigLIP; no new package is needed for the first candidate. Candidate code from the papers' repositories has not been run here, so its compatibility with Python 3.13 is unverified.
