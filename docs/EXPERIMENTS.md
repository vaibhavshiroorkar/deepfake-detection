# EXPERIMENTS

Every experiment is logged here. The hypothesis and the success and failure criteria are written BEFORE the run and are never edited afterwards. Failed runs stay in the log.

## Environment: ENV-DESKTOP-GPU

- Date measured: 2026-10-24
- Measured by: Session 0, commands run on the machine itself. Nothing below is copied from the user's description except where marked.

### Hardware (measured)

| Item | Measured value | User's description (unconfirmed) | Matches? |
|---|---|---|---|
| GPU | NVIDIA GeForce RTX 5070, PCI bus 01:00.0 | "RTX 5070 class" | yes |
| VRAM | 12227 MiB reported by nvidia-smi (11.94 GiB as seen by PyTorch). About 1047 MiB was already in use by desktop apps at idle | "8 or 12 GB" | 12 GB |
| Compute capability | 12.0 (Blackwell, sm_120) | n/a | n/a |
| NVIDIA driver | 595.79 (WDDM mode) | n/a | n/a |
| CUDA version reported by driver | 13.2 | n/a | n/a |
| CPU | AMD Ryzen 9 7900X, 12 cores, 24 threads, max clock 4701 MHz, L3 cache 64 MiB | "Ryzen 7900K" | The model is the 7900X, not 7900K |
| RAM | 2 x 16 GiB Crucial CP16G60C36U5W (DDR5), configured at 5200 MT/s, 31.1 GiB usable | "32 GB DDR5" | yes (modules are rated higher than the 5200 they run at) |
| Second GPU | AMD Radeon integrated graphics (display only, not used for compute) | n/a | n/a |

### Disk (measured)

| Drive | Size | Free |
|---|---|---|
| C: | 400 GB | between 46 GB and 103 GB, the reading changed during the session |
| D: | 531 GB | 404 GB |

The repository lives on D:. Data, checkpoints and extracted frames must go on D:.

### Software (measured)

| Item | Version |
|---|---|
| OS | Windows 10 Pro, version 10.0.19045, 64-bit |
| System Python | 3.14.0 (not used for the project) |
| Project Python | CPython 3.13.15, managed by uv, virtual environment in `.venv` inside the repository |
| uv | 0.12.12 |
| git | 2.51.1.windows.1 |
| GitHub CLI (gh) | 2.103.2, installed in this session with winget. It was not present before. |
| gh login | not logged in at the end of Session 0 |
| git remote | none configured at the end of Session 0 |
| ffmpeg / ffprobe | 9.0.1 (gyan.dev full build), already installed, working, includes libx264, libx265, libvpx, libopus, libmp3lame and NVENC / NVDEC support |
| PyTorch | 2.14.1+cu130 (CUDA runtime 13.0, cuDNN 9.24.0) |
| NumPy | 2.5.4 |

Choice of PyTorch build: the wheel index at download.pytorch.org was queried directly. It offers 2.14.1 builds for CUDA 12.6, 13.0 and 13.2, while the CUDA 12.8 index stops at 2.11.0. The cu130 build was installed because the driver reports CUDA 13.2, which is newer than 13.0, and because the build ships kernels for `sm_120`. The official install page (pytorch.org/get-started/locally) is rendered by script and returned no command text to the fetch tool, so the wheel index was the source used.

The exact resolved versions of all packages are pinned in `uv.lock`; the direct dependencies are in `pyproject.toml`.

### GPU proof (measured)

- `torch.cuda.is_available()` = True
- `torch.cuda.get_device_name(0)` = NVIDIA GeForce RTX 5070
- `torch.cuda.get_device_capability(0)` = (12, 0)
- `torch.cuda.get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 (native Blackwell kernels are present)
- 4096 x 4096 float32 matrix product on the GPU ran; maximum absolute difference to a float64 CPU product on the first 8 rows was 6.0e-4 (the entries are sums of 4096 products of standard normals, so a size near 64 is typical; this is ordinary float32 rounding)
- Mixed precision: `torch.autocast` with bfloat16 and with float16 both produced outputs of the requested dtype, and a short training loop under each ran with finite loss

### Sizing measurement: tiny network training step (measured)

A deliberately trivial network was timed so later plans are sized from measurement. This is a throughput and memory probe, not a model proposal. The network is three 3x3 convolution layers with 32, 64 and 128 channels, max pooling, global average pooling and a linear layer with 2 outputs (93,506 parameters). Input: random tensors, batch of 256 images of 3 x 224 x 224. AdamW optimiser, 5 warm-up steps then 50 timed steps, data already on the GPU (so this excludes data loading and decoding).

| Precision | Time per step | Images per second | Peak allocated GPU memory |
|---|---|---|---|
| float32 | 134.5 ms | 1903 | 5771 MiB |
| bfloat16 autocast | 77.6 ms | 3300 | 3787 MiB |
| float16 autocast with GradScaler | 83.4 ms | 3068 | 3787 MiB |

What this tells us and what it does not:
- Mixed precision cut peak memory by about 34 percent and step time by about 42 percent for this network, so it should be the default assumption when sizing, to be re-checked per model.
- The memory here is dominated by activations at full 224 x 224 resolution with batch 256, not by parameters. Activation memory is what will limit batch size for larger backbones within 12 GB, of which about 1 GiB is taken by the desktop.
- The figures are for one synthetic workload. They say nothing about the speed of any real backbone, and real data loading (video decoding, face cropping) will probably be the bottleneck. This must be measured when the pipeline exists.

### Network download throughput (measured, Session 1)

- Date: 2026-10-24, between about 16:55 and 17:10 local time (reconstructed from file timestamps; an earlier version of this line said about 18:00, which was a wrong guess and is corrected here). One HTTP range request of the first 200 MiB (209,715,200 bytes) of the PyTorch cu130 wheel from download-r2.pytorch.org, discarded.
- Result: 51.3 seconds, 4.09 MB/s (about 33 Mbit/s), a single connection.
- Caveats: one server, one connection, one moment, and five background research agents were also using the network. Other servers and parallel connections may be faster or slower. At 4 MB/s a full 404 GB would take about 27 hours, so the download order in DATA_LOG.md must put the most important datasets first.

### Environment changes in Session 1

- CPU name: the measured CPU is the AMD Ryzen 9 7900X. The notes in CLAUDE.md section 6 and in the first prompt say 7900K; the measured value stands.
- LaTeX tool versions (MiKTeX): recorded by the template stream; see below when merged.

### Reproduction

```
uv sync
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

The probe script was kept outside the repository (scratch directory) because Session 0 forbids training code in the repository.

## Experiment template

```
## E-NNN: <title>   (Iteration I-N)
- Date / Environment / Git commit / Seed(s):
- Hypothesis (written BEFORE the run):
- Success criterion and failure criterion (written BEFORE the run):
- Data and split IDs:
- Command and config:
- Result (numbers, with seeds and intervals):
- Was the hypothesis supported? Why or why not:
- Decision triggered (D-NNN) / next step:
```

## Stage 2 verification log (not experiments on models; acceptance checks of the harness)

### V-001: evaluation harness acceptance checks A1 to A5 and A8 (PROTOCOL.md section 8)
- Date: 2026-10-24. Environment: ENV-DESKTOP-GPU, CPU only, `.venv` (CPython 3.13.15, numpy, scikit-learn only inside tests). Code: `src/dfd/eval/`, tests: `tests/eval/` (written by subagent stream S2-A, reviewed by the main agent).
- Command: `.venv/Scripts/python.exe -m pytest tests/eval -q`. Result in the main agent's own run: 80 passed in 11.6 s.
- Independent cross-check by the main agent (5 random score sets of 2,000 samples with heavily tied scores): AUC equals scikit-learn's to 10 decimals; log loss and Brier differ by at most 1.1e-16; a 4-sample hand example gives AUC 0.75.
- A3 (as reported by S2-A, from its simulation of 200 data sets): the 95 percent cluster bootstrap interval covered the true AUC in 192 of 200 (0.96); the test requires at least 0.90.
- A8 (as reported by S2-A): a random scorer through the whole pipeline gave every interval containing 0.5 and every AUC within 0.06 of 0.5 (fixed seed chosen before running); a one-off check with 30 other synthetic files had 117 of 120 intervals containing 0.5.
- Bootstrap speed (as reported by S2-A): 2000 resamples on 50,000 samples in 5,000 groups take 1.0 to 2.1 s; a full run of the command line on 10,000 validation and 50,000 test rows took 9 s.
- Not covered here: A6 and A7 (stream S2-B, manifests and leakage check).
- Status: A1 to A5 and A8 pass. The real ledger `ledger/final_ledger.jsonl` has not been written; tests use temporary ledgers.

### V-002: Stage 1 and Stage 2 gate checks A6 and A7 (PROTOCOL.md section 8)
- Date: 2026-10-24. Code: `src/dfd/data/`, tests `tests/data/` (stream S2-B, reviewed by the main agent). Command: `.venv/Scripts/python.exe -m pytest tests/data -q -rs`. Main agent's own run after FF++ completed: 49 passed, 0 skipped.
- A7 counts measured on the real files (S2-B, with the commands in docs/_agents/S2B.md): ASVspoof 2019 LA manifest 121,461 rows (25,380 train, 24,844 dev, 71,237 eval; 20, 20 and 67 speakers, none in two partitions); ProGAN validation and test 8,000 images each (4,000 real, 4,000 fake, 20 classes); FF++ c23 5,000 rows (1,000 originals and 1,000 each of the four methods); Celeb-DF v2 6,529 videos (590, 300, 5,639) with 59 identity groups; CNN subset 56,003 and diffusion test 10,000 images.
- A6: the leakage check exits non-zero on a manufactured overlap and zero on clean splits (tests). On the real image sets with content hashing it first failed (800 hashes shared across splits, then 1,200 samples including stylegan2 horse); after the exclusion rule of D-026 it passes on 80,803 files. ASVspoof, FF++ and Celeb-DF were not content-hashed.
- Status: A6 and A7 pass. Together with V-001 the Stage 2 gate A1 to A8 passes.
- Update after D-035 (same day): `dev_x` slices written and A9 implemented on both sides. tests/data 67 passed, tests/eval 89 passed. Measured slices: Celeb-DF v2 dev_x 1,728 videos (146 real, 1,582 fake; 15 identity groups: id0, id1, id10, id12, id16, id20, id22, id26, id31, id41, id42, id47, id48, id50, id53), test_x 4,801 videos (744 real incl. 300 YouTube-real, 4,057 fake); CNN subset (7 draw units, Amendment 1) dev_x stargan 1,999 / 1,999 and whichfaceisreal 1,000 / 1,000 (real / fake), test_x 48,805 images after removing 1,200 duplicate reals, worst real share stylegan2 7,188 / 7,988; In-the-Wild 54 speakers, 31,779 clips; DFD 28 actors, 3,431 videos (test_x only). The leakage check with content hashing passes on the image sets (80,803 files) and on Celeb-DF + In-the-Wild + DFD (41,739 files).

### M-001: data-preparation speed on FF++ c23 (measurement, not an experiment)
- Date: 2026-10-24. Environment: ENV-DESKTOP-GPU, CPU only, Python 3.13, OpenCV 5.0.0.93 (headless). Produced by stream S2-C with the commands in docs/_agents/S2C.md; the main agent re-ran only the tests (26 passed) and checked that none of 15 sampled FF++ and Celeb-DF videos has an audio stream.
- Caveat on every number: downloads and extraction were running, with 13 to 17 percent of the CPU busy before each run; the 200-video sample is about 220 MB, so reads were warm; cold-disk reads were not measured.
- Decode of 16 frames per video, one thread, 67 videos: OpenCV sequential 0.74 s per video, ffmpeg pipe 0.74 s, OpenCV seek 2.88 s; pixel-identical frames.
- Detectors on 7,488 sampled FF++ frames (see D-034): YuNet 12 ms per frame, UltraFace 5 ms, SCRFD 30 ms; all found a face in 100 percent.
- Pipeline throughput (200 videos, 16 frames, 224 px, JPEG 95, YuNet): 1.25, 4.13, 5.91, 6.35 and 6.22 videos per second with 1, 4, 8, 16 and 24 workers; growth stops after 8 workers and the cause was not found. Projection for the 5,000 FF++ c23 videos: about 13 minutes at 16 workers; crops 222,786 bytes per video, about 1.1 GB for 5,000 videos.
- Audio: ASVspoof 2019 LA FLAC loads at 4.76 ms per file (200 random files); about 582 s single-threaded for all 122,299 files. No FF++ (0 of 20) and no Celeb-DF v2 (0 of 250) video has an audio stream, so the audio branch of video needs LAV-DF or FakeAVCeleb.
- LAV-DF speed is a projection from 40 re-encoded FF++ videos (315 videos per second at 16 workers), not a measurement.
- Full FF++ crop run started 2026-10-24 at about 21:55 (log data/_logs/crops_ffpp_c23.log); its measured rate in the first minute was about 6 videos per second, consistent with the benchmark.

### M-002: face detector comparison on FF++ val and Celeb-DF dev_x frames (measurement, not an experiment)
- Date: 2026-10-24. Stream S2-C (docs/_agents/S2C.md, Addendum 2; code src/dfd/prep/detcmp.py); the main agent re-ran tests/prep (26 passed) and checked the licence lines against the InsightFace README. CPU only, machine busy (14 to 20 percent CPU before each run); frames only, no labels or scores used.
- Frames, seed 20261024, lists saved: 300 frames from 300 distinct FF++ official-val videos (all 140 real val videos and 40 per manipulation method) and 300 frames from 300 distinct Celeb-DF v2 dev_x videos (all 146 real and 154 fake).
- Face found in 300 of 300 frames for YuNet, UltraFace and SCRFD on both sets; the detectors never disagreed on whether a face exists (0 of 300 for every pair).
- Speed, ms per frame (FF++ / Celeb-DF): UltraFace 4.8 / 4.8, YuNet 12.1 / 12.4, SCRFD 29.7 / 30.9.
- Box agreement of the largest box between detectors: mean IoU 0.887 to 0.923; 100 percent of frames with IoU at least 0.5; no frame where one box is under half the area of another; median box area ratio 0.97 to 1.05; median centre shift about 2 percent of the box side. The 8 lowest-agreement frames per set (lowest pairwise IoU 0.70 to 0.81) show the same face in every tile.
- Licences (read from the repositories' own files): YuNet MIT; UltraFace MIT; SCRFD weights non-commercial research only (InsightFace: "ALL models are available for non-commercial research purposes only").
- What this does not show: these frames contain none of the hard cases (rotated, tiny, face-free images) where the earlier stress test separated the detectors, and the effect of a few percent of box shift on any model's accuracy was not measured. JPEG quality 95 for the stored crops is a candidate preprocessing iteration and was not changed.
- Reading for D-034: no geometric reason to re-crop; D-034 stays Decided autonomously until the user confirms it.

### M-003: fine-tuning step time and peak VRAM of the image encoders (measurement, not an experiment)
- Date / Environment / Git commit: 2026-10-24 night to 2026-10-25 00:02 / ENV-DESKTOP-GPU (RTX 5070, 12 GB, torch 2.14.1+cu130) / src/dfd/sweep/step_time.py at the commit that follows this entry.
- What: one training step = forward, backward and AdamW update of the whole encoder plus a linear head, fp16 autocast with gradient scaling, random uint8 224 px frames, median of 10 timed steps after 3 warm-up steps. Weights as in the sweep (the Xception row used random weights, which does not change the cost).
- Caveat that limits these numbers: the run was PARTLY OVERLAPPED. The E-001 queue and the LAV-DF job started on the same GPU and disk at 00:01 while the later rows (IMG-13, IMG-15) were being measured, so those step times may be too long; the rows for IMG-01, IMG-04, IMG-07 and IMG-08 were measured before the start but this was not recorded, so they are also unconfirmed. The run was stopped after 11 of 14 rows. A clean rerun with the GPU free is chained (scripts/sweep/after_e001_steptime.sh, output data/_logs/step_time_alone.json) and supersedes this table. Raw lines: data/_logs/step_time_partial_overlapped.jsonl.
- Result (single run, no interval):

| Candidate | Batch | Step (ms) | Frames per second | Peak VRAM (GiB) | Trainable parameters |
|---|---|---|---|---|---|
| IMG-07 DINOv2-S | 16 / 32 | 28.4 / 45.8 | 563 / 700 | 1.19 / 2.00 | 22.1 M |
| IMG-08 DINOv2-B | 16 / 32 | 72.2 / 133.8 | 222 / 239 | 2.87 / 4.49 | 86.6 M |
| IMG-04 Xception | 16 / 32 | 56.1 / 110.7 | 285 / 289 | 2.36 / 4.29 | 20.8 M |
| IMG-01 EfficientNet-B4 | 16 / 32 | 71.8 / 140.8 | 223 / 227 | 3.82 / 7.28 | 17.5 M |
| IMG-13 CLIP ViT-L/14 (OpenAI) | 16 / 32 | 246.3 / 19,078 | 65 / 1.7 | 9.13 / 13.75 | 304.0 M |
| IMG-15 CLIP ViT-L/14 (LAION) | 16 | 274.5 | 58 | 8.38 | 304.0 M |
| IMG-15 at batch 32, IMG-17 SigLIP base | | not measured | | | |

- Reading: a peak above the 12 GB card (13.75 GiB at batch 32 for IMG-13) means the allocation spilled into shared system memory and the step took 19 s; it is a failure to fit, not a speed. These numbers are inputs to D-042 and are not a result about detection.

## Diagnostic battery (defined once, run after every iteration)

Every iteration, I-0 included, runs the same battery before its verdict; the next iteration's lever is chosen from it (ITERATIONS.md). Each item is a table or figure under data/_results/<run>/ and is listed in the iteration's experiment entry. All AUC values carry the 95 percent group-bootstrap interval of the protocol; seeds 0, 1, 2.
1. Per-dataset and per-generator (or per-method) AUC with intervals.
2. Train versus validation gap, to tell bias (both low) from variance (large gap).
3. In-dataset validation versus dev_x gap, to tell distribution shift from the other two.
4. Calibration curve (15 equal-width bins), ECE and Brier before and after temperature scaling.
5. Score histograms of real and fake examples, per dataset.
6. Results split by compression level, face size and number of faces (the MULTIPLE_FACES and NO_FACE_DETECTED counts of D-034 included).
7. The 20 worst errors (10 false alarms, 10 misses) reviewed by eye and classified; the classification is written down, not summarised from memory.
8. Seed variance: standard deviation over the 3 seeds next to the bootstrap half-width, to tell training noise from sampling noise.
Diagnosis vocabulary: bias, variance, shift, calibration, aggregation, missing signal. Entry template for an iteration: Observation (numbers) -> Diagnosis -> Mathematical hypothesis -> Change -> Prediction (sign and size) -> Result -> Verdict (supported or not). The hypothesis, prediction and failure line are written before the run (CLAUDE.md rule 2).

## Model comparison table (defined once)

One table per task (T-V face video, T-I generated images, T-A speech audio); one row per candidate encoder, including candidates that were dropped, with the reason. Columns: candidate ID; checkpoint and revision; weights source URL; licence and research-only label; SHA-256 (local, verified); parameters; feature size; input size; FLOPs per forward pass; peak VRAM at the section 7 batch; extraction time per 1,000 inputs (model-only and end-to-end); in-dataset validation AUC; dev_x AUC with 95 percent interval; gap; rank under the D-036 rule; tied with the leader (yes or no); entered I-1 (yes or no). Generated by a script from the feature metadata, data/_logs/encoder_weights.json and the score files; not edited by hand.

## Model comparison tables (generated, do not edit by hand)

One block per task, written by src/dfd/sweep/report.py after each ranking run.

<!-- AUTO TABLE T-V begin -->
#### T-V (frozen image encoders on face video frames), generated 2026-10-25 00:37 from run 221e12e

Weights and cost (one end-to-end run each; FLOPs and the model-only cost are not measured yet):

| Candidate | Checkpoint and revision | Weights source | Licence and label | SHA-256 (local, verified) | Parameters | Feature size | Input size | FLOPs | Peak VRAM, extraction (GiB) | Extraction s per 1,000 inputs, end-to-end | Model-only s per 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG-07 | facebook/dinov2-small @ ed25f3a31f01 | facebook/dinov2-small | Apache-2.0; commercial allowed by tag | ae1e99fcefd534ed... | 22,056,576 | 384 | 224 | NR (not measured) | 1.23 | 1.5 | NR (section 7 cost record not run) |
| IMG-17 | google/siglip-base-patch16-224 @ 7fd15f0689c7 | google/siglip-base-patch16-224 | Apache-2.0 tag; commercial allowed by tag | 2c63cb7d1f2e95ba... | 92,884,224 | 768 | 224 | NR (not measured) | 2.45 | 1.5 | NR (section 7 cost record not run) |
| IMG-08 | facebook/dinov2-base @ f9e44c814b77 | facebook/dinov2-base | Apache-2.0; commercial allowed by tag | d73036b56966966d... | 86,580,480 | 768 | 224 | NR (not measured) | 2.36 | 2.1 | NR (section 7 cost record not run) |
| IMG-15 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K @ 162703219714 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K | MIT tag; card says research use | 0fce40ac684817a1... | 303,966,208 | 768 | 224 | NR (not measured) | 4.52 | 4.1 | NR (section 7 cost record not run) |
| IMG-13 | openai/clip-vit-large-patch14 @ 32bd64288804 | openai/clip-vit-large-patch14 | NR in its repo; research-only (accepted, D-038) | a2bf730a0c7debf1... | 303,966,208 | 768 | 224 | NR (not measured) | 5.02 | 4.6 | NR (section 7 cost record not run) |
| IMG-01 | timm/efficientnet_b4.ra2_in1k @ 442c00e15609 | timm/efficientnet_b4.ra2_in1k | Apache-2.0 tag; ImageNet-trained, treated as research-only | 8030f9c929ed71a0... | 17,548,616 | 1792 | 320 | NR (not measured) | 5.08 | 2.4 | NR (section 7 cost record not run) |
| IMG-04 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth @ release v0.1 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth | apache-2.0 declared by timm; upstream Keras licence NR; research-only | 43020ad2812c333d... | 20,806,952 | 2048 | 299 | NR (not measured) | 10.52 | 3.7 | NR (section 7 cost record not run) |

Results (exploratory, dev_x; step-time file tags are in the fit column):

| Candidate | Validation AUC | dev_x AUC [95%] | Gap | Rank (I-1 order) | Tied with leader | Entered I-1 candidates | I-1 fit (D-042, video only) |
|---|---|---|---|---|---|---|---|
| IMG-07 | 0.7985 | 0.6933 [0.6639, 0.7314] | +0.1052 | 1 | yes | yes | not measured |
| IMG-17 | 0.8767 | 0.7396 [0.7031, 0.7832] | +0.1371 | 2 | yes | yes | not measured |
| IMG-08 | 0.8170 | 0.6745 [0.6323, 0.7239] | +0.1425 | 3 | yes | yes | not measured |
| IMG-15 | 0.8635 | 0.7331 [0.6764, 0.7966] | +0.1304 | 4 | yes | no | not measured |
| IMG-13 | 0.9046 | 0.7464 [0.6721, 0.8306] | +0.1582 | 5 | yes | no | not measured |
| IMG-01 | 0.7478 | 0.6471 [0.5967, 0.6987] | +0.1007 | 6 | no | no | not measured |
| IMG-04 | 0.7184 | 0.6162 [0.5744, 0.6633] | +0.1022 | 7 | no | no | not measured |

Considered and not in the sweep:
- IMG-05: Xception41 (aligned variant): replaced by the original Xception IMG-04 (D-038)
- IMG-09: DINOv2 large: reserve, fine-tuning estimate tight on 12 GB (O007_SWEEP 3.1)
- IMG-10..12,19: DINOv3: manual access gate and acknowledgement duty, excluded on purpose (D-038)
- IMG-14: OpenAI CLIP timm re-upload: fallback only, not needed because IMG-13 was accepted (D-038)
- From-scratch reference (IMG-20): not run (excluded from selection in any case).
<!-- AUTO TABLE T-V end -->

## Experiment log

No experiment has been run yet. Planned entries follow; their hypotheses were written before any run.

## E-001: frozen image encoders on face video frames, ranked on dev_x   (Iteration I-0)
- Status: RUN on 2026-10-25 (automatic ranking, dfd.sweep.rank). The hypothesis and criteria below were written on 2026-10-24 before any feature extraction and are not edited after the run. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / 221e12e / head seeds 0, 1, 2.
- Hypothesis (written BEFORE the run): Among the frozen encoders of the shortlist, at least one self-supervised or vision-language encoder (the DINOv2 or CLIP-style families) reaches a higher dev_x clip AUC on Celeb-DF v2 identities not used in training than the best supervised ImageNet CNN (EfficientNet, Xception), when all are trained on FF++ c23 frames with the same linear head. Basis: R-011 (a frozen CLIP feature space generalised across generators better than trained networks, protocol: trained on ProGAN, tested on other generators), R-017 (strong supervised detectors lose about 0.17 mean AUC from within-domain to cross-domain), R-018 (some supervised features transfer worse). Counter-evidence: R-044 (face detectors learn identity cues) and the absence of any frozen-feature result for face forgery in the papers read, so the direction is uncertain. Every candidate is also expected to score lower on dev_x than on the in-dataset validation split (a generalisation gap).
- Success criterion (written BEFORE the run): (a) the paired 95 percent bootstrap interval of dev_x AUC (best non-CNN candidate minus best supervised CNN) lies above zero; and (b) the best candidate's lower interval bound on dev_x AUC is above 0.60.
- Failure criterion (written BEFORE the run): (f1) the interval of the difference in (a) includes zero or lies below zero: the hypothesis is not supported, and the selection rule of O007_SWEEP.md section 8 decides (the cheaper candidate wins ties); (f2) the best candidate's lower bound on dev_x AUC is 0.55 or below: frozen features are judged inadequate for this task and the sweep is declared inconclusive.
- Data and split IDs: FF++ c23 official split (train, val) and the Celeb-DF v2 dev_x slice (15 of 59 identity groups, D-035); IDs to be filled from data/_splits at the run.
- Command and config: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.rank --task T-V`; config in data/_results/T-V/rank.json.
- Result (numbers, with seeds and intervals): see the generated block below (data/_results/T-V/rank.md).
- Was the hypothesis supported? Why or why not: mechanical verdict on the pre-declared criteria: SUPPORTED (details in the generated block; exploratory dev_x result, D-041).
- Decision triggered (D-NNN) / next step: the selection decision for T-V and the I-1 candidates.

<!-- AUTO E-001 begin -->
### E-001 ranking (T-V): frozen image encoders on face video frames
Run 2026-10-25 00:37, commit 221e12e, seeds [0, 1, 2], 2000 bootstrap resamples (base seed 20261024), head: linear, class-balanced BCE, AdamW lr 0.001, weight decay 0.0001, batch 1024, at most 200 epochs, patience 10. Train ffpp_train, validation ffpp_val, dev_x celebdf_dev_x: 1728 clips (146 real, 1582 fake) in 15 groups, 0 excluded without a score.

| Rank (I-1 order) | Candidate | dev_x AUC (mean of seeds) | 95% interval | seed sd | validation AUC | gap | diff vs leader [95%] | permutation p | tied with leader |
|---|---|---|---|---|---|---|---|---|---|
| 1 | IMG-07 | 0.6933 | [0.6639, 0.7314] | 0.0107 | 0.7985 | +0.1052 | -0.053 [-0.126, +0.010] | 0.220 | yes |
| 2 | IMG-17 | 0.7396 | [0.7031, 0.7832] | 0.0081 | 0.8767 | +0.1371 | -0.007 [-0.113, +0.091] | 0.930 | yes |
| 3 | IMG-08 | 0.6745 | [0.6323, 0.7239] | 0.0072 | 0.8170 | +0.1425 | -0.072 [-0.171, +0.017] | 0.202 | yes |
| 4 | IMG-15 | 0.7331 | [0.6764, 0.7966] | 0.0084 | 0.8635 | +0.1304 | -0.013 [-0.104, +0.082] | 0.795 | yes |
| 5 | IMG-13 | 0.7464 | [0.6721, 0.8306] | 0.0044 | 0.9046 | +0.1582 | leader | n/a | yes |
| 6 | IMG-01 | 0.6471 | [0.5967, 0.6987] | 0.0025 | 0.7478 | +0.1007 | -0.099 [-0.172, -0.023] | 0.110 | no |
| 7 | IMG-04 | 0.6162 | [0.5744, 0.6633] | 0.0066 | 0.7184 | +0.1022 | -0.130 [-0.191, -0.071] | 0.007 | no |

Leader (highest mean dev_x AUC): IMG-13. Tie set (paired interval against the leader includes zero), cheapest first: IMG-07, IMG-17, IMG-08, IMG-15, IMG-13. Winner under the D-036 rule (cheapest in the tie set): IMG-07 (the cost rule decided).
Cost used for the rule: end-to-end extraction time per 1,000 inputs, then parameters, from extraction sidecar (end-to-end, one run).
Order for I-1 (tie set first, then the rest by AUC; ordering decided autonomously, D-042 fit rule applied later): IMG-07, IMG-17, IMG-08, IMG-15, IMG-13, IMG-01, IMG-04.
From-scratch reference: not run (excluded from selection in any case).

| Candidate | dev_x EER | balanced accuracy at the validation threshold | ECE before / after temperature | Brier before / after | mean temperature |
|---|---|---|---|---|---|
| IMG-07 | 0.3582 | 0.6170 | 0.2048 / 0.2217 | 0.2963 / 0.3050 | 0.877 |
| IMG-17 | 0.3356 | 0.6696 | 0.0312 / 0.0430 | 0.2116 / 0.2122 | 0.902 |
| IMG-08 | 0.3788 | 0.6270 | 0.1118 / 0.1401 | 0.2704 / 0.2815 | 0.786 |
| IMG-15 | 0.3270 | 0.6684 | 0.1768 / 0.1958 | 0.2803 / 0.2883 | 0.880 |
| IMG-13 | 0.3173 | 0.6581 | 0.2845 / 0.2874 | 0.3458 / 0.3476 | 0.984 |
| IMG-01 | 0.4062 | 0.6166 | 0.0636 / 0.0710 | 0.2199 / 0.2189 | 1.462 |
| IMG-04 | 0.4137 | 0.5751 | 0.0492 / 0.0696 | 0.2396 / 0.2428 | 0.813 |

Pre-declared criteria, evaluated by script: best candidate IMG-13, lower bound 0.6721 (criterion (b): above 0.6: met; inconclusive line: not crossed).
Family comparison, best IMG-13 minus best IMG-01: +0.0994 [+0.0228, +0.1723], interval above zero (met).
Mechanical verdict on the pre-declared criteria: SUPPORTED. This is an exploratory dev_x result (D-041); the sensitivity column is a check, not a second rule.
<!-- AUTO E-001 end -->

## E-002: frozen image encoders on generated images, ranked on dev_x   (Iteration I-0)
- Status: PLANNED. Written 2026-10-24 before any extraction. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): to be filled at the run / ENV-DESKTOP-GPU / to be filled / head seeds 0, 1, 2.
- Hypothesis (written BEFORE the run): A CLIP-style frozen encoder reaches a higher dev_x AUC on the two held-out CNN-synthesis generators than every supervised ImageNet CNN, when all are trained on the ProGAN validation images (14 classes). Basis: R-011 (trained on ProGAN, a frozen CLIP-feature linear probe reached 99.31 mean AP on GAN-family test sets and 95.00 on unseen diffusion and autoregressive ones, against 94.19 and 75.51 for the best trained network; its training set was far larger than ours).
- Success criterion (written BEFORE the run): the paired 95 percent interval of dev_x AUC (best CLIP-style candidate minus best supervised CNN) lies above zero, and the best candidate's lower bound on dev_x AUC is above 0.65.
- Failure criterion (written BEFORE the run): the interval includes zero or lies below zero (not supported; the cost rule decides), or the best candidate's lower bound is 0.55 or below (inconclusive).
- Data and split IDs: ProGAN validation set split by class (14 train, 6 val) and the CNN-synthesis dev_x generator units (2 of 7, D-035 Amendment 1); IDs to be filled at the run.
- Command and config: to be filled at the run.
- Result (numbers, with seeds and intervals): none yet.
- Was the hypothesis supported? Why or why not: none yet.
- Decision triggered (D-NNN) / next step: the selection decision for T-I and the I-1 candidates.

## E-003: frozen speech encoders on speech deepfakes, ranked on dev_x   (Iteration I-0)
- Status: PLANNED. Written 2026-10-24 before any extraction. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): to be filled at the run / ENV-DESKTOP-GPU / to be filled / head seeds 0, 1, 2.
- Hypothesis (written BEFORE the run): A self-supervised speech encoder (wav2vec 2.0 or WavLM) reaches a higher dev_x AUC on In-the-Wild speakers not used in training than both the Whisper encoder and the small from-scratch reference, when trained on ASVspoof 2019 LA. Basis: R-029 (a wav2vec 2.0 front end lowered EER from 11.47 to 6.15 percent, best seed, train ASVspoof 2019 LA and test ASVspoof 2021 LA) and R-033 (models trained on ASVspoof 2019 reach EERs far above their in-corpus values on In-the-Wild, about 33.9 percent for RawNet2).
- Success criterion (written BEFORE the run): the paired 95 percent intervals of dev_x AUC (best self-supervised candidate minus the Whisper candidate, and minus the from-scratch reference) both lie above zero, and the best candidate's lower bound on dev_x AUC is above 0.60.
- Failure criterion (written BEFORE the run): either interval includes zero or lies below zero (not supported; the cost rule decides), or the best candidate's lower bound is 0.55 or below (inconclusive).
- Data and split IDs: ASVspoof 2019 LA train and dev (val) and the In-the-Wild dev_x slice (12 of 54 speakers, D-035); IDs to be filled at the run.
- Command and config: to be filled at the run.
- Result (numbers, with seeds and intervals): none yet.
- Was the hypothesis supported? Why or why not: none yet.
- Decision triggered (D-NNN) / next step: the selection decision for T-A and the I-1 candidates.

