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

### M-004: audio label mapping, trivial audio features and per-generator image AUC (diagnosis, not an experiment)
- Date 2026-10-25, CPU only, commit see git log (scripts/sweep/m004.py, output data/_results/M-004/m004.json). Reads train, validation and dev_x data only. It changes no selection (D-049) and tests no pre-declared hypothesis: the questions were put by the orchestrator after E-003 and the answers are reported as sign, size and interval. Intervals: group bootstrap over speakers (2,000 resamples) for audio; for the two image units, which are one group each, the interval resamples images inside the unit and is not a population interval.
- (a) Label mapping. The In-the-Wild dev_x features (6,991 items) agree with meta.csv on every label and speaker (0 mismatches); the ASVspoof 2019 LA train (25,380) and dev (24,844) features agree with the cm protocol files (0 mismatches). The from-scratch reference AU-REF reaches AUC 0.99996, 0.99997 and 0.99997 (seeds 0, 1, 2) on ASVspoof dev, in the training domain, and 0.232, 0.219 and 0.286 on In-the-Wild dev_x. Reading: no label inversion; the below-chance value on dev_x is a property of the data shift, not of the label mapping. This rejects the label-bug hypothesis.
- (b) AUC of single trivial features used as the fake score (0.5 means no information; below 0.5 means the fake class has the smaller value). Leading silence is the time before the first sample with absolute value above 0.005 of full scale in the 4 s window; the repeat-padded fraction is one minus the original length over 64,600 samples, floored at zero.

| feature | ASVspoof train (20 speakers) | ASVspoof dev (20 speakers) | In-the-Wild dev_x (12 speakers) |
|---|---|---|---|
| leading silence | 0.416 [0.350, 0.488] | 0.375 [0.299, 0.453] | 0.719 [0.425, 0.933] |
| clip duration | 0.479 [0.428, 0.531] | 0.467 [0.393, 0.540] | 0.622 [0.532, 0.670] |
| RMS energy | 0.572 [0.525, 0.618] | 0.594 [0.537, 0.649] | 0.248 [0.073, 0.546] |
| repeat-padded fraction | 0.530 [0.478, 0.581] | 0.543 [0.471, 0.614] | 0.394 [0.348, 0.481] |

  Medians (real / fake): leading silence 0.43 s / 0.12 s on ASVspoof dev, 0.002 s / 0.249 s on In-the-Wild; RMS 0.113 / 0.123 and 0.066 / 0.038; duration 3.39 s / 3.27 s and 2.88 s / 3.86 s. Reading: the sign of leading silence and of RMS energy reverses between the training corpus and In-the-Wild (fakes start earlier and are louder in ASVspoof, start later and are quieter in In-the-Wild), and the RMS AUC on dev_x (0.248) is of the same size as the from-scratch reference (0.246). This supports, in sign, the hypothesis that recording-condition cues of ASVspoof reverse in In-the-Wild; it does not establish it in size: with 12 speakers the intervals of leading silence and RMS include 0.5, only duration (0.622 [0.532, 0.670]) and the repeat-padded fraction (0.394 [0.348, 0.481]) exclude it. Single features of this kind are weak in-domain (all ASVspoof intervals are within 0.30 to 0.65), so the frozen and from-scratch models use more than these four.
- (c) Per-generator AUC on the two T-I dev_x units (mean of 3 head seeds; image-level interval inside the unit):

| encoder | StarGAN (3,998 images) | whichfaceisreal (2,000 images) | pooled, 2 units |
|---|---|---|---|
| IMG-13 | 0.9988 [0.9983, 0.9992] | 0.9830 [0.9788, 0.9870] | 0.9946 [0.9934, 0.9956] |
| IMG-15 | 0.9978 [0.9971, 0.9985] | 0.7790 [0.7591, 0.8000] | 0.9578 [0.9533, 0.9620] |

  Reading: both encoders separate StarGAN almost perfectly; they differ on the second unit (0.983 against 0.779), which the pooled AUC (dominated by StarGAN, two thirds of the images) hides. The pre-declared winner IMG-13 is the better of the two on both units; with two units nothing is said about other generators. The file extension does not separate the classes in any T-I split (ProGAN real and fake are all png, StarGAN png, whichfaceisreal jpeg for both classes).

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
#### T-V (frozen image encoders on face video frames), generated 2026-10-25 02:20 from run e998f04

Weights and cost (one end-to-end run each; FLOPs and the model-only cost are not measured yet):

| Candidate | Checkpoint and revision | Weights source | Licence and label | SHA-256 (local, verified) | Parameters | Feature size | Input size | FLOPs | Peak VRAM, extraction (GiB) | Extraction s per 1,000 inputs, end-to-end | Model-only s per 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG-07 | facebook/dinov2-small @ ed25f3a31f01 | facebook/dinov2-small | Apache-2.0; commercial allowed by tag | ae1e99fcefd534ed... | 22,056,576 | 384 | 224 | NR (not measured) | 0.38 | 0.9 | NR (section 7 cost record not run) |
| IMG-17 | google/siglip-base-patch16-224 @ 7fd15f0689c7 | google/siglip-base-patch16-224 | Apache-2.0 tag; commercial allowed by tag | 2c63cb7d1f2e95ba... | 92,884,224 | 768 | 224 | NR (not measured) | 1.3 | 1.4 | NR (section 7 cost record not run) |
| IMG-08 | facebook/dinov2-base @ f9e44c814b77 | facebook/dinov2-base | Apache-2.0; commercial allowed by tag | d73036b56966966d... | 86,580,480 | 768 | 224 | NR (not measured) | 0.95 | 1.7 | NR (section 7 cost record not run) |
| IMG-15 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K @ 162703219714 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K | MIT tag; card says research use | 0fce40ac684817a1... | 303,966,208 | 768 | 224 | NR (not measured) | 2.72 | 4.0 | NR (section 7 cost record not run) |
| IMG-13 | openai/clip-vit-large-patch14 @ 32bd64288804 | openai/clip-vit-large-patch14 | NR in its repo; research-only (accepted, D-038) | a2bf730a0c7debf1... | 303,966,208 | 768 | 224 | NR (not measured) | 2.85 | 4.5 | NR (section 7 cost record not run) |
| IMG-01 | timm/efficientnet_b4.ra2_in1k @ 442c00e15609 | timm/efficientnet_b4.ra2_in1k | Apache-2.0 tag; ImageNet-trained, treated as research-only | 8030f9c929ed71a0... | 17,548,616 | 1792 | 320 | NR (not measured) | 1.31 | 1.7 | NR (section 7 cost record not run) |
| IMG-04 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth @ release v0.1 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth | apache-2.0 declared by timm; upstream Keras licence NR; research-only | 43020ad2812c333d... | 20,806,952 | 2048 | 299 | NR (not measured) | 1.96 | 3.3 | NR (section 7 cost record not run) |

Results (exploratory, dev_x; step-time file tags are in the fit column):

| Candidate | Validation AUC | dev_x AUC [95%] | Gap | Rank (I-1 order) | Tied with leader | Entered I-1 candidates | I-1 fit (D-042, video only) |
|---|---|---|---|---|---|---|---|
| IMG-07 | 0.7985 | 0.6933 [0.6639, 0.7314] | +0.1052 | 1 | yes | yes | fits (2.0 GiB at batch 32, 1.4 min per budget; step-time run 'alone') |
| IMG-17 | 0.8767 | 0.7396 [0.7031, 0.7832] | +0.1371 | 2 | yes | yes | fits (3.99 GiB at batch 32, 3.4 min per budget; step-time run 'alone') |
| IMG-08 | 0.8170 | 0.6745 [0.6323, 0.7239] | +0.1425 | 3 | yes | yes | fits (4.49 GiB at batch 32, 4.0 min per budget; step-time run 'alone') |
| IMG-15 | 0.8635 | 0.7331 [0.6764, 0.7966] | +0.1304 | 4 | yes | no | does not fit (12.23 GiB at batch 32, 300.0 min per budget; step-time run 'alone') |
| IMG-13 | 0.9046 | 0.7464 [0.6721, 0.8306] | +0.1582 | 5 | yes | no | does not fit (13.75 GiB at batch 32, 457.1 min per budget; step-time run 'alone') |
| IMG-01 | 0.7478 | 0.6471 [0.5967, 0.6987] | +0.1007 | 6 | no | no | fits (7.28 GiB at batch 32, 4.2 min per budget; step-time run 'alone') |
| IMG-04 | 0.7184 | 0.6162 [0.5744, 0.6633] | +0.1022 | 7 | no | no | fits (4.29 GiB at batch 32, 3.3 min per budget; step-time run 'alone') |

Considered and not in the sweep:
- IMG-05: Xception41 (aligned variant): replaced by the original Xception IMG-04 (D-038)
- IMG-09: DINOv2 large: reserve, fine-tuning estimate tight on 12 GB (O007_SWEEP 3.1)
- IMG-10..12,19: DINOv3: manual access gate and acknowledgement duty, excluded on purpose (D-038)
- IMG-14: OpenAI CLIP timm re-upload: fallback only, not needed because IMG-13 was accepted (D-038)
- From-scratch reference (IMG-20): reported, excluded from the selection. dev_x AUC 0.5166 [0.5033, 0.5324] over 3 seeds; protocol difference: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.
<!-- AUTO TABLE T-V end -->

<!-- AUTO TABLE T-I begin -->
#### T-I (frozen image encoders on generated images), generated 2026-10-25 02:31 from run 214076b

Weights and cost (one end-to-end run each; FLOPs and the model-only cost are not measured yet):

| Candidate | Checkpoint and revision | Weights source | Licence and label | SHA-256 (local, verified) | Parameters | Feature size | Input size | FLOPs | Peak VRAM, extraction (GiB) | Extraction s per 1,000 inputs, end-to-end | Model-only s per 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG-13 | openai/clip-vit-large-patch14 @ 32bd64288804 | openai/clip-vit-large-patch14 | NR in its repo; research-only (accepted, D-038) | a2bf730a0c7debf1... | 303,966,208 | 768 | 224 | NR (not measured) | 2.85 | 4.9 | NR (section 7 cost record not run) |
| IMG-15 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K @ 162703219714 | laion/CLIP-ViT-L-14-laion2B-s32B-b82K | MIT tag; card says research use | 0fce40ac684817a1... | 303,966,208 | 768 | 224 | NR (not measured) | 2.72 | 4.5 | NR (section 7 cost record not run) |
| IMG-08 | facebook/dinov2-base @ f9e44c814b77 | facebook/dinov2-base | Apache-2.0; commercial allowed by tag | d73036b56966966d... | 86,580,480 | 768 | 224 | NR (not measured) | 0.95 | 2.0 | NR (section 7 cost record not run) |
| IMG-17 | google/siglip-base-patch16-224 @ 7fd15f0689c7 | google/siglip-base-patch16-224 | Apache-2.0 tag; commercial allowed by tag | 2c63cb7d1f2e95ba... | 92,884,224 | 768 | 224 | NR (not measured) | 1.3 | 1.8 | NR (section 7 cost record not run) |
| IMG-07 | facebook/dinov2-small @ ed25f3a31f01 | facebook/dinov2-small | Apache-2.0; commercial allowed by tag | ae1e99fcefd534ed... | 22,056,576 | 384 | 224 | NR (not measured) | 0.38 | 1.2 | NR (section 7 cost record not run) |
| IMG-01 | timm/efficientnet_b4.ra2_in1k @ 442c00e15609 | timm/efficientnet_b4.ra2_in1k | Apache-2.0 tag; ImageNet-trained, treated as research-only | 8030f9c929ed71a0... | 17,548,616 | 1792 | 320 | NR (not measured) | 1.31 | 2.5 | NR (section 7 cost record not run) |
| IMG-04 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth @ release v0.1 | https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth | apache-2.0 declared by timm; upstream Keras licence NR; research-only | 43020ad2812c333d... | 20,806,952 | 2048 | 299 | NR (not measured) | 1.96 | 2.2 | NR (section 7 cost record not run) |

Results (exploratory, dev_x; step-time file tags are in the fit column):

| Candidate | Validation AUC | dev_x AUC [95%] | Gap | Rank (I-1 order) | Tied with leader | Entered I-1 candidates | I-1 fit (D-042, video only) |
|---|---|---|---|---|---|---|---|
| IMG-13 | 1.0000 | 0.9935 [0.9803, 0.9982] | +0.0065 | 1 | yes | yes | n/a |
| IMG-15 | 1.0000 | 0.9563 [0.7760, 0.9975] | +0.0437 | 2 | no | yes | n/a |
| IMG-08 | 0.9899 | 0.9023 [0.8496, 0.9316] | +0.0876 | 3 | no | yes | n/a |
| IMG-17 | 0.9975 | 0.8657 [0.7591, 0.9166] | +0.1318 | 4 | no | no | n/a |
| IMG-07 | 0.9685 | 0.8340 [0.7401, 0.8823] | +0.1345 | 5 | no | no | n/a |
| IMG-01 | 0.8990 | 0.6688 [0.6238, 0.7137] | +0.2302 | 6 | no | no | n/a |
| IMG-04 | 0.8140 | 0.6380 [0.4912, 0.7115] | +0.1759 | 7 | no | no | n/a |

Considered and not in the sweep:
- IMG-05: Xception41 (aligned variant): replaced by the original Xception IMG-04 (D-038)
- IMG-09: DINOv2 large: reserve, fine-tuning estimate tight on 12 GB (O007_SWEEP 3.1)
- IMG-10..12,19: DINOv3: manual access gate and acknowledgement duty, excluded on purpose (D-038)
- IMG-14: OpenAI CLIP timm re-upload: fallback only, not needed because IMG-13 was accepted (D-038)
- From-scratch reference (IMG-20): reported, excluded from the selection. dev_x AUC 0.5500 [0.5080, 0.5735] over 3 seeds; protocol difference: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.
<!-- AUTO TABLE T-I end -->

<!-- AUTO TABLE T-A begin -->
#### T-A (frozen speech encoders on speech deepfakes), generated 2026-10-25 02:49 from run 214076b

Weights and cost (one end-to-end run each; FLOPs and the model-only cost are not measured yet):

| Candidate | Checkpoint and revision | Weights source | Licence and label | SHA-256 (local, verified) | Parameters | Feature size | Input size | FLOPs | Peak VRAM, extraction (GiB) | Extraction s per 1,000 inputs, end-to-end | Model-only s per 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AU-8 | facebook/hubert-base-ls960 @ dba3bb02fda4 | facebook/hubert-base-ls960 | Apache-2.0 tag; commercial allowed by tag | 062249fffb353eab... | 94,371,712 | 1536 | 64600 | NR (not measured) | 2.4 | 2.0 | NR (section 7 cost record not run) |
| AU-4 | microsoft/wavlm-base-plus @ 4c66d4806a42 | microsoft/wavlm-base-plus | no tag, CC BY-SA 3.0 upstream; research-only (accepted, D-038) | 3bb273a6ace99408... | 94,381,936 | 1536 | 64600 | NR (not measured) | 2.38 | 2.5 | NR (section 7 cost record not run) |
| AU-6 | openai/whisper-small @ 973afd24965f | openai/whisper-small | Apache-2.0 tag, MIT upstream | 1d7734884874f1a1... | 88,154,112 | 1536 | 64600 | NR (not measured) | 1.62 | 11.5 | NR (section 7 cost record not run) |
| AU-1 | facebook/wav2vec2-base @ 0b5b8e868dd8 | facebook/wav2vec2-base | Apache-2.0 tag; commercial allowed by tag | 3249fe98bfc62fcb... | 94,371,712 | 1536 | 64600 | NR (not measured) | 2.34 | 2.0 | NR (section 7 cost record not run) |

Results (exploratory, dev_x; step-time file tags are in the fit column):

| Candidate | Validation AUC | dev_x AUC [95%] | Gap | Rank (I-1 order) | Tied with leader | Entered I-1 candidates | I-1 fit (D-042, video only) |
|---|---|---|---|---|---|---|---|
| AU-8 | 0.9942 | 0.6100 [0.5434, 0.7256] | +0.3842 | 1 | yes | yes | n/a |
| AU-4 | 0.9924 | 0.6334 [0.5291, 0.7714] | +0.3590 | 2 | yes | yes | n/a |
| AU-6 | 0.9995 | 0.6380 [0.5338, 0.7700] | +0.3615 | 3 | yes | yes | n/a |
| AU-1 | 0.9965 | 0.4975 [0.3451, 0.7092] | +0.4990 | 4 | no | no | n/a |

Considered and not in the sweep:
- IMG-05: Xception41 (aligned variant): replaced by the original Xception IMG-04 (D-038)
- IMG-09: DINOv2 large: reserve, fine-tuning estimate tight on 12 GB (O007_SWEEP 3.1)
- IMG-10..12,19: DINOv3: manual access gate and acknowledgement duty, excluded on purpose (D-038)
- IMG-14: OpenAI CLIP timm re-upload: fallback only, not needed because IMG-13 was accepted (D-038)
- From-scratch reference (AU-REF): reported, excluded from the selection. dev_x AUC 0.2458 [0.0706, 0.4404] over 3 seeds; protocol difference: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.
<!-- AUTO TABLE T-A end -->

## Experiment log

No experiment has been run yet. Planned entries follow; their hypotheses were written before any run.

## E-001: frozen image encoders on face video frames, ranked on dev_x   (Iteration I-0)
- Status: RUN on 2026-10-25 (automatic ranking, dfd.sweep.rank). The hypothesis and criteria below were written on 2026-10-24 before any feature extraction and are not edited after the run. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / e998f04 / head seeds 0, 1, 2. (The header first read 221e12e, the commit before the run; corrected 2026-10-25 to the commit named in the generated block, D-049.)
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
Run 2026-10-25 02:20, commit e998f04, seeds [0, 1, 2], 2000 bootstrap resamples (base seed 20261024), head: linear, class-balanced BCE, AdamW lr 0.001, weight decay 0.0001, batch 1024, at most 200 epochs, patience 10. Train ffpp_train, validation ffpp_val, dev_x celebdf_dev_x: 1728 clips (146 real, 1582 fake) in 15 groups, 0 excluded without a score.

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
Cost used for the rule: end-to-end extraction time per 1,000 inputs, then parameters, from cost_record.json (section 7: end-to-end, median of 3 repeats).
Order for I-1 (tie set first, then the rest by AUC; ordering decided autonomously, D-042 fit rule applied later): IMG-07, IMG-17, IMG-08, IMG-15, IMG-13, IMG-01, IMG-04.
From-scratch reference: reported, excluded from the selection. dev_x AUC 0.5166 [0.5033, 0.5324], seed sd 0.0045; difference to the leader -0.2298 [-0.3068, -0.1637]; protocol: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.

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
- Status: RUN on 2026-10-25 (automatic ranking, dfd.sweep.rank). Written 2026-10-24 before any extraction. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / 214076b / head seeds 0, 1, 2.
- Hypothesis (written BEFORE the run): A CLIP-style frozen encoder reaches a higher dev_x AUC on the two held-out CNN-synthesis generators than every supervised ImageNet CNN, when all are trained on the ProGAN validation images (14 classes). Basis: R-011 (trained on ProGAN, a frozen CLIP-feature linear probe reached 99.31 mean AP on GAN-family test sets and 95.00 on unseen diffusion and autoregressive ones, against 94.19 and 75.51 for the best trained network; its training set was far larger than ours).
- Success criterion (written BEFORE the run): the paired 95 percent interval of dev_x AUC (best CLIP-style candidate minus best supervised CNN) lies above zero, and the best candidate's lower bound on dev_x AUC is above 0.65.
- Failure criterion (written BEFORE the run): the interval includes zero or lies below zero (not supported; the cost rule decides), or the best candidate's lower bound is 0.55 or below (inconclusive).
- Data and split IDs: ProGAN validation set split by class (14 train, 6 val) and the CNN-synthesis dev_x generator units (2 of 7, D-035 Amendment 1); IDs to be filled at the run.
- Command and config: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.rank --task T-I`; config in data/_results/T-I/rank.json.
- Result (numbers, with seeds and intervals): see the generated block below (data/_results/T-I/rank.md).
- Was the hypothesis supported? Why or why not: mechanical verdict on the pre-declared criteria: SUPPORTED (details in the generated block; exploratory dev_x result, D-041).
- Decision triggered (D-NNN) / next step: the selection decision for T-I and the I-1 candidates.

<!-- AUTO E-002 begin -->
### E-002 ranking (T-I): frozen image encoders on generated images
Run 2026-10-25 02:31, commit 214076b, seeds [0, 1, 2], 2000 bootstrap resamples (base seed 20261024), head: linear, class-balanced BCE, AdamW lr 0.001, weight decay 0.0001, batch 1024, at most 200 epochs, patience 10. Train progan_train, validation progan_val, dev_x cnn_dev_x: 5998 clips (2999 real, 2999 fake) in 2 groups, 0 excluded without a score.

| Rank (I-1 order) | Candidate | dev_x AUC (mean of seeds) | 95% interval | seed sd | validation AUC | gap | diff vs leader [95%] | permutation p | tied with leader |
|---|---|---|---|---|---|---|---|---|---|
| 1 | IMG-13 | 0.9935 | [0.9803, 0.9982] | 0.0006 | 1.0000 | +0.0065 | leader | n/a | yes |
| 2 | IMG-15 | 0.9563 | [0.7760, 0.9975] | 0.0018 | 1.0000 | +0.0437 | -0.037 [-0.204, -0.001] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |
| 3 | IMG-08 | 0.9023 | [0.8496, 0.9316] | 0.0039 | 0.9899 | +0.0876 | -0.091 [-0.131, -0.067] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |
| 4 | IMG-17 | 0.8657 | [0.7591, 0.9166] | 0.0034 | 0.9975 | +0.1318 | -0.128 [-0.221, -0.082] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |
| 5 | IMG-07 | 0.8340 | [0.7401, 0.8823] | 0.0021 | 0.9685 | +0.1345 | -0.159 [-0.240, -0.116] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |
| 6 | IMG-01 | 0.6688 | [0.6238, 0.7137] | 0.0018 | 0.8990 | +0.2302 | -0.325 [-0.357, -0.285] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |
| 7 | IMG-04 | 0.6380 | [0.4912, 0.7115] | 0.0019 | 0.8140 | +0.1759 | -0.355 [-0.489, -0.287] | n/a (fewer than 6 groups: the permutation test cannot reach 0.05) | no |

Leader (highest mean dev_x AUC): IMG-13. Tie set (paired interval against the leader includes zero), cheapest first: IMG-13. Winner under the D-036 rule (cheapest in the tie set): IMG-13 (no tie).
Cost used for the rule: end-to-end extraction time per 1,000 inputs, then parameters, from cost_record.json (section 7: end-to-end, median of 3 repeats).
Order for I-1 (tie set first, then the rest by AUC; ordering decided autonomously, D-042 fit rule applied later): IMG-13, IMG-15, IMG-08, IMG-17, IMG-07, IMG-01, IMG-04.
From-scratch reference: reported, excluded from the selection. dev_x AUC 0.5500 [0.5080, 0.5735], seed sd 0.0544; difference to the leader -0.4435 [-0.4723, -0.4247]; protocol: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.

| Candidate | dev_x EER | balanced accuracy at the validation threshold | ECE before / after temperature | Brier before / after | mean temperature |
|---|---|---|---|---|---|
| IMG-13 | 0.0401 | 0.9551 | 0.0462 / 0.0050 | 0.0334 / 0.0299 | 0.582 |
| IMG-15 | 0.1196 | 0.8683 | 0.0222 / 0.0998 | 0.1074 / 0.1242 | 0.375 |
| IMG-08 | 0.1803 | 0.7962 | 0.1914 / 0.1534 | 0.2272 / 0.2103 | 1.374 |
| IMG-17 | 0.2147 | 0.6893 | 0.2084 / 0.2345 | 0.2463 / 0.2620 | 0.766 |
| IMG-07 | 0.2446 | 0.7336 | 0.2074 / 0.1771 | 0.2592 / 0.2450 | 1.240 |
| IMG-01 | 0.3716 | 0.5920 | 0.2880 / 0.2463 | 0.3284 / 0.3043 | 1.337 |
| IMG-04 | 0.4069 | 0.5794 | 0.2868 / 0.1728 | 0.3362 / 0.2785 | 2.120 |

Pre-declared criteria, evaluated by script: best candidate IMG-13, lower bound 0.9803 (criterion (b): above 0.65: met; inconclusive line: not crossed).
Family comparison, best IMG-13 minus best IMG-01: +0.3247 [+0.2845, +0.3565], interval above zero (met).
Mechanical verdict on the pre-declared criteria: SUPPORTED. This is an exploratory dev_x result (D-041); the sensitivity column is a check, not a second rule.
<!-- AUTO E-002 end -->

## E-003: frozen speech encoders on speech deepfakes, ranked on dev_x   (Iteration I-0)
- Status: RUN on 2026-10-25 (automatic ranking, dfd.sweep.rank). Written 2026-10-24 before any extraction. Protocol: docs/O007_SWEEP.md.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / 214076b / head seeds 0, 1, 2.
- Hypothesis (written BEFORE the run): A self-supervised speech encoder (wav2vec 2.0 or WavLM) reaches a higher dev_x AUC on In-the-Wild speakers not used in training than both the Whisper encoder and the small from-scratch reference, when trained on ASVspoof 2019 LA. Basis: R-029 (a wav2vec 2.0 front end lowered EER from 11.47 to 6.15 percent, best seed, train ASVspoof 2019 LA and test ASVspoof 2021 LA) and R-033 (models trained on ASVspoof 2019 reach EERs far above their in-corpus values on In-the-Wild, about 33.9 percent for RawNet2).
- Success criterion (written BEFORE the run): the paired 95 percent intervals of dev_x AUC (best self-supervised candidate minus the Whisper candidate, and minus the from-scratch reference) both lie above zero, and the best candidate's lower bound on dev_x AUC is above 0.60.
- Failure criterion (written BEFORE the run): either interval includes zero or lies below zero (not supported; the cost rule decides), or the best candidate's lower bound is 0.55 or below (inconclusive).
- Data and split IDs: ASVspoof 2019 LA train and dev (val) and the In-the-Wild dev_x slice (12 of 54 speakers, D-035); IDs to be filled at the run.
- Command and config: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.rank --task T-A`; config in data/_results/T-A/rank.json.
- Result (numbers, with seeds and intervals): see the generated block below (data/_results/T-A/rank.md).
- Was the hypothesis supported? Why or why not: mechanical verdict on the pre-declared criteria: INCONCLUSIVE (best lower bound at or below 0.55) (details in the generated block; exploratory dev_x result, D-041).
- Decision triggered (D-NNN) / next step: the selection decision for T-A and the I-1 candidates.

<!-- AUTO E-003 begin -->
### E-003 ranking (T-A): frozen speech encoders on speech deepfakes
Run 2026-10-25 02:49, commit 214076b, seeds [0, 1, 2], 2000 bootstrap resamples (base seed 20261024), head: linear, class-balanced BCE, AdamW lr 0.001, weight decay 0.0001, batch 1024, at most 200 epochs, patience 10. Train asvspoof_train, validation asvspoof_dev, dev_x itw_dev_x: 6991 clips (3804 real, 3187 fake) in 12 groups, 0 excluded without a score.

| Rank (I-1 order) | Candidate | dev_x AUC (mean of seeds) | 95% interval | seed sd | validation AUC | gap | diff vs leader [95%] | permutation p | tied with leader |
|---|---|---|---|---|---|---|---|---|---|
| 1 | AU-8 | 0.6100 | [0.5434, 0.7256] | 0.0026 | 0.9942 | +0.3842 | -0.028 [-0.107, +0.039] | 0.599 | yes |
| 2 | AU-4 | 0.6334 | [0.5291, 0.7714] | 0.0031 | 0.9924 | +0.3590 | -0.005 [-0.044, +0.040] | 0.894 | yes |
| 3 | AU-6 | 0.6380 | [0.5338, 0.7700] | 0.0089 | 0.9995 | +0.3615 | leader | n/a | yes |
| 4 | AU-1 | 0.4975 | [0.3451, 0.7092] | 0.0212 | 0.9965 | +0.4990 | -0.140 [-0.238, -0.050] | 0.170 | no |

Leader (highest mean dev_x AUC): AU-6. Tie set (paired interval against the leader includes zero), cheapest first: AU-8, AU-4, AU-6. Winner under the D-036 rule (cheapest in the tie set): AU-8 (the cost rule decided).
Cost used for the rule: end-to-end extraction time per 1,000 inputs, then parameters, from cost_record.json (section 7: end-to-end, median of 3 repeats).
Order for I-1 (tie set first, then the rest by AUC; ordering decided autonomously, D-042 fit rule applied later): AU-8, AU-4, AU-6, AU-1.
From-scratch reference: reported, excluded from the selection. dev_x AUC 0.2458 [0.0706, 0.4404], seed sd 0.0355; difference to the leader -0.3922 [-0.6527, -0.1434]; protocol: end to end, no pretrained weights, 10 epochs, AdamW 1e-3, no augmentation, last-epoch weights; excluded from the selection.

| Candidate | dev_x EER | balanced accuracy at the validation threshold | ECE before / after temperature | Brier before / after | mean temperature |
|---|---|---|---|---|---|
| AU-8 | 0.4183 | 0.5289 | 0.3293 / 0.4052 | 0.3699 / 0.4247 | 0.485 |
| AU-4 | 0.3930 | 0.5256 | 0.3700 / 0.4420 | 0.3893 / 0.4494 | 0.490 |
| AU-6 | 0.4006 | 0.5787 | 0.1819 / 0.2875 | 0.2796 / 0.3292 | 0.454 |
| AU-1 | 0.5020 | 0.5208 | 0.3482 / 0.4165 | 0.3993 / 0.4418 | 0.541 |

Pre-declared criteria, evaluated by script: best candidate AU-6, lower bound 0.5338 (criterion (b): above 0.6: not met; inconclusive line: crossed).
Family comparison, best AU-4 minus best AU-6: -0.0046 [-0.0437, +0.0399], interval includes zero or is below (not met).
Comparison with the from-scratch reference: {"estimate": 0.38759700153236, "low": 0.15308106499361954, "high": 0.6533440810769281, "se": 0.14355747696297294, "seed_sd": null, "n_nan": 0, "met": true}
Mechanical verdict on the pre-declared criteria: INCONCLUSIVE (best lower bound at or below 0.55). This is an exploratory dev_x result (D-041); the sensitivity column is a check, not a second rule.
<!-- AUTO E-003 end -->


## E-004: the working version I-0 (MVP): heads of the three sweep winners, calibration, parity and page check   (Iteration I-0)
- Status: RUN on 2026-10-25 (scripts/mvp/train_mvp.py, parity_check.py, web_real_check.py, battery.py, face_stats.py; CPU for the heads and the battery, the GPU only for the parity check and the page). The criteria are those of D-040 (written 2026-10-24, approved by the user), the heads and statistics those of the sweep (D-036, D-049); nothing was changed after the result.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / 8405365 (checkpoint metadata; the scripts are committed in 9d1fa9c and 249ce5c) / head seeds 0, 1, 2; the shipped head is seed 0 (fixed before the run, decided autonomously, D-050).
- Checkpoint: checkpoints/I-0-mvp (not in git), checkpoint id I-0-mvp-20261025-0319. T-V DINOv2 small (preprocess id 960bf135c75f, temperature 0.907, threshold 0.4774), T-I OpenAI CLIP ViT-L/14 (ccbaba9519dc, 0.636, 0.2073), T-A HuBERT base (c8964f41ccfb, 0.486, 0.1451); temperature and threshold fitted on the validation split of each task with the clip score, the threshold in calibrated space.
- Hypothesis and criteria: D-040 items 1 to 7: a file goes in through the page, the winners score it, the score is calibrated on validation, the checkpoint is checked at load, metrics with intervals exist on validation and dev_x, the page shows a real result with the D-034 warnings, the visual and audio scores of a video are separate, tests pass. No accuracy claim beyond the intervals.
- Data and split IDs: T-V train ffpp-037e2976fade train (3,600 clips), validation its val (700), dev_x celebdf_v2-82a3a70594e9 dev_x (1,728); T-I progan train (5,600 images), val (2,400), dev_x cnn_dev_x (5,998 images, 2 units); T-A asvspoof2019_la-875bf337c379 train (25,380), dev (24,844), in_the_wild-5cf2ea54a724 dev_x (6,991, 12 speakers). No test_in or test_x item was read.
- Result (exploratory dev_x, D-041; mean of seeds 0, 1, 2 with the 95 percent group-bootstrap interval; reproduces the E-001 to E-003 values):

| task | encoder | train AUC | validation AUC | dev_x AUC | dev_x EER | ECE before / after TS, validation | ECE before / after TS, dev_x | Brier before / after, dev_x |
|---|---|---|---|---|---|---|---|---|
| T-V | IMG-07 | 0.8367 [0.8267, 0.8475] | 0.7985 [0.7731, 0.8269] | 0.6933 [0.6639, 0.7314] | 0.3582 | 0.0325 / 0.0321 | 0.2048 / 0.2217 | 0.2963 / 0.3050 |
| T-I | IMG-13 | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9935 [0.9803, 0.9982] | 0.0401 | 0.0050 / 0.0012 | 0.0462 / 0.0050 | 0.0334 / 0.0299 |
| T-A | AU-8 | 0.9997 [0.9996, 0.9999] | 0.9942 [0.9910, 0.9967] | 0.6100 [0.5434, 0.7256] | 0.4183 | 0.0883 / 0.0078 | 0.3293 / 0.4052 | 0.3699 / 0.4247 |

  Seed sd on dev_x 0.0107, 0.0006 and 0.0026 against bootstrap half-widths 0.0338, 0.0089 and 0.0911. Balanced accuracy at the validation threshold, validation and dev_x: T-V 0.750 and 0.617, T-I 0.998 and 0.955, T-A 0.971 and 0.529. The T-I dev_x interval rests on 2 units and the T-A one on 12 speakers (D-049 caveats).
- Parity check (D-044), training items scored from the stored features and by the real adapter from the raw file: 7 items (one fake and one real FF++ video, 2 ProGAN images in the unsaturated range, 3 ASVspoof utterances), maximum absolute difference of the calibrated score 0.0019 (T-V 0.0002, T-I 0.0019, T-A 0.0005), tolerance 0.02: PASS (data/_results/I-0/parity.json). The two dependences that D-044 named (the JPEG round trip and the colour order) agree.
- Page check: the page started with `python -m dfd.web --adapter real --checkpoint checkpoints/I-0-mvp` (the health route answers adapter real, the MOCK banner is hidden). In real Chromium one image (a face crop of an FF++ training video), one video with audio (a LAV-DF training video) and one audio file (an ASVspoof training utterance) gave results with inference_source real, no MOCK banner and no page error. The video shows a visual score and an audio score separately and the verdict "Undetermined" (no combining rule); the audio score carries the new warning AUDIO_CALIBRATION_EXPLORATORY (D-050); the upload folder was empty after the three requests (data/_results/I-0/web_real_check.json, assets/screens/real_*.png). These three files are training items: the check shows that the path works and says nothing about accuracy.
- Diagnostic battery (generated: data/_results/I-0/battery.md, figures in data/_results/I-0/figs, contact sheets of the worst errors local only because of the dataset licences). Observations in numbers:
  1. Per method or generator. T-V validation: Deepfakes 0.884, Face2Face 0.834, FaceSwap 0.778, NeuralTextures 0.697 (intervals within 0.035); dev_x Celeb-DF has one fake label, 0.693. T-I: all six validation classes 1.000; dev_x StarGAN 0.999 and whichfaceisreal 0.983. T-A: validation attacks A01 to A05 0.996 to 0.997 and A06 0.984; dev_x per speaker minimum 0.503, median 0.692, maximum 0.990, none below 0.5, pooled 0.610.
  2. Train minus validation AUC: T-V +0.038, T-I 0.000, T-A +0.006.
  3. Validation minus dev_x AUC: T-V +0.105, T-I +0.007, T-A +0.384.
  4. Calibration: temperature scaling fitted on validation did not improve the dev_x calibration for T-V (ECE 0.205 to 0.222) or T-A (0.329 to 0.405) and did for T-I (0.046 to 0.005). All three temperatures are below 1 (0.907, 0.636, 0.486): the validation scores are sharpened, and dev_x scores are then over-confident.
  5. Histograms: T-V dev_x calibrated score at 10, 50 and 90 percent, real 0.140, 0.304, 0.597 and fake 0.204, 0.434, 0.725 (strong overlap); T-A dev_x real 0.167, 0.971, 1.000 and fake 0.591, 0.994, 1.000 (real In-the-Wild speech scores as fake); T-I dev_x real 0.000, 0.004, 0.141 and fake 0.819, 0.996, 1.000.
  6. T-V splits: all 700 validation and 1,728 dev_x clips have faces (two dev_x clips have one frame without a face); validation clips with two or more faces (127) 0.754 against 0.811 for one face; dev_x one face 0.692 and two or more (59 clips) 0.762 [0.667, 0.939]; face-height tertiles on dev_x 0.689, 0.696 and 0.697 (no trend). Compression: FF++ c23 is one level and the compression of Celeb-DF is not recorded, so no split by compression was possible. Aggregation with the seed-0 head on dev_x (validation in brackets): frame-level 0.656 (0.759), clip mean 0.685 (0.799), median 0.688 (0.798), top-3 mean 0.663 (0.786), max 0.662 (0.775); within-clip sd of the frame scores 0.104 for real and 0.110 for fake clips.
  7. Worst errors on dev_x, classified by eye from local contact sheets (T-V and T-I) and from the audio features (T-A). T-V false alarms (10 real Celeb-DF clips of 7 identities): sharp, well lit, near-frontal faces with smooth skin or interview graphics, no visible defect; misses (10 fakes): 6 of 10 come from one identity group (id1) and 3 from another (id16); several clips show the same source scene (the same background) with different swapped faces, low contrast or a turned head. T-I false alarms: real StarGAN images (cropped CelebA faces, several with stretched, replicated border pixels at the top or side) and two real whichfaceisreal photographs; misses: 9 of 10 are whichfaceisreal fakes (photorealistic StyleGAN faces, most with no visible artefact and one clearly distorted) and one is a StarGAN fake. T-A: both error kinds are short clips (median duration 1.9 s for false alarms and 1.2 s for misses, against 2.9 s for real and 3.9 s for fake clips of dev_x) with a repeat-padded fraction of 0.52 and 0.70 (against 0.29 and 0.04); three misses are under 0.8 s with RMS 0.003 (near silence); all 20 scores are saturated at 0.000 or 1.000.
  8. Seed variance is a third or less of the bootstrap half-width for all three tasks: the sampling of identities and speakers, not training noise, limits the intervals.
- Candidate diagnosis per task (vocabulary of the battery; for the orchestrator to decide; no lever is chosen here). T-V: bias first (train AUC only 0.837 and a train-validation gap of 0.038, so the frozen DINOv2-small feature separates even the training data only moderately), then shift (validation minus dev_x 0.105) and calibration under shift (the temperature does not transfer); aggregation is not indicated (the clip mean is above frame level by 0.03 and max or top-3 are not better); face size and face count show no effect; the weakest manipulations are NeuralTextures and FaceSwap. T-I: no bias or variance visible (train and validation 1.000); dev_x has only 2 units, so the shift to other generators cannot be measured; calibration is good; the errors are preprocessing artefacts in some real images and photorealistic StyleGAN faces. T-A: shift is dominant (validation minus dev_x 0.384 at an in-domain AUC of 0.994), the from-scratch reference inverts (M-004), real In-the-Wild speech scores as fake, trivial recording features reverse sign (M-004 b), the pooled AUC (0.610) is below the per-speaker median (0.692), calibration worsens under shift, and the errors concentrate in short repeat-padded clips with saturated scores (a window or padding effect is a candidate: missing signal or shortcut).
- Was the hypothesis supported? D-040 items 1 to 7 hold on the checks above; the claim is limited to "the path works and metrics with intervals exist". No accuracy claim is made beyond the intervals, and the dev_x numbers are exploratory.
- Decision triggered (D-050) / next step: the choice of the I-1 lever and of the audio treatment is for the orchestrator; I-1 is not started.


## E-005: full fine-tuning of the video-task encoder (T-V), three candidates, three learning rates, then three seeds for the winner   (Iteration I-1)
- Status: PLANNED on 2026-10-25, written before the first run (CLAUDE.md rule 2). Lever and reasons: D-051 (decided autonomously - needs user review); scope and budget: D-042; settings: O007_SWEEP.md section 9 as amended by D-042. The result block is written by script after the run and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU (RTX 5070, 12 GB, PyTorch 2.14.1+cu130) / the commit that holds this entry and src/dfd/train/finetune.py, scripts/i1/run_i1.py (see git log) / stage 1 seed 0, stage 2 seeds 1 and 2 for the winner.
- Observation (E-004, battery): T-V train AUC 0.8367, validation 0.7985, dev_x 0.6933 [0.6639, 0.7314]; the frozen DINOv2-small feature plus a linear head fits even the training data only moderately (D-051).
- Diagnosis: bias first (the frozen feature lacks the signal for a linear separation of the training data), then shift (validation minus dev_x 0.105).
- Mathematical hypothesis (MATH_NOTES 11): let the probe class be $\{x \mapsto \sigma(w^\top\phi_{\theta_0}(x)+b)\}$ with the encoder $\phi_{\theta_0}$ fixed, and the fine-tuned class $\{x \mapsto \sigma(w^\top\phi_{\theta}(x)+b)\}$ with $\theta$ free. The first class is contained in the second, so the minimum of the training risk $R_{\text{train}}$ over the larger class is no larger; if the optimiser finds it, the training AUC rises. Whether the validation and dev_x AUC rise is not implied: a larger class can fit identity- and source-specific cues (R-044).
- Change: all vision-tower parameters of the encoder and a linear head are trained end to end: AdamW (betas 0.9 and 0.999), weight decay 0.05, learning rate in {1e-5, 3e-5, 1e-4} with 5 percent linear warm-up (90 steps) and cosine decay to zero, fp16 autocast with gradient scaling, batch 32 (no accumulation), 1,800 optimiser steps (57,600 frames, one pass over the training frames), no augmentation, class-balanced binary cross-entropy $w_y = n/(2n_y)$ over the training frames (the I-0 head's weights), head with the default PyTorch initialisation, no gradient clipping. Candidates (D-049 item 2): IMG-07 (DINOv2 small), IMG-17 (SigLIP base), IMG-08 (DINOv2 base). Same preprocessing as the I-0 extraction (D-034 crops, 224 px, the model card's mean and std, no JPEG change). Inference of the fine-tuned models: clip score = mean of the frame probabilities (D-024).
- Budget (D-042, with the clean step times of data/_logs/step_time_alone.json): per run 1,800 steps; compute at batch 32 is 86 s (IMG-07, 47.7 ms per step), 203 s (IMG-17, 112.5 ms) and 238 s (IMG-08, 132.2 ms); peak VRAM 2.00, 3.99 and 4.49 GiB, all under the 10 GiB fit rule, so no substitution. Cap: a run whose training exceeds 600 s, or whose wall time (training plus the evaluation pass of about 53,000 frames) exceeds 900 s, is stopped and recorded as exceeded; no score is made from a stopped run. Stage 1: 3 encoders x 3 learning rates x seed 0 = 9 runs. Stage 2: seeds 1 and 2 at the selected learning rate of the selected encoder = 2 runs. Expected total: about 40 to 60 minutes (training 28 minutes of compute plus data loading, evaluation and model loading). Data-loader workers 10 (cap 16). One GPU job at a time.
- Selection (validation only, rule 3): within an encoder, the learning rate with the highest validation AUC; across encoders the rule of D-036 on validation clips (paired group bootstrap of the clip AUC against the leader, 2,000 resamples; the tie set is every candidate whose interval for the difference to the leader includes zero; the cheapest of the tie set wins; cost = measured fine-tuning time per 1,000 frames, then trainable parameters). The selection is written to data/_results/I-1/selection.json before stage 2 starts.
- Data and split IDs: train ffpp-037e2976fade train (3,600 clips, about 57,600 frames with a face), validation the same split's val (700 clips, 11,200 frames), dev_x celebdf_v2-82a3a70594e9 dev_x (1,728 clips, 27,648 frames), crops data/_crops/ffpp_c23_yunet_n16 and celebdf_v2_yunet_n16 (preprocess id 960bf135c75f for FF++). The training AUC is computed on a fixed strided subset of 900 training clips (every 4th clip in the order of the I-0 score file); the I-0 probe's AUC on the same 900 clips is recomputed from its stored scores. No test_in or test_x item is read.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/i1/run_i1.py` (background, log data/_logs/i1_run.log, resumable per run); analysis `scripts/i1/analyse_i1.py` writes the result block below.
- Prediction (sign and size; from ITERATIONS.md I-1, written before the battery): validation AUC of the fine-tuned model above the probe of the same encoder by +0.02 to +0.05; dev_x between slightly negative and +0.03. The battery (train AUC 0.837) suggests the validation gain could be larger than this range; a larger gain is reported as "larger than predicted" and does not change the verdict. Additional prediction: the training-subset AUC rises to 0.90 or more (the bias trigger's own threshold).
- Measured for every run: validation, dev_x and training-subset AUC, time per 1,000 steps, peak VRAM, instability events (a non-finite loss, or the 50-step mean loss above five times its running minimum after the warm-up), the loss curve every 50 steps; learning-rate sensitivity per encoder (largest minus smallest validation AUC over the three rates).
- Success / failure lines (mechanical, on the winner's three seeds against the three-seed linear probe of the same encoder, paired group bootstrap of the AUC difference, 2,000 resamples, seeds averaged within each resample, 95 percent interval): C1 validation gain: the lower bound of (fine-tuned minus probe) on validation is above 0. C2 bias resolved: the AUC of the fine-tuned model on the 900-clip training subset is at least 0.90. C3 transfer: the lower bound of (fine-tuned minus probe) on dev_x is above 0. Verdict: SUPPORTED if C1, C2 and C3 hold. PARTIAL if C1 and C3 hold and C2 fails. NOT SUPPORTED, failure line of ITERATIONS.md I-1 (capacity went into source-specific cues), if C1 holds and the dev_x interval includes zero or lies below zero. NOT SUPPORTED, bias diagnosis rejected, if C1 fails. NOT EVALUATED if a run of the winner is stopped by the cap or runs out of memory. Failure by resources: peak VRAM above 12 GB at batch 32, or a stage 1 run beyond the cap, is recorded for that candidate. All dev_x verdicts are exploratory (D-041); no sealed data is touched.
- Result (generated by scripts/i1/analyse_i1.py from data/_results/I-1, rerunnable):
<!-- E-005 RESULT begin (generated by scripts/i1/analyse_i1.py) -->

#### Stage 1 (seed 0): every run

| encoder | learning rate | status | validation AUC | dev_x AUC | training-subset AUC (900 clips) | training s | wall s | peak VRAM GiB | instability events | non-finite losses |
|---|---|---|---|---|---|---|---|---|---|---|
| IMG-07 | 1e-05 | ok | 0.9869 | 0.9031 | 0.9982 | 107.5 | 199.5 | 2.08 | 0 | 0 |
| IMG-07 | 3e-05 | ok | 0.7056 | 0.6412 | 0.7293 | 107.1 | 192.4 | 2.08 | 0 | 0 |
| IMG-07 | 0.0001 | ok | 0.5871 | 0.5679 | 0.5980 | 108.2 | 195.8 | 2.08 | 0 | 0 |
| IMG-17 | 1e-05 | ok | 0.9893 | 0.9210 | 0.9993 | 215.2 | 344.1 | 4.33 | 0 | 0 |
| IMG-17 | 3e-05 | ok | 0.9814 | 0.8947 | 0.9971 | 258.5 | 374.0 | 4.33 | 0 | 0 |
| IMG-17 | 0.0001 | ok | 0.5186 | 0.5432 | 0.5170 | 216.7 | 331.7 | 4.33 | 0 | 0 |
| IMG-08 | 1e-05 | ok | 0.9918 | 0.9036 | 0.9992 | 263.3 | 387.1 | 4.79 | 0 | 0 |
| IMG-08 | 3e-05 | ok | 0.5218 | 0.5379 | 0.5280 | 262.1 | 388.7 | 4.8 | 0 | 0 |
| IMG-08 | 0.0001 | ok | 0.5583 | 0.5667 | 0.5491 | 260.4 | 432.6 | 4.79 | 0 | 0 |

Learning-rate sensitivity (largest minus smallest AUC over the three rates): IMG-07 validation 0.3998, dev_x 0.3352; IMG-17 validation 0.4707, dev_x 0.3778; IMG-08 validation 0.4700, dev_x 0.3657.

#### Selection (validation only, D-036 rule on validation clips)

Best learning rate per encoder: IMG-07 1e-05, IMG-17 1e-05, IMG-08 1e-05. Leader IMG-08; tie set (cheapest first) IMG-17, IMG-08; winner IMG-17 at learning rate 1e-05 (decided by cost inside a tie set). Validation AUC of the best-rate runs: IMG-07 0.9869 [0.9782, 0.9945]; IMG-17 0.9893 [0.9801, 0.9967]; IMG-08 0.9918 [0.9847, 0.9974]. Paired difference to the leader: IMG-07 -0.0049 [-0.0092, -0.0010]; IMG-17 -0.0025 [-0.0068, 0.0009].

#### Winner IMG-17, learning rate 1e-05: seeds 0, 1, 2

| seed | status | validation AUC | dev_x AUC | training-subset AUC | training s | wall s | peak VRAM GiB | instability events |
|---|---|---|---|---|---|---|---|---|
| 0 | ok | 0.9893 | 0.9210 | 0.9993 | 215.2 | 344.1 | 4.33 | 0 |
| 1 | ok | 0.9896 | 0.9166 | 0.9991 | 211.5 | 347.1 | 4.34 | 0 |
| 2 | ok | 0.9908 | 0.8912 | 0.9999 | 210.3 | 317.8 | 4.33 | 0 |

| split | fine-tuned (3 seeds) | linear probe of the same encoder (3 seeds) | fine-tuned minus probe |
|---|---|---|---|
| validation | 0.9899 [0.9809, 0.9967] | 0.8767 [0.8582, 0.8966] | +0.1132 [+0.0948, +0.1299] |
| dev_x | 0.9096 [0.8776, 0.9388] | 0.7396 [0.7031, 0.7832] | +0.1700 [+0.1228, +0.2114] |
| training subset | 0.9994 [0.9986, 1.0000] | 0.9245 [0.9093, 0.9388] | +0.0749 [+0.0607, +0.0902] |

Criteria (E-005): C1 validation gain lower bound above 0: met (+0.0948). C2 training-subset AUC at least 0.90: met (0.9994). C3 dev_x gain lower bound above 0: met (+0.1228).

**VERDICT (mechanical, exploratory on dev_x, D-041): SUPPORTED.**

Prediction check (written before the run): validation gain +0.02 to +0.05, found +0.1132 (larger than the range); dev_x between slightly negative and +0.03, found +0.1700 (larger than the range); training-subset AUC at least 0.90: 0.9994.

#### Diagnostic battery on the fine-tuned winner (exploratory; train, validation and dev_x only)

Item 1, validation, real against each manipulation method (3 seeds):

| method | fakes | AUC [95% interval] |
|---|---|---|
| Deepfakes | 140 | 1.000 [0.999, 1.000] |
| Face2Face | 140 | 0.991 [0.978, 1.000] |
| FaceSwap | 140 | 0.999 [0.997, 1.000] |
| NeuralTextures | 140 | 0.969 [0.946, 0.989] |

Items 2, 3 and 8: gaps and seed variance (training subset 900 clips):

| split | AUC [95% interval] | seed sd | interval half-width |
|---|---|---|---|
| training subset | 0.9994 [0.9986, 1.0000] | 0.0004 | 0.0007 |
| validation | 0.9899 [0.9809, 0.9967] | 0.0008 | 0.0079 |
| dev_x | 0.9096 [0.8776, 0.9388] | 0.0161 | 0.0306 |

Training-subset minus validation AUC: +0.0095. Validation minus dev_x AUC: +0.0803 (I-0: +0.1052).

Item 4, calibration (15 equal-width bins; temperature fitted on the validation split of each seed; ECE before / after, Brier before / after):

| split | ECE before / after | Brier before / after | temperatures (seeds 0, 1, 2) | balanced accuracy at the validation threshold |
|---|---|---|---|---|
| validation | 0.0261 / 0.0174 | 0.0307 / 0.0291 | 0.832, 0.834, 0.758 | 0.974 |
| dev_x | 0.0522 / 0.0331 | 0.1052 / 0.1027 | 0.832, 0.834, 0.758 | 0.824 |

Item 5, quantiles of the seed-mean clip score (not calibrated), 10 / 50 / 90 percent:

| split | real | fake |
|---|---|---|
| validation | 0.004 / 0.050 / 0.285 | 0.794 / 0.992 / 1.000 |
| dev_x | 0.071 / 0.310 / 0.620 | 0.450 / 0.828 / 0.984 |

Item 6, aggregation and face statistics (seed 0 run):

| split | frame-level AUC | clip mean (shipped rule) | clip median | clip top-3 mean | clip max |
|---|---|---|---|---|---|
| validation | 0.981 [0.972, 0.989] | 0.989 [0.980, 0.997] | 0.990 [0.982, 0.997] | 0.988 [0.979, 0.996] | 0.986 [0.977, 0.994] |
| dev_x | 0.852 [0.818, 0.881] | 0.921 [0.890, 0.950] | 0.921 [0.889, 0.950] | 0.926 [0.895, 0.953] | 0.918 [0.888, 0.947] |

| split | faces in three sampled frames | clips | real | fake | AUC [95% interval] |
|---|---|---|---|---|---|
| validation | 1 | 573 | 118 | 455 | 0.993 [0.986, 0.998] |
| validation | 2 or more | 127 | 22 | 105 | 0.981 [0.948, 0.998] |
| dev_x | 1 | 1669 | 141 | 1528 | 0.911 [0.879, 0.941] |
| dev_x | 2 or more | 59 | 5 | 54 | 0.884 [0.845, 0.987] |

Item 7, the 20 worst errors on dev_x (10 false alarms, 10 misses), seed-mean score:

| kind | sample | group | score |
|---|---|---|---|
| false alarm | celebdf_v2:Celeb-real/id12_0003.mp4 | celebdf_v2:id12 | 0.831 |
| false alarm | celebdf_v2:Celeb-real/id48_0006.mp4 | celebdf_v2:id48 | 0.815 |
| false alarm | celebdf_v2:Celeb-real/id0_0003.mp4 | celebdf_v2:id0 | 0.758 |
| false alarm | celebdf_v2:Celeb-real/id10_0008.mp4 | celebdf_v2:id10 | 0.755 |
| false alarm | celebdf_v2:Celeb-real/id48_0007.mp4 | celebdf_v2:id48 | 0.738 |
| false alarm | celebdf_v2:Celeb-real/id31_0008.mp4 | celebdf_v2:id31 | 0.723 |
| false alarm | celebdf_v2:Celeb-real/id41_0005.mp4 | celebdf_v2:id41 | 0.708 |
| false alarm | celebdf_v2:Celeb-real/id1_0003.mp4 | celebdf_v2:id1 | 0.696 |
| false alarm | celebdf_v2:Celeb-real/id12_0006.mp4 | celebdf_v2:id12 | 0.691 |
| false alarm | celebdf_v2:Celeb-real/id1_0007.mp4 | celebdf_v2:id1 | 0.674 |
| miss | celebdf_v2:Celeb-synthesis/id16_id2_0011.mp4 | celebdf_v2:id16 | 0.104 |
| miss | celebdf_v2:Celeb-synthesis/id16_id2_0002.mp4 | celebdf_v2:id16 | 0.109 |
| miss | celebdf_v2:Celeb-synthesis/id1_id6_0004.mp4 | celebdf_v2:id1 | 0.146 |
| miss | celebdf_v2:Celeb-synthesis/id1_id17_0004.mp4 | celebdf_v2:id1 | 0.146 |
| miss | celebdf_v2:Celeb-synthesis/id10_id7_0006.mp4 | celebdf_v2:id10 | 0.157 |
| miss | celebdf_v2:Celeb-synthesis/id26_id1_0001.mp4 | celebdf_v2:id26 | 0.158 |
| miss | celebdf_v2:Celeb-synthesis/id12_id7_0001.mp4 | celebdf_v2:id12 | 0.175 |
| miss | celebdf_v2:Celeb-synthesis/id1_id3_0004.mp4 | celebdf_v2:id1 | 0.180 |
| miss | celebdf_v2:Celeb-synthesis/id1_id20_0004.mp4 | celebdf_v2:id1 | 0.184 |
| miss | celebdf_v2:Celeb-synthesis/id1_id2_0002.mp4 | celebdf_v2:id1 | 0.185 |

Worst errors per group: celebdf_v2:id1 7, celebdf_v2:id12 3, celebdf_v2:id48 2, celebdf_v2:id10 2, celebdf_v2:id16 2, celebdf_v2:id0 1, celebdf_v2:id31 1, celebdf_v2:id41 1, celebdf_v2:id26 1.

<!-- E-005 RESULT end -->
- Run log note (appended 2026-10-25 after the result, CLAUDE.md rule 9): the first attempt of the stage-1 run IMG-08 at learning rate 1e-4 (seed 0) stopped at about step 1,750 on 2026-10-25 04:37 when a data-loader worker died (Windows shared-memory error, `Couldn't open shared event`; cause not established). Its error record is kept in data/_results/I-1_failed_attempts/IMG-08_lr0.0001_s0_attempt1/result.json. The run was repeated once on 2026-10-25 from 13:52 with the identical configuration and seed and finished (validation AUC 0.5583, a collapsed run like the other high-rate runs); the other eight stage-1 runs and the two stage-2 runs had no failure. The result above uses the repeated run. Stage 1 and stage 2 were therefore run in two sessions with the same code.

## E-007: speech preprocessing that removes leading and trailing silence and peak-normalises each clip (hypothesis H-A1)   (Iteration I-2 candidate, task T-A)
- Status: PLANNED on 2026-10-25, written before any run of the variant (CLAUDE.md rule 2). Hypothesis and reasons: D-051 item 5 (decided autonomously - needs user review); it is justified by measurement M-004 and not by any tuning on dev_x. It runs after the I-1 runs, when the GPU is idle. The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry and scripts/i2/run_ha1.py (see git log) / head seeds 0, 1, 2.
- Observation (M-004, E-003, E-004): T-A validation AUC 0.9942 [0.9910, 0.9967] and dev_x AUC 0.6100 [0.5434, 0.7256] (12 speakers); the from-scratch reference is 0.246 on dev_x; leading silence as a fake-score has AUC 0.375 [0.299, 0.453] on ASVspoof dev and 0.719 [0.425, 0.933] on In-the-Wild dev_x, RMS energy 0.594 [0.537, 0.649] and 0.248 [0.073, 0.546]; the worst dev_x errors are short repeat-padded clips with saturated scores.
- Diagnosis: shift through recording-condition cues (missing signal or shortcut), supported in sign and not established in size.
- Mathematical hypothesis: let a clip be $x=c(t)\,s(t)$ with a content signal $s$ and a recording factor $c$ (level, leading silence). A score $f(x)$ that depends on $c$ separates classes in a corpus where $c$ differs between classes and reverses where it differs the other way. A map $T$ with $T(c\,s)=T(s)$ for the level and the leading and trailing silence removes this dependence: trimming sets $c$ to its support and peak normalisation sets $\max|x|=1$. Then $f(T(x))$ cannot use level or leading silence, and if the head used them, its out-of-corpus AUC moves towards the content-based value.
- Change: before the 4 s window is cut, the leading and trailing samples with absolute value at most 0.005 of full scale (the threshold of M-004) are removed and the clip is divided by its peak (src/dfd/prep/audio.py trim_and_peak_normalise); a clip silent throughout is left unchanged; then the first 64,600 samples, repeat-padded, as before. Encoder AU-8 (HuBERT base) frozen, mean and standard deviation over time, the head recipe of dfd.sweep.heads, seeds 0, 1, 2; nothing else changes. Features: data/_features_trimnorm/AU-8/T-A/ (new files; the E-003 files are not touched).
- Data and split IDs: ASVspoof 2019 LA train and dev (asvspoof2019_la-875bf337c379, 25,380 and 24,844 items), In-the-Wild dev_x (in_the_wild-5cf2ea54a724, 6,991 items, 12 speakers). No test item is read.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_audio --cand AU-8 --variant trim_norm --out-root data/_features_trimnorm` then `PYTHONPATH=src .venv/Scripts/python.exe scripts/i2/run_ha1.py --write`. Budget: decoding 57,000 files plus one HuBERT pass, expected 10 to 20 minutes, the head training about 1 minute on CPU.
- Prediction (sign and size): dev_x AUC rises by +0.05 to +0.15 over the I-0 probe (from 0.6100); validation AUC changes by at most -0.01 (the in-domain gain from the shortcut is given up); the single-feature AUC of leading silence, RMS and duration on ASVspoof dev moves towards 0.5.
- Manipulation check (the hypothesis is tested only if it holds): on the preprocessed signal the 95 percent group-bootstrap intervals of leading silence, RMS energy and clip duration (as fake-scores) on ASVspoof dev include 0.5. If not, the verdict is NOT TESTED.
- Success / failure lines (mechanical): SUPPORTED if the manipulation check holds, the lower bound of the paired dev_x difference (new minus I-0 probe, speaker bootstrap, 3 seeds) is above 0, and the validation AUC has not fallen by more than 0.02. NOT SUPPORTED if the dev_x interval includes zero or lies below zero, or validation AUC falls by more than 0.02. With 12 speakers the dev_x interval is wide (half-width 0.09 in I-0), so a true gain below about +0.09 cannot be shown; this limits the power and is stated, not hidden. All dev_x verdicts are exploratory (D-041).
- Result (generated by scripts/i2/run_ha1.py from data/_results/I-2, rerunnable):
<!-- E-007 RESULT begin (generated by scripts/i2/run_ha1.py) -->

| split | new (trimmed, normalised) | I-0 probe | new minus I-0 |
|---|---|---|---|
| validation | 0.9889 [0.9831, 0.9933] | 0.9942 [0.9910, 0.9967] | -0.0054 [-0.0115, -0.0006] |
| dev_x | 0.6723 [0.6126, 0.7584] | 0.6100 [0.5434, 0.7256] | +0.0623 [+0.0114, +0.1048] |

Manipulation check, AUC of the single feature on the preprocessed signal (0.5 = no information; M-004 before: ASVspoof dev leading silence 0.375, RMS 0.594, duration 0.467, repeat-padded fraction 0.543):

| feature | ASVspoof train | ASVspoof dev | In-the-Wild dev_x |
|---|---|---|---|
| leading_silence_s | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.500 [0.498, 0.500] |
| duration_s | 0.485 [0.440, 0.529] | 0.513 [0.457, 0.563] | 0.591 [0.505, 0.647] |
| rms | 0.549 [0.494, 0.606] | 0.541 [0.467, 0.605] | 0.135 [0.054, 0.382] |
| repeat_pad_fraction | 0.517 [0.473, 0.563] | 0.490 [0.439, 0.546] | 0.421 [0.365, 0.506] |

ECE before / after temperature scaling (fitted on validation): validation 0.0835 / 0.0048, dev_x 0.2252 / 0.3116 (I-0: 0.0883 / 0.0078 and 0.3293 / 0.4052).

Manipulation check passed (the intervals of leading silence, RMS and duration on ASVspoof dev include 0.5).
Prediction check: dev_x gain +0.05 to +0.15, found +0.0623 (inside the range); validation change at most -0.01, found -0.0054.

**VERDICT (mechanical, exploratory on dev_x, D-041): SUPPORTED.**

<!-- E-007 RESULT end -->


## E-006: rPPG signal-sanity check S-1 on real clips of FF++, Celeb-DF v2 and LAV-DF   (Iteration I-7, part S-1)
- Status: PLANNED on 2026-10-25, written before the full run (CLAUDE.md rule 2). The hypothesis, thresholds and the pass rule are those of ITERATIONS.md I-7, frozen by D-041; nothing in them is changed. The full design with every fixed parameter is docs/S1_SPEC.md (written before any measurement; a 9-clip smoke test only checked that the pipeline runs and changed no parameter). The result block is written by script.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU, CPU only, at most 8 worker processes / the commit that holds this entry and docs/S1_SPEC.md (see git log) / sampling seed 0.
- Observation: no pulse ground truth is held; the videos are H.264 compressed; the 16-frame crops of D-034 cannot carry a pulse (ITERATIONS.md I-7).
- Source of the method: R-057 (Wang, den Brinker, Stuijk, de Haan, IEEE TBME 64(7) 2017, author's pre-review version): READ for the abstract, Section II (model), III.A and III.B, IV.A and IV.B (equations 28 to 34 and Algorithm 1), V.A to V.C and the footnotes; the published journal text and the CHROM paper (de Haan and Jeanne 2013) were not read. POS: temporal normalisation over a window of l frames (l = 32 at 20 fps, 1.6 s), $S_1=G_n-B_n$, $S_2=-2R_n+G_n+B_n$, $h=S_1+\frac{\sigma(S_1)}{\sigma(S_2)}S_2$, overlap-add of the zero-mean $h$, no band-pass (eq. 32 to 34, Algorithm 1). This agrees with MATH_NOTES.md section 17.
- Hypothesis (ITERATIONS.md I-7): a pulse signal can be recovered from our real videos at all, in the sense of the rule below. Prior expectation written there: S-1 passes only partially, because compression attenuates the pulse; this is a hypothesis, not a result.
- Method (summary of docs/S1_SPEC.md): real clips only, from FF++ c23 original sequences (train and val), Celeb-DF v2 real videos (dev_x) and LAV-DF real videos (official train split, from its metadata file because no manifest exists); never a test split. Seed 0, up to 150 clips per dataset among the videos with at least 10 s of frames (FF++ 839 eligible, Celeb-DF 132, LAV-DF 3,731). From each, 10 s of contiguous frames at the native frame rate (start 0.5 s) from the original video; YuNet on every frame, largest face at least 48 px, causal box smoothing, central ROI 0.2 to 0.8 of the box, YCrCb skin mask, mean RGB; missing frames interpolated; usable if the share of frames with a face is at least 0.8 and no gap exceeds 1 s; POS with l = round(1.6 fps); linear detrend and Hann window, zero-padding to a bin spacing of at most 0.01 Hz, peak in 0.7 to 3.0 Hz; SNR in dB = power within 0.2 Hz of the peak and of its second harmonic over the remaining power in 0.7 to 3.0 Hz. Negative control: 20 independent time shuffles of the RGB trace per usable clip through the same code, the threshold is the 95th percentile of the pooled shuffled SNR values.
- Success / failure lines (frozen, mechanical): S-1 passes if at least 60 percent of the sampled real clips (pooled over the three datasets; clips that are unusable stay in the denominator as "not above") have a peak SNR strictly above that threshold. If it does not, the result is reported exactly as "rPPG signal not recoverable from these data" with the numbers, and no separation, fusion or added-value experiment is run. The heart-rate error clause does not apply (no pulse ground truth held). If it passes, the steps (b) to (d) of I-7 follow on dev_x only.
- Limits written before the run: the time-shuffle null is lenient (it destroys the slow coloured drift and motion of real traces), so a pass shows more narrow-band structure than a time-shuffled trace and does not show that a heart rate was recovered; there is no ground truth to check the peak frequency; our videos are compressed; Celeb-DF dev_x has 15 identity groups, so its Wilson interval is optimistic; the pass rule is pooled and the per-dataset fractions are reported without entering the rule.
- Prediction (sign and size): the pooled fraction above the threshold is between 0.30 and 0.80 (the prior expectation is a partial pass); per dataset the order FF++ and Celeb-DF below LAV-DF is not predicted.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/rppg/run_s1.py` (8 workers, about 5 minutes expected). Output data/_results/S-1/.
- Result (generated by scripts/rppg/run_s1.py, data/_results/S-1/result_block.md, run 2026-10-25, 432 clips, 280 s wall):

Threshold (95th percentile of the pooled shuffled SNR, 8520 values): 1.118 dB.

| Dataset | Eligible / candidates | Sampled | Usable | Above threshold, of sampled (Wilson 95%) | Above, of usable (Wilson 95%) | Median real SNR dB (usable) |
|---|---|---|---|---|---|---|
| ffpp | 839 / 860 | 150 | 148 | 42/150 = 0.280 [0.214, 0.357] | 42/148 = 0.284 [0.217, 0.361] | -0.277 |
| celebdf_v2 | 132 / 146 | 132 | 129 | 38/132 = 0.288 [0.218, 0.370] | 38/129 = 0.295 [0.223, 0.378] | -0.272 |
| lavdf | 3731 / 21254 | 150 | 149 | 50/150 = 0.333 [0.263, 0.412] | 50/149 = 0.336 [0.265, 0.415] | 0.002 |
| pooled | all | 432 | 426 | 130/432 = 0.301 [0.260, 0.346] | 130/426 = 0.305 [0.263, 0.350] | -0.222 |

Pass rule (mechanical): pooled share of sampled real clips with SNR above the 95th percentile of the pooled shuffled control is at least 0.6. Pooled fraction 0.301, required 0.6: S-1 passes = False.

Unusable clips (counted as not above the threshold): {"ffpp": {"face_share": 2}, "celebdf_v2": {"face_share": 3}, "lavdf": {"face_share": 1}, "pooled": {"face_share": 6}}.

- **Verdict (mechanical, frozen rule of ITERATIONS.md I-7): S-1 FAILS. Pooled fraction 0.301 [0.260, 0.346] against the required 0.6. rPPG signal not recoverable from these data** (threshold 1.118 dB from 8,520 pooled shuffled SNR values; per dataset 0.280 FF++, 0.288 Celeb-DF v2, 0.333 LAV-DF; 6 of 432 clips unusable). The median SNR of the real clips is below zero dB in FF++ and Celeb-DF v2 and about zero in LAV-DF, so for most clips the strongest peak carries less power than the rest of the band. The prediction written before the run (pooled fraction between 0.30 and 0.80) is met at its lower edge (0.301). The null was lenient (limits above), so the failure is not explained by a hard null. No pulse ground truth exists, so the 30 percent of clips above the threshold are not claimed to carry a pulse. No separation, fusion or added-value experiment (I-7 b to d) is run. This is the result of the I-7 slot, reported as a negative result with the negative control (CLAUDE.md rule 9 spirit: failed runs stay in the log).
- Decision triggered: I-7 reports exactly the sentence above; the rPPG GPU/CPU budget is released to the other slots (decision D-052).

## E-008: nuisance augmentation of the speech training clips (hypothesis H-A2)   (Iteration I-3, task T-A)
- Status: PLANNED on 2026-10-25, written before any augmented feature exists (CLAUDE.md rule 2). Lever and reasons: D-055 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry, src/dfd/prep/audio_aug.py and scripts/i3/run_ha2.py (see git log) / head seeds 0, 1, 2.
- Observation (E-007): with silence trimming and peak normalisation the T-A head has validation AUC 0.9889 [0.9831, 0.9933] and dev_x AUC 0.6723 [0.6126, 0.7584] (12 speakers); the validation minus dev_x gap is 0.3166. After the preprocessing, the single feature RMS energy still has AUC 0.135 [0.054, 0.382] on In-the-Wild dev_x (0.541 on ASVspoof dev), so a level-related cue remains and reverses sign out of domain.
- Diagnosis: shift. The head is fitted on one corpus whose recording and channel conditions are nearly constant within a class; its features can encode channel properties that do not carry over.
- Mathematical hypothesis (MATH_NOTES 14): replace the empirical risk by $\min_\theta \mathbb{E}_{(x,y)}\mathbb{E}_{t\sim\mathcal T} L(f_\theta(t(x)),y)$ with $\mathcal T$ a family of nuisance transforms (additive noise, low-pass, reverberation, MP3 round trip). The minimiser is fitted to the pushed-forward distribution, so a cue that $\mathcal T$ destroys or randomises cannot be used reliably and the learned score has to rely on cues that survive $\mathcal T$. Whether the surviving cues are the speech-synthesis artefacts or other channel cues is not implied. Invariance is learned only to the transformations in $\mathcal T$ (the In-the-Wild conditions may lie outside it).
- Change: the training set of the head is the E-007 window of every ASVspoof 2019 LA training clip (the original) plus 3 augmented copies per clip (76,140 extra rows, 101,520 rows in all). Each copy applies independently: reverberation (probability 0.5, synthetic exponential-decay impulse response, RT60 uniform in 0.15 to 0.8 s), low-pass (0.5, 6th-order Butterworth, cut-off uniform in 3,400 to 7,000 Hz), MP3 round trip (0.5, bit rate from 24, 32, 48, 64, 96, 128 kbit/s), additive white or one-pole-coloured noise (0.7, SNR uniform in 5 to 30 dB); if none was drawn the noise is applied; the peak is renormalised to 1 afterwards. The random generator is seeded by the sample id and the copy number, so the copies are reproducible. Ranges are fixed here from the nuisance types of M-004 and are not tuned on dev_x. Encoder AU-8 (HuBERT base) frozen, mean and standard deviation over time, the head recipe of dfd.sweep.heads (class-weighted over rows, early stopping on the clean validation AUC), seeds 0, 1, 2. Validation and dev_x features are the clean E-007 features (unchanged). Features: data/_features_aug/AU-8/T-A/ (new files; E-003 and E-007 files are not touched). Code: src/dfd/prep/audio_aug.py, the variants aug1 to aug3 of dfd.sweep.extract_audio.
- Data and split IDs: ASVspoof 2019 LA train and dev (asvspoof2019_la-875bf337c379, 25,380 and 24,844 items), In-the-Wild dev_x (in_the_wild-5cf2ea54a724, 6,991 items, 12 speakers). One augmented copy of ASVspoof dev (copy 1) is also extracted, used only for the robustness measurement below. No test item is read.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_audio --cand AU-8 --variant aug1 --out-root data/_features_aug --jobs asvspoof_train,asvspoof_dev` (and aug2, aug3 with `--jobs asvspoof_train`), then `PYTHONPATH=src .venv/Scripts/python.exe scripts/i3/run_ha2.py --write`. Budget: about 100,000 augmented windows, decoding and augmentation on the CPU (8 workers) and one HuBERT pass, expected 10 to 20 minutes; the head training a few minutes on the CPU.
- Prediction (sign and size): dev_x AUC rises by +0.03 to +0.12 over the E-007 head (from 0.6723); validation AUC changes by at most -0.01 (the clean validation split loses the shortcut share); the AUC of the E-007 head on the augmented validation copy is lower than on the clean copy by at least 0.01, and the new head loses less than the E-007 head.
- Precondition check (the hypothesis is tested only if it holds): the AUC of the E-007 head on the augmented validation copy is at least 0.01 below its AUC on clean validation (the augmentation is a perturbation for the baseline). If not, the verdict is NOT TESTED.
- Success / failure lines (mechanical): SUPPORTED if the precondition holds, the lower bound of the paired dev_x difference (new minus E-007 head, speaker bootstrap, 3 seeds, 2,000 resamples) is above 0, and the validation AUC has not fallen by more than 0.02. NOT SUPPORTED if the dev_x interval includes zero or lies below zero, or the validation AUC falls by more than 0.02. With 12 speakers the dev_x interval is wide, so a true gain below about +0.09 cannot be shown; this limits the power and is stated, not hidden. All dev_x verdicts are exploratory (D-041). The comparison with the I-0 probe (E-004) is reported as well.
- Result (generated by scripts/i3/run_ha2.py from data/_results/I-3, rerunnable):
<!-- E-008 RESULT begin (generated by scripts/i3/run_ha2.py) -->

| split | new (augmented training) | E-007 head | I-0 probe | new minus E-007 | new minus I-0 |
|---|---|---|---|---|---|
| validation | 0.9869 [0.9818, 0.9911] | 0.9889 [0.9831, 0.9933] | 0.9942 [0.9910, 0.9967] | -0.0020 [-0.0043, +0.0006] | -0.0074 [-0.0132, -0.0024] |
| dev_x | 0.7455 [0.7052, 0.8121] | 0.6723 [0.6126, 0.7584] | 0.6100 [0.5434, 0.7256] | +0.0732 [+0.0122, +0.1225] | +0.1355 [+0.0515, +0.2088] |

Robustness on ASVspoof dev, clean against one augmented copy (copy 1), AUC [95% interval]:

| head | clean | augmented copy | drop |
|---|---|---|---|
| E-007 head | 0.9889 [0.9831, 0.9933] | 0.6569 [0.6229, 0.6875] | +0.3320 |
| new head | 0.9869 [0.9818, 0.9911] | 0.9436 [0.9299, 0.9558] | +0.0433 |

Training rows: E-007 head 25380, new head 101520. Early-stopping epochs (seeds 0, 1, 2): E-007 head [79, 54, 60], new head [40, 39, 31]. The retrained E-007 head gives validation 0.9889 and dev_x 0.6723 against 0.9889 and 0.6723 logged in E-007.

ECE before / after temperature scaling (fitted on clean validation), new head: validation 0.0305 / 0.0020, dev_x 0.1733 / 0.2253 (E-007 head: 0.0835 / 0.0048 and 0.2252 / 0.3116).

Precondition passed (the E-007 head loses +0.3320 AUC on the augmented validation copy; required at least 0.01).
Prediction check: dev_x gain +0.03 to +0.12 over the E-007 head, found +0.0732 (inside the range); validation change at most -0.01, found -0.0020; the new head loses +0.0433 on the augmented copy against +0.3320 for the E-007 head.

**VERDICT (mechanical, exploratory on dev_x, D-041): SUPPORTED.**

<!-- E-008 RESULT end -->

## E-009: averaging the logits of three speech encoders (hypothesis H-A3)   (Iteration I-4, task T-A)
- Status: PLANNED on 2026-10-25, written before any feature of AU-4 or AU-6 under the new preprocessing exists (CLAUDE.md rule 2). Lever and reasons: D-056 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry and scripts/i4/run_ha3.py (see git log) / head seeds 0, 1, 2.
- Observation (E-008, E-003, D-051): with the E-007 preprocessing and the E-008 augmentation the AU-8 head has validation AUC 0.9869 [0.9818, 0.9911] and dev_x AUC 0.7455 [0.7052, 0.8121] (12 speakers). On the frozen E-003 features the three tied speech encoders (AU-8 HuBERT base, AU-4, AU-6 Whisper) had a mean pairwise score rank correlation of 0.428 on validation and 0.257 on dev_x (D-051), and the best dev_x lower bound was 0.5338 (INCONCLUSIVE).
- Diagnosis: variance and complementary errors (ITERATIONS I-5): three encoders with weakly correlated scores, no member clearly the best.
- Mathematical hypothesis (MATH_NOTES 15): for $m$ scores with equal variance $\sigma^2$ and mean pairwise correlation $\bar\rho$, the variance of the average is $\sigma^2(\bar\rho+(1-\bar\rho)/m)$; the bias of the average is the average of the biases, so only the variance part falls. With $m=3$ and $\bar\rho=0.3$ the variance falls to 53 percent of a single member's, with $\bar\rho=0.8$ to 87 percent. The average of logits adds no parameter. It does not help if the members share the same bias (the same shortcut), which the shift measured in E-007 suggests may be the case for encoders trained on the same corpus.
- Change: for each of AU-8, AU-4 and AU-6 the same pipeline as E-008 (E-007 window, three augmented copies of every training clip, head recipe of dfd.sweep.heads, seeds 0, 1, 2); the clip score of the combination for seed $s$ is the mean of the three members' logits of the clip probability (seed $s$ of each member); all three are on the logit scale, no recalibration before averaging. Nothing else changes. Features: data/_features_trimnorm/AU-4, AU-6 and data/_features_aug/AU-4, AU-6 (new files; the AU-8 files of E-007 and E-008 are not touched). Whisper input as in E-003 (30 s window, the 4 s window first, 202 frames pooled).
- Data and split IDs: ASVspoof 2019 LA train and dev (asvspoof2019_la-875bf337c379), In-the-Wild dev_x (in_the_wild-5cf2ea54a724, 12 speakers); the same augmented copies as E-008 (seeded by sample id and copy number). No test item is read.
- Command: for AU-4 and AU-6: `PYTHONPATH=src .venv/Scripts/python.exe -m dfd.sweep.extract_audio --cand <AU-x> --variant trim_norm --out-root data/_features_trimnorm`, then `--variant aug1 --out-root data/_features_aug --jobs asvspoof_train,asvspoof_dev`, and aug2, aug3 with `--jobs asvspoof_train`; then `PYTHONPATH=src .venv/Scripts/python.exe scripts/i4/run_ha3.py --write`. Budget: about 160,000 windows per encoder, decoding and augmentation on 8 CPU workers and one encoder pass, expected 15 to 30 minutes per encoder; the head training a few minutes on the CPU.
- Prediction (sign and size): dev_x AUC of the combination above the AU-8 head of E-008 by +0.02 to +0.06 (from 0.7455); validation AUC changes by at most -0.01; the mean pairwise correlation of the members' dev_x scores under the new pipeline is below 0.8 (measured and reported; it is not a gate).
- Success / failure lines (mechanical): SUPPORTED if the lower bound of the paired dev_x difference (combination minus the AU-8 head of E-008, speaker bootstrap on the seed-averaged scores, 2,000 resamples) is above 0 and the validation AUC has not fallen by more than 0.02. NOT SUPPORTED if the dev_x interval includes zero or lies below zero, or the validation AUC falls by more than 0.02 (the cheaper single encoder is then kept, CLAUDE.md rule 4). The combination is also compared with the best single member under the same pipeline, and each member is reported alone. With 12 speakers the dev_x interval is wide, so a true gain below about +0.07 cannot be shown; this limits the power and is stated, not hidden. All dev_x verdicts are exploratory (D-041). The cost of the combination (three encoder passes at inference) is reported next to the gain.
- Result (generated by scripts/i4/run_ha3.py from data/_results/I-4, rerunnable):
<!-- E-009 RESULT begin (generated by scripts/i4/run_ha3.py) -->

| scores | validation AUC | dev_x AUC | ensemble minus this, validation | ensemble minus this, dev_x |
|---|---|---|---|---|
| ensemble of AU-8, AU-4, AU-6 | 0.9964 [0.9943, 0.9982] | 0.8130 [0.7523, 0.8808] | | |
| AU-8 alone (E-008 head) | 0.9869 [0.9818, 0.9911] | 0.7455 [0.7052, 0.8121] | +0.0095 [+0.0067, +0.0131] | +0.0675 [+0.0300, +0.0929] |
| AU-4 alone | 0.9699 [0.9595, 0.9786] | 0.7823 [0.7172, 0.8436] | +0.0265 [+0.0195, +0.0349] | +0.0307 [+0.0160, +0.0579] |
| AU-6 alone | 0.9951 [0.9924, 0.9973] | 0.7526 [0.6851, 0.8406] | +0.0013 [+0.0003, +0.0024] | +0.0604 [+0.0208, +0.0822] |

Mean pairwise Spearman correlation of the members' clip scores (mean over the seeds): validation 0.624, dev_x 0.534; pairs on dev_x: AU-8-AU-4 0.646, AU-8-AU-6 0.472, AU-4-AU-6 0.485.
Best single member by validation AUC: AU-6. Early-stopping epochs (seeds 0, 1, 2): AU-8 [40, 39, 31]; AU-4 [32, 20, 45]; AU-6 [34, 40, 42].
The retrained AU-8 head gives validation 0.9869 and dev_x 0.7455 against 0.9869 and 0.7455 logged in E-008.
ECE before / after temperature scaling (fitted on validation), ensemble: validation 0.0417 / 0.0016, dev_x 0.0692 / 0.1670.
Extraction speed of the members on In-the-Wild dev_x (items per second, end to end, one run): AU-8 237.3, AU-4 209.8, AU-6 64.0; parameters: AU-8 94,371,712, AU-4 94,381,936, AU-6 88,154,112.

Prediction check: dev_x gain over the AU-8 head +0.02 to +0.06, found +0.0675 (larger than the range); validation change at most -0.01, found +0.0095; mean pairwise correlation on dev_x 0.534 (below 0.8).

**VERDICT (mechanical, exploratory on dev_x, D-041): SUPPORTED.**

<!-- E-009 RESULT end -->

## E-010: pre-check of the frozen visual and audio heads on LAV-DF, then late fusion of the two   (Iteration I-5, the I-6 lever of ITERATIONS.md)
- Status: PLANNED on 2026-10-25, written before any LAV-DF score exists (CLAUDE.md rule 2). Lever and reasons: D-057 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry, scripts/i6/score_lavdf.py and scripts/i6/analyse_i6.py (see git log) / head seeds 0, 1, 2 (the heads are frozen; no head is trained here).
- Observation (E-005, E-008, E-009, D-051 item 4): the visual head (fine-tuned IMG-17, FF++ training) has validation AUC 0.9899 and dev_x 0.9096 on face-swap material; the audio heads (ASVspoof 2019 LA training) have dev_x AUC around 0.75 on In-the-Wild with 12 speakers. Neither head has been scored on a dataset where video and audio are manipulated independently. LAV-DF (136,304 videos, all with audio; D-051) provides the two per-modality labels `modify_video` and `modify_audio`. Its fakes are content-driven manipulations of segments of a talking-head video (the generators are not those of FF++ or ASVspoof), so a loss of both heads is expected.
- Diagnosis: shift (domain and manipulation type) for both heads; fusion has value only if both heads carry evidence on the same data and their errors are not strongly correlated.
- Mathematical hypothesis (MATH_NOTES 16): if the visual and audio log-odds $\ell_v,\ell_a$ are conditionally independent given the label, the posterior log-odds are $\ell_v+\ell_a-\ln\frac{\pi}{1-\pi}$ (a sum, no parameter); if they are dependent or mis-scaled, a logistic regression $\sigma(w_0+w_a\ell_a+w_v\ell_v)$ fitted on labelled fusion data has weights that differ from 1 and absorbs the scale. Fusion cannot lift the AUC above the better member by more than the independent part of the evidence; a member at chance contributes variance only.
- Data: LAV-DF subset (D-057): 500 videos for each of the four cells (real; video fake only; audio fake only; both fake) in each official split (train, dev as `val`, test as `test_in`), fixed seed 20261025, manifest data/_manifests/lavdf_subset.manifest.csv, split data/_splits/lavdf_subset-e9d12decaf1f.csv (2,000 clips per split). Group = source video (a fake and its real original share a group); the official split is identity-disjoint by the authors' rule (identities are not in the metadata). Fused label: fake if `modify_video` or `modify_audio`. Only `train` and `val` are scored; `test_in` is touched only at the freeze under `--final`; its face crops are not prepared yet. Fusion layers are fitted on `train` and compared on `val`.
- Change: none to any head. Visual score: mean of the frame probabilities (D-024) of IMG-17 (E-005 winner, learning rate 1e-5, seeds 0, 1, 2) on the same 16-frame crops as before (YuNet, 224 px). Audio score: mean of the logits of the AU-8, AU-4 and AU-6 heads of E-009 on the E-007 window of the audio track of the video (per seed). Temperature of each modality fitted on its source-domain validation split (FF++ val and ASVspoof dev) and, for comparison, refitted on LAV-DF `train`; both are reported. Fusion rules compared on LAV-DF `val`: (a) sum of the two calibrated log-odds (the prior term is left out: AUC is invariant to it); (b) logistic regression on the two calibrated log-odds fitted on LAV-DF `train` (3 parameters, no regularisation tuning). Clips without a face frame (visual score missing) are counted and scored with the audio side only in the fused result (their visual log-odds set to 0, i.e. no evidence); this is reported.
- Pre-check (D-051 item 4, written before the scores): for each head the AUC on LAV-DF `val` against its own modality label (visual head against `modify_video`, audio head against `modify_audio`), 95 percent group bootstrap, seeds averaged. A head whose lower bound is at or below 0.55 is declared near chance on LAV-DF. If either head is near chance, no fused accuracy is claimed, the single-modality results and the fallback claim of PLAN.md risk 7 stand, the verdict is NOT TESTED for fusion, and the alternative named in ITERATIONS I-6 (a head retrained on LAV-DF) is a separate decision.
- Prediction (sign and size): both heads lose AUC against their in-domain values; the visual head AUC against `modify_video` falls to 0.55 to 0.80 and the audio head AUC against `modify_audio` to 0.50 to 0.75 (I have no measurement of either; these ranges are guesses from the shift seen on Celeb-DF and In-the-Wild); if both pass the pre-check, the fused AUC against the fused label exceeds the better single head by 0 to +0.05, with the logistic-regression weights below 1.
- Success / failure lines (mechanical): SUPPORTED if both heads pass the pre-check and the lower bound of (best fusion rule minus the better single head), paired group bootstrap on `val`, 2,000 resamples, on the fused label is above 0. NOT SUPPORTED if both pass and that interval includes zero or lies below zero. NOT TESTED if a head fails the pre-check. The fusion rule that is reported as best is chosen on `val` AUC point estimate; with two rules and 2,000 clips this choice is itself noisy and is stated. All verdicts are exploratory (D-041); LAV-DF `test_in` gives the one confirmatory number at the freeze.
- Result (generated by scripts/i6/analyse_i6.py from data/_results/I-6, rerunnable):
<!-- E-010 RESULT begin (generated by scripts/i6/analyse_i6.py) -->

Audio side: AU-8 + AU-4 + AU-6 (E-009 verdict: SUPPORTED). Clips without a visual score: train 0, val 0 of 2,000. Temperatures (seeds 0, 1, 2): visual [0.832, 0.834, 0.758], audio [0.454, 0.445, 0.48]; refitted on LAV-DF train: visual 100.000, audio 8.581.

Pre-check on LAV-DF val (frozen heads, no fitting; AUC with 95 percent group bootstrap interval):

| head | against its own modality label | against the fused label |
|---|---|---|
| visual (IMG-17, E-005) | 0.3838 [0.3599, 0.4079] | 0.2786 [0.2534, 0.3042] |
| audio (AU-8 + AU-4 + AU-6) | 0.6476 [0.6244, 0.6701] | 0.6359 [0.6094, 0.6614] |

Pre-check rule: lower bound above 0.55. Visual: FAIL; audio: pass.

Per cell on val (the cell against the real videos), AUC [95%]:

| cell | visual head | audio head |
|---|---|---|
| video_fake | 0.2788 [0.2488, 0.3087] | 0.5564 [0.5215, 0.5895] |
| audio_fake | 0.2876 [0.2568, 0.3174] | 0.6804 [0.6464, 0.7119] |
| both_fake | 0.2692 [0.2393, 0.2986] | 0.6708 [0.6378, 0.7029] |

ECE on val against the own modality label, before any LAV-DF fitting (source-domain temperature) and with the temperature refitted on LAV-DF train: visual 0.3323 / 0.0483, audio 0.2735 / 0.0299.
Pearson correlation of the calibrated visual and audio log-odds on val: real clips 0.049, fake clips 0.002.

Fusion on val against the fused label (computed for the record; the verdict follows the pre-check). Logistic regression fitted on LAV-DF train: intercept +0.324, visual weight -0.700, audio weight 0.123.

| scores | AUC [95%] fused label | AUC against modify_video | AUC against modify_audio |
|---|---|---|---|
| visual head | 0.2786 [0.2534, 0.3042] | 0.3838 [0.3599, 0.4079] | 0.3873 [0.3627, 0.4101] |
| audio head | 0.6359 [0.6094, 0.6614] | 0.5254 [0.5009, 0.5501] | 0.6476 [0.6244, 0.6701] |
| sum of log-odds | 0.5783 [0.5513, 0.6046] | 0.4983 [0.4737, 0.5227] | 0.6160 [0.5921, 0.6391] |
| logistic regression | 0.7536 [0.7289, 0.7781] | 0.6039 [0.5797, 0.6285] | 0.6787 [0.6559, 0.7016] |

Best single head by val AUC: audio head; best fusion rule by val AUC: logistic regression; difference (best rule minus best single head, paired group bootstrap): +0.1178 [+0.0942, +0.1424].

**VERDICT (mechanical, exploratory, D-041): NOT TESTED (pre-check: the visual head is near chance on LAV-DF, lower bound at or below 0.55; no fused result is claimed).**

<!-- E-010 RESULT end -->

## E-011: in-domain visual and audio heads on LAV-DF, then late fusion   (Iteration I-6)
- Status: PLANNED on 2026-10-25, written after E-010 and before any in-domain head exists (CLAUDE.md rule 2). Lever and reasons: D-058 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry, scripts/i6/train_visual_lavdf.py, scripts/i6/train_audio_lavdf.py and scripts/i6/analyse_e011.py (see git log) / seeds 0, 1, 2.
- Observation (E-010): on LAV-DF val the frozen FF++-trained visual head has AUC 0.3838 [0.3599, 0.4079] against `modify_video` and 0.2786 against the fused label, below chance; it also scores clips whose video is untouched (audio fakes) as more real than real clips (AUC 0.2876 for that cell against real), so the inversion is not a response to the visual manipulation. The frozen audio head has AUC 0.6476 [0.6244, 0.6701] against `modify_audio`. The pre-check failed for the visual head; fusion was NOT TESTED. The scoring code reproduces the E-005 validation scores on FF++ (maximum difference 0.00012 on 175 clips; a check of the code, not a result).
- Diagnosis: shift in domain and manipulation type for both heads, and for the visual head a reversed sign (a cue specific to FF++ and its compression pipeline, as for the speech heads in M-004).
- Mathematical hypothesis (MATH_NOTES 16, 11): a head fitted on the target distribution minimises the risk where it is evaluated, so the sign reversal and the loss of margin caused by the shift cannot persist as they do for a head fitted elsewhere; the fused log-odds then follow the conditional-independence argument of MATH_NOTES 16 (sum) or the fitted weights of the logistic regression.
- Change: (a) visual: IMG-17 fine-tuned with the E-005 recipe unchanged (learning rate 1e-5, 1,800 steps, batch 32, AdamW with weight decay 0.05, class-balanced BCE, no augmentation, no selection on any split) on the face crops of 10,000 LAV-DF head-training videos, label `modify_video` (5,000 positive: video fake and both fake; 5,000 negative: real and audio fake), seeds 0, 1, 2. (b) audio: linear heads (dfd.sweep.heads recipe, seeds 0, 1, 2) on AU-8, AU-4 and AU-6 features of the E-007 window of the audio track, same 10,000 videos, label `modify_audio`, early stopping on the AUC against `modify_audio` on LAV-DF val (as every head of this project stops on its validation split; this makes the audio val values slightly optimistic and is stated); no augmentation copies. (c) calibration: a temperature per modality and seed fitted on the fusion-fit subset (the 2,000 `train` videos of the E-010 subset, disjoint in source video from the head-training videos). (d) fusion on the fused label (fake if `modify_video` or `modify_audio`): sum of the calibrated log-odds, and logistic regression fitted on the fusion-fit subset.
- Data and split IDs: head-training subset data/_manifests/lavdf_headtrain.manifest.csv, split lavdf_headtrain-b5544b99f6bf (2,500 videos per cell from the official train split, fixed seed 20261026, 7,808 source groups, none shared with the E-010 subset `train` or `val`); fusion-fit subset and val: lavdf_subset-e9d12decaf1f `train` and `val` (2,000 videos each). `test_in` is not scored before the freeze.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/train_visual_lavdf.py --seeds 0 1 2`, `PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/train_audio_lavdf.py`, then `PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/analyse_e011.py --write`. Budget: visual about 5 to 6 minutes per seed (the E-005 cost of IMG-17) plus scoring, audio about 15 minutes of extraction for three encoders, analysis a minute; one GPU job at a time.
- Prediction (sign and size): visual val AUC against `modify_video` between 0.70 and 0.95 (the manipulation is content-driven; I have no in-domain measurement); audio val AUC against `modify_audio` between 0.80 and 0.99; on the fused label the best fusion rule exceeds the better single head by 0 to +0.10, because a fake can sit in either modality and each head sees only one; logistic-regression weights both positive.
- Success / failure lines (mechanical): C1 visual pass: lower bound of the visual AUC against `modify_video` on val above 0.55. C2 audio pass: same for the audio head against `modify_audio`. C3 fusion gain: lower bound of (best fusion rule minus better single head, fused label, paired group bootstrap on val, 2,000 resamples) above 0. SUPPORTED if C1, C2 and C3 hold. PARTIAL if C1 and C2 hold and C3 fails, or if exactly one of C1 and C2 holds (then no fused result is claimed). NOT SUPPORTED if neither holds. The best rule is chosen on val AUC, a noisy choice between two rules, stated. All verdicts are exploratory (D-041); the one confirmatory LAV-DF number is the `test_in` evaluation at the freeze.
- Result (generated by scripts/i6/analyse_e011.py from data/_results/I-6, rerunnable):
<!-- E-011 RESULT begin (generated by scripts/i6/analyse_e011.py) -->

Temperatures fitted on the fusion-fit subset (seeds 0, 1, 2): visual [0.859, 0.838, 0.881], audio [0.939, 0.947, 0.952]. Clips without a visual score: train 0, val 0 of 2,000.

Pre-check on LAV-DF val (in-domain heads; AUC [95% group bootstrap]; E-010 values of the frozen out-of-domain heads in the last column):

| head | against its own modality label | against the fused label | against the other modality's label | E-010 frozen head, own label |
|---|---|---|---|---|
| visual (IMG-17 fine-tuned on LAV-DF) | 0.7491 [0.7280, 0.7697] | 0.9913 [0.9871, 0.9949] | 0.7473 [0.7268, 0.7676] | 0.3838 [0.3599, 0.4079] |
| audio (AU-8 + AU-4 + AU-6 on LAV-DF) | 0.8604 [0.8435, 0.8769] | 0.7480 [0.7260, 0.7689] | 0.5098 [0.4848, 0.5341] | 0.6476 [0.6244, 0.6701] |

Pre-check rule: lower bound above 0.55. Visual: pass; audio: pass.

Per cell on val (the cell against the real videos), AUC [95%]:

| cell | visual head | audio head |
|---|---|---|
| video_fake | 0.9905 [0.9857, 0.9945] | 0.5175 [0.4836, 0.5532] |
| audio_fake | 0.9912 [0.9864, 0.9951] | 0.8711 [0.8474, 0.8920] |
| both_fake | 0.9921 [0.9878, 0.9959] | 0.8556 [0.8327, 0.8779] |

ECE on val against the own modality label (temperature from the fusion-fit subset): visual 0.0345, audio 0.0268. Pearson correlation of the calibrated log-odds on val: real clips 0.026, fake clips 0.032.

Fusion on val. Logistic regression fitted on the fusion-fit subset: intercept +2.963, visual weight 4.930, audio weight 0.813.

| scores | AUC [95%] fused label | AUC against modify_video | AUC against modify_audio |
|---|---|---|---|
| visual head | 0.9913 [0.9871, 0.9949] | 0.7491 [0.7280, 0.7697] | 0.7473 [0.7268, 0.7676] |
| audio head | 0.7480 [0.7260, 0.7689] | 0.5098 [0.4848, 0.5341] | 0.8604 [0.8435, 0.8769] |
| sum of log-odds | 0.9795 [0.9726, 0.9857] | 0.6553 [0.6319, 0.6793] | 0.9212 [0.9103, 0.9320] |
| logistic regression | 0.9935 [0.9905, 0.9962] | 0.6820 [0.6589, 0.7058] | 0.8887 [0.8767, 0.9019] |

Best single head by val AUC: visual head; best fusion rule by val AUC: logistic regression; difference (best rule minus best single head, paired group bootstrap): +0.0023 [+0.0009, +0.0040].
Criteria: C1 visual pass True, C2 audio pass True, C3 fusion lower bound above 0 True.

**VERDICT (mechanical, exploratory, D-041): SUPPORTED.**

<!-- E-011 RESULT end -->
- Post-hoc control of E-011 (written after the result above, labelled exploratory; it does not change the mechanical verdict):
<!-- E-011 CONTROL begin (generated by scripts/i6/control_pipeline.py) -->

Post-hoc control (not a pre-declared criterion; scripts/i6/control_pipeline.py): AUC [95% group bootstrap] on LAV-DF val within the produced videos only (cells video_fake, audio_fake, both_fake; the real originals are left out), so that every clip went through the dataset's own pipeline.

| score | label asked | AUC within produced videos |
|---|---|---|
| visual head | modify_video | 0.5070 [0.4771, 0.5361] |
| audio head | modify_audio | 0.8574 [0.8394, 0.8752] |
| visual head | modify_audio (should be near 0.5) | 0.5029 [0.4743, 0.5320] |
| audio head | modify_video (should be near 0.5) | 0.3330 [0.3059, 0.3602] |

Visual head, real originals against audio-only fakes (video unmodified): 0.9912 [0.9864, 0.9951].

<!-- E-011 CONTROL end -->

## E-012: visual head trained on produced videos only, so that the production pipeline is not a cue   (Iteration I-6 follow-up, task T-V on LAV-DF)
- Status: PLANNED on 2026-10-25, written after the E-011 control and before any E-012 head exists (CLAUDE.md rule 2). Lever and reasons: D-059 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry, the `--produced-only` option of scripts/i6/train_visual_lavdf.py and scripts/i6/analyse_e012.py (see git log) / seeds 0, 1, 2.
- Observation (E-011 and its post-hoc control): the in-domain visual head separates the real originals from the audio-only fakes (video untouched) with AUC 0.9912 [0.9864, 0.9951], and within the produced videos (video fake and both fake against audio-only fake) its AUC against `modify_video` is 0.5070 [0.4771, 0.5361], at chance. The AUC of 0.7491 against `modify_video` over all videos and the fused AUC of 0.9935 come from the cue "original or produced by the dataset's pipeline", not from a visual manipulation cue.
- Diagnosis: shortcut (a nuisance variable, the production pipeline, is perfectly aligned with the label in the training set: every original is real, every produced video is fake or an audio fake).
- Mathematical hypothesis (MATH_NOTES 14, 10): the head minimises the risk over the training distribution, in which the nuisance $c$ (original or produced) predicts the visual label better than the manipulation $m$ does within the real/fake split of the original-versus-produced contrast; removing the originals from the training set makes $c$ constant (all produced), so $c$ carries no information about the label, and the minimiser has to use $m$ if $m$ is recoverable at all from the 224 px crops by this recipe. If $m$ is not recoverable, the within-produced AUC stays at 0.5.
- Change: the E-005/E-011 recipe unchanged (IMG-17, learning rate 1e-5, 1,800 steps, batch 32, class-balanced BCE, no augmentation, no selection) on the 7,500 produced videos of the head-training subset (2,500 video fake and 2,500 both fake as positives, 2,500 audio fake as negatives; the 2,500 real originals are left out), label `modify_video`, seeds 0, 1, 2. Nothing else changes.
- Data and split IDs: data/_manifests/lavdf_headtrain.manifest.csv (split lavdf_headtrain-b5544b99f6bf), val of lavdf_subset-e9d12decaf1f (2,000 videos, 1,500 produced). `test_in` is not scored.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/i6/train_visual_lavdf.py --seeds 0 1 2 --produced-only`, then `PYTHONPATH="src;scripts/i6" .venv/Scripts/python.exe scripts/i6/analyse_e012.py --write`. Budget: about 6 minutes per seed.
- Prediction (sign and size): the within-produced AUC against `modify_video` is between 0.55 and 0.80; the AUC of real originals against audio-only fakes (reported, not trained for) falls from 0.99 toward 0.5 to 0.7. I have no measurement to anchor the first range: lip-synchronised edits may be hard to see in 16 face crops per video.
- Success / failure lines (mechanical): C1 SUPPORTED if the lower bound of the within-produced AUC against `modify_video` on val is above 0.55 and the paired group bootstrap difference to the E-011 head (same clips) has a lower bound above 0. NOT SUPPORTED if the within-produced lower bound is at or below 0.55 (the visual manipulation is not recoverable by this recipe on this subset, which is then the result). The audio-visual fusion is not re-run in this entry. Exploratory (D-041).
- Result (generated by scripts/i6/analyse_e012.py from data/_results/I-6, rerunnable):
<!-- E-012 RESULT begin (generated by scripts/i6/analyse_e012.py) -->

AUC [95% group bootstrap] on LAV-DF val, seeds averaged. Within produced = cells video_fake, both_fake (positives) and audio_fake (negatives); the head of E-011 is scored on the same clips.

| quantity | E-012 head (produced only) | E-011 head (all videos) |
|---|---|---|
| within produced videos, against modify_video | 0.5169 [0.4945, 0.5392] | 0.5069 [0.4793, 0.5333] |
| all val videos, against modify_video | 0.5342 [0.5163, 0.5530] | 0.7481 [0.7280, 0.7679] |

Difference within produced videos (E-012 minus E-011, paired group bootstrap): +0.0100 [-0.0159, +0.0372].
E-012 head, fused label over all videos: 0.5459 [0.5227, 0.5677]. Real originals against audio-only fakes (video unmodified): 0.5347 [0.5088, 0.5598]; against video-only fakes: 0.5572 [0.5304, 0.5847].
Within produced videos, per fake cell against audio-only fakes: video_fake 0.5229 [0.4967, 0.5493]; both_fake 0.5108 [0.4859, 0.5367].

**VERDICT (mechanical, exploratory, D-041): NOT SUPPORTED (the within-produced lower bound is at or below 0.55, or the difference to the E-011 head includes zero).**

<!-- E-012 RESULT end -->

## E-013: re-run published baseline for the comparison of D-025: Xception fine-tuned on FF++ with the E-005 recipe   (baseline, task T-V)
- Status: PLANNED on 2026-10-25, written before any baseline run (CLAUDE.md rule 2). Choice and delta rule: D-060 (decided autonomously - needs user review). The result block is written by script and nothing above it is edited after the first run.
- Date / Environment / Git commit / Seed(s): 2026-10-25 / ENV-DESKTOP-GPU / the commit that holds this entry, scripts/baseline/run_baseline.py and scripts/baseline/analyse_baseline.py (see git log) / stage 1 seed 0, stage 2 seeds 1 and 2.
- Observation: the confirmatory claim (2) of the PROTOCOL addendum (D-041, D-025) compares the final system with a published detector family re-run under our protocol. No such run exists yet. The frozen sweep candidate IMG-04 is the original Xception (R-017 and the FF++ benchmark baseline family) with ImageNet weights (D-038).
- Diagnosis: not applicable (a baseline measurement).
- Mathematical hypothesis: none beyond E-005 (MATH_NOTES 11): the fine-tuned class contains the probe class for each encoder; the comparison of two different encoders is empirical. Stated as a prediction below.
- Change: IMG-04 (Xception, 20.8 M parameters, all weights trained) with the E-005 recipe unchanged: AdamW, weight decay 0.05, learning rate in {1e-5, 3e-5, 1e-4} with 5 percent warm-up and cosine decay, fp16 autocast, batch 32, 1,800 steps, class-balanced BCE, no augmentation, the same FF++ train frames, the same crops and 224 px input. The learning rate is chosen by validation AUC at seed 0 (the same selection rule as E-005 within an encoder); seeds 1 and 2 at that rate. Budget per run is the same cap as E-005 (600 s training, 900 s wall).
- Data and split IDs: as E-005 (ffpp-037e2976fade train and val, celebdf_v2-82a3a70594e9 dev_x). No test item is read.
- Command: `PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline/run_baseline.py` (log data/_logs/baseline_run.log), then `PYTHONPATH=src .venv/Scripts/python.exe scripts/baseline/analyse_baseline.py --write`.
- Prediction (sign and size): the Xception baseline reaches a validation AUC of 0.95 to 0.995 and a dev_x AUC below that of fine-tuned IMG-17 (0.9096): between 0.70 and 0.90. IMG-17 minus Xception on dev_x is positive. I have no measurement of Xception on this protocol (the sweep ranked it as a frozen feature only).
- Success / failure lines (mechanical): SUPPORTED if the lower bound of (IMG-17 fine-tuned, 3 seeds, minus Xception, 3 seeds), paired group bootstrap on dev_x, 2,000 resamples, is above 0. NOT SUPPORTED otherwise (then the non-inferiority comparison of D-025 at the freeze still applies and is reported as it comes out). Exploratory on dev_x (D-041). The seed standard deviations of the baseline are the input of the delta rule of D-060.
- Result (generated by scripts/baseline/analyse_baseline.py from data/_results/baseline, rerunnable):
<!-- E-013 RESULT begin (generated by scripts/baseline/analyse_baseline.py) -->

Selected learning rate (validation AUC only): 0.0001; stage 1 validation AUC by rate: 1e-05 0.9646, 3e-05 0.9809, 0.0001 0.9883.

| split | Xception baseline (3 seeds) | IMG-17 fine-tuned (E-005, 3 seeds) | IMG-17 minus Xception, paired | I-0 probe (IMG-07) | Xception minus I-0 |
|---|---|---|---|---|---|
| validation | 0.9872 [0.9784, 0.9942] (seed sd 0.0010) | 0.9899 [0.9809, 0.9967] | +0.0027 [-0.0009, +0.0060] | 0.7985 [0.7731, 0.8269] | +0.1887 [+0.1602, +0.2152] |
| dev_x | 0.8750 [0.8435, 0.9089] (seed sd 0.0072) | 0.9096 [0.8776, 0.9388] | +0.0346 [+0.0136, +0.0550] | 0.6933 [0.6639, 0.7314] | +0.1817 [+0.1396, +0.2166] |

Hypothesis of E-013 (IMG-17 fine-tuned is superior to the Xception baseline on dev_x, lower bound of the paired difference above 0): **SUPPORTED** (mechanical, exploratory, D-041). The seed standard deviation of the baseline on each split (above) is the quantity from which delta of D-025 is chosen before the final comparison (D-060). Delta by the rule of D-060 (2 times the larger seed standard deviation, floor 0.01): 0.0143.


<!-- E-013 RESULT end -->
