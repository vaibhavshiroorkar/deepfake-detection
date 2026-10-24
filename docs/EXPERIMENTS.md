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

## Experiment log

No experiments have been run yet.
