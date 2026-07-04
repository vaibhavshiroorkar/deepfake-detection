# Audio-Visual Deepfake Detection

A deepfake detector that fuses five independent streams — three visual models (Xception, EfficientNet, DINOv2) and two cross-modal mismatch signals (lip-sync/semantic and emotion) — into one real-or-fake decision. Partial fakes like lip-sync deepfakes barely disturb the video frames; they show up in the *disagreement* between audio and video, which is what the cross-modal streams measure.

**Start here:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — architecture, datasets, phases, team roles.
**Current phase:** Phase 0.5 (foundations) — day-by-day plan in [docs/phase-0.5-plan.md](docs/phase-0.5-plan.md).

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python version and virtual environment — it doesn't touch or depend on whatever `python` resolves to on your system PATH. Install uv once ([instructions](https://docs.astral.sh/uv/getting-started/installation/)), and [ffmpeg](https://ffmpeg.org/download.html) on your PATH (Windows: `winget install Gyan.FFmpeg`, then reopen the terminal).

The repo pins its Python version in `.python-version` (3.13) — uv reads that automatically and downloads that exact interpreter into its own managed store if you don't already have it. You never need to hunt for the right `python`/`py` command yourself.

### Default (CPU — works everywhere)

```bash
uv venv
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
```

Then either activate as usual (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux) or skip activation entirely and prefix commands with `uv run`, e.g. `uv run python preprocessing/extract_faces.py`.

### NVIDIA GPU (CUDA)

Install the CUDA build of PyTorch *first*, then the rest (uv keeps the CUDA build because the versions match requirements.txt):

```bash
uv pip install torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

CUDA 13 wheels need a recent NVIDIA driver — if `torch.cuda.is_available()` is False after installing, update the driver first.

Verify with:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**30, "GiB")
```

**Confirmed working (2026-07-04):** RTX 5070 Ti, ~16 GB VRAM, driver 596.49 — `torch.cuda.is_available()` returns `True` and a GPU matmul runs correctly on the cu130 wheels above.

### Google Colab

Colab ships with a GPU build of torch preinstalled — don't reinstall it. Install only the extras:

```python
!pip install opencv-python librosa ffmpeg-python timm facenet-pytorch
```

## Repo layout (current, lean)

```
data/            datasets, splits, manifests (raw media is git-ignored)
preprocessing/   face + audio extraction module (frozen interface, Phase 0.5)
models/baseline/ practice models and the first end-to-end slice
evaluation/      metrics, ablation, robustness tests
notebooks/       exploration and training notebooks
docs/            glossary, math writeups, phase plans, report drafts
```

Stream, fusion, and feature-store folders get added when those phases start — see PROJECT_OVERVIEW.md §8.
