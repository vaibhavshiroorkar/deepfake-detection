# Audio-Visual Deepfake Detection

A deepfake detector that fuses five signals, three visual models (Xception, EfficientNet, DINOv2) and two cross-modal mismatch checks (lip-sync/semantic, emotion), into one real-or-fake decision. Partial fakes like lip-sync deepfakes barely change the video frames; the giveaway is usually a mismatch between audio and video, which is what the cross-modal streams catch.

**Start here:** [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - architecture, datasets, phases, team roles.

**Current phase:** Phase 0.5 (foundations) - see [docs/phase-0.5-plan.md](docs/phase-0.5-plan.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and [ffmpeg](https://ffmpeg.org/download.html) on PATH (Windows: `winget install Gyan.FFmpeg`).

### CPU

```bash
uv venv
uv pip install -r requirements.txt
```

Activate with `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux) or skip activation and prefix commands with `uv run`.

### GPU (NVIDIA / CUDA)

```bash
uv pip install torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

Requires a recent NVIDIA driver. Verify with:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Google Colab

Torch is preinstalled, just add the extras:

```python
!pip install opencv-python librosa ffmpeg-python timm facenet-pytorch
```

## Repo layout

```
data/            datasets, splits, manifests (raw media is git-ignored)
preprocessing/   face + audio extraction
models/baseline/ early practice models
evaluation/      metrics, ablation, robustness tests
notebooks/       exploration and training
docs/            glossary, math writeups, phase plans
```

Stream, fusion, and feature-store folders get added as those phases start, see PROJECT_OVERVIEW.md.
