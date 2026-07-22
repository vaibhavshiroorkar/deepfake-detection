# Audio-Visual Deepfake Detection

A deepfake detector that fuses five signals into one real-or-fake decision: three visual streams (Xception, EfficientNet, DINOv2) and two cross-modal streams (lip-sync, emotion) that use cross-attention over embeddings to catch mismatches between audio and video. Partial fakes like lip-sync deepfakes barely change the video frames; the giveaway is usually a mismatch between audio and video, which is what the cross-modal streams catch. Fusion is feature-level: stream embeddings are concatenated and passed through an MLP + sigmoid.

**Start here:** [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) - the project's single main document: architecture, data, tooling, compute assumptions, build order, current status.

**Current stage:** Stage 3 (remaining visual streams). Stages 1-2 are done - see [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) §14.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`, so every machine resolves to identical versions.

Pick **exactly one** of the two extras — they install the same torch version from different indexes and cannot be combined.

### CPU

```bash
uv sync --extra cpu
```

### GPU (NVIDIA / CUDA)

```bash
uv sync --extra cu130
```

Requires a recent NVIDIA driver. Verify with:

```bash
uv run --extra cu130 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Running things

`uv sync` creates `.venv` for you. The simplest safe workflow is to activate it once per shell and then use plain `python`:

```bash
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
python training/train_visual_stream.py
```

> **Careful:** if you use `uv run` instead, you must pass the extra **every single time**:
> `uv run --extra cu130 python ...`. A bare `uv run python ...` re-syncs the
> environment to the no-extra state, which silently swaps your CUDA torch for the
> CPU build — `torch.cuda.is_available()` starts returning `False` and training
> quietly drops to CPU. Activating the venv avoids this entirely, because plain
> `python` never re-syncs.

To add or change a dependency, edit `pyproject.toml`, then run `uv lock` and commit the updated `uv.lock`.

Note: PyAV (`av`) bundles its own ffmpeg libraries, so no system ffmpeg is needed for clip extraction. A system [ffmpeg](https://ffmpeg.org/download.html) on PATH (Windows: `winget install Gyan.FFmpeg`) is only required for the `ffmpeg-python` paths in `preprocessing/crop_faces.py`.

### Google Colab

Torch is preinstalled, just add the extras:

```python
!pip install opencv-python av librosa ffmpeg-python timm facenet-pytorch
```

## Repo layout

```
data/            datasets, splits, manifests (raw media is git-ignored)
preprocessing/   face + audio extraction
models/baseline/ early practice models
evaluation/      metrics, ablation, robustness tests
notebooks/       exploration and training
docs/            glossary, math writeups, stage plans
```

Stream, fusion, and feature-store folders get added as those stages start, see [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md).
