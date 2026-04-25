# Training Veritas heads on free Colab

This guide walks you through training the optional classification heads on Google Colab's free T4 runtime. All three heads (image, audio, video) fit inside the free quota.

## Why this works on free compute

The expensive part of any deepfake detector is the backbone: the big self-supervised vision/audio model that turns raw pixels or waveforms into useful features. Veritas uses the bundled HF checkpoints with their backbones **frozen** and only trains a small MLP head on top. That changes the math from "fine-tune a 1B parameter model" to "train a 2 MB MLP", which Colab free can handle in under an hour.

## What you need

- A Google account for Colab.
- 2 GB to 20 GB of labelled training data (real and fake examples), depending on modality and how thorough you want to be.
- A Hugging Face account for downloading the bundled models. No paid plan needed.

## 1. Image head (DINOv2 + MLP)

Open a fresh Colab notebook, set the runtime type to **T4 GPU** (Runtime, Change runtime type, T4).

```python
# Colab cell 1: clone and install
!git clone https://github.com/vaibhavshiroorkar/deepfake-detection.git
%cd deepfake-detection/backend
!pip install -q -r requirements.txt scikit-learn
```

```python
# Colab cell 2: get data into /content/data/{real,fake}
# Option A: download a small public dataset directly
# Option B: mount your Drive and copy from there
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/deepfake_train /content/data
```

For datasets to use, see the README's free-dataset list. FF++ is the canonical starting point.

```python
# Colab cell 3: train
!python -m training.train_image \
    --data-real /content/data/real \
    --data-fake /content/data/fake \
    --output /content/dinov2_head.pt \
    --epochs 15 \
    --batch-size 32
```

```python
# Colab cell 4: download the trained head
from google.colab import files
files.download('/content/dinov2_head.pt')
```

To use it in the deployed backend, push `dinov2_head.pt` to your Hugging Face Space (Files tab) and set the env var in Space settings:
```
VERITAS_DINOV2_WEIGHTS=/home/user/app/dinov2_head.pt
```

## 2. Audio head (Whisper + MLP)

```python
!python -m training.train_audio \
    --data-real /content/audio_data/real \
    --data-fake /content/audio_data/fake \
    --output /content/whisper_head.pt \
    --epochs 20 \
    --batch-size 16
```

Recommended datasets: ASVspoof 2019 LA, WaveFake. Both are free for research.

Set on the Space:
```
VERITAS_WHISPER_WEIGHTS=/home/user/app/whisper_head.pt
```

## 3. Video temporal model

This one extracts DINOv2 frame embeddings first (slow, but cached), then trains a small Transformer over them (fast).

```python
!python -m training.train_video \
    --data-real /content/video_data/real \
    --data-fake /content/video_data/fake \
    --output /content/temporal_transformer.pt \
    --cache-dir /content/embedding_cache \
    --n-frames 16 \
    --epochs 20
```

The cache directory means a second training run reuses the embeddings instead of recomputing them. Embed once, train many times.

Set on the Space:
```
VERITAS_TEMPORAL_WEIGHTS=/home/user/app/temporal_transformer.pt
```

## Tips for fitting inside the free quota

- **T4 has 15 GB VRAM**. Image head training peaks around 4 GB, so you have room. If you hit OOM, drop `--batch-size`.
- **Colab free disconnects after 12 hours** of continuous use. All three heads finish inside that. Save your `.pt` to Drive periodically with `--save-every` if your dataset is large.
- **Mount Drive for data**, don't `wget` the same dataset every session. Free Drive gives you 15 GB which is enough for a starter set.
- **Start small**. A few thousand real and fake samples per class is enough to validate the pipeline. Scale up after you know it works end-to-end.

## What good results look like

Validation AUROC of 0.85+ is a meaningful improvement over the bundled pretrained classifier. Below 0.75 means your training data is too narrow or noisy.

For honest evaluation, hold out a **different generator family** in the validation set (e.g. train on SDXL, test on Flux). The published research finds in-distribution accuracy of 92-96% but cross-distribution closer to 65-75%. Don't be discouraged by the gap, that's the open problem in the field.

## After training

The backend gracefully detects whether weights are present at the configured path. With weights, the trained head replaces the corresponding pretrained signal. Without, the original ensemble stack runs unchanged. So you can train one head at a time and ship incremental improvements.
