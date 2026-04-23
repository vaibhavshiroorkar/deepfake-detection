# Phase 2 — Training Scripts

These scripts train the classification heads for Veritas Phase 2.
**Backbone models (DINOv2, Whisper) are frozen** — only the small heads are trained,
so training is fast and doesn't require massive GPU memory.

## Prerequisites

```bash
pip install scikit-learn   # for roc_auc_score during validation
# (all other deps are already in requirements.txt)
```

## Training the image classifier (DINOv2 head)

```bash
cd backend
python -m training.train_image \
    --data-real /path/to/real_images/ \
    --data-fake /path/to/fake_images/ \
    --output ./weights/dinov2_head.pt \
    --epochs 15 \
    --batch-size 32
```

**Datasets:** FaceForensics++, Celeb-DF, or any directory pair of real vs fake images.

After training, set the environment variable:
```
VERITAS_DINOV2_WEIGHTS=/absolute/path/to/weights/dinov2_head.pt
```

---

## Training the audio classifier (Whisper head)

```bash
cd backend
python -m training.train_audio \
    --data-real /path/to/real_audio/ \
    --data-fake /path/to/fake_audio/ \
    --output ./weights/whisper_head.pt \
    --epochs 20 \
    --batch-size 16
```

**Datasets:** ASVspoof 2019 LA, In-the-Wild TTS, or any directory pair.

After training:
```
VERITAS_WHISPER_WEIGHTS=/absolute/path/to/weights/whisper_head.pt
```

---

## Training the video temporal transformer

```bash
cd backend
python -m training.train_video \
    --data-real /path/to/real_videos/ \
    --data-fake /path/to/fake_videos/ \
    --output ./weights/temporal_transformer.pt \
    --cache-dir ./cache/video_embeddings \
    --n-frames 16 \
    --epochs 20
```

**Note:** This pre-extracts DINOv2 embeddings per frame and caches them,
so subsequent training runs are fast (the backbone only runs once per video).

**Datasets:** FaceForensics++ (video split), Celeb-DF (video split).

After training:
```
VERITAS_TEMPORAL_WEIGHTS=/absolute/path/to/weights/temporal_transformer.pt
```

---

## Text — GPT-2 Perplexity

**No training required.** The text detector uses pretrained GPT-2 directly
for perplexity scoring. It works out of the box.

---

## GPU Requirements

| Script | Min VRAM | Recommended |
|--------|----------|-------------|
| `train_image.py` | 4 GB | 8+ GB |
| `train_audio.py` | 2 GB | 4+ GB |
| `train_video.py` | 4 GB (extraction), 2 GB (training) | 8+ GB |

All scripts support `--cpu` for CPU-only training (slower but works).

## Without trained weights

The system **works without any trained weights**. Each modality gracefully
falls back:

- **Image:** Falls back to Swin-v2 (Organika/sdxl-detector)
- **Video:** Falls back to per-frame Swin-v2 analysis
- **Audio:** Falls back to Phase 1 Whisper heuristics
- **Text:** GPT-2 perplexity works immediately (no training needed)
