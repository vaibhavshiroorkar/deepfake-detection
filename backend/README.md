---
title: Deepfake Detection API
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Veritas — Deepfake Detection API

FastAPI backend for image, video, audio, and text deepfake detection.
Served on Hugging Face Spaces from the `backend/` directory of the
`deepfake-detection` repo.

## Endpoints

- `GET  /health` — service liveness
- `POST /api/detect/image` — multipart `file`
- `POST /api/detect/video` — multipart `file`
- `POST /api/detect/audio` — multipart `file`
- `POST /api/detect/text`  — JSON `{ "text": "..." }`

## Required Space secrets

Set these in Space → Settings → Variables and secrets:

- `ALLOWED_ORIGINS` — comma-separated list of your Vercel URL(s)
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWT_SECRET`
- `SUPABASE_JWT_PUBLIC_KEY` (optional, only for ES256)

## Model configuration

On the free CPU tier the container defaults to lighter models:

- `VERITAS_FORCE_CPU=1`
- `VERITAS_WHISPER_MODEL=openai/whisper-tiny`

The primary image classifier is the pretrained Swin-v2
(`Organika/sdxl-detector`) unless `VERITAS_DINOV2_WEIGHTS` points at a
trained DINOv2 head. Without that file, DINOv2 is skipped entirely so
the model load stays well under the 16 GB Space limit.
