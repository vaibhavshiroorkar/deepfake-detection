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

# Veritas Deepfake Detection API

FastAPI backend for image, video, audio, and text deepfake detection.
Served on Hugging Face Spaces from the `backend/` directory of the
`deepfake-detection` repo.

## Endpoints

- `GET  /health` service liveness
- `POST /api/detect/image` multipart `file`
- `POST /api/detect/video` multipart `file`
- `POST /api/detect/audio` multipart `file`
- `POST /api/detect/text`  JSON `{ "text": "..." }`

All detection endpoints are rate-limited. Anonymous callers get 10 per
minute per IP. Authenticated callers get 30 per minute per user id.
Override with `VERITAS_ANON_RATE` and `VERITAS_AUTH_RATE`.

## Required Space secrets

Set these in Space, Settings, Variables and secrets.

**Security:**
- `ALLOWED_ORIGINS` comma-separated Vercel URL(s), e.g.
  `https://veritas.vercel.app,https://your-preview.vercel.app`.
  Default is `*` (wide open) for backward compatibility, with a loud
  startup warning.
- `VERITAS_STRICT_CORS` default `0`. Set to `1` in prod to refuse the
  wildcard at startup and force an explicit origin list.

**Auth and persistence (all optional, scans work anonymously):**
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWT_SECRET`
- `SUPABASE_JWT_PUBLIC_KEY` (only for ES256)

**Rate limit overrides (optional):**
- `VERITAS_ANON_RATE` default `10/minute`
- `VERITAS_AUTH_RATE` default `30/minute`

## Model configuration

On the free CPU tier the container defaults to lighter models:

- `VERITAS_FORCE_CPU=1`
- `VERITAS_WHISPER_MODEL=openai/whisper-tiny`

The primary image classifier is the pretrained Swin-v2
(`Organika/sdxl-detector`) unless `VERITAS_DINOV2_WEIGHTS` points at a
trained DINOv2 head. Without that file, DINOv2 is skipped entirely so
the model load stays well under the 16 GB Space limit.

Model weights are pre-fetched at image-build time via
`scripts/prefetch_models.py`, so cold-start only pays the load cost,
not the download cost.
