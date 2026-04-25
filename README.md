# Veritas

A small tool that looks at text, images, audio, or video and tries to tell you whether they're real or AI-generated. It doesn't give you a yes or no. It runs a few independent checks, shows you what each one saw, and gives you the score with the working attached.

> Detection is always a step behind generation. If a verdict matters, don't trust one reading. Find the original source, look for a witness, ask who benefits if you believe it.

---

## What it is

A web app with a workspace where you drop a file or paste some writing, and you get back:
- a 0 to 100 suspicion score
- a four-level verdict: likely authentic, inconclusive, likely synthetic, highly likely synthetic
- every signal that contributed, with its individual score and a one-line note on what it measured
- for images, ELA and noise heatmaps showing where the suspicious pixels live
- for video, a per-frame timeline plus an analysis of the audio track

It runs as a Next.js frontend on Vercel and a FastAPI backend on Hugging Face Spaces. Auth and history are optional and use Supabase. Everything works without an account.

## What it isn't

- Not a final answer. It's a second opinion.
- Not trained from scratch on labelled deepfake datasets. It stacks pretrained Hugging Face checkpoints and adds calibration logic on top. Training scripts exist for the heads, see [Training](#training).
- Not perfect at cross-generator generalisation. Detectors trained on one generator family (SDXL, GPT-3-era) degrade on newer ones (Flux, GPT-4o). The published research calls this out as the open problem in the field, and we don't pretend to have solved it. The honest answer for borderline cases is "inconclusive", which is why that's a real verdict tier here.

## How a detection actually works

For any submission, several independent checks run in parallel and each emits a `Signal(name, score, detail)`. The final verdict is a weighted aggregate, with calibration to dampen overconfident pretrained classifiers.

### Image
- Two pretrained classifiers vote: `Organika/sdxl-detector` (Swin-v2 on SDXL output) and `umm-maybe/AI-image-detector` (different generator mix).
- MTCNN crops faces and re-runs the same classifiers per face.
- Camera-physics signals: focus uniformity, chromatic aberration, sensor noise, error-level analysis.
- Capture metadata: EXIF presence, camera-make consistency.
- C2PA Content Credentials manifest verification (when present).
- Sigmoid calibration tempers the classifiers' habit of returning 0.95+ on ordinary modern photography.
- Agreement adjustment pulls a secondary classifier toward 0.5 when it disagrees with the primary by more than 0.35.

### Video
- Frames sampled along the timeline are each run through the image pipeline.
- A temporal-flicker check looks for the per-frame wobble face generators leave behind.
- Optional `TemporalTransformer` (DINOv2 frame embeddings + transformer encoder) when weights are configured; otherwise per-frame DINOv2 plus the heuristic.
- The audio track is extracted with ffmpeg and run through the standalone audio detector. It contributes 18% weight to the final score, zero when there's no audio.
- Result includes a frame-by-frame timeline.

### Audio
- `MelodyMachine/Deepfake-audio-detection` (wav2vec2-based binary classifier) is the primary signal.
- Whisper-base encoder features feed an adjacent-frame cosine heuristic.
- Classical signals: pitch variability, silence-floor cleanliness, energy-envelope rhythm.

### Text
- `roberta-base-openai-detector` for binary discriminator.
- GPT-2 perplexity, including per-sentence variance ("burstiness").
- Heuristic signals: sentence-length burstiness, lexical rhythm, function-word frequency, scaffolding-phrase repetition, punctuation patterns.

The frontend reads the signal payload and generates a narrative summary that names the strongest and quietest signals and highlights any disagreement between learned classifiers and forensic checks. See [`frontend/lib/narrate.ts`](frontend/lib/narrate.ts).

---

## The stack

### Frontend
- **Next.js 14** (App Router), **React 18**, **TypeScript**
- **Tailwind CSS**, **framer-motion**, **lucide-react**
- **Supabase** (auth, scan history, API keys) — optional
- Deployed on **Vercel** free tier
- Direct-to-backend uploads when `NEXT_PUBLIC_BACKEND_URL` is set, otherwise routed through a Vercel serverless proxy (4.5 MB body cap)

### Backend
- **FastAPI** + **Uvicorn**, Python 3.11
- **PyTorch** (CPU on the free tier, GPU when configured)
- **Transformers** for Hugging Face checkpoints, **facenet-pytorch** for MTCNN
- **OpenCV**, **scipy**, **soundfile** for forensic signal computation
- **slowapi** for rate limiting
- **ffmpeg** for video audio extraction
- **c2pa-python** for Content Credentials (optional)
- Dockerised, deployed on **Hugging Face Spaces** free CPU tier (16 GB RAM, 2 vCPU)
- Model weights pre-fetched at image build time so cold starts only pay the load cost, not the download cost

### Storage and auth (optional)
- **Supabase Postgres** for `scans` and `api_keys` tables
- **Supabase Auth** with JWT verification on the backend (HS256 or ES256)

---

## Models

Bundled and used by default (downloaded by Hugging Face Transformers, no training required):

| Modality | Model | Role |
|---|---|---|
| Image | [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) | Primary AI-image classifier (Swin-v2) |
| Image | [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) | Ensemble vote, different training mix |
| Image | MTCNN via `facenet-pytorch` | Face detection for per-face classification |
| Audio | [`MelodyMachine/Deepfake-audio-detection`](https://huggingface.co/MelodyMachine/Deepfake-audio-detection) | wav2vec2 binary classifier |
| Audio | `openai/whisper-base` (or `whisper-tiny` on free tier) | Encoder for spectral feature extraction |
| Text | [`roberta-base-openai-detector`](https://huggingface.co/roberta-base-openai-detector) | Binary discriminator |
| Text | `gpt2` | Perplexity scoring and per-sentence burstiness |

Optional, scaffolded but require trained weights to activate:

| Modality | Model | Set env var to use |
|---|---|---|
| Image | DINOv2-large + classification head | `VERITAS_DINOV2_WEIGHTS=/path/to/dinov2_head.pt` |
| Audio | Whisper-base encoder + classification head | `VERITAS_WHISPER_WEIGHTS=/path/to/whisper_head.pt` |
| Video | Temporal Transformer over DINOv2 frame embeddings | `VERITAS_TEMPORAL_WEIGHTS=/path/to/temporal_transformer.pt` |

These three are where most of the future accuracy improvement lives. They run with frozen backbones, so training the small heads is tractable on free compute. See [Training](#training).

---

## Training

The backbone models (DINOv2, Whisper) stay frozen. Only the small classification heads are trained, which is fast and fits inside Google Colab's free T4 quota. See [`backend/training/README.md`](backend/training/README.md) for the full training reference.

### What runs on free compute

| Head | Free GPU sufficient? | Approx time on Colab T4 free | Min VRAM |
|---|---|---|---|
| DINOv2 image head | Yes | 30 to 90 min for 15 epochs | 4 GB |
| Whisper audio head | Yes | 20 to 60 min for 20 epochs | 2 GB |
| Temporal Transformer | Yes (with embedding cache) | 30 to 60 min after extraction | 4 GB extract, 2 GB train |

All scripts accept `--cpu` for CPU-only training as a fallback, much slower but possible.

### Free datasets to train on

- **FaceForensics++**: classic deepfake video benchmark. Request access at [GitHub](https://github.com/ondyari/FaceForensics).
- **Celeb-DF v2**: high-quality deepfake faces. Free with a Google Form request.
- **DeepFake Detection Challenge (DFDC)** sample set: free download via Kaggle.
- **WildDeepFake**: in-the-wild deepfakes, free for research.
- **ASVspoof 2019 LA + 2021**: synthetic speech, free for research.
- **WaveFake**: TTS / voice-clone audio, free.

### Quick start (image head)

```bash
cd backend
python -m training.train_image \
    --data-real /path/to/real_images/ \
    --data-fake /path/to/fake_images/ \
    --output ./weights/dinov2_head.pt \
    --epochs 15 \
    --batch-size 32
```

Then point the backend at the weights:
```
VERITAS_DINOV2_WEIGHTS=/absolute/path/to/weights/dinov2_head.pt
```

### Running on Colab free

Open a fresh Colab notebook with the T4 runtime and:

```python
!git clone https://github.com/vaibhavshiroorkar/deepfake-detection.git
%cd deepfake-detection/backend
!pip install -q -r requirements.txt scikit-learn
# Mount Drive or wget your dataset somewhere under /content/data
!python -m training.train_image \
    --data-real /content/data/real \
    --data-fake /content/data/fake \
    --output /content/dinov2_head.pt \
    --epochs 15
```

Download the resulting `.pt` file and either upload it to your HF Space (set `VERITAS_DINOV2_WEIGHTS` to its mounted path) or push it to the Hub and have the loader fetch it on startup.

---

## Running locally

You'll need Python 3.10+ and Node 18+.

Open two terminals.

**Terminal 1: backend**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate         # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

**Terminal 2: frontend**
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

For the frontend to call the local backend directly (and skip the Vercel-style proxy), add to `frontend/.env.local`:
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

---

## Deployment

### Backend on Hugging Face Spaces
Free CPU tier, 16 GB RAM, 2 vCPU. Push to the GitHub repo and the Space rebuilds automatically. Model weights are baked into the Docker image at build time, so cold starts are around 15 seconds instead of 90.

Required Space secrets are documented in [`backend/README.md`](backend/README.md). At minimum set `ALLOWED_ORIGINS` to your Vercel URL once you go to production. For optional auth and history, add the Supabase secrets.

### Frontend on Vercel
Connect the GitHub repo, set the project root to `frontend/`. Required env vars:
- `NEXT_PUBLIC_BACKEND_URL` — your HF Space URL, e.g. `https://your-username-deepfake-detection-api.hf.space`. Without this, uploads route through a Vercel serverless function and hit a 4.5 MB body cap.
- `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` for auth.

---

## Project structure

```
deepfake-detection/
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint, rate limiting, CORS
│   │   ├── auth.py             # Supabase JWT verification (HS256 + ES256)
│   │   ├── db.py               # Async Postgres client for scans/keys
│   │   ├── models.py           # Lazy singletons for every ML model
│   │   └── detectors/
│   │       ├── signal.py       # Shared Signal dataclass
│   │       ├── image.py        # Image pipeline: classifiers + forensic signals
│   │       ├── video.py        # Video pipeline + audio extraction
│   │       ├── audio.py        # Audio pipeline: wav2vec2 + Whisper + classical
│   │       ├── text.py         # Text pipeline: RoBERTa + GPT-2 + heuristics
│   │       ├── heatmap.py      # ELA + noise visualisation
│   │       ├── c2pa.py         # Content Credentials reading
│   │       ├── dinov2_head.py        # Optional trained image head
│   │       ├── whisper_classifier.py # Optional trained audio head
│   │       ├── temporal_transformer.py # Optional trained video head
│   │       └── gpt2_perplexity.py    # GPT-2 perplexity scoring
│   ├── scripts/
│   │   ├── prefetch_models.py  # Bake models into the Docker image
│   │   └── calibrate.py        # Build calibration histograms from labelled data
│   ├── training/
│   │   ├── train_image.py      # DINOv2 head training
│   │   ├── train_audio.py      # Whisper head training
│   │   └── train_video.py      # Temporal Transformer training
│   ├── Dockerfile              # HF Spaces deployable
│   ├── README.md               # HF Spaces config + secrets
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Landing
│   │   ├── detect/             # Workspace (image/video/audio/text)
│   │   ├── compare/            # Side-by-side image comparison
│   │   ├── method/             # How it works
│   │   ├── calibration/        # Histograms + accuracy table + research comparison
│   │   ├── history/            # Past scans (signed-in)
│   │   ├── keys/               # API key management
│   │   ├── login/, auth/       # Supabase auth flows
│   │   └── api/detect/         # Vercel proxy (fallback when env var unset)
│   ├── components/
│   │   ├── DetectorConsole.tsx # Tabbed workspace + preview pane
│   │   ├── Uploader.tsx, TextPane.tsx, PreviewPanel.tsx
│   │   ├── ResultPanel.tsx     # Verdict + signals + heatmaps + timeline
│   │   ├── HeatmapOverlay.tsx
│   │   ├── ErrorBoundary.tsx, Toast.tsx
│   │   └── (landing sections)  # Hero, Capabilities, HowItWorks, Principles, etc
│   └── lib/
│       ├── api.ts              # Detection client + auth headers
│       ├── narrate.ts          # AI-style narrative summary generator
│       └── supabase/
└── .github/workflows/ci.yml    # Typecheck + Python syntax on every push
```

---

## Where the project sits today

Honest accuracy estimates, not benchmark numbers. There's a full breakdown including a comparison with a recent systematic review on the [Calibration page](https://your-deployed-url/calibration).

| Modality | Today | Realistic ceiling with free training |
|---|---|---|
| Image | 75-80% on SDXL-era generators, 55-65% on newer | 88-92% with DINOv2 head trained on FF++/DFDC/Celeb-DF/WildDeepFake |
| Video | 60-70% on FF++/DFDC | 85-90% with trained temporal model + audio-visual fusion |
| Audio | ~90% on ASVspoof-style TTS, weaker on modern voice clones | 93-95% with Whisper head fine-tuned on WaveFake + 2024 clones |
| Text | ~65% on modern LLM output | 80-85% with discriminator fine-tuned on GPT-4 / Claude / Gemini / Llama 3 |

Cross-dataset generalisation in deepfake detection is unsolved. The best published models hit 92-96% on the dataset they were trained on, and 65-75% on everything else. A realistic ceiling for a careful ensemble like this is around 85% with honest hedging on the other 15%.

---

## Reference

Ramanaharan, R., Guruge, D. B., & Agbinya, J. I. (2025). DeepFake video detection: Insights into model generalisation. A systematic review. *Data and Information Management*, 9(2), 100099.

The Calibration page in the app contains a point-by-point comparison between Veritas and the findings of this review.

---

## License

MIT.
