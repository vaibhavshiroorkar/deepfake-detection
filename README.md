# Veritas — deepfake detection

A careful second opinion on synthetic media. Upload an image, a clip, or a
paragraph of prose; Veritas examines each through several independent
forensic channels and returns a reading with its working shown.

- **Images** — error-level analysis, noise stationarity, spectral falloff,
  face-boundary continuity, EXIF metadata.
- **Video** — per-frame forensic sampling plus a temporal-flicker check.
- **Text** — burstiness, lexical rhythm, phrase-tic detection, punctuation
  entropy.

No single signal is decisive. The verdict is the agreement across channels.

---

## Repository layout

```
deepfake-detection/
├── backend/      # FastAPI — forensic analysis
│   ├── app/
│   │   ├── main.py
│   │   └── detectors/{image,video,text}.py
│   └── requirements.txt
└── frontend/     # Next.js 14 — editorial interface
    ├── app/
    ├── components/
    └── lib/
```

---

## Running locally

Open two terminals.

### 1. Backend (Python 3.10+)

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

The service listens on `http://localhost:8000`. A health probe sits at
`/health`; the detection endpoints are `/api/detect/{image,video,text}`.

### 2. Frontend (Node 18+)

```bash
cd frontend
npm install
npm run dev
```

The site runs at [http://localhost:3000](http://localhost:3000) and proxies
detection calls to the backend through `/api/detect`.

If your backend lives elsewhere, set `BACKEND_URL` in the frontend
environment:

```bash
BACKEND_URL=http://my-backend:8000 npm run dev
```

---

## A note on expectations

This project demonstrates a forensic pipeline and a careful interface. It is
not a magic oracle. The signals implemented here are real — they are the
same ones used in the research literature — but generators improve quickly,
and every detector lags its quarry. Treat the reading as what it says it is:
an informed second opinion, not a verdict.

For anything with real stakes, seek provenance: the first-party publisher,
a higher-resolution original, a witness. Tools like this are a supplement,
never a substitute.

---

## License

Educational use. No warranty, no retention of uploads, no user tracking.
