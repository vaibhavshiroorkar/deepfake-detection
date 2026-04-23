# Veritas

A small tool that looks at text, images, audio, or video and tries to tell you whether they're real or generated. 
It doesn't give you a yes or no, it measures a few things and tells you what it noticed.

## Running it

You'll need Python 3.10+ and Node 18+.

Open two terminals.

**Terminal 1 — backend:**

```
cd backend
python -m venv .venv
.venv\Scripts\activate        # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

**Terminal 2 — frontend:**

```
cd frontend
npm install
npm run dev
```

Then open http://localhost:3000.

## What it checks

- **Text** —> sentence rhythm, word variety, common AI phrases, punctuation patterns
- **Images** —> compression artifacts, noise, spectrum, faces, metadata
- **Audio** —> pitch movement, background noise, frequency shape, speech rhythm
- **Video** —> each sampled frame, plus flicker between frames

## A note

Detection is always a step behind generation. If it matters, don't trust
one reading. Find the original source, look for a witness, ask who
benefits if you believe it.

Uploads are processed in memory and thrown away. Nothing is stored.
