# Deploying Veritas

Three services, in this order:

1. **Supabase** — auth + Postgres
2. **Render** — Python detection backend
3. **Vercel** — Next.js frontend

## 1. Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. **SQL editor** → paste the contents of `supabase/migrations/0001_init.sql` → **Run**.
3. **Project Settings → API** — collect:
   - `Project URL` → used as `SUPABASE_URL` (backend) and `NEXT_PUBLIC_SUPABASE_URL` (frontend)
   - `anon public` key → `NEXT_PUBLIC_SUPABASE_ANON_KEY` (frontend)
   - `service_role` key → `SUPABASE_SERVICE_ROLE_KEY` (backend, **never** ship to browser)
   - `JWT secret` → `SUPABASE_JWT_SECRET` (backend)
4. **Authentication → URL configuration** — add:
   - Site URL: your Vercel domain, e.g. `https://veritas.vercel.app`
   - Additional redirect URLs: `https://veritas.vercel.app/auth/callback`
5. **Authentication → Providers** — enable Email (default). Optional: Google/GitHub OAuth.

## 2. Render (backend)

1. Push this repo to GitHub.
2. [Render](https://render.com) → **New → Blueprint** → point at your repo.
   `backend/render.yaml` is auto-detected.
3. After the service is created, set environment variables (Render dashboard → service → Environment):
   - `ALLOWED_ORIGINS` = your Vercel URL (no trailing slash, comma-separated if multiple)
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET` from step 1
4. Wait for first deploy (5–10 min — opencv + scipy take a while). The build will fail if `c2pa-python` can't compile on Render's Python image — that's fine, just remove it from `requirements.txt` and redeploy. C2PA is optional.
5. Verify: `curl https://your-backend.onrender.com/health` should return `{"status":"ok","db":true}`.

> Free Render dynos sleep after 15 min of inactivity. First scan after sleep takes ~30s.

## 3. Vercel (frontend)

1. [Vercel](https://vercel.com) → **Add New → Project** → import the repo.
2. **Root Directory** = `frontend`.
3. **Environment Variables**:
   - `BACKEND_URL` = `https://your-backend.onrender.com`
   - `NEXT_PUBLIC_SUPABASE_URL` = from step 1
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` = from step 1
4. Deploy. Once live, go back to Supabase **Authentication → URL configuration** and confirm the Vercel URL is in the allowed list.

## Local dev

```bash
# Terminal 1 — backend
cd backend
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env   # fill in Supabase values
bash run.sh

# Terminal 2 — frontend
cd frontend
npm install
cp .env.example .env.local   # fill in Supabase values
npm run dev
```

Open http://localhost:3000.

The site works without Supabase configured (anonymous scans only — history/keys pages will redirect to login).

## What writes where

- **Frontend, signed-in user** → reads `scans` and `api_keys` directly via Supabase REST (RLS-protected, can only see their own).
- **Backend** → uses the service role key to insert scan rows tagged with the user_id pulled from the JWT or API key. Bypasses RLS for writes; never reads user data on behalf of someone else.
- **Anonymous scans** → not persisted.

## After it's live

- Lock down CORS: change `ALLOWED_ORIGINS` from `*` to just your Vercel URL.
- (Optional) Set `c2pa-python` to a known-good version once you confirm it compiles on Render.
- Replace the synthetic histograms in `app/calibration/page.tsx` with measurements from a real labeled test set.
