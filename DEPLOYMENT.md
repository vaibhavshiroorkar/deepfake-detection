# Deploying Veritas

Three services, in this order:

1. **Supabase** — auth + Postgres
2. **Hugging Face Spaces** — Python detection backend (Docker SDK, free CPU)
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

## 2. Hugging Face Spaces (backend)

1. On [huggingface.co](https://huggingface.co), create a **New Space**:
   - SDK: **Docker**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public** (private Spaces sleep on free tier)
2. Push the `backend/` subtree to the Space's git remote:
   ```bash
   git subtree split --prefix backend -b hf-deploy
   git push --force https://huggingface.co/spaces/<user>/<space-name> hf-deploy:main
   ```
   You'll need a Write-scoped access token from huggingface.co/settings/tokens.
3. In the Space → **Settings → Variables and secrets**, add:
   - `ALLOWED_ORIGINS` = your Vercel URL (no trailing slash, comma-separated if multiple)
   - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_JWT_SECRET` from step 1
4. Wait for first build (10–15 min — torch + transformers are heavy). Watch the **Logs** tab.
5. Verify: `curl https://<user>-<space-name>.hf.space/health` should return `{"status":"ok","db":true}`.

> Free CPU Spaces have no persistent disk, so model weights re-download on every container restart. The first request after a restart takes 30–90s while HF caches repopulate.

### Re-deploying after code changes

```bash
git branch -D hf-deploy
git subtree split --prefix backend -b hf-deploy
git push --force https://huggingface.co/spaces/<user>/<space-name> hf-deploy:main
```

## 3. Vercel (frontend)

1. [Vercel](https://vercel.com) → **Add New → Project** → import the repo.
2. **Root Directory** = `frontend`.
3. **Environment Variables**:
   - `BACKEND_URL` = `https://<user>-<space-name>.hf.space`
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
- Replace the synthetic histograms in `app/calibration/page.tsx` with measurements from a real labeled test set.
- When you train a DINOv2 head, upload the weights file to the Space (via web UI or `git lfs`) and set `VERITAS_DINOV2_WEIGHTS=/home/user/app/weights/dinov2_head.pt` in Space secrets.
