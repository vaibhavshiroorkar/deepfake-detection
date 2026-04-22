-- Veritas schema: scans + per-user API keys.
-- Run in the Supabase SQL editor (project dashboard → SQL).

create extension if not exists "pgcrypto";

create table if not exists public.scans (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users (id) on delete cascade,
  api_key_id  uuid,
  kind        text not null check (kind in ('image', 'video', 'audio', 'text')),
  filename    text,
  suspicion   real not null,
  verdict     text not null,
  confidence  real not null,
  signals     jsonb not null default '[]'::jsonb,
  result      jsonb not null,
  created_at  timestamptz not null default now()
);

create index if not exists scans_user_created_idx
  on public.scans (user_id, created_at desc);

create table if not exists public.api_keys (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  name        text not null,
  prefix      text not null,
  key_hash    text not null,
  last_used   timestamptz,
  created_at  timestamptz not null default now(),
  revoked_at  timestamptz
);

create unique index if not exists api_keys_hash_idx on public.api_keys (key_hash);
create index if not exists api_keys_user_idx on public.api_keys (user_id, created_at desc);

alter table public.scans      enable row level security;
alter table public.api_keys   enable row level security;

-- Owner can read their own scans. Inserts are done by the backend with the
-- service role and bypass RLS, so no insert policy is needed for users.
drop policy if exists "scans owner read" on public.scans;
create policy "scans owner read"
  on public.scans for select
  using (auth.uid() = user_id);

drop policy if exists "scans owner delete" on public.scans;
create policy "scans owner delete"
  on public.scans for delete
  using (auth.uid() = user_id);

-- API keys: owner manages their own.
drop policy if exists "api_keys owner read" on public.api_keys;
create policy "api_keys owner read"
  on public.api_keys for select
  using (auth.uid() = user_id);

drop policy if exists "api_keys owner insert" on public.api_keys;
create policy "api_keys owner insert"
  on public.api_keys for insert
  with check (auth.uid() = user_id);

drop policy if exists "api_keys owner update" on public.api_keys;
create policy "api_keys owner update"
  on public.api_keys for update
  using (auth.uid() = user_id);

drop policy if exists "api_keys owner delete" on public.api_keys;
create policy "api_keys owner delete"
  on public.api_keys for delete
  using (auth.uid() = user_id);
