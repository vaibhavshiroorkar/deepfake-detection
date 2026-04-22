"""Thin async wrapper over the Supabase PostgREST endpoint, using the
service-role key so the backend can write rows on behalf of users without
fighting RLS. Reads done from the frontend use the anon key + RLS instead.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE)


async def supabase_request(
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    json: Any | None = None,
    prefer: str | None = None,
) -> Any:
    if not is_configured():
        raise RuntimeError("Supabase is not configured on the backend.")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    url = f"{SUPABASE_URL}{path}"
    client = _get_client()
    resp = await client.request(method, url, params=params, json=json, headers=headers)
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"{resp.status_code}: {resp.text}",
            request=resp.request,
            response=resp,
        )
    if resp.status_code == 204 or not resp.content:
        return None
    return resp.json()


async def insert_scan(row: dict[str, Any]) -> str | None:
    """Insert a scan row, return its id (or None on failure)."""
    try:
        data = await supabase_request(
            "POST",
            "/rest/v1/scans",
            json=row,
            prefer="return=representation",
        )
        if isinstance(data, list) and data:
            return data[0].get("id")
    except Exception:
        # Persistence is best-effort; never fail the scan because of DB issues.
        return None
    return None
