"""Identity for two callers: signed-in users (Supabase JWT) and machines
(per-user API keys). Both resolve to the same shape: an `Identity` carrying
a user_id and, when applicable, the api_key row that authorized the call.

Anonymous calls are permitted; they simply produce `Identity(None, None)`
and the scan is not persisted.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import httpx
import jwt
from fastapi import Header, HTTPException

from .db import supabase_request

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")


@dataclass
class Identity:
    user_id: str | None
    api_key_id: str | None = None


def _verify_jwt(token: str) -> str | None:
    """Returns the Supabase user id (`sub`) if the token is valid."""
    if not SUPABASE_JWT_SECRET:
        return None
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
        sub = payload.get("sub")
        return sub if isinstance(sub, str) else None
    except jwt.PyJWTError:
        return None


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _verify_api_key(raw: str) -> Identity | None:
    """API keys look like 'vrt_<prefix>_<secret>'. We hash the full string
    and look it up in the api_keys table."""
    if not SUPABASE_URL:
        return None
    key_hash = hash_api_key(raw)
    rows = await supabase_request(
        "GET",
        "/rest/v1/api_keys",
        params={
            "select": "id,user_id,revoked_at",
            "key_hash": f"eq.{key_hash}",
            "limit": "1",
        },
    )
    if not rows:
        return None
    row = rows[0]
    if row.get("revoked_at"):
        return None
    # Touch last_used (best-effort).
    try:
        await supabase_request(
            "PATCH",
            "/rest/v1/api_keys",
            params={"id": f"eq.{row['id']}"},
            json={"last_used": "now()"},
        )
    except httpx.HTTPError:
        pass
    return Identity(user_id=row["user_id"], api_key_id=row["id"])


async def resolve_identity(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Identity:
    if x_api_key:
        ident = await _verify_api_key(x_api_key.strip())
        if ident is None:
            raise HTTPException(401, "Invalid or revoked API key.")
        return ident
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        sub = _verify_jwt(token)
        if sub:
            return Identity(user_id=sub)
        # Token present but invalid — treat as anonymous rather than 401,
        # so anonymous browse-and-scan still works if a stale cookie hangs around.
    return Identity(user_id=None)
