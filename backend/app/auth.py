"""Identity for two callers: signed-in users (Supabase JWT) and machines
(per-user API keys). Both resolve to the same shape: an `Identity` carrying
a user_id and, when applicable, the api_key row that authorized the call.

Anonymous calls are permitted; they simply produce `Identity(None, None)`
and the scan is not persisted.

Supports both HS256 (SUPABASE_JWT_SECRET — the classic shared secret) and
ES256 (SUPABASE_JWT_PUBLIC_KEY — the JWK JSON for newer EC-signed projects).
HS256 is tried first; ES256 is the fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import httpx
import jwt
from jwt.algorithms import ECAlgorithm
from fastapi import Header, HTTPException

from .db import supabase_request

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

_EC_PUBLIC_KEY = None
_JWK_JSON = os.getenv("SUPABASE_JWT_PUBLIC_KEY", "")
if _JWK_JSON:
    try:
        _EC_PUBLIC_KEY = ECAlgorithm.from_jwk(json.loads(_JWK_JSON))
    except Exception:
        pass


@dataclass
class Identity:
    user_id: str | None
    api_key_id: str | None = None


def _verify_jwt(token: str) -> str | None:
    """Try HS256 first (most Supabase projects), then ES256."""
    if _JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                _JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
            )
            sub = payload.get("sub")
            if isinstance(sub, str):
                return sub
        except jwt.PyJWTError:
            pass

    if _EC_PUBLIC_KEY is not None:
        try:
            payload = jwt.decode(
                token,
                _EC_PUBLIC_KEY,
                algorithms=["ES256"],
                audience="authenticated",
            )
            sub = payload.get("sub")
            if isinstance(sub, str):
                return sub
        except jwt.PyJWTError:
            pass

    return None


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _verify_api_key(raw: str) -> Identity | None:
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
    return Identity(user_id=None)
