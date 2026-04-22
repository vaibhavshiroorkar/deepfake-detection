"""C2PA (Content Credentials) manifest reader. Provenance is the
opposite signal from forensic detection: a valid signed manifest tells
you who claims to have made the file and how it was edited.

The `c2pa` Python package is optional — if it isn't installed, this
returns None and the rest of the pipeline keeps working.
"""
from __future__ import annotations

from typing import Any

try:
    import c2pa as _c2pa  # type: ignore
    _AVAILABLE = True
except Exception:
    _c2pa = None
    _AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


def read_manifest(data: bytes, fmt: str = "image/jpeg") -> dict[str, Any] | None:
    if not _AVAILABLE:
        return None
    try:
        reader = _c2pa.Reader.from_stream(fmt, data)
        manifest_json = reader.json()
    except Exception:
        return None

    import json
    try:
        parsed = json.loads(manifest_json)
    except Exception:
        return None

    active_id = parsed.get("active_manifest")
    manifests = parsed.get("manifests", {}) or {}
    active = manifests.get(active_id) if active_id else None
    if not active:
        return {"present": False}

    claim_generator = active.get("claim_generator", "")
    signature = active.get("signature_info", {}) or {}
    issuer = signature.get("issuer", "")
    actions = []
    for assertion in active.get("assertions", []) or []:
        if assertion.get("label", "").startswith("c2pa.actions"):
            for action in (assertion.get("data", {}) or {}).get("actions", []) or []:
                a = action.get("action", "")
                if a:
                    actions.append(a)

    return {
        "present": True,
        "claim_generator": claim_generator,
        "signed_by": issuer,
        "actions": actions[:12],
        "trusted": bool(issuer),
    }
