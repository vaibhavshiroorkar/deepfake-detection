"""Real C2PA Content Credentials verification through the ``c2pa-python`` Reader API (WS3).

``verify(path)`` never raises and never touches the network (remote manifest fetch and OCSP are switched off in the Settings
passed to the Reader). It reports what the library validated; it does not decide whether content is authentic.

Trust: the library validates the manifest cryptographically (claim signature, assertion hashes, data hash) and separately
checks the signing certificate against a trust list. No trust list is bundled here, so by default a correctly signed manifest
comes back as validation state ``Valid`` with the status ``signingCredential.untrusted``: integrity was checked, the signer
was not established as trusted. ``Trusted`` is reported only when the certificate chains to an anchor the caller configured
(``trust_anchors_pem`` or the file named by the DFD_C2PA_TRUST_ANCHORS environment variable).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

AI_SOURCE_TYPES = ("trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia")
ACTION_LABELS = ("c2pa.actions", "c2pa.actions.v2")
UNTRUSTED_CODE = "signingCredential.untrusted"


def available() -> bool:
    try:
        import c2pa  # noqa: F401
        return True
    except Exception:
        return False


def library_versions() -> dict:
    out = {"c2pa_python": None, "c2pa_sdk": None}
    try:
        import importlib.metadata as im
        out["c2pa_python"] = im.version("c2pa-python")
    except Exception:
        pass
    try:
        import c2pa
        out["c2pa_sdk"] = c2pa.sdk_version()
    except Exception:
        pass
    return out


def _result(status: str, reason: str | None = None, **extra) -> dict:
    base = {"status": status, "reason": reason, "validation_state": None, "trust": None, "signer_trusted": None,
            "validation_statuses": [], "failure_codes": [], "signer": None, "claim_generator": None, "actions": [],
            "digital_source_types": [], "ai_generated_declared": False, "library": library_versions()}
    base.update(extra)
    return base


def _context(trust_anchors_pem: str | None):
    import c2pa
    cfg: dict = {"verify": {"remote_manifest_fetch": False, "ocsp_fetch": False}}
    pem = trust_anchors_pem
    if pem is None and os.environ.get("DFD_C2PA_TRUST_ANCHORS"):
        pem = Path(os.environ["DFD_C2PA_TRUST_ANCHORS"]).read_text(encoding="utf-8")
    if pem:
        cfg["trust"] = {"user_anchors": pem}
    return c2pa.Context(c2pa.Settings.from_dict(cfg))


def _statuses(items) -> list[dict]:
    return [{"code": s.get("code"), "explanation": s.get("explanation")} for s in (items or []) if isinstance(s, dict)]


def _generator(m: dict) -> str | None:
    gens = m.get("claim_generator_info")
    if isinstance(gens, list) and gens:
        g = gens[0] if isinstance(gens[0], dict) else {}
        name = g.get("name")
        return f"{name} {g['version']}".strip() if name and g.get("version") else name
    return m.get("claim_generator")


def _actions(m: dict) -> tuple[list[dict], list[str]]:
    acts, dst = [], []
    for a in m.get("assertions") or []:
        if a.get("label") not in ACTION_LABELS:
            continue
        data = a.get("data") or {}
        for act in data.get("actions") or []:
            d = act.get("digitalSourceType")
            acts.append({"action": act.get("action"), "digital_source_type": d})
            if d:
                dst.append(d)
        for t in data.get("templates") or []:
            if isinstance(t, dict) and t.get("digitalSourceType"):
                dst.append(t["digitalSourceType"])
    return acts, sorted(set(dst))


def verify(path: str | Path, trust_anchors_pem: str | None = None) -> dict:
    """Validate the C2PA manifest in ``path``. ``status`` is one of: verified (a manifest was read and validated, see
    ``validation_state``), absent, unsupported, error, unavailable."""
    if not available():
        return _result("unavailable", "the c2pa-python package is not importable")
    import c2pa
    try:
        ctx = _context(trust_anchors_pem)
        reader = c2pa.Reader(str(path), context=ctx)
    except Exception as e:
        name, msg = type(e).__name__, str(e)
        if "ManifestNotFound" in name:
            return _result("absent", "no C2PA manifest found in the file")
        if "NotSupported" in name:
            return _result("unsupported", f"file type not supported by the C2PA library ({msg})")
        return _result("error", f"{name}: {msg}")
    try:
        with reader:
            doc = json.loads(reader.json())
            state = reader.get_validation_state()
    except Exception as e:
        return _result("error", f"manifest could not be read: {type(e).__name__}: {e}")
    m = (doc.get("manifests") or {}).get(doc.get("active_manifest")) or {}
    statuses = _statuses(doc.get("validation_status"))
    failures = [s["code"] for s in statuses]
    untrusted = UNTRUSTED_CODE in failures
    state = str(state)
    if state == "Trusted":
        trust = "trusted"
    elif state == "Valid" and untrusted:
        trust = "untrusted_signer"
    else:
        trust = "unknown" if state == "Valid" else None
    acts, dst = _actions(m)
    si = m.get("signature_info") or {}
    signer = {"issuer": si.get("issuer"), "common_name": si.get("common_name"), "algorithm": si.get("alg"),
              "signing_time": si.get("time")} if si else None
    ai = any(any(t in d for t in AI_SOURCE_TYPES) for d in dst)
    return _result("verified", None, validation_state=state, trust=trust, signer_trusted=(trust == "trusted"),
                   validation_statuses=statuses, failure_codes=failures, signer=signer, claim_generator=_generator(m),
                   actions=acts, digital_source_types=dst, ai_generated_declared=ai)
