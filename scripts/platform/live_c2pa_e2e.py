"""Live API check of the C2PA module on signed, trusted, tampered and unsigned files (integration, not an accuracy experiment).

Builds an ephemeral test PKI (the helpers of tests/platform_c2pa/conftest.py; keys exist only in this process), signs tests/platform_c2pa/fixtures/A.jpg
with c2pa-python, and uploads the variants through the real HTTP API of the live dashboard server. The trust anchor for the "trusted" case is set
explicitly through DFD_C2PA_TRUST_ANCHORS, as an operator would; nothing is trusted by default.

  PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/live_c2pa_e2e.py
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_face_e2e as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests" / "platform_c2pa"


def load_conftest():
    spec = importlib.util.spec_from_file_location("c2pa_conftest", TESTS / "conftest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_pki(m):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID
    rk, ik, lk = (ec.generate_private_key(ec.SECP256R1()) for _ in range(3))
    root = m._cert(m._name("Ephemeral Root CA", "DFD Test Root"), m._name("Ephemeral Root CA", "DFD Test Root"), rk.public_key(), rk, True,
                   m._ku(key_cert_sign=True, crl_sign=True, digital_signature=True))
    inter = m._cert(m._name("Ephemeral Intermediate CA", "DFD Test Intermediate"), root.subject, ik.public_key(), rk, True,
                    m._ku(key_cert_sign=True, crl_sign=True, digital_signature=True), issuer_pub=rk.public_key())
    leaf = m._cert(m._name("DFD Ephemeral Signer", "DFD Test Signer"), inter.subject, lk.public_key(), ik, False,
                   m._ku(digital_signature=True, content_commitment=True), ExtendedKeyUsageOID.EMAIL_PROTECTION, issuer_pub=ik.public_key())
    pem = lambda c: c.public_bytes(serialization.Encoding.PEM)
    key = lk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return {"chain": pem(leaf) + pem(inter), "key": key, "root_pem": pem(root).decode("ascii")}


def sign(pki, data: bytes) -> bytes:
    import c2pa
    ai = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    manifest = {"claim_generator_info": [{"name": "dfd-live-test", "version": "1"}],
                "assertions": [{"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": ai}]}}]}
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, pki["chain"], pki["key"], None))
    out = io.BytesIO()
    c2pa.Builder.from_json(json.dumps(manifest)).sign(signer, "image/jpeg", io.BytesIO(data), out)
    return out.getvalue()


def upload(base, path: Path) -> dict:
    with open(path, "rb") as fh:
        r = requests.post(base + "/api/analyse", files={"file": (path.name, fh)}, timeout=60)
    jid = r.json()["job_id"]
    t0 = time.time()
    while time.time() - t0 < 120:
        j = requests.get(f"{base}/api/jobs/{jid}", timeout=30).json()
        if j["state"] in ("done", "failed"):
            return j
        time.sleep(0.3)
    return {"state": "timeout"}


def main() -> int:
    td = Path(tempfile.mkdtemp(prefix="dfd_c2pa_live_"))
    pki = build_pki(load_conftest())
    anchors = td / "anchors.pem"
    anchors.write_text(pki["root_pem"], encoding="ascii")
    src = (TESTS / "fixtures" / "A.jpg").read_bytes()
    signed = sign(pki, src)
    tampered = bytearray(signed)
    tampered[len(tampered) // 2] ^= 0xFF               # flip one byte inside the image data
    files = {"unsigned.jpg": src, "signed.jpg": signed, "tampered.jpg": bytes(tampered)}
    for n, b in files.items():
        (td / n).write_bytes(b)
    srv, base = E.start_server()
    rows, checks = [], []

    def run(name, trusted):
        if trusted:
            os.environ["DFD_C2PA_TRUST_ANCHORS"] = str(anchors)
        else:
            os.environ.pop("DFD_C2PA_TRUST_ANCHORS", None)
        j = upload(base, td / name)
        m = {x["module_id"]: x for x in (j.get("report") or {}).get("modules", [])}.get("content_credentials", {})
        codes = [o["code"] for o in m.get("observations", [])]
        rows.append({"file": name, "trust_anchor_configured": trusted, "state": j["state"], "status": m.get("status"), "finding": m.get("finding"),
                     "observation_codes": codes, "warnings": [w["code"] for w in m.get("warnings", [])]})
        return rows[-1]

    exp = [("unsigned.jpg", False, "no_credentials_found"), ("signed.jpg", False, "inconclusive"), ("signed.jpg", True, "credentials_valid"),
           ("tampered.jpg", False, "credentials_invalid"), ("tampered.jpg", True, "credentials_invalid")]
    for name, trusted, want in exp:
        r = run(name, trusted)
        checks.append({"check": f"{name} trusted={trusted} -> {want}", "ok": r["finding"] == want, "got": r["finding"], "codes": r["observation_codes"]})
    os.environ.pop("DFD_C2PA_TRUST_ANCHORS", None)
    srv.shutdown()
    res = {"purpose": "integration check of content_credentials through the live API (ephemeral test PKI); not an accuracy experiment",
           "rows": rows, "checks": checks, "failed": [c for c in checks if not c["ok"]]}
    (E.OUT / "live_c2pa_e2e.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
