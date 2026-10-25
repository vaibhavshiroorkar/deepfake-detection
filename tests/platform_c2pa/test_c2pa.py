"""Real C2PA verification tests: signed, tampered, unsigned, unsupported and corrupted files, using upstream test fixtures."""
import io
import json
from pathlib import Path

import pytest

c2pa = pytest.importorskip("c2pa")

from dfd.platform.coordinator import Coordinator  # noqa: E402
from dfd.platform.module import Probe  # noqa: E402
from dfd.platform.provenance import c2pa_verify as V  # noqa: E402
from dfd.platform.provenance.modules import ContentCredentialsModule  # noqa: E402
from dfd.platform.schema import validate_report  # noqa: E402

FX = Path(__file__).parent / "fixtures"
AI = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


@pytest.fixture(scope="module")
def signed(tmp_path_factory):
    if not (FX / "es256_private.key").exists():      # the upstream public test key is git-ignored (*.key); see fixtures/README.md to fetch it again
        pytest.skip("es256_private.key not present (git-ignored); fetch it as described in fixtures/README.md")
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, (FX / "es256_certs.pem").read_bytes(),
                                                       (FX / "es256_private.key").read_bytes(), None))
    manifest = {"claim_generator_info": [{"name": "dfd-test", "version": "1"}],
                "assertions": [{"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": AI}]}}]}
    out = io.BytesIO()
    with open(FX / "A.jpg", "rb") as src:
        c2pa.Builder.from_json(json.dumps(manifest)).sign(signer, "image/jpeg", src, out)
    return out.getvalue()


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def run_module(path, mime="image/jpeg"):
    probe = Probe(path=str(path), filename=path.name, media_type="image", mime_type=mime, size_bytes=path.stat().st_size,
                  sha256="0" * 64, has_video=False, has_audio=False)
    r = Coordinator([ContentCredentialsModule()]).analyse(probe, "t")
    assert validate_report(r) == []
    return r["modules"][0]


def test_signed_file_integrity_ok_but_signer_untrusted(tmp_path, signed):
    p = write(tmp_path, "s.jpg", signed)
    v = V.verify(p)
    assert v["status"] == "verified" and v["validation_state"] == "Valid" and v["trust"] == "untrusted_signer"
    assert v["signer_trusted"] is False and v["signer"]["issuer"] == "C2PA Test Signing Cert"
    assert v["claim_generator"] == "dfd-test 1" and v["actions"][0]["action"] == "c2pa.created"
    assert v["digital_source_types"] == [AI] and v["ai_generated_declared"] is True
    assert v["library"]["c2pa_python"] and v["library"]["c2pa_sdk"]
    m = run_module(p)
    assert m["finding"] == "inconclusive"              # integrity checked, signer not trusted: not reported as valid
    ver = [o for o in m["observations"] if o["kind"] == "verification"]
    assert ver and ver[0]["code"] == "C2PA_VALID_SIGNER_UNTRUSTED" and "not on a trust list" in ver[0]["text"]
    ai = [o for o in m["observations"] if o["code"] == "C2PA_DECLARED_SOURCE_AI"]
    assert ai and "not a detection" in ai[0]["text"]
    text = " ".join(m["limitations"])
    assert "does not prove" in text and "Absence of credentials proves nothing" in text


def test_signed_file_with_configured_trust_anchor_is_valid(tmp_path, signed, monkeypatch):
    p = write(tmp_path, "s.jpg", signed)
    monkeypatch.setenv("DFD_C2PA_TRUST_ANCHORS", str(FX / "es256_certs.pem"))
    assert V.verify(p)["trust"] == "trusted"
    m = run_module(p)
    assert m["finding"] == "credentials_valid"
    assert [o["code"] for o in m["observations"] if o["kind"] == "verification"] == ["C2PA_VALID_TRUSTED"]


def test_tampered_pixels_are_invalid(tmp_path, signed):
    b = bytearray(signed)
    b[len(b) // 2 + 5000] ^= 0xFF
    p = write(tmp_path, "t.jpg", bytes(b))
    v = V.verify(p)
    assert v["validation_state"] == "Invalid" and "assertion.dataHash.mismatch" in v["failure_codes"]
    m = run_module(p)
    assert m["finding"] == "credentials_invalid"
    assert any(o["kind"] == "verification" and "dataHash.mismatch" in o["text"] for o in m["observations"])


def test_unsigned_file_has_no_credentials(tmp_path):
    p = FX / "A.jpg"
    assert V.verify(p)["status"] == "absent"
    m = run_module(p)
    assert m["finding"] == "no_credentials_found" and not any(o["kind"] == "verification" for o in m["observations"])
    assert any("proves nothing" in x for x in m["limitations"])


def test_upstream_signed_file_not_reported_valid():
    m = run_module(FX / "C.jpg")
    assert m["finding"] in ("inconclusive", "credentials_invalid")
    assert m["finding"] != "credentials_valid"


def test_unsupported_format_is_inconclusive_with_reason(tmp_path):
    p = write(tmp_path, "note.txt", b"hello")
    assert V.verify(p)["status"] == "unsupported"
    m = run_module(p, "text/plain")
    assert m["finding"] == "inconclusive" and m["observations"][0]["code"] == "C2PA_FORMAT_UNSUPPORTED"
    assert not any(o["kind"] == "verification" for o in m["observations"])
    assert any("No credential was verified" in x for x in m["limitations"])


def test_corrupted_manifest_is_never_valid(tmp_path, signed):
    b = bytearray(signed)
    j = b.find(b"jumb")
    for k in range(j + 40, j + 140):
        b[k] ^= 0x55
    p = write(tmp_path, "c.jpg", bytes(b))
    m = run_module(p)
    assert m["finding"] in ("inconclusive", "credentials_invalid")
    assert not any(o["kind"] == "verification" and o["code"].startswith("C2PA_VALID") for o in m["observations"])


def test_missing_file_does_not_raise(tmp_path):
    v = V.verify(tmp_path / "nope.jpg")
    assert v["status"] in ("error", "absent", "unsupported")
