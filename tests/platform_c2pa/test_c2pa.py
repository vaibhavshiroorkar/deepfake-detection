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


MANIFEST = {"claim_generator_info": [{"name": "dfd-test", "version": "1"}],
            "assertions": [{"label": "c2pa.actions", "data": {"actions": [{"action": "c2pa.created", "digitalSourceType": AI}]}}]}


def sign_bytes(pki, data: bytes, mime: str) -> bytes:
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, pki["chain"], pki["key"], None))
    out = io.BytesIO()
    c2pa.Builder.from_json(json.dumps(MANIFEST)).sign(signer, mime, io.BytesIO(data), out)
    return out.getvalue()


@pytest.fixture(scope="module")
def signed(pki):
    return sign_bytes(pki, (FX / "A.jpg").read_bytes(), "image/jpeg")


@pytest.fixture
def anchor_env(pki, tmp_path, monkeypatch):
    """Point DFD_C2PA_TRUST_ANCHORS at the ephemeral root only (explicit configuration, as in real use)."""
    f = tmp_path / "anchors.pem"
    f.write_text(pki["root_pem"], encoding="ascii")
    monkeypatch.setenv("DFD_C2PA_TRUST_ANCHORS", str(f))


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
    assert v["signer_trusted"] is False and v["signer"]["common_name"] == "DFD Ephemeral Signer"
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


def test_signed_file_with_configured_trust_anchor_is_valid(tmp_path, signed, anchor_env):
    p = write(tmp_path, "s.jpg", signed)
    assert V.verify(p)["trust"] == "trusted"
    m = run_module(p)
    assert m["finding"] == "credentials_valid"
    assert [o["code"] for o in m["observations"] if o["kind"] == "verification"] == ["C2PA_VALID_TRUSTED"]
    assert any("does not prove" in x for x in m["limitations"])      # valid never reads as authentic


def test_unrelated_anchor_keeps_signer_untrusted(tmp_path, signed, monkeypatch):
    """Anchors are configured but the signer does not chain to them: still untrusted, never valid."""
    p = write(tmp_path, "s.jpg", signed)
    monkeypatch.setenv("DFD_C2PA_TRUST_ANCHORS", str(FX / "es256_certs.pem"))
    assert run_module(p)["finding"] == "inconclusive"


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


def _make_media(tmp_path, name, args):
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg not on PATH")
    p = tmp_path / name
    subprocess.run([ff, "-loglevel", "error", "-y", *args, str(p)], check=True)
    return p


MEDIA = {
    "mp4": ("video/mp4", "video", ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1", "-f", "lavfi", "-i",
                                   "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"], b"mdat"),
    "wav": ("audio/wav", "audio", ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"], b"data"),
}


@pytest.mark.parametrize("ext", sorted(MEDIA))
def test_signed_video_and_audio(tmp_path, pki, anchor_env, monkeypatch, ext):
    mime, kind, args, marker = MEDIA[ext]
    src = _make_media(tmp_path, f"src.{ext}", args)
    signed_bytes = sign_bytes(pki, src.read_bytes(), mime)
    p = write(tmp_path, f"s.{ext}", signed_bytes)
    # trusted via explicit anchor
    m = run_module(p, mime)
    assert m["finding"] == "credentials_valid"
    # default: no anchors configured -> integrity ok, signer not established
    monkeypatch.delenv("DFD_C2PA_TRUST_ANCHORS")
    m = run_module(p, mime)
    assert m["finding"] == "inconclusive"
    assert [o["code"] for o in m["observations"] if o["kind"] == "verification"] == ["C2PA_VALID_SIGNER_UNTRUSTED"]
    assert any(o["code"] == "C2PA_DECLARED_SOURCE_AI" and "not a detection" in o["text"] for o in m["observations"])
    # tamper one byte of the media payload
    b = bytearray(signed_bytes)
    b[b.find(marker, 64) + 100] ^= 0xFF
    t = write(tmp_path, f"t.{ext}", bytes(b))
    v = V.verify(t)
    assert v["validation_state"] == "Invalid", v
    assert run_module(t, mime)["finding"] == "credentials_invalid"
