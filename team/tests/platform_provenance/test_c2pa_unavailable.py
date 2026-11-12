"""Behaviour when the optional c2pa-python package is absent (simulated, so this runs with or without the package installed).

Without the library nothing can be cryptographically verified: the module may only report a marker scan and must never say credentials are
valid or invalid, nor claim that a manifest was validated."""
import pytest

from dfd.platform.coordinator import Coordinator
from dfd.platform.module import Probe
from dfd.platform.provenance import c2pa_verify as V
from dfd.platform.provenance.modules import ContentCredentialsModule
from dfd.platform.schema import validate_report


def _probe(path):
    return Probe(path=str(path), filename=path.name, media_type="image", mime_type="image/jpeg", size_bytes=path.stat().st_size, sha256="0" * 64,
                 duration_s=None, has_video=False, has_audio=False)


@pytest.fixture
def no_lib(monkeypatch):
    monkeypatch.setattr(V, "available", lambda: False)


@pytest.mark.parametrize("payload,finding", [(b"plain bytes", "no_credentials_found"), (b"xx jumb yy c2pa zz", "inconclusive")])
def test_without_library_only_marker_scan_is_reported(tmp_path, no_lib, payload, finding):
    f = tmp_path / "a.jpg"
    f.write_bytes(payload)
    r = Coordinator([ContentCredentialsModule()]).analyse(_probe(f), "t")
    m = r["modules"][0]
    assert validate_report(r) == [] and m["status"] == "completed" and m["finding"] == finding
    assert m["finding"] not in ("credentials_valid", "credentials_invalid")
    assert "no manifest was validated" in m["tested_scope"]
    assert any("No credential was verified" in x for x in m["limitations"])


def test_verify_reports_unavailable_status(tmp_path, no_lib):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    assert V.verify(f)["status"] == "unavailable"
