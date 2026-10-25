"""WS3 tests: local metadata and content-credential marker scan. Real ffprobe, small synthetic files, no model."""
import shutil
import subprocess

import pytest

from dfd.platform.coordinator import Coordinator
from dfd.platform.module import Probe
from dfd.platform.provenance import metadata as M
from dfd.platform.provenance.modules import build_provenance_modules
from dfd.platform.schema import validate_report


@pytest.fixture(scope="module")
def mp4(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")
    p = tmp_path_factory.mktemp("prov") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(p)], check=True)
    return p


def probe_of(path):
    return Probe(path=str(path), filename=path.name, media_type="video", mime_type="video/mp4", size_bytes=path.stat().st_size,
                 sha256="0" * 64, has_video=True, has_audio=True)


def run(path):
    return Coordinator(build_provenance_modules()).analyse(probe_of(path), "t")


def test_plain_file_has_metadata_and_no_credentials(mp4):
    r = run(mp4)
    assert validate_report(r) == []
    by = {m["module_id"]: m for m in r["modules"]}
    assert by["media_metadata"]["finding"] == "inconclusive"
    kinds = {o["kind"] for o in by["media_metadata"]["observations"]}
    assert "file_metadata" in kinds and "verification" not in kinds
    assert any(o["code"] == "VIDEO_STREAM" for o in by["media_metadata"]["observations"])
    assert by["content_credentials"]["finding"] == "no_credentials_found"
    assert r["status"]["state"] == "ok"


def test_marker_bytes_reported_as_unverified_never_valid(mp4, tmp_path):
    p = tmp_path / "marked.mp4"
    p.write_bytes(mp4.read_bytes() + b"\x00\x00\x00\x20jumb\x00\x00\x00\x18jumdc2pa")
    cc = next(m for m in run(p)["modules"] if m["module_id"] == "content_credentials")
    assert cc["finding"] == "inconclusive"
    assert cc["observations"][0]["code"] == "C2PA_MARKER_PRESENT"
    assert any("No credential was verified" in x for x in cc["limitations"])


def test_single_marker_is_not_enough(mp4, tmp_path):
    p = tmp_path / "one.mp4"
    p.write_bytes(mp4.read_bytes() + b"jumb")
    assert next(m for m in run(p)["modules"] if m["module_id"] == "content_credentials")["finding"] == "no_credentials_found"


def test_byte_scan_helper_never_reports_verification_even_with_verifier_installed(mp4):
    assert M.c2pa_verifier_available() is True        # c2pa-python 0.38.0 installed 2026-10-26; the byte-scan helper itself never verifies
    out = M.inspect_provenance(mp4)
    assert out["verified_credentials"] == [] and not any(o["kind"] == "verification" for o in out["observations"])


def test_unreadable_file_fails_metadata_module_only(tmp_path):
    p = tmp_path / "junk.mp4"
    p.write_bytes(b"not a media file")
    r = run(p)
    assert validate_report(r) == []
    by = {m["module_id"]: m for m in r["modules"]}
    assert by["media_metadata"]["status"] == "failed" and by["media_metadata"]["errors"][0]["code"] == "MODULE_EXCEPTION"
    assert by["content_credentials"]["status"] == "completed"       # the byte scan does not need ffprobe


def test_heuristic_flags_are_descriptive_only(mp4):
    obs = M.heuristics(M.ffprobe(mp4))
    assert all(o["kind"] == "heuristic" for o in obs)
    low = " ".join(o["text"].lower() for o in obs)
    assert "fake" not in low and "manipulat" not in low and "genuine" not in low
