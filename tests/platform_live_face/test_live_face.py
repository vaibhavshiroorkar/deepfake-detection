"""Face-bearing live path. Fast tests (always, no GPU) use the real pipeline and server with a fake adapter on synthetic media.
The slow test (RUN_LIVE_FACE=1) runs the real checkpoint on non-sealed fixtures through the HTTP API and the dashboard in a browser;
it is an integration check, not an accuracy experiment. Fixtures are referenced in place and the test skips if they are absent."""
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.platform import pipeline
from dfd.platform.dashboard import serve
from dfd.platform.modules import build_default_modules
from dfd.platform.modules.base import AdapterProvider
from dfd.platform.pipeline import planned_modules
from dfd.platform.provenance.modules import build_provenance_modules
from dfd.platform.schema import validate_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "platform"))


class FaceAdapter:
    """Image: a face is found (path face_crop). Video: no face is found."""
    meta = {"checkpoint_id": "fake-ckpt", "files": {}, "tasks": {"T-V": {"threshold": {"value": 0.6}}, "T-I": {"threshold": {"value": 0.5}},
                                                                    "T-A": {"threshold": {"value": 0.5}}}}

    def analyse(self, media, progress=None):
        pm = {}
        if media.has_audio:
            pm["audio"] = ModalityScore(True, 0.2, 1)
        if media.media_type == "image":
            pm["image"] = ModalityScore(True, 0.9, 1, path="face_crop")
        if media.media_type == "video":
            pm["visual"] = ModalityScore(False, reason="no face found")
        return AdapterResult(pm, [{"code": "NO_FACE_DETECTED", "message": "no face"}] if media.media_type == "video" else [],
                             {"checkpoint_id": "fake-ckpt"}, 0.5, {})


def mods():
    return build_default_modules(AdapterProvider(adapter=FaceAdapter())) + planned_modules() + build_provenance_modules()


def by_id(rep):
    return {m["module_id"]: m for m in rep["modules"]}


def test_no_face_video_is_not_applicable_and_never_a_finding(media):
    rep = pipeline.analyse_file(media / "clip_audio.mp4", modules=mods())
    assert validate_report(rep) == []
    fv = by_id(rep)["face_video"]
    assert fv["status"] == "not_applicable" and fv["finding"] is None and fv["score"] is None
    assert fv["reason"]
    assert "face_video" in [c["module_id"] for c in rep["coverage"]["not_applicable"]]
    assert "face_video" not in [c["module_id"] for c in rep["coverage"]["completed"]]


def test_image_with_face_routes_to_image_detector_face_crop_with_limitation(media):
    rep = pipeline.analyse_file(media / "img.png", modules=mods())
    assert validate_report(rep) == []
    mm = by_id(rep)
    assert mm["face_video"]["status"] == "not_applicable"
    im = mm["image_detector"]
    assert im["status"] == "completed" and im["finding"] is not None
    assert "path=face_crop" in im["tested_scope"]
    assert im["score"]["calibrated"] is False
    text = " ".join(im["limitations"]).lower()
    assert "video-trained" in text and "still" in text and "not evaluated on still face-swapped photos" in text
    assert "not evaluated on still face-swapped photos" in im["model"]["evaluation"].lower()


def test_server_path_no_face_video_over_http(media, tmp_path):
    from werkzeug.serving import make_server
    import threading
    import requests
    app = serve.create_app(live=True, modules_factory=mods, upload_dir=tmp_path / "up")
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        with open(media / "clip_audio.mp4", "rb") as fh:
            jid = requests.post(base + "/api/analyse", files={"file": ("clip_audio.mp4", fh)}, timeout=30).json()["job_id"]
        t0 = time.time()
        while time.time() - t0 < 30:
            j = requests.get(f"{base}/api/jobs/{jid}", timeout=10).json()
            if j["state"] in ("done", "failed"):
                break
            time.sleep(0.1)
        assert j["state"] == "done"
        assert by_id(j["report"])["face_video"]["status"] == "not_applicable"
        assert list((tmp_path / "up").glob("*")) == []
    finally:
        srv.shutdown()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_LIVE_FACE") != "1", reason="set RUN_LIVE_FACE=1 to run the real checkpoint (GPU) on non-sealed FF++/LAV-DF/ASVspoof fixtures")
def test_real_checkpoint_face_path_end_to_end():
    import live_face_e2e as E
    import live_face_fixtures as F
    pytest.importorskip("playwright.sync_api")
    if not (F.DATA / "ffpp").is_dir():
        pytest.skip(f"FF++ dataset path absent: {F.DATA / 'ffpp'}")
    fixtures, skipped = F.build(Path(tempfile.mkdtemp(prefix="dfd_lf_fx_")))
    assert any(f["name"] == "ffpp_fake_video" for f in fixtures), skipped
    srv, base = E.start_server()
    try:
        rows = [E.run_api(base, fx) for fx in fixtures]
        assert all(r["state"] == "done" and r["source_summary"] == "real" for r in rows), [(r["fixture"], r["state"]) for r in rows]
        cons = E.parity(rows, fixtures)
        assert cons and all(c["consistent"] for c in cons), cons
        by = {r["fixture"]: {m["module_id"]: m for m in r["modules"]} for r in rows}
        assert by["synthetic_noface_video"]["face_video"]["status"] == "not_applicable"
        assert by["ffpp_fake_frame_image"]["image_detector"]["status"] == "completed"
        face_fx = next(f for f in fixtures if f["name"] == "ffpp_fake_video")
        b = E.browser_check(base, face_fx)
        assert b["live_banner"] and not b["mock_banner"] and not b["forbidden_hits"] and not b["page_errors"]
        assert b["face_status"] == "completed"
        assert re.search(r"Source\s+real", b["face_card_text"]) and re.search(r"Version\s+I-", b["face_card_text"]), b["face_card_text"][:400]
    finally:
        srv.shutdown()
