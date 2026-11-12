"""Real-checkpoint check of the local web page in real Chromium (D-040 item 5): upload one image, one video and one audio file, check that
the result is real (no MOCK banner, inference_source real, a score or the honest unavailable note), and save screenshots.

    PYTHONPATH=src PLAYWRIGHT_BROWSERS_PATH=D:/cache/playwright .venv/Scripts/python.exe scripts/mvp/web_real_check.py [--checkpoint checkpoints/I-0-mvp]

The files are TRAINING-split items, copied to a scratch folder (this shows that the page works; it says nothing about accuracy): a face crop of an
FF++ c23 training video, a LAV-DF training video with audio, an ASVspoof 2019 LA training utterance. Screenshots: assets/screens/real_<kind>.png.
Binds 127.0.0.1 only (a free port); the page deletes the upload after each request.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", r"D:\cache\playwright")

from playwright.sync_api import sync_playwright  # noqa: E402
from werkzeug.serving import make_server  # noqa: E402

from dfd.infer.factory import make_adapter  # noqa: E402
from dfd.web.app import create_app  # noqa: E402

SCREENS = Path("assets/screens")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/I-0-mvp")
    a = ap.parse_args(argv)
    tmp = Path(tempfile.mkdtemp(prefix="dfd_demo_"))
    files = {
        "image": next(Path("data/_crops/ffpp_c23_yunet_n16").glob("*/0.jpg")),
        "video": Path("data/lavdf/LAV-DF/train/000470.mp4"),
        "audio": next(Path("data/asvspoof2019_la/LA/ASVspoof2019_LA_train/flac").glob("LA_T_1*.flac")),
    }
    local = {k: shutil.copy(v, tmp / f"demo_{k}{v.suffix}") for k, v in files.items()}
    app = create_app(make_adapter("real", a.checkpoint), upload_dir=tmp / "up")
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    out, ok_all = {}, True
    with sync_playwright() as p:
        b = p.chromium.launch()
        for kind, path in local.items():
            pg = b.new_page(viewport={"width": 1100, "height": 1000})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto(url)
            mock_banner_visible = pg.locator("#mock-banner").is_visible()
            with pg.expect_file_chooser() as fc:
                pg.click("#choose")
            fc.value.set_files(str(path))
            pg.click("#go")
            pg.wait_for_selector("#result:not([hidden])", timeout=300000)
            raw = json.loads(pg.locator("#raw").text_content())
            res = raw["result"] or {}
            rec = dict(file=str(files[kind]), mock_banner_visible=mock_banner_visible, result_mock_banner_visible=pg.locator("#result-mock-banner").is_visible(),
                       inference_source=raw["inference_source"], state=raw["status"]["state"], verdict=pg.locator("#verdict-label").inner_text(),
                       verdict_score=pg.locator("#verdict-score").inner_text(), per_modality=res.get("per_modality"), warnings=[w["code"] for w in raw["warnings"]],
                       errors=raw["errors"], checkpoint=(raw.get("model") or {}).get("checkpoint_id"), page_errors=errs)
            ok = (not rec["mock_banner_visible"] and not rec["result_mock_banner_visible"] and rec["inference_source"] == "real"
                  and rec["state"] != "failed" and not errs and rec["checkpoint"])
            rec["passed"] = bool(ok)
            ok_all &= bool(ok)
            out[kind] = rec
            SCREENS.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(SCREENS / f"real_{kind}.png"), full_page=True)
            print(kind, json.dumps({k: rec[k] for k in ("inference_source", "state", "verdict", "verdict_score", "warnings", "passed")}), flush=True)
            pg.close()
        b.close()
    srv.shutdown()
    leftovers = list((tmp / "up").glob("*")) if (tmp / "up").is_dir() else []
    out["upload_folder_empty_after_requests"] = not leftovers
    Path("data/_results/I-0/web_real_check.json").write_text(json.dumps(dict(passed=bool(ok_all and not leftovers), cases=out), indent=1, default=str))
    shutil.rmtree(tmp, ignore_errors=True)
    print("WEB REAL CHECK", "PASS" if ok_all and not leftovers else "FAIL")
    return 0 if ok_all and not leftovers else 1


if __name__ == "__main__":
    raise SystemExit(main())
