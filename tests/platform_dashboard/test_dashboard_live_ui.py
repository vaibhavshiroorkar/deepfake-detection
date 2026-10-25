"""Live-analysis UI tests for the dashboard, against a stub server that implements the documented contract.

The stub serves the real dashboard static files and canned jobs. No model runs; reports come from the
mock fixtures (some relabelled as source "real" ONLY to exercise the banner logic, never as a result).
Test hooks in the page: ?poll_interval_ms=N and ?poll_timeout_ms=N (defaults 1000 and 600000).
Skipped when playwright or a usable browser is missing.
"""
import copy
import json
import re
import threading
import uuid
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "src" / "dfd" / "platform" / "dashboard"
FIX = ROOT / "src" / "dfd" / "platform" / "fixtures"
LIVE_TEXT = "Results come from the modules listed below; each states its own tested scope and limitations."
EVIL = "<img src=x onerror=window.__pwned=1>.mp4"
FORBIDDEN = [r"likely real", r"\bgenuine\b", r"\bauthentic\b", r"\bsafe\b"]

STATE = {"live": True, "mode_status": 200}
JOBS = {}


def _fx(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def _real(name, mixed=False):
    r = _fx(name)
    for i, m in enumerate(r["modules"]):
        m["source"] = "mock" if (mixed and i > 0) else "real"
    r["source_summary"] = "mixed" if mixed else "real"
    return r


def _running(c, mod):
    return {"state": "running", "progress": {"module_id": mod, "completed": c, "total": 3}, "report": None, "error": None}


def _done(rep):
    return {"state": "done", "progress": {"module_id": None, "completed": 3, "total": 3}, "report": rep, "error": None}


def _script(fname):
    queued = {"state": "queued", "progress": {"module_id": None, "completed": 0, "total": 3}, "report": None, "error": None}
    steps = [queued, _running(0, "face_video"), _running(1, "speech")]
    if fname == "bad.mp4":
        return steps[:1] + [{"state": "failed", "progress": {"module_id": None, "completed": 0, "total": 3}, "report": None,
                             "error": {"code": "UNREADABLE_FILE", "message": "The file could not be decoded."}}]
    if fname == "slow.mp4":
        return steps[:2] + [_running(1, "speech")] * 1000
    if fname == "real.mp4":
        return steps + [_done(_real("video_nothing_found"))]
    if fname == "mixed.mp4":
        return steps + [_done(_real("video_nothing_found", mixed=True))]
    if fname == "partial.mp4":
        return steps + [_done(_fx("module_failure"))]
    return steps + [_done(_fx("video_nothing_found"))]


def make_stub():
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(DASH, "index.html")

    @app.get("/dashboard.js")
    def js():
        return send_from_directory(DASH, "dashboard.js")

    @app.get("/dashboard.css")
    def css():
        return send_from_directory(DASH, "dashboard.css")

    @app.get("/api/fixtures")
    def fixtures():
        return jsonify(sorted(p.stem for p in FIX.glob("*.json")))

    @app.get("/api/fixtures/<name>")
    def fixture(name):
        if name not in {p.stem for p in FIX.glob("*.json")}:
            abort(404)
        return send_from_directory(FIX, name + ".json", mimetype="application/json")

    @app.get("/api/mode")
    def mode():
        if STATE["mode_status"] != 200:
            return jsonify({"error": {"code": "NOPE", "message": "x"}}), STATE["mode_status"]
        return jsonify({"live": STATE["live"], "mock_only": not STATE["live"]})

    @app.post("/api/analyse")
    def analyse():
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": {"code": "NO_FILE", "message": "No file."}}), 400
        if f.filename == "huge.mp4":
            return jsonify({"error": {"code": "FILE_TOO_LARGE", "message": "File exceeds the size limit."}}), 413
        jid = uuid.uuid4().hex
        JOBS[jid] = _script(f.filename)
        return jsonify({"job_id": jid}), 202

    @app.get("/api/jobs/<jid>")
    def job(jid):
        steps = JOBS.get(jid)
        if steps is None:
            return jsonify({"error": {"code": "NOT_FOUND", "message": "Unknown job."}}), 404
        cur = steps[0]
        if len(steps) > 1:
            steps.pop(0)
        return jsonify(cur)

    return app


@pytest.fixture(scope="module")
def env():
    pw_mod = pytest.importorskip("playwright.sync_api")
    pw = pw_mod.sync_playwright().start()
    browser, err = None, None
    for kw in ({}, {"channel": "msedge"}, {"channel": "chrome"}):
        try:
            browser = pw.chromium.launch(**kw)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if browser is None:
        pw.stop()
        pytest.skip(f"no usable browser for playwright: {str(err)[:120]}")
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 0, make_stub(), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield browser, base
    srv.shutdown()
    browser.close()
    pw.stop()


@pytest.fixture()
def open_page(env):
    browser, base = env
    pages = []
    STATE.update(live=True, mode_status=200)

    def make(query="?poll_interval_ms=150&poll_timeout_ms=20000"):
        pg = browser.new_page()
        pages.append(pg)
        pg.goto(base + "/" + query)
        pg.wait_for_selector("#report:not([hidden])")      # first scenario loaded
        pg.wait_for_function("() => document.querySelector('#live-panel').getAttribute('data-live') !== null"
                             " && (document.querySelector('#live-panel').dataset.live === 'true' || !document.querySelector('#live-unavailable').hidden)")
        return pg

    yield make
    for p in pages:
        p.close()


def _upload(pg, name):
    pg.set_input_files("#analyse-file", {"name": name, "mimeType": "video/mp4", "buffer": b"stub"})
    pg.click("#analyse-btn")


def _wait_status(pg, text):
    pg.wait_for_function("t => document.querySelector('#live-status').textContent.includes(t)", arg=text, timeout=15000)


def test_mode_false_hides_upload_and_keeps_scenarios(open_page):
    STATE["live"] = False
    pg = open_page()
    assert pg.is_disabled("#analyse-btn") and pg.is_disabled("#analyse-file")
    assert "Live analysis is not available on this server" in pg.inner_text("#live-panel")
    assert pg.locator("#scenario option").count() >= 1 and pg.locator("article.card").count() > 0


def test_mode_endpoint_missing_is_treated_as_not_live(open_page):
    STATE["mode_status"] = 404
    pg = open_page()
    assert pg.is_disabled("#analyse-btn")
    assert pg.is_visible("#live-unavailable")
    assert pg.locator("article.card").count() > 0


def test_success_flow_shows_progress_then_mock_report(open_page):
    pg = open_page("?poll_interval_ms=300&poll_timeout_ms=20000")
    assert pg.is_enabled("#analyse-btn") and not pg.is_visible("#live-unavailable")
    _upload(pg, "ok.mp4")
    assert pg.is_disabled("#analyse-btn")                  # disabled while the job runs
    _wait_status(pg, "Running")
    assert re.search(r"face_video|speech", pg.inner_text("#live-status"))
    assert re.search(r"\d of 3", pg.inner_text("#live-status"))
    _wait_status(pg, "Done")
    assert pg.is_enabled("#analyse-btn")
    # mock-sourced content keeps the MOCK banner and never the LIVE one
    banner = pg.inner_text("#banner")
    assert "MOCK DATA" in banner and "LIVE ANALYSIS" not in banner
    assert pg.locator("article.card").count() == len(_fx("video_nothing_found")["modules"])


def test_real_report_shows_live_banner_without_mock(open_page):
    pg = open_page()
    _upload(pg, "real.mp4")
    _wait_status(pg, "Done")
    banner = pg.inner_text("#banner")
    assert "LIVE ANALYSIS" in banner and LIVE_TEXT in banner
    assert "MOCK" not in banner
    body = pg.inner_text("body").lower()
    for pat in FORBIDDEN:
        assert not re.search(pat, body.replace(_fx("video_nothing_found")["statement"].lower(), "")), pat


def test_mixed_report_shows_both_facts(open_page):
    pg = open_page()
    _upload(pg, "mixed.mp4")
    _wait_status(pg, "Done")
    banner = pg.inner_text("#banner")
    assert "LIVE ANALYSIS" in banner and LIVE_TEXT in banner and "MOCK DATA" in banner


def test_failed_job_shows_code_and_message(open_page):
    pg = open_page()
    _upload(pg, "bad.mp4")
    _wait_status(pg, "UNREADABLE_FILE")
    assert "The file could not be decoded." in pg.inner_text("#live-status")
    assert pg.is_enabled("#analyse-btn")


def test_upload_refused_413_shows_code(open_page):
    pg = open_page()
    _upload(pg, "huge.mp4")
    _wait_status(pg, "FILE_TOO_LARGE")
    assert pg.is_enabled("#analyse-btn")


def test_partial_report_shows_note_with_counts(open_page):
    pg = open_page()
    _upload(pg, "partial.mp4")
    _wait_status(pg, "Done")
    note = pg.inner_text("#partial-note")
    assert "1 module(s) did not complete" in note and "1 failed" in note and "0 timed out" in note


def test_html_in_filename_is_inert(open_page):
    pg = open_page()
    _upload(pg, EVIL)
    _wait_status(pg, "Done")
    assert EVIL in pg.inner_text("#live-status")
    assert pg.locator("#live-status img, #live-panel img").count() == 0
    assert pg.evaluate("() => window.__pwned === undefined")


def test_polling_timeout_shows_message_and_reenables(open_page):
    pg = open_page("?poll_interval_ms=100&poll_timeout_ms=800")
    assert pg.evaluate("() => window.dashboard.config.pollTimeoutMs") == 800
    _upload(pg, "slow.mp4")
    assert pg.is_disabled("#analyse-btn")
    _wait_status(pg, "Timed out")
    assert pg.is_enabled("#analyse-btn")


def test_default_timeout_is_ten_minutes(open_page):
    pg = open_page("")
    assert pg.evaluate("() => window.dashboard.config.pollTimeoutMs") == 600000
    assert pg.evaluate("() => window.dashboard.config.pollIntervalMs") == 1000
