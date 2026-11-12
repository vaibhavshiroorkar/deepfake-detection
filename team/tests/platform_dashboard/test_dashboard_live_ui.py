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

    @app.get("/api/vocabulary")
    def vocabulary():
        from dfd.platform.explain import vocabulary as vocab
        return jsonify(vocab())

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
        # ready = mode resolved and scenarios listed; a scenarios-only server has also opened its first MOCK scenario
        pg.wait_for_function("() => document.querySelectorAll('#scenario option').length > 1"
                             " && (document.querySelector('#live-panel').dataset.live === 'true'"
                             "     || (!document.querySelector('#live-unavailable').hidden && !document.querySelector('#report').hidden))")
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
    assert re.search(r"Face manipulation analysis|Synthetic speech analysis", pg.inner_text("#live-status"))   # human names, not ids
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


def test_failed_or_refused_analysis_does_not_leave_a_mock_or_old_report_on_screen(open_page):
    """Regression (2026-10-26 demo run): a failed live job left the initial MOCK scenario report visible next to the failure message.
    Since D14 live mode opens on the upload panel, so the MOCK report is put on screen on purpose first."""
    pg = open_page()
    assert not pg.is_visible("#report")                      # live mode: no report of any kind at page load
    pg.select_option("#scenario", "video_manipulation_indicated")
    pg.wait_for_selector("#report:not([hidden])")
    assert "MOCK DATA" in pg.inner_text("#banner")
    _upload(pg, "bad.mp4")
    _wait_status(pg, "UNREADABLE_FILE")
    assert not pg.is_visible("#report") and not pg.is_visible("#banner")
    _upload(pg, "huge.mp4")
    _wait_status(pg, "FILE_TOO_LARGE")
    assert not pg.is_visible("#report")
    pg.select_option("#scenario", index=1)                   # scenarios can still be shown on request
    pg.wait_for_selector("#report:not([hidden])")


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
    assert "1 of 3 checks did not complete" in note and "1 failed" in note and "0 timed out" in note


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


# ---------------------------------------------------------------- redesign (D14, UX-4, accessibility)

def test_live_mode_opens_on_the_upload_panel_with_no_invented_report(open_page):
    pg = open_page()
    assert not pg.is_visible("#report") and not pg.is_visible("#banner")
    assert pg.is_visible("#dropzone") and "Live analysis" in pg.inner_text("#mode-pill")
    assert pg.locator("article.card").count() == 0
    assert pg.input_value("#scenario") == ""


def test_choosing_a_file_shows_name_size_and_preview_before_analysis(open_page):
    pg = open_page()
    pg.set_input_files("#analyse-file", {"name": "clip.mp4", "mimeType": "video/mp4", "buffer": b"x" * 2048})
    assert pg.is_visible("#file-card") and pg.inner_text("#file-name") == "clip.mp4"
    assert "2.0 KB" in pg.inner_text("#file-sub")
    assert pg.locator("#preview video").count() == 1
    assert not pg.is_visible("#report")                      # nothing is analysed until Analyse is pressed


def test_mode_false_shows_scenarios_only_pill(open_page):
    STATE["live"] = False
    pg = open_page()
    assert "Demo data only" in pg.inner_text("#mode-pill") and "MOCK DATA" in pg.inner_text("#banner")


def _real_shaped_report():
    """A schema 2.1 report built by the real coordinator and explanation layer from SYNTHETIC module results (made-up numbers)."""
    import base64

    import cv2
    import numpy as np
    from dfd.platform.coordinator import Coordinator
    from dfd.platform.module import Applicability, Probe
    from dfd.platform.schema import ModuleResult, validate_report

    class Fixed:
        timeout_s, source = 5.0, "real"

        def __init__(self, res):
            self.res, self.module_id, self.version, self.section = res, res.module_id, res.version, res.section

        def applicability(self, p):
            return Applicability.yes()

        def run(self, ctx):
            return copy.deepcopy(self.res)
    jpg = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", np.zeros((112, 112, 3), np.uint8))[1].tobytes()).decode()
    model = {"name": "test model", "version": "t", "weights_sha256": "0" * 64, "licence": "test", "evaluation": "E-TEST synthetic"}
    face = ModuleResult("face_video", "t", "authenticity", "real", "completed", finding="indicates_manipulation",
                        score={"value": 0.97231, "meaning": "test", "calibrated": True, "threshold": 0.6138830927259571}, units_analysed=16,
                        model=model, tested_scope="Face manipulation of the largest face in 16 sampled frames (at most 16).",
                        limitations=["Trained and sealed-tested on face-swap material (test text)."])
    gen = ModuleResult("generated_video", "t", "authenticity", "real", "completed", finding="inconclusive",
                       score={"value": 0.99, "meaning": "Uncalibrated ranking score; not a probability.", "calibrated": False, "threshold": None},
                       units_analysed=14, model={**model, "evaluation": "E-020, experimental, NOT validated: DF26 held-out AUC 0.4957 [0.4755, 0.5168]"},
                       tested_scope="chance level on DF26", warnings=[{"code": "EXPERIMENTAL_UNVALIDATED", "message": "Experimental capability."}],
                       limitations=["Only the first 2.0 s of the clip is analysed.", "Face-independent: it does not detect face swaps."],
                       evidence=[{"kind": "frames", "frames": [jpg] * 8, "times_s": [0, 0.29, 0.57, 0.86, 1.0, 1.29, 1.57, 1.86], "n_frames": 8,
                                  "n_decoded": 14, "text": "The 8 frames this model analysed."}])
    lip = ModuleResult("lip_sync", "t", "cross_modal", "real", "completed", finding="consistent", score={"value": 0.45, "meaning": "m",
                       "calibrated": False, "threshold": None}, units_analysed=80, model=model, tested_scope="(at most 30 s)",
                       limitations=["A synchronised fake passes: test."],
                       observations=[{"kind": "heuristic", "code": "av_offset_frames", "text": "offset", "value": 1},
                                     {"kind": "heuristic", "code": "sync_confidence", "text": "conf", "value": 4.5}],
                       evidence=[{"kind": "offset_curve", "offsets": list(range(-15, 16)), "mean_dist": [10.0 + abs(o - 1) for o in range(-15, 16)],
                                  "best_offset": 1, "band": [-6, 2], "min_confidence": 3.0, "confidence": 4.5, "text": "offset search"}])
    p = Probe(path="x", filename="talk.mp4", media_type="video", mime_type="video/mp4", size_bytes=5_900_000, sha256="b" * 64, duration_s=5.9,
              has_video=True, has_audio=True, width=1280, height=720)
    r = Coordinator([Fixed(face), Fixed(gen), Fixed(lip)]).analyse(p, "synthetic-ui-test")
    assert validate_report(r) == [], validate_report(r)
    return r


def test_real_shaped_report_hierarchy_and_experimental_deemphasis(open_page):
    pg = open_page()
    r = _real_shaped_report()
    assert pg.evaluate("r => window.dashboard.renderReport(r, true)", r) is True
    groups = pg.eval_on_selector_all("#evidence section.group", "els => els.map(e => e.id)")
    assert groups[0] == "group-attention" and "group-experimental" in groups      # attention first; experimental separate
    face = pg.locator("article.card[data-module='face_video']")
    assert "0.97 (decision line 0.61)" not in face.inner_text()       # UI v2 (contract section 1): a raw score is shown only under Technical details
    exp = pg.locator("article.card[data-module='generated_video']")
    t = exp.inner_text()
    assert "EXPERIMENTAL" in t and "Experimental signal, not a finding" in t and exp.locator(".keyfig").count() == 0
    visible = pg.inner_text("#report")
    assert "0.6138830927259571" not in visible and "E-TEST" not in visible           # raw float and experiment ids stay in details
    face.locator("details.tech-box > summary").click()
    assert "0.6138830927259571" in face.inner_text() and "0.97 (decision line 0.61)" in face.inner_text()
    exp.locator("details.why-box > summary").click()
    assert exp.locator(".filmstrip img").count() == 8
    lip = pg.locator("article.card[data-module='lip_sync']")
    lip.locator("details.why-box > summary").click()
    assert lip.locator("svg.curve polyline").count() == 1 and lip.locator("svg.curve rect.band").count() == 1
    assert r["summary"]["headline"] in pg.inner_text("#summary")
    for pat in FORBIDDEN:
        assert not re.search(pat, pg.inner_text("body").lower().replace(r["statement"].lower(), "")), pat


def test_keyboard_reaches_the_file_input_and_every_disclosure(open_page):
    pg = open_page()
    assert pg.evaluate("r => window.dashboard.renderReport(r, true)", _real_shaped_report()) is True
    pg.focus("#analyse-file")
    assert pg.evaluate("() => document.activeElement.id") == "analyse-file"
    n = pg.locator("#report details > summary").count()
    reached = set()
    for _ in range(200):
        pg.keyboard.press("Tab")
        info = pg.evaluate("() => { const a = document.activeElement; return [a.tagName, a.closest('#report') ? 'report' : 'page', a.closest('article') ? a.closest('article').dataset.module : '', a.textContent.slice(0, 40)].join('|'); }")
        if info.startswith("SUMMARY"):
            reached.add(info)
    in_report = {x for x in reached if x.startswith("SUMMARY|report|")}
    assert len(in_report) == n and n >= 7              # every disclosure of the report is reachable by Tab
    assert any("Which checks run" in x for x in reached - in_report)   # and the upload panel's list of checks
    pg.focus("#report article.card[data-module='face_video'] details.why-box > summary")
    pg.keyboard.press("Enter")
    assert pg.evaluate("() => document.activeElement.parentElement.open") is True


def test_glance_table_lists_every_check_and_links_to_its_card(open_page):
    pg = open_page()
    rep = _real_shaped_report()
    assert pg.evaluate("r => window.dashboard.renderReport(r, true)", rep) is True
    assert pg.locator(".glance-t tbody tr").count() == len(rep["modules"])
    for m in rep["modules"]:
        row = pg.locator(f".glance-t tr[data-module='{m['module_id']}']")
        href = row.locator("a").get_attribute("href")
        assert href == "#check-" + m["module_id"] and pg.locator(href).count() == 1
        assert m["explanation"]["headline"] in row.inner_text()
    exp = pg.locator(".glance-t tr[data-module='generated_video']")
    assert "EXPERIMENTAL" in exp.inner_text() and not exp.locator(".keyfig").count()


def test_explanation_headings_follow_the_server_parts(open_page):
    pg = open_page()
    assert pg.evaluate("r => window.dashboard.renderReport(r, true)", _real_shaped_report()) is True
    card = pg.locator("article.card[data-module='face_video']")
    card.locator("details.why-box > summary").click()
    heads = card.locator(".why-part h4").all_inner_texts()
    assert heads[:2] == ["What we observed", "Why this was reported"] or "Why this was reported" in heads
    order = [h for h in ["What we did", "What we examined", "What we observed", "Why this was reported", "What it means",
                         "What it does not establish", "Limitations"] if h in heads]
    assert heads == order                                       # parts appear in the fixed order, each once


def test_landing_explains_what_happens_and_choosing_a_video_lists_its_checks(open_page):
    pg = open_page()
    assert pg.locator("ol.how li").count() == 3 and "no overall score" in pg.inner_text("ol.how").lower()
    pg.locator("#which-checks > summary").click()
    from dfd.platform.explain import APPLIES_TO
    assert pg.locator("#which-checks-body tbody tr").count() == len(APPLIES_TO) == 8       # 8 since D-083 (generated_image)
    pg.set_input_files("#analyse-file", {"name": "clip.mp4", "mimeType": "video/mp4", "buffer": b"x" * 2048})
    assert pg.is_visible("#will-run")
    items = pg.locator("#will-run li").all_inner_texts()
    assert "Face manipulation analysis" in items and "Image analysis" not in items and len(items) == 6
    pg.set_input_files("#analyse-file", {"name": "pic.png", "mimeType": "image/png", "buffer": b"x" * 64})
    items = pg.locator("#will-run li").all_inner_texts()
    assert "Image analysis" in items and "Synthetic speech analysis" not in items
    assert "AI-generated image, whole image (experimental)" in items and "AI-generated video (experimental research)" not in items


def test_a_slow_earlier_scenario_load_cannot_replace_a_newer_one(open_page):
    """Regression (2026-10-27): a scenario request that answers late (for example the automatic first load) must not overwrite the
    report the user chose afterwards and close its open panels."""
    import time as _t
    pg = open_page()
    def slow(route):
        _t.sleep(1.0)
        route.continue_()
    pg.route("**/api/fixtures/no_face", slow)
    pg.evaluate("() => { window.dashboard.loadFixture('no_face'); window.dashboard.loadFixture('video_nothing_found'); }")
    pg.wait_for_function("() => document.querySelector('#meta').textContent.includes('mock-video_nothing_found')")
    pg.evaluate("() => document.querySelectorAll('#report details').forEach(d => { d.open = true; })")
    pg.wait_for_timeout(1500)                                     # the slow answer arrives now
    assert "mock-video_nothing_found" in pg.inner_text("#meta")
    assert pg.evaluate("() => [...document.querySelectorAll('#evidence details')].every(d => d.open)")


@pytest.mark.parametrize("width,height", [(1366, 900), (390, 844)])
def test_report_overview_is_not_hidden_under_the_header_after_analysis(open_page, width, height):
    # Regression of 2026-10-28: the page scrolled the overview to a fixed 84 px offset while the sticky header was
    # 129 px (desktop, two rows) or 213 px (phone) tall, so the file name of the report sat under the header.
    pg = open_page("?poll_interval_ms=150&poll_timeout_ms=20000")
    pg.set_viewport_size({"width": width, "height": height})
    _upload(pg, "real.mp4")
    _wait_status(pg, "Done")
    pg.wait_for_timeout(300)
    geo = pg.evaluate("""() => { const h = document.querySelector('header.top'), m = document.querySelector('#meta');
        return {pos: getComputedStyle(h).position, head: h.getBoundingClientRect().bottom, meta: m.getBoundingClientRect().top}; }""")
    if geo["pos"] == "sticky":
        assert geo["meta"] >= geo["head"], geo
    else:
        assert width <= 720, geo                       # only the narrow layout un-pins the header
        assert geo["meta"] >= 0, geo
