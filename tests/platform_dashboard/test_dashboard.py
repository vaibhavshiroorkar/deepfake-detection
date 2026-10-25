"""Mock dashboard tests: routes, fixtures, static content rules, and (if a browser is available) rendering."""
import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from dfd.platform import schema

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "src" / "dfd" / "platform" / "dashboard"
FIX = ROOT / "src" / "dfd" / "platform" / "fixtures"

_spec = importlib.util.spec_from_file_location("dash_serve", DASH / "serve.py")
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)

NAMES = sorted(p.stem for p in FIX.glob("*.json"))
FORBIDDEN = [r"likely real", r"\bgenuine\b", r"\bauthentic\b", r"\bsafe\b", r"\bverified real\b", r"\bis real\b", r"\bfake probability\b"]
NEEDED_LABELS = ("Indicates manipulation", "No indication within tested scope", "Inconclusive", "Potential mismatch")


@pytest.fixture(scope="module")
def client():
    return serve.create_app().test_client()


def _load(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def test_all_scenarios_have_fixtures():
    from dfd.platform.mocks import SCENARIOS
    assert sorted(NAMES) == sorted(SCENARIOS)


def test_index_and_assets_served(client):
    for path, ctype in (("/", "text/html"), ("/dashboard.js", "javascript"), ("/dashboard.css", "text/css")):
        r = client.get(path)
        assert r.status_code == 200 and ctype in r.content_type, path
        r.close()


def test_fixture_list_and_each_fixture_served_valid(client):
    assert client.get("/api/fixtures").get_json() == NAMES
    for n in NAMES:
        r = client.get(f"/api/fixtures/{n}")
        assert r.status_code == 200
        rep = r.get_json()
        r.close()
        assert rep == _load(n)
        assert schema.validate_report(rep) == [], n


def test_unknown_and_traversal_rejected(client):
    for bad in ("nope", "..%2Fschema", "%2e%2e/schema.py", "conflicting_modules.json"):
        assert client.get(f"/api/fixtures/{bad}").status_code == 404, bad


def test_no_upload_or_write_routes(client):
    assert client.post("/api/fixtures/no_audio").status_code == 405
    assert client.post("/api/upload").status_code in (404, 405)
    rules = {r.rule for r in serve.create_app().url_map.iter_rules()}
    assert not any("upload" in r or "analy" in r for r in rules)


def test_binds_loopback_only():
    assert serve.HOST == "127.0.0.1"
    src = (DASH / "serve.py").read_text(encoding="utf-8")
    assert "0.0.0.0" not in src and "host=HOST" in src


def test_forbidden_phrases_absent_in_sources():
    for f in ("index.html", "dashboard.js", "dashboard.css", "serve.py"):
        text = (DASH / f).read_text(encoding="utf-8").lower()
        for pat in FORBIDDEN:
            assert not re.search(pat, text), f"{f}: {pat}"


def test_no_em_dash_and_no_external_resources():
    for f in ("index.html", "dashboard.js", "dashboard.css", "serve.py"):
        text = (DASH / f).read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text, f
    for f in ("index.html", "dashboard.js", "dashboard.css"):
        text = (DASH / f).read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f"external URL in {f}"


def test_js_uses_textcontent_and_no_html_injection():
    js = (DASH / "dashboard.js").read_text(encoding="utf-8")
    assert "textContent" in js
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert bad not in js, bad


def test_js_has_all_required_labels_and_sections():
    js = (DASH / "dashboard.js").read_text(encoding="utf-8")
    for s in NEEDED_LABELS + ("Authenticity Analysis", "Cross-Modal Consistency", "Provenance and Supporting Checks",
                              "MOCK DATA", "MOCK", "Coverage", "Conflicts between modules", "Media routing",
                              "Tested scope", "Limitations", "Warnings", "Errors", "Calibrated"):
        assert s in js, s
    for st in schema.STATUSES:
        assert st in js
    for fnd in sum((list(v) for v in schema.FINDINGS.values()), []):
        assert fnd in js, fnd       # every finding code has a wording


def test_status_styles_distinct():
    css = (DASH / "dashboard.css").read_text(encoding="utf-8")
    bodies = []
    for st in schema.STATUSES:
        m = re.search(r"\.card\.st-%s|\n\.st-%s\{([^}]*)\}" % (st, st), css)
        mm = re.search(r"\n\.st-%s\{([^}]*)\}" % st, css)
        assert mm, st
        bodies.append(mm.group(1))
    assert len(set(bodies)) == len(bodies)


def test_every_fixture_has_needed_labels_and_banner_trigger():
    seen = set()
    for n in NAMES:
        rep = _load(n)
        assert rep["source_summary"] in ("mock", "mixed"), n
        assert rep["status"]["state"] in ("ok", "partial", "failed")
        for m in rep["modules"]:
            assert m["source"] == "mock"
            if m["status"] == "completed":
                seen.add(m["finding"])
            else:
                assert m["reason"], (n, m["module_id"])
    # all four human wordings are reachable from the fixtures
    for f in ("indicates_manipulation", "no_indication_within_tested_scope", "inconclusive", "potential_mismatch"):
        assert f in seen, f
    statuses = {m["status"] for n in NAMES for m in _load(n)["modules"]}
    assert {"completed", "not_applicable", "unsupported", "failed", "timed_out"} <= statuses


def test_fixtures_carry_no_forbidden_phrases_in_wording():
    # The dashboard never adds claims; check the fixture text it would display as well.
    for n in NAMES:
        text = json.dumps(_load(n)).lower()
        for pat in (r"likely real", r"\bgenuine\b", r"\bsafe\b"):
            assert not re.search(pat, text), (n, pat)


# ---------------------------------------------------------------- browser tests (skipped if no browser)

@pytest.fixture(scope="module")
def page_factory():
    pw_mod = pytest.importorskip("playwright.sync_api")
    pw = pw_mod.sync_playwright().start()
    browser, err = None, None
    for kw in ({}, {"channel": "msedge"}, {"channel": "chrome"}):   # bundled browser, then an installed Edge or Chrome
        try:
            browser = pw.chromium.launch(**kw)
            break
        except Exception as e:  # noqa: BLE001
            err = e
    if browser is None:
        pw.stop()
        pytest.skip(f"no usable browser for playwright: {str(err)[:120]}")
    import threading
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 0, serve.create_app(), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    def make():
        pg = browser.new_page()
        pg.goto(base + "/")
        pg.wait_for_selector("#report:not([hidden])")
        return pg

    yield make
    srv.shutdown()
    browser.close()
    pw.stop()


def test_browser_renders_every_fixture(page_factory):
    pg = page_factory()
    for n in NAMES:
        pg.select_option("#scenario", n)
        pg.wait_for_function("name => document.querySelector('#meta').textContent.includes('mock-' + name)", arg=n)
        assert pg.is_visible("#banner") and "MOCK DATA" in pg.inner_text("#banner")
        rep = _load(n)
        assert pg.locator("article.card").count() == len(rep["modules"])
        assert pg.locator("article.card .tag.mock").count() == len(rep["modules"])
        body = pg.inner_text("body")
        assert rep["statement"] in body
        free = body.replace(rep["statement"], "")      # the fixed statement negates "authentic" by design
        for pat in FORBIDDEN:
            assert not re.search(pat, free.lower()), (n, pat)
        for st in schema.STATUSES:
            assert pg.locator(f"[data-coverage={st}]").count() == 1
    pg.close()


def test_browser_edge_cases_and_escaping(page_factory):
    pg = page_factory()
    rep = _load("module_failure")
    # empty modules
    empty = copy.deepcopy(rep)
    empty["modules"] = []
    empty["sections"] = {s: [] for s in schema.SECTIONS}
    empty["coverage"] = {s: [] for s in schema.STATUSES}
    empty["source_summary"] = "none"
    empty["status"]["state"] = "failed"
    assert pg.evaluate("r => window.dashboard.renderReport(r)", empty) is True
    assert pg.locator("article.card").count() == 0
    assert "failed" in pg.inner_text("#meta")
    assert not pg.is_visible("#banner")
    # malformed reports do not crash
    for bad in (None, 5, [], {}, {"schema_version": "2"}, {**rep, "modules": "x"}, {**rep, "modules": [None]}, {**rep, "schema_version": "1"}):
        assert pg.evaluate("r => window.dashboard.renderReport(r)", bad) is False
        assert pg.is_visible("#error")
    # HTML in report strings stays text
    inj = copy.deepcopy(rep)
    inj["modules"][0]["reason"] = "<img src=x onerror=window.__pwned=1>"
    inj["modules"][1]["tested_scope"] = "<b id=injected>bold</b>"
    inj["modules"][1]["limitations"] = ["<script>window.__pwned=1</script>"]
    inj["request_id"] = "<i id=injected2>x</i>"
    assert pg.evaluate("r => window.dashboard.renderReport(r)", inj) is True
    assert pg.locator("#injected, #injected2, #report img").count() == 0
    assert pg.evaluate("() => window.__pwned === undefined")
    assert "<img src=x" in pg.inner_text("#report")
    pg.close()


def test_browser_negative_finding_wording_and_no_green_check(page_factory):
    pg = page_factory()
    pg.select_option("#scenario", "video_nothing_found")
    pg.wait_for_function("() => document.querySelector('#meta').textContent.includes('video_nothing_found')")
    body = pg.inner_text("body")
    assert "No indication within tested scope" in body
    assert "✓" not in body and "✔" not in body and "OK" not in re.findall(r"\bOK\b", body)
    pg.close()
