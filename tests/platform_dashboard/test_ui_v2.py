"""UI v2 tests (2026-10-27): assessed module cards (D-087), top navigation, Evaluation Lab view against a MOCKED API, and layout at 360 px.

The mock-mode dashboard server is the real one (serve.create_app); the Evaluation Lab API and the two lab front-end files are answered by
Playwright route interception, so these tests do not depend on the lab backend. Every number in the mocked lab payloads is synthetic test
data written in this file. Browser tests skip when no browser is available (same mechanism as test_dashboard.py)."""
import copy
import importlib.util
import json
import re
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "src" / "dfd" / "platform" / "dashboard"
FIX = ROOT / "src" / "dfd" / "platform" / "fixtures"

_spec = importlib.util.spec_from_file_location("dash_serve_v2", DASH / "serve.py")
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)

NAMES = sorted(p.stem for p in FIX.glob("*.json"))
ORDER = ["verdict-label", "headline", "signals", "model-line", "limits-list", "uncert"]
DETECTOR_FLAG_CLASSES = ("v-flag", "v-bad")


def _load(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- static checks

NEW_FILES = ("index.html", "dashboard.js", "dashboard.css", "evallab.js", "evallab.css")


def test_new_front_end_files_follow_the_house_rules():
    forbidden = [r"likely real", r"\bgenuine\b", r"\bauthentic\b", r"\bsafe\b", r"\bverified real\b", r"\bis real\b", r"\bfake probability\b"]
    for f in NEW_FILES:
        text = (DASH / f).read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, f
        assert not re.search(r"https?://", text.replace('"http://www.w3.org/2000/svg"', "")), f
        for pat in forbidden:
            assert not re.search(pat, text.lower()), (f, pat)
    for f in ("dashboard.js", "evallab.js"):
        js = (DASH / f).read_text(encoding="utf-8")
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
            assert bad not in js, (f, bad)
        assert "textContent" in js


def test_no_font_or_script_from_a_cdn_and_system_font_stack():
    css = (DASH / "dashboard.css").read_text(encoding="utf-8")
    assert "@import" not in css and "@font-face" not in css and "url(" not in css
    assert "system-ui" in css


def test_js_has_no_overall_verdict_or_probability_wording():
    js = (DASH / "dashboard.js").read_text(encoding="utf-8") + (DASH / "evallab.js").read_text(encoding="utf-8")
    for pat in (r"overall verdict", r"combined verdict", r"final verdict", r"overall score:"):
        assert not re.search(pat, js.lower()), pat
    # a card never prints "probability" or "confidence" itself; the server's calibration statement is shown verbatim instead
    card_src = js[js.index("function renderAssessedCard"):js.index("function renderCard")]
    assert "probability" not in card_src.lower() and "confidence" not in card_src.lower() and "%" not in card_src


def test_fixtures_used_by_the_ui_tests_carry_assessments():
    for n in NAMES:
        for m in _load(n)["modules"]:
            assert m["assessment"]["verdict_label"], (n, m["module_id"])


# ---------------------------------------------------------------- browser fixtures

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
    srv = make_server("127.0.0.1", 0, serve.create_app(), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield browser, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    browser.close()
    pw.stop()


def _serve_lab_assets(pg):
    """The lab files are answered from disk whatever the server does, so the tests do not depend on agent C's static routes."""
    for name, ctype in (("evallab.js", "application/javascript"), ("evallab.css", "text/css")):
        pg.route(f"**/{name}", lambda route, _req=None, n=name, c=ctype: route.fulfill(status=200, content_type=c, body=(DASH / n).read_bytes()))


@pytest.fixture()
def make_page(env):
    browser, base = env
    pages = []

    def make(width=1280, height=900, query="?poll_interval_ms=100", lab=None, goto=True):
        pg = browser.new_page(viewport={"width": width, "height": height})
        pages.append(pg)
        pg.lab_calls = []
        _serve_lab_assets(pg)
        if lab is not None:
            _mock_lab(pg, lab)
        if goto:
            pg.goto(base + "/" + query)
            pg.wait_for_selector("#report:not([hidden])")
        return pg

    make.base = base
    yield make
    for p in pages:
        p.close()


def _show(pg, name):
    pg.select_option("#scenario", "")
    pg.wait_for_function("() => document.querySelector('#report').hidden")
    pg.select_option("#scenario", name)
    pg.wait_for_function("name => !document.querySelector('#report').hidden && document.querySelector('#meta').textContent.includes('mock-' + name)", arg=name)


# ---------------------------------------------------------------- assessed cards

def test_card_parts_follow_the_contract_order(make_page):
    pg = make_page()
    _show(pg, "video_manipulation_indicated")
    card = pg.locator("article.card[data-module='face_video']")
    idx = card.evaluate("""(c, order) => { const kids = Array.from(c.children);
        return order.map(cls => kids.findIndex(k => k.classList.contains(cls))); }""", ORDER)
    assert all(i >= 0 for i in idx), idx
    assert idx == sorted(idx), idx                      # verdict, headline, evidence, model, limitations, uncertainty
    rep = _load("video_manipulation_indicated")
    face = next(m for m in rep["modules"] if m["module_id"] == "face_video")
    assert card.locator(".verdict-label").inner_text().strip() == face["assessment"]["verdict_label"]
    assert face["explanation"]["headline"] in card.locator(".headline").inner_text()
    ev = card.locator(".signals").inner_text()
    assert all(e["text"] in ev for e in face["assessment"]["evidence"])
    assert face["assessment"]["uncertainty"]["text"] in card.locator(".uncert").inner_text()
    assert face["assessment"]["calibration"]["statement"] in card.locator(".uncert").inner_text()
    assert (face["model"] or {}).get("name", "") in card.locator(".model-line").inner_text()
    assert "Check: " + face["explanation"]["title"] in card.locator(".model-line").inner_text()
    assert all(lim in card.locator(".limits-list").inner_text() for lim in face["limitations"])


def test_every_fixture_card_has_its_tier_class_and_label(make_page):
    pg = make_page()
    for n in NAMES:
        _show(pg, n)
        for m in _load(n)["modules"]:
            a = m["assessment"]
            card = pg.locator(f"article.card[data-module='{m['module_id']}']")
            cls = card.get_attribute("class").split()
            assert "tier-" + a["tier"] in cls and "assessed" in cls, (n, m["module_id"], cls)
            assert a["verdict_label"] in card.locator(".verdict-label").inner_text(), (n, m["module_id"])


def test_experimental_card_never_takes_a_detector_verdict_style(make_page):
    pg = make_page()
    _show(pg, "conflicting_modules")
    exp = pg.locator("article.card[data-module='generated_video']")
    flag = pg.locator("article.card[data-module='face_video']")
    ec, fc = exp.get_attribute("class").split(), flag.get_attribute("class").split()
    assert "tier-experimental" in ec and not any(c in ec for c in DETECTOR_FLAG_CLASSES)
    assert "v-flag" in fc
    assert "Experimental signal observed, no reliable verdict available" in exp.locator(".verdict-label").inner_text()
    style = lambda loc: loc.locator(".verdict-label").evaluate("e => { const s = getComputedStyle(e); return [s.backgroundColor, s.color, s.borderTopStyle]; }")  # noqa: E731
    es, fs = style(exp), style(flag)
    assert es[0] != fs[0] and es[1] != fs[1]
    assert es[2] == "dashed" and fs[2] != "dashed"
    assert exp.locator(".keyfig").count() == 0                       # an experimental signal is never a key figure
    # hostile report: an experimental module whose stored verdict code claims a detector verdict still gets no detector style
    rep = _load("conflicting_modules")
    gen = next(m for m in rep["modules"] if m["module_id"] == "generated_video")
    gen["assessment"]["verdict"] = "possible_face_manipulation"
    pg.evaluate("r => window.dashboard.renderReport(r, false)", rep)
    cls = pg.locator("article.card[data-module='generated_video']").get_attribute("class").split()
    assert "tier-experimental" in cls and not any(c in cls for c in DETECTOR_FLAG_CLASSES)


def test_no_report_level_verdict_anywhere(make_page):
    pg = make_page()
    for n in NAMES:
        _show(pg, n)
        assert pg.locator("#meta .verdict-label, #summary .verdict-label, #meta [data-verdict], #summary [data-verdict]").count() == 0, n
        assert pg.locator(".verdict-label:not(article.card .verdict-label)").count() == 0, n
        n_cards = pg.locator("article.card").count()
        assert pg.locator("article.card .verdict-label").count() == n_cards == len(_load(n)["modules"])
        summary = pg.inner_text("#summary").lower()
        assert "overall verdict" not in summary and "combined" not in summary
        assert "no overall score" in pg.inner_text("#summary").lower() or "no overall" in pg.inner_text("#summary").lower()
        assert _load(n)["statement"] in pg.inner_text("#report")                    # the fixed statement stays


def test_scores_are_never_percentages_probability_or_confidence(make_page):
    pg = make_page()
    for n in ("video_manipulation_indicated", "conflicting_modules", "no_face", "video_nothing_found"):
        _show(pg, n)
        pg.evaluate("() => document.querySelectorAll('article.card details').forEach(d => { d.open = true; })")
        rep = _load(n)
        for m in rep["modules"]:
            a = m["assessment"]
            card = pg.locator(f"article.card[data-module='{m['module_id']}']")
            # the visible part of a card carries no score next to a percent sign, "probability" or "confidence" other than the server's
            # explicit "not a probability" statements (calibration and uncertainty blocks)
            for sel in (".keyfig", ".exp-signal", ".headline", ".verdict-label"):
                for t in card.locator(sel).all_inner_texts():
                    assert not re.search(r"\d\s*%|probab|confiden", t.lower()), (n, m["module_id"], sel, t)
            if isinstance(a["raw"]["score"], (int, float)):
                label = "Ranking score" if a["raw"]["score_kind"] == "ranking_score" else "Model output"
                dts = card.locator("details.tech-box dt").all_inner_texts()
                assert label in dts, (n, m["module_id"], dts)
                assert "Raw score" not in dts
                dd = card.locator(f"details.tech-box dt:text-is('{label}') + dd").inner_text()
                assert re.match(r"^\d+(\.\d+)?", dd) and "%" not in dd and not re.search(r"probab|confiden", dd.lower()), dd
                if a["tier"] == "experimental":       # an experimental number appears only under Technical details
                    visible = card.evaluate("c => Array.from(c.children).filter(k => k.tagName !== 'DETAILS').map(k => k.innerText).join(' ')")
                    assert str(a["raw"]["score"]) not in visible, (n, m["module_id"])


def test_old_reports_without_assessment_still_render(make_page):
    pg = make_page()
    rep = _load("video_manipulation_indicated")
    for m in rep["modules"]:
        m.pop("assessment", None)
    assert pg.evaluate("r => window.dashboard.renderReport(r, false)", rep) is True
    assert pg.locator("article.card").count() == len(rep["modules"])
    assert pg.locator("article.card.assessed").count() == 0
    assert "Raw score" in pg.locator("article.card[data-module='face_video']").evaluate("c => c.querySelector('details.tech-box').textContent")


def test_failed_modules_show_the_failed_label(make_page):
    pg = make_page()
    _show(pg, "module_failure")
    card = pg.locator("article.card[data-module='face_video']")
    assert "Analysis failed" in card.locator(".verdict-label").inner_text()
    assert "v-failed" in card.get_attribute("class").split()


def test_job_error_shows_the_error_assessment_label(make_page):
    pg = make_page(goto=False)
    pg.route("**/api/mode", lambda r: r.fulfill(json={"live": True, "accept": {"image": [".png"], "video": [".mp4"], "audio": [".wav"]}, "max_bytes": 1000000}))
    pg.route("**/api/analyse", lambda r: r.fulfill(status=202, json={"job_id": "j1"}))
    err = {"code": "UNREADABLE_FILE", "message": "The file could not be decoded.",
           "assessment": {"verdict": "analysis_failed", "verdict_label": "Analysis failed"}}
    pg.route("**/api/jobs/j1", lambda r: r.fulfill(json={"state": "failed", "error": err}))
    pg.goto(make_page.base + "/?poll_interval_ms=100")
    pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
    pg.set_input_files("#analyse-file", {"name": "bad.mp4", "mimeType": "video/mp4", "buffer": b"x"})
    pg.click("#analyse-btn")
    pg.wait_for_selector("#live-status.failed")
    t = pg.inner_text("#live-status")
    assert "Analysis failed" in t and "The file could not be decoded." in t
    assert pg.locator("#live-status [data-error-verdict='analysis_failed']").count() == 1
    err2 = dict(err, assessment={"verdict": "unsupported_input", "verdict_label": "Unsupported input"})
    pg.unroute("**/api/jobs/j1")
    pg.route("**/api/jobs/j1", lambda r: r.fulfill(json={"state": "failed", "error": err2}))
    pg.click("#analyse-btn")
    pg.wait_for_function("() => document.querySelector('#live-status').textContent.includes('Unsupported input')")


# ---------------------------------------------------------------- navigation

def test_nav_switches_between_the_two_views(make_page):
    pg = make_page(lab={"catalogue": {"score_files": [], "roles": {}, "default_dirs": ["data/_final/scores"]}})
    assert pg.is_visible("#view-analyse") and not pg.is_visible("#view-lab")
    assert pg.get_attribute("#nav-analyse", "aria-current") == "page" and pg.get_attribute("#nav-lab", "aria-current") is None
    pg.click("#nav-lab")
    pg.wait_for_selector("#lab-h")
    assert pg.is_visible("#view-lab") and not pg.is_visible("#view-analyse")
    assert pg.get_attribute("#nav-lab", "aria-current") == "page" and pg.get_attribute("#nav-analyse", "aria-current") is None
    assert not pg.is_visible("#scenario")                      # the scenario picker belongs to the analysis view
    assert pg.evaluate("() => location.hash") == "#lab"
    pg.click("#nav-analyse")
    assert pg.is_visible("#view-analyse") and not pg.is_visible("#view-lab") and pg.is_visible("#report")
    assert [b.strip() for b in pg.locator("nav.nav button").all_inner_texts()] == ["Analyse a file", "Evaluation Lab"]


def test_direct_link_opens_the_lab(make_page, env):
    pg = make_page(lab={"catalogue": {"score_files": [], "roles": {}, "default_dirs": []}}, goto=False)
    pg.goto(make_page.base + "/#lab")
    pg.wait_for_selector("#lab-h")
    assert pg.is_visible("#view-lab") and not pg.is_visible("#view-analyse")


# ---------------------------------------------------------------- Evaluation Lab against a mocked API

def _block(dataset, split, n_fake, n_real, groups, runs, seeds):
    return {"dataset": dataset, "split": split, "role": "test role text", "n_samples": n_fake + n_real, "n_fake": n_fake, "n_real": n_real,
            "n_groups": groups, "runs": runs, "seeds": seeds}


CAT = {"score_files": [{"id": "sf1", "path": "data/_final/scores/synthetic_scores.csv", "sha256": "ab" * 32, "n_rows": 1234,
                        "iterations": ["I-test"], "modalities": ["video"],
                        "blocks": [_block("synthA", "val", 40, 60, 25, ["r1", "r2"], [1, 2]),
                                   _block("synthA", "dev_x", 30, 70, 20, ["r1", "r2"], [1, 2]),
                                   _block("synthB", "test_x", 50, 50, 30, ["r1", "r2"], [1, 2])]}],
       "roles": {"train": "model development", "val": "validation (threshold and temperature are fitted here)",
                 "dev_x": "development slice (repeatable, not final)", "test_in": "final held-out (sealed)", "test_x": "final held-out (sealed)"},
       "default_dirs": ["data/_final/scores", "data/_evallab/imports"]}

ROC = {"fpr": [0.0, 0.1, 0.3, 0.6, 1.0], "tpr": [0.0, 0.5, 0.8, 0.95, 1.0]}


def _record(rid, split="dev_x", label="synthetic run", state="done", final=False, auc=0.8123, breakdown=True):
    def m(e, lo, hi):
        return {"estimate": e, "low": lo, "high": hi}
    return {"id": rid, "created": "2026-10-27T20:00:00", "state": state, "progress": {"stage": "bootstrap", "done": 2000, "total": 2000},
            "config": {"score_file_id": "sf1", "iteration": "I-test", "modality": "video", "dataset": "synthA" if not final else "synthB", "split": split,
                       "run_ids": ["r1", "r2"], "threshold_source": "val_fit", "n_boot": 2000, "seed": 20261027, "label": label,
                       "purpose": "final_reanalysis" if final else "development"},
            "config_sha256": "cd" * 32, "score_file": {"path": "data/_final/scores/synthetic_scores.csv", "sha256": "ab" * 32, "n_rows": 1234},
            "role": "final held-out (sealed)" if final else "development slice (repeatable, not final)", "purpose": "development",
            "final_guard": {"required": final, "ledger_entries": ["E-TESTLEDGER"] if final else []},
            "threshold": {"value": 0.4321, "source": "val_fit", "provenance": "maximum balanced accuracy on the validation rows of runs r1, r2",
                          "fitted_on_these_rows": False},
            "data": {"n_samples": 100, "n_fake": 30, "n_real": 70, "n_groups": 20, "class_balance": 0.3,
                     "skipped": [{"sample_id": "s-17", "reason": "score is not finite"}], "warnings": ["test warning: small block"],
                     "leakage": {"shared_groups": 0, "duplicate_sample_ids": 0}},
            "metrics": {"auc": m(auc, auc - 0.05, auc + 0.04), "eer": m(0.2111, 0.15, 0.27), "recall": m(0.7, 0.55, 0.82),
                        "specificity": m(0.88, 0.8, 0.94), "balanced_accuracy": m(0.79, 0.7, 0.86)},
            "confusion": {"tp": 21, "fp": 8, "tn": 62, "fn": 9}, "roc": ROC,
            "breakdown": {"column": "generator", "rows": [{"value": "gen-a", "n_fake": 10, "n_real": 0, "auc": None, "fpr": None, "fnr": 0.2},
                                                           {"value": "gen-b", "n_fake": 20, "n_real": 70, "auc": 0.77, "fpr": 0.11, "fnr": 0.35}]} if breakdown else None,
            "runtime_s": 3.4, "error": None, "environment": {"python": "3.12.0", "git_head": "deadbeef", "dirty_entries": 4}}


def _mock_lab(pg, cfg):
    calls = pg.lab_calls

    def handler(route):
        req = route.request
        path = req.url.split("/api/evallab", 1)[1].split("?")[0]
        body = req.post_data_json if req.method == "POST" and req.post_data else None
        calls.append((req.method, path, body))
        if path == "/catalogue":
            return route.fulfill(status=cfg.get("catalogue_status", 200), json=cfg["catalogue"])
        if path == "/runs" and req.method == "POST":
            if "post" in cfg:
                return route.fulfill(status=cfg["post"][0], json=cfg["post"][1])
            return route.fulfill(status=202, json={"id": "run-new"})
        if path == "/runs":
            return route.fulfill(json=cfg.get("runs", []))
        if path.startswith("/runs/"):
            rid = path.split("/")[2]
            recs = cfg.get("records", {})
            seq = recs.get(rid)
            rec = seq.pop(0) if isinstance(seq, list) and len(seq) > 1 else (seq[0] if isinstance(seq, list) else seq)
            return route.fulfill(json=rec) if rec else route.fulfill(status=404, json={"error": {"code": "NOT_FOUND", "message": "no such run"}})
        if path == "/compare":
            return route.fulfill(json=cfg.get("compare", {"available": False, "reason": "not mocked"}))
        return route.fulfill(status=404, json={"error": {"code": "NOT_FOUND", "message": path}})
    pg.route("**/api/evallab/**", handler)


def _open_lab(pg):
    pg.click("#nav-lab")
    pg.wait_for_selector("#lab-files")


def test_lab_empty_catalogue_shows_an_explained_empty_state(make_page):
    pg = make_page(lab={"catalogue": {"score_files": [], "roles": {}, "default_dirs": ["data/_final/scores", "data/_evallab/imports"]}})
    pg.click("#nav-lab")
    pg.wait_for_selector("#lab-empty")
    t = pg.inner_text("#lab-empty")
    assert "No score files found" in t and "data/_final/scores" in t and "no example data is invented" in t
    assert not pg.is_visible("#lab-config") and not pg.is_visible("#lab-results")


def test_lab_catalogue_error_is_shown_with_retry(make_page):
    pg = make_page(lab={"catalogue": {"error": {"code": "NO_BACKEND", "message": "boom"}}, "catalogue_status": 500})
    pg.click("#nav-lab")
    pg.wait_for_selector("#lab-catalogue [role=alert]")
    assert "boom" in pg.inner_text("#lab-catalogue") and pg.locator("#lab-catalogue button:text('Try again')").count() == 1


def test_lab_flow_from_catalogue_to_results_and_export(make_page):
    rec = _record("run-new")
    pg = make_page(lab={"catalogue": CAT, "records": {"run-new": [dict(_record("run-new"), state="running", progress={"stage": "bootstrap", "done": 5, "total": 20}), rec]},
                        "runs": [rec, _record("run-old", label="older run", auc=0.7)],
                        "compare": {"available": False, "reason": "The two runs cover different samples."}})
    _open_lab(pg)
    # step 1: files and blocks come from the API, with role badges
    assert pg.locator("#lab-files .pick").count() == 1 and "synthetic_scores.csv" in pg.inner_text("#lab-files")
    blocks = pg.locator("#lab-blocks .pick")
    assert blocks.count() == 3
    badges = pg.locator("#lab-blocks .role-badge").all_inner_texts()
    assert badges == ["Validation", "Development", "Final held-out (sealed)"]
    assert "(50 fake, 50 real)" in pg.inner_text("#lab-blocks")
    assert not pg.is_visible("#lab-config")
    # step 2: choosing a development block opens the configuration; the final purpose is not offered
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.wait_for_selector("#lab-form")
    assert "Fake (positive)" in pg.inner_text("#lab-block-summary")
    assert pg.locator("#lab-purpose option[value=final_reanalysis]").is_disabled()
    assert pg.input_value("#lab-purpose") == "development"
    assert pg.locator("input[name=lab-run-id]").count() == 2
    # validation: a fixed threshold needs a value and a provenance
    pg.check("input[name=lab-thr][value=fixed]")
    pg.click("#lab-run-btn")
    assert "Enter the fixed threshold" in pg.inner_text("#lab-formerr")
    pg.fill("#lab-thr-value", "0.5")
    pg.click("#lab-run-btn")
    assert "provenance" in pg.inner_text("#lab-formerr")
    pg.fill("#lab-thr-prov", "test provenance text")
    pg.fill("#lab-label", "synthetic run")
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-metrics")
    post = [c for c in pg.lab_calls if c[0] == "POST" and c[1] == "/runs"][0][2]
    assert post["score_file_id"] == "sf1" and post["dataset"] == "synthA" and post["split"] == "dev_x" and post["run_ids"] == ["r1", "r2"]
    assert post["threshold_source"] == "fixed" and post["threshold"] == 0.5 and post["threshold_provenance"] == "test provenance text"
    assert post["purpose"] == "development" and post["n_boot"] == 2000 and post["seed"] == 20261027 and post["iteration"] == "I-test"
    # step 4: everything on screen comes from the record
    rows = pg.locator("#lab-metrics tbody tr")
    assert rows.count() == len(rec["metrics"])
    auc = pg.locator("#lab-metrics tr[data-metric=auc]").inner_text()
    assert "0.8123" in auc and "[0.7623, 0.8523]" in auc
    cells = {c: pg.locator(f"#lab-confusion td[data-cell={c}] b").inner_text() for c in ("tp", "fp", "tn", "fn")}
    assert cells == {"tp": "21", "fp": "8", "tn": "62", "fn": "9"}
    pts = pg.locator("#lab-roc polyline").get_attribute("points").split()
    assert len(pts) == len(ROC["fpr"])
    assert pg.locator("#lab-roc circle.op").count() == 1
    assert pg.get_attribute("#lab-roc", "role") == "img" and "AUC 0.8123" in pg.get_attribute("#lab-roc", "aria-label")
    assert pg.locator("#lab-breakdown tbody tr").count() == 2 and "not defined" in pg.inner_text("#lab-breakdown")
    assert "generator" in pg.inner_text("#lab-breakdown")
    assert "test warning: small block" in pg.inner_text("#lab-warnings")
    pg.locator("#lab-skipped > summary").click()
    assert "s-17: score is not finite" in pg.inner_text("#lab-skipped")
    assert "maximum balanced accuracy" in pg.inner_text("#lab-threshold") and "0.4321" in pg.inner_text("#lab-threshold")
    assert not pg.is_visible("#lab-final-badge")
    assert "does not tell you" in pg.inner_text("#lab-results")
    # step 5: export links and a comparison that is honestly unavailable
    hrefs = pg.eval_on_selector_all("#lab-export a", "els => els.map(a => a.getAttribute('href'))")
    assert hrefs == ["/api/evallab/runs/run-new/export?format=json", "/api/evallab/runs/run-new/export?format=md"]
    pg.wait_for_selector("#lab-baseline")
    assert pg.locator("#lab-baseline option").count() == 1 and "run-old" in pg.inner_text("#lab-baseline")
    pg.click("#lab-cmp-btn")
    pg.wait_for_selector("#lab-unavailable")
    assert "different samples" in pg.inner_text("#lab-unavailable")
    cmp = [c for c in pg.lab_calls if c[1] == "/compare"][0][2]
    assert cmp == {"run_a": "run-new", "run_b": "run-old", "metric": "auc"}


def test_lab_candidates_drive_the_modality_and_run_choice(make_page):
    cat = copy.deepcopy(CAT)
    cat["score_files"][0]["candidates"] = [
        {"iteration": "I-test", "modality": "audio", "run_ids": ["I-test-audio-s0"], "seeds": [0], "blocks": [["synthA", "dev_x"]]},
        {"iteration": "I-test", "modality": "visual", "run_ids": ["I-test-visual-s0", "I-test-visual-s1"], "seeds": [0, 1], "blocks": [["synthA", "dev_x"]]},
        {"iteration": "I-other", "modality": "visual", "run_ids": ["I-other-visual-s0"], "seeds": [0], "blocks": [["synthB", "test_x"]]}]
    pg = make_page(lab={"catalogue": cat})
    _open_lab(pg)
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.wait_for_selector("#lab-form")
    assert pg.locator("#lab-iteration option").all_inner_texts() == ["I-test"]          # the candidate of another block is not offered
    assert pg.locator("#lab-modality option").all_inner_texts() == ["audio", "visual"]
    assert pg.locator("input[name=lab-run-id]").count() == 1
    pg.select_option("#lab-modality", "visual")
    assert pg.eval_on_selector_all("input[name=lab-run-id]", "els => els.map(e => e.value)") == ["I-test-visual-s0", "I-test-visual-s1"]
    assert "seed 1" in pg.inner_text("#lab-runs")


def test_lab_available_comparison_is_shown_with_its_interval(make_page):
    rec = _record("run-new")
    pg = make_page(lab={"catalogue": CAT, "records": {"run-new": [rec]}, "runs": [rec, _record("run-old")],
                        "compare": {"available": True, "metric": "auc", "run_a": "run-new", "run_b": "run-old",
                                    "block": {"dataset": "synthA", "split": "dev_x", "n_samples": 100, "n_groups": 20},
                                    "a": {"estimate": 0.8123, "low": 0.76, "high": 0.85}, "b": {"estimate": 0.7123, "low": 0.66, "high": 0.76},
                                    "difference": {"estimate": 0.1, "low": 0.02, "high": 0.18, "contains_zero": False},
                                    "protocol": {"n_boot": 2000, "level": 0.95, "seed": 1, "rule": "paired cluster bootstrap over group_id"},
                                    "notes": ["synthetic note about thresholds"]}})
    _open_lab(pg)
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-baseline")
    pg.click("#lab-cmp-btn")
    pg.wait_for_selector("#lab-cmp-result")
    t = pg.inner_text("#lab-cmp-result")
    assert "0.1000 [0.0200, 0.1800]" in t and "excludes 0" in t and "includes 0" not in t
    assert "0.8123 [0.7600, 0.8500]" in t and "100 in 20 groups" in t and "interval level 0.95" in t and "synthetic note about thresholds" in t


def test_lab_final_block_is_a_labelled_reanalysis(make_page):
    rec = _record("run-new", split="test_x", final=True)
    pg = make_page(lab={"catalogue": CAT, "records": {"run-new": [rec]}, "runs": [rec]})
    _open_lab(pg)
    pg.locator("#lab-blocks .pick[data-split='test_x']").click()
    pg.wait_for_selector("#lab-form")
    assert "sealed" in pg.inner_text("#lab-block-summary")
    assert pg.input_value("#lab-purpose") == "final_reanalysis"
    assert pg.locator("#lab-purpose option[value=development]").is_disabled()
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-final-badge")
    assert pg.inner_text("#lab-final-badge").strip() == "Final held-out, re-analysis of a logged evaluation"
    assert "E-TESTLEDGER" in pg.inner_text("#lab-results")
    assert [c for c in pg.lab_calls if c[1] == "/runs" and c[0] == "POST"][0][2]["purpose"] == "final_reanalysis"
    pg.wait_for_selector("#lab-cmp-body")
    assert "No other finished run" in pg.inner_text("#lab-cmp-body")


def test_lab_refused_and_failed_runs_are_explained(make_page):
    pg = make_page(lab={"catalogue": CAT, "post": (400, {"error": {"code": "FINAL_NOT_LOGGED", "message": "no ledger entry for this evaluation"}})})
    _open_lab(pg)
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-error")
    t = pg.inner_text("#lab-error")
    assert "not started" in t and "FINAL_NOT_LOGGED" in t and "no ledger entry" in t
    assert pg.is_enabled("#lab-run-btn")
    assert not pg.is_visible("#lab-results")
    failed = dict(_record("run-new"), state="failed", error={"code": "BAD_SCORES", "message": "labels are all one class"})
    pg2 = make_page(lab={"catalogue": CAT, "records": {"run-new": [failed]}})
    _open_lab(pg2)
    pg2.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg2.click("#lab-run-btn")
    pg2.wait_for_selector("#lab-error")
    assert "The run failed" in pg2.inner_text("#lab-error") and "BAD_SCORES" in pg2.inner_text("#lab-error")


def test_lab_progress_is_shown_while_running(make_page):
    running = dict(_record("run-new"), state="running", progress={"stage": "bootstrap", "done": 5, "total": 20})
    pg = make_page(lab={"catalogue": CAT, "records": {"run-new": [running]}}, query="?poll_interval_ms=100&lab_poll_ms=100")
    _open_lab(pg)
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-run [role=progressbar]")
    assert "bootstrap (5 of 20)" in pg.inner_text("#lab-status")
    assert pg.get_attribute("#lab-run [role=progressbar]", "aria-valuenow") == "5"
    assert pg.is_disabled("#lab-run-btn")


# ---------------------------------------------------------------- layout

@pytest.mark.parametrize("width", [360, 390, 768])
def test_no_horizontal_scroll_on_narrow_screens(make_page, width):
    rec = _record("run-new")
    pg = make_page(width=width, lab={"catalogue": CAT, "records": {"run-new": [rec]}, "runs": [rec]})
    over = "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    assert pg.evaluate(over) <= 0, "landing / default scenario"
    for n in NAMES:
        _show(pg, n)
        pg.evaluate("() => document.querySelectorAll('article.card details').forEach(d => { d.open = true; })")
        assert pg.evaluate(over) <= 0, n
    _open_lab(pg)
    assert pg.evaluate(over) <= 0, "lab step 1"
    pg.locator("#lab-blocks .pick[data-split='dev_x']").click()
    pg.click("#lab-run-btn")
    pg.wait_for_selector("#lab-metrics")
    assert pg.evaluate(over) <= 0, "lab results"


def test_focus_is_visible_on_nav_and_buttons(make_page):
    pg = make_page()
    pg.focus("#nav-lab")
    ow = pg.evaluate("() => { const s = getComputedStyle(document.activeElement); return [s.outlineStyle, parseFloat(s.outlineWidth)]; }")
    assert ow[0] != "none" and ow[1] >= 2
    pg.keyboard.press("Enter")
    pg.wait_for_selector("#lab-h")
    assert pg.is_visible("#view-lab")


# ---------------------------------------------------------------- deliverables_v2 website: guided "Evaluate a dataset" flow

WEB = ROOT / "deliverables_v2" / "website" / "static"
BANNER = "Experimental evaluation output. Evaluating a model on a dataset does not approve it for deployment."


def test_website_lab_js_guided_flow_static_checks():
    js = (WEB / "evallab.js").read_text(encoding="utf-8")
    assert chr(0x2014) not in js and chr(0x2014) not in (WEB / "app.css").read_text(encoding="utf-8")
    assert "/api/evallab/groups" in js and "/api/evallab/group-runs" in js and "confirm_full" in js
    low = js.lower()
    for pat in (r"overall verdict", r"combined verdict", r"final verdict", r"overall score:", r"approved"):
        assert not re.search(pat, low), pat
    assert "innerHTML" not in js


def _group(gid, title, kind, n_fake, n_real, units, desc, missing=0):
    return {"id": gid, "title": title, "kind": kind, "description": desc, "overlap": "nested: contains every smaller group", "group_unit": "source video",
            "n_samples": n_fake + n_real, "n_fake": n_fake, "n_real": n_real, "n_groups": units, "n_declared": n_fake + n_real + missing,
            "n_missing_scores": missing}


GROUPS = {"datasets": [
    {"dataset": "dsA", "title": "Dataset A, face-video head", "experiment": "E-033", "iteration": "E-033", "modality": "visual", "head": "T-V", "split": "post_x",
     "status": "experimental evaluation output: it is not an approval for deployment and does not change any product verdict",
     "score_file_id": "sfA", "score_file": "data/_evallab/imports/posteval/A.csv", "threshold": {"value": 0.6139, "provenance": "fixed at the checkpoint of E-033"},
     "score_kind": "ranking score of the head", "eligibility": "videos never read before the freeze", "caveats": ["caveat one about reuse"], "provenance": {},
     "groups": [_group("G1", "Small sample", "general", 87, 103, 60, "A small nested sample for a quick look.", missing=2),
                _group("G2", "Hard cases", "diagnostic", 40, 20, 15, "Only the hard cases.")],
     "full": {"id": "FULL", "kind": "full", "title": "Whole pool", "n_samples": 5000, "n_fake": 2500, "n_real": 2500, "n_groups": 900,
              "warning": "This takes a long time on one GPU.", "estimate": "About twenty minutes."}},
    {"dataset": "dsB", "title": "Dataset B, audio head", "experiment": "E-040", "iteration": "E-040", "modality": "audio", "head": "T-A", "split": "post_x",
     "status": "experimental evaluation output: not an approval", "threshold": {"value": 0.5, "provenance": "p"}, "caveats": [],
     "groups": [_group("G1", "B only group", "general", 5, 5, 4, "Group of dataset B.")], "full": None}], "skipped": [], "dirs": []}


def _group_record(rid="run-g", dataset="dsA", group="G1", state="done", progress=None):
    def m(e, lo, hi):
        return {"estimate": e, "low": lo, "high": hi}
    rec = _record(rid, split="post_x")
    rec["state"] = state
    if progress:
        rec["progress"] = progress
    rec["config"].update({"dataset": dataset, "split": "post_x", "purpose": "post_freeze_evaluation"})
    rec["threshold"] = {"value": 0.6139, "source": "fixed", "provenance": "fixed at the checkpoint of E-033"}
    rec["data"].update({"n_samples": 190, "n_fake": 87, "n_real": 103, "n_groups": 60, "class_balance": 0.519, "group_missing_scores": 2,
                        "warnings": ["group warning: few videos"]})
    rec["metrics"].update({"f1": m(0.7311, 0.65, 0.8), "precision": m(0.6911, 0.6, 0.77), "accuracy": m(0.7511, 0.7, 0.8), "fpr": m(0.12, 0.08, 0.17),
                           "fnr": m(0.3, 0.2, 0.4), "recall": m(0.7, 0.55, 0.82), "specificity": m(0.88, 0.8, 0.94)})
    rec["eval_group"] = {"dataset": dataset, "id": group, "title": "Small sample", "kind": "general", "description": "A small nested sample.",
                         "overlap": "nested", "group_unit": "source video", "n_declared": 192, "experiment": "E-033", "groups_file": "groups/A.json",
                         "groups_sha256": "ef" * 32, "descriptor": "descriptor-text", "provenance": {"k": "v"}, "caveats": ["eval caveat"], "score_kind": "ranking"}
    rec["protocol"] = {"n_boot": 2000, "seed": 20261027}
    return rec


@pytest.fixture(scope="module")
def env_v2(env):
    browser, _ = env
    spec = importlib.util.spec_from_file_location("serve_v2_for_tests", ROOT / "deliverables_v2" / "website" / "serve_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 0, mod.build_app(), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield browser, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _mock_groups(pg, cfg):
    calls = pg.lab_calls
    post = cfg.get("post", (202, {"id": "run-g"}))

    def handler(route):
        req = route.request
        path = req.url.split("/api/evallab", 1)[1].split("?")[0]
        body = req.post_data_json if req.method == "POST" and req.post_data else None
        calls.append((req.method, path, body))
        if path == "/groups":
            return route.fulfill(json=cfg.get("groups", GROUPS))
        if path == "/recommended":
            st, payload = cfg.get("recommended", (200, RECO))
            return route.fulfill(status=st, json=payload)
        if path == "/group-runs" and req.method == "POST":
            return route.fulfill(status=post[0], json=post[1])
        if path.startswith("/runs/"):
            seq = cfg.get("records", {}).get(path.split("/")[2])
            rec = seq.pop(0) if isinstance(seq, list) and len(seq) > 1 else (seq[0] if seq else None)
            return route.fulfill(json=rec) if rec else route.fulfill(status=404, json={"error": {"code": "NOT_FOUND", "message": "no such run"}})
        if path == "/catalogue":
            return route.fulfill(json=CAT)
        if path == "/runs":
            return route.fulfill(json=[])
        return route.fulfill(status=404, json={"error": {"code": "NOT_FOUND", "message": path}})
    pg.route("**/api/evallab/**", handler)


@pytest.fixture()
def v2_page(env_v2):
    browser, base = env_v2
    pages = []

    def make(cfg=None, width=1280):
        pg = browser.new_page(viewport={"width": width, "height": 900})
        pages.append(pg)
        pg.lab_calls = []
        _mock_groups(pg, cfg or {})
        pg.goto(base + "/?poll_interval_ms=100&lab_poll_ms=100")
        pg.click("#nav-lab")
        pg.wait_for_selector("#ev-dataset, #ev-empty")
        return pg

    yield make
    for p in pages:
        p.close()


def test_website_dataset_dropdown_lists_datasets_and_shows_only_its_groups(v2_page):
    pg = v2_page()
    assert pg.locator("#ev-dataset option").all_inner_texts() == ["Choose a dataset ...", "Face video: Dataset A, face-video head",
                                                                   "Speech / audio: Dataset B, audio head"]
    assert pg.locator("#ev-groups").count() == 0
    pg.select_option("#ev-dataset", "dsA")
    assert pg.locator("#ev-groups .ev-group").count() == 2
    assert "B only group" not in pg.inner_text("#ev-area")
    t = pg.inner_text("#ev-summary")
    assert "E-033" in t and "T-V" in t and "post_x" in t and "previously unread data" in t and "videos never read before the freeze" in t and "caveat one" in t
    assert "not an approval for deployment" in pg.inner_text("#ev-status-banner")
    pg.select_option("#ev-dataset", "dsB")
    assert pg.locator("#ev-groups .ev-group").count() == 1 and "B only group" in pg.inner_text("#ev-groups")
    assert "Small sample" not in pg.inner_text("#ev-groups")
    assert pg.locator("#ev-full").count() == 0


def test_website_group_cards_show_counts_before_any_run_and_nothing_posts(v2_page):
    pg = v2_page()
    pg.select_option("#ev-dataset", "dsA")
    g1 = pg.inner_text("#ev-groups .ev-group[data-group=G1]")
    assert "A small nested sample" in g1 and "190 samples: 103 real, 87 fake" in g1 and "60 source videos" in g1 and "nested" in g1 and "General" in g1
    assert "2 declared sample(s) have no stored score" in g1
    g2 = pg.inner_text("#ev-groups .ev-group[data-group=G2]")
    assert "Diagnostic" in g2 and "no stored score" not in g2
    assert pg.is_disabled("#ev-run-btn")
    pg.locator("#ev-groups .ev-group[data-group=G1]").click()
    assert pg.is_enabled("#ev-run-btn")
    assert [c for c in pg.lab_calls if c[0] == "POST"] == []


def test_website_run_posts_dataset_and_group_only_and_shows_results(v2_page):
    running = dict(_group_record(), state="running", progress={"stage": "bootstrap", "done": 5, "total": 20})
    pg = v2_page({"records": {"run-g": [running, _group_record()]}})
    pg.select_option("#ev-dataset", "dsA")
    pg.locator("#ev-groups .ev-group[data-group=G1]").click()
    pg.click("#ev-run-btn")
    pg.wait_for_selector("#ev-out [role=progressbar]")
    assert "bootstrap (5 of 20)" in pg.inner_text("#ev-progress")
    pg.wait_for_selector("#ev-headline")
    posts = [c for c in pg.lab_calls if c[0] == "POST"]
    assert posts == [("POST", "/group-runs", {"dataset": "dsA", "group": "G1"})]
    tiles = {k: pg.inner_text(f"#ev-headline [data-metric={k}]") for k in ("auc", "f1", "precision", "recall", "specificity", "accuracy")}
    assert "0.8123" in tiles["auc"] and "[0.7623, 0.8523]" in tiles["auc"] and "95% interval" in tiles["auc"]
    assert "0.7311" in tiles["f1"] and "0.6911" in tiles["precision"] and "0.7000" in tiles["recall"] and "0.8800" in tiles["specificity"]
    assert "[0.8000, 0.9400]" in tiles["specificity"] and "51.9% fake" in tiles["accuracy"]
    r = pg.inner_text("#ev-results")
    assert BANNER in r and "fixed at the checkpoint of E-033" in r and "0.6139" in r and "190" in r and "E-033" in r and "post_x" in r
    # progressive disclosure: details are closed until opened
    assert not pg.locator("#ev-more").evaluate("d => d.open") and not pg.is_visible("#ev-more table")
    pg.locator("#ev-more > summary").click()
    assert "0.3000" in pg.inner_text("#ev-more") and "Balanced accuracy" in pg.inner_text("#ev-more")
    pg.locator("#ev-cr > summary").click()
    assert pg.locator("#ev-cr td[data-cell=tp] b").inner_text() == "21" and pg.locator("#ev-cr svg polyline").count() == 1
    pg.locator("#ev-warnings > summary").click()
    assert "group warning: few videos" in pg.inner_text("#ev-warnings")
    pg.locator("#ev-prov > summary").click()
    p = pg.inner_text("#ev-prov")
    assert "ef" * 32 in p and "groups/A.json" in p and "descriptor-text" in p and "deadbeef" in p and "2000" in p
    assert pg.eval_on_selector_all("#ev-export a", "els => els.map(a => a.getAttribute('href'))") == [
        "/api/evallab/runs/run-g/export?format=json", "/api/evallab/runs/run-g/export?format=md"]
    for bad in ("approved", "verdict", "overall score"):
        assert bad not in pg.inner_text("#ev-results").lower()


def test_website_whole_pool_needs_confirmation_and_sends_confirm_full(v2_page):
    pg = v2_page({"records": {"run-g": [_group_record(group="FULL")]}})
    pg.select_option("#ev-dataset", "dsA")
    assert pg.locator("#ev-groups .ev-group[data-group=FULL]").count() == 0 and pg.locator("#ev-groups input:checked").count() == 0
    pg.click("#ev-full-btn")
    t = pg.inner_text("#ev-full-confirm")
    assert "5000" in t and "This takes a long time on one GPU." in t and "About twenty minutes." in t
    assert [c for c in pg.lab_calls if c[0] == "POST"] == []
    pg.click("#ev-full-cancel-btn")
    assert pg.locator("#ev-full-confirm").count() == 0 and [c for c in pg.lab_calls if c[0] == "POST"] == []
    pg.click("#ev-full-btn")
    pg.click("#ev-full-confirm-btn")
    pg.wait_for_selector("#ev-headline")
    assert [c for c in pg.lab_calls if c[0] == "POST"] == [("POST", "/group-runs", {"dataset": "dsA", "group": "FULL", "confirm_full": True})]


def test_website_api_error_is_shown_and_run_can_be_retried(v2_page):
    pg = v2_page({"post": (404, {"error": {"code": "UNKNOWN_GROUP", "message": "no such group in this dataset"}})})
    pg.select_option("#ev-dataset", "dsA")
    pg.locator("#ev-groups .ev-group[data-group=G2]").click()
    pg.click("#ev-run-btn")
    pg.wait_for_selector("#ev-error")
    t = pg.inner_text("#ev-error")
    assert "UNKNOWN_GROUP" in t and "no such group in this dataset" in t
    assert pg.is_enabled("#ev-run-btn") and pg.locator("#ev-results").count() == 0


def test_website_empty_groups_state_and_advanced_section_remain(v2_page):
    pg = v2_page({"groups": {"datasets": [], "skipped": [], "dirs": []}})
    assert "No evaluation groups are installed" in pg.inner_text("#ev-empty")
    adv = pg.locator("#lab-advanced")
    assert adv.count() == 1 and not adv.evaluate("d => d.open")
    assert "Advanced: re-analyse stored score files" in pg.inner_text("#lab-advanced > summary")
    pg.locator("#lab-advanced > summary").click()
    pg.wait_for_selector("#lab-files")
    assert pg.locator("#lab-blocks .pick").count() == 3
    assert [c for c in pg.lab_calls if c[0] == "POST"] == []


def test_website_guided_flow_has_no_horizontal_scroll_at_390(v2_page):
    pg = v2_page({"records": {"run-g": [_group_record()]}}, width=390)
    over = "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
    pg.select_option("#ev-dataset", "dsA")
    pg.locator("#ev-groups .ev-group[data-group=G1]").click()
    pg.click("#ev-full-btn")
    assert pg.evaluate(over) <= 0
    pg.click("#ev-run-btn")
    pg.wait_for_selector("#ev-headline")
    pg.evaluate("() => document.querySelectorAll('#ev-results details').forEach(d => { d.open = true; })")
    assert pg.evaluate(over) <= 0


# ---------------------------------------------------------------- recommended demonstrations (2026-10-31)

def _reco(rid, section, modality, label, kind, auc, n_real, n_fake, lab=None, seed0=None):
    def m(e, lo=None, hi=None, derived=False):
        out = {"estimate": e, "low": lo, "high": hi}
        if derived:
            out["derived"] = True
        return out
    sealed = kind == "stored_sealed"
    return {"id": rid, "section": section, "modality": modality, "modality_label": label, "task": "T-X", "title": "Title " + rid,
            "setting": "Setting " + rid, "why": "Why " + rid, "caveats": ["Caveat " + rid], "lab": lab, "kind": kind,
            "kind_text": "Kind text " + rid, "source_file": "data/x/" + rid + ".json", "source_sha256": "cd" * 32, "primary_metric": "auc",
            "available": True, "n_real": n_real, "n_fake": n_fake, "n_samples": n_real + n_fake, "n_groups": 40, "seed0_auc": seed0,
            "n_seeds": 3 if sealed else 1, "interval": "95 percent group bootstrap, 2000 resamples", "experiment": "E-900", "block": rid + " block",
            "threshold": {"values": {"s0": 0.61} if sealed else None, "provenance": "threshold provenance " + rid},
            "derived_note": "derived note" if sealed else None, "lab_run_id": None if sealed else "run-x",
            "metrics": {"auc": m(auc, auc - 0.02, auc + 0.02), "balanced_accuracy": m(0.8111, 0.79, 0.83), "recall": m(0.7222, 0.7, 0.74),
                        "precision": m(0.9333, 0.92, 0.95), "specificity": m(0.9001, derived=sealed), "f1": m(0.8123, derived=sealed),
                        "fpr": m(0.0999, derived=sealed), "accuracy": m(0.7555, derived=sealed), "eer": m(0.2, 0.18, 0.22)}}


RECO = {"status": "Experimental research results: test banner.", "note": "Footer note.", "modalities": {},
        "not_shown": [{"title": "Shortcut set", "reason": "explained by a shortcut"}],
        "entries": [_reco("r1", "recommended", "speech", "Speech / audio", "stored_sealed", 0.9148, 300, 200, seed0=0.9167),
                    _reco("r2", "recommended", "face_video", "Face video", "stored_sealed", 0.8999, 100, 400),
                    _reco("r3", "recommended", "speech", "Speech / audio", "stored_post_freeze", 0.6447, 103, 87, lab={"dataset": "dsA", "group": "G1"}),
                    _reco("o1", "other", "image", "Image", "stored_sealed", 0.9354, 50, 50),
                    _reco("l1", "limitation", "face_video", "Face video", "stored_post_freeze", 0.3803, 20, 20, lab={"dataset": "dsA", "group": "G2"})]}


def test_website_recommended_cards_show_counts_metrics_and_kind_before_the_dataset_list(v2_page):
    pg = v2_page()
    pg.wait_for_selector("#reco-recommended .reco-card")
    order = pg.evaluate("() => [...document.querySelectorAll('#lab-reco, #lab-guided, #lab-advanced')].map(e => e.id)")
    assert order == ["lab-reco", "lab-guided", "lab-advanced"]
    assert pg.locator("#reco-recommended .reco-card").count() == 3 and pg.locator("#reco-limitations .reco-card").count() == 1
    c = pg.inner_text("[data-reco=r1]")
    assert "Speech / audio" in c and "Stored result (sealed test, E-014)" in c and "Setting r1" in c and "Why r1" in c
    assert "300 real · 200 fake · 500 total · 40 independent groups" in c
    assert "0.9148" in c and "[0.8948, 0.9348]" in c and "mean of 3 seeds" in c and "Product checkpoint (seed 0): 0.9167" in c
    assert "Balanced accuracy 0.8111" in c and "Recall 0.7222" in c and "Specificity 0.9001" in c
    assert "Stored result (post-freeze)" in pg.inner_text("[data-reco=r3]")
    assert pg.locator("[data-reco=r1] [data-action=lab]").count() == 0 and pg.locator("[data-reco=r3] [data-action=lab]").count() == 1
    assert "test banner" in pg.inner_text("#reco-status-banner")
    legend = pg.inner_text("#reco-legend")
    assert "Stored evaluation result" in legend and "Lab evaluation" in legend and "model is not run again" in legend and "Experimental" in legend
    assert not pg.locator("#reco-more").evaluate("d => d.open")
    pg.locator("#reco-more > summary").click()
    assert pg.locator("#reco-other .reco-card").count() == 1 and "explained by a shortcut" in pg.inner_text("#reco-notshown")
    # opening the lab and reading recommendations never starts or reads an evaluation run
    assert [c for c in pg.lab_calls if c[0] == "POST" or c[1].startswith("/runs/")] == []


def test_website_recommended_filter_by_modality(v2_page):
    pg = v2_page()
    pg.wait_for_selector("#reco-filter")
    assert pg.locator("#reco-filter [data-filter]").all_inner_texts() == ["All (5)", "Face video (2)", "Speech / audio (2)", "Image (1)"]
    pg.click("#reco-filter [data-filter=speech]")
    assert pg.get_attribute("#reco-filter [data-filter=speech]", "aria-pressed") == "true"
    assert pg.eval_on_selector_all("#lab-reco [data-reco]", "els => els.map(e => e.getAttribute('data-reco'))") == ["r1", "r3"]
    assert "No listed limitation for this modality." in pg.inner_text("#reco-limitations")
    pg.click("#reco-filter [data-filter=all]")
    assert pg.locator("#lab-reco [data-reco]").count() == 5


def test_website_show_stored_result_lists_logged_and_derived_metrics(v2_page):
    pg = v2_page()
    pg.wait_for_selector("[data-reco=r1] [data-action=show]")
    assert pg.locator("#reco-detail-r1").count() == 0
    pg.click("[data-reco=r1] [data-action=show]")
    d = pg.inner_text("#reco-detail-r1")
    assert "Kind text r1" in d and "derived note" in d and "threshold provenance r1" in d and "cd" * 32 in d and "data/x/r1.json" in d and "s0 0.6100" in d
    assert "derived (see note)" in pg.inner_text("#reco-detail-r1 tr[data-metric=specificity]")
    assert "no interval" in pg.inner_text("#reco-detail-r1 tr[data-metric=f1]")
    assert "[0.7000, 0.7400]" in pg.inner_text("#reco-detail-r1 tr[data-metric=recall]")
    assert "40.0% fake" in pg.inner_text("#reco-detail-r1 tr[data-metric=accuracy]")
    assert pg.get_attribute("[data-reco=r1] [data-action=show]", "aria-expanded") == "true"
    pg.click("[data-reco=r1] [data-action=show]")
    assert pg.locator("#reco-detail-r1").count() == 0
    assert [c for c in pg.lab_calls if c[0] == "POST"] == []


def test_website_open_in_lab_preselects_the_group_without_running(v2_page):
    pg = v2_page()
    pg.wait_for_selector("[data-reco=r3] [data-action=lab]")
    pg.click("[data-reco=r3] [data-action=lab]")
    assert pg.input_value("#ev-dataset") == "dsA"
    assert pg.is_checked("#ev-groups .ev-group[data-group=G1] input") and pg.is_enabled("#ev-run-btn")
    assert "190 samples: 103 real, 87 fake" in pg.inner_text("#ev-groups .ev-group[data-group=G1]")
    assert [c for c in pg.lab_calls if c[0] == "POST"] == []


def test_website_recommended_error_does_not_block_the_guided_flow(v2_page):
    pg = v2_page({"recommended": (500, {"error": {"code": "BROKEN", "message": "results unreadable"}})})
    pg.wait_for_selector("#reco-load-error")
    assert "could not be loaded" in pg.inner_text("#reco-load-error")
    pg.select_option("#ev-dataset", "dsA")
    assert pg.locator("#ev-groups .ev-group").count() == 2


@pytest.mark.parametrize("width", [390, 1280])
def test_website_recommended_has_no_horizontal_scroll(v2_page, width):
    pg = v2_page(width=width)
    pg.wait_for_selector("[data-reco=r1] [data-action=show]")
    pg.click("[data-reco=r1] [data-action=show]")
    pg.evaluate("() => document.querySelectorAll('#lab-reco details').forEach(d => { d.open = true; })")
    assert pg.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 0
