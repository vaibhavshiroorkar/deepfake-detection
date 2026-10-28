"""Browser check of the v2 front end (Playwright, Edge channel) against a RUNNING v2 server in live mode.

Start the server first (software check only; the demo inputs are non-sealed or synthetic files, analysing them is not an experiment):
    PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live
Then:
    PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/ui_v2_check.py --base http://127.0.0.1:8012 --out deliverables_v2/website/test_evidence

Drives the NEW page through: landing, content-type selector, image / face-swap video / video with audio / audio / no-face tone uploads,
an unsupported type and a corrupt video (error paths), report rendering with the "Checks by outcome" chart, expand and collapse of details,
the Evaluation Lab (read-only: catalogue and settings form, no run is started) and the About page, a MOCK scenario, at 1366x900 and 390x844
with a horizontal-overflow check, plus a dark-colour-scheme screenshot. Per report it checks on the page and in the API JSON that no
probability or percentage of fake is shown, that the face-video flag reads "Possible manipulation" when flagged, that experimental cards
make no finding, that the chart counts equal the module counts per outcome group, and that the JSON has no FORBIDDEN_REPORT_KEYS.
Writes only under --out (screenshots in screens/, ui_v2_run.json). Does not write anywhere else.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True      # reading the project's schema must not create __pycache__ folders under src/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
INPUTS = ROOT / "data" / "_demo_inputs"

from dfd.platform import schema  # noqa: E402

FACE_FLAG = "Possible manipulation"
EXPERIMENTAL_LABEL = "Experimental signal observed, no reliable verdict available"
VIEWPORTS = [("1366", 1366, 900), ("390", 390, 844)]
# file -> (content-type choice in the selector, kind of outcome expected)
CASES = [
    ("5_face_image_frame_ffpp_val.png", "image", "report"),
    ("1_face_swap_video_ffpp_val.mp4", "video", "report"),
    ("3_talking_head_with_audio_lavdf_train.mp4", "video", "report"),
    ("4_speech_audio_asvspoof_dev.flac", "audio", "report"),
    ("6_synthetic_no_face_tone_audio.mp4", "auto", "report"),
    ("7_unsupported_notes.txt", "auto", "error"),
    ("8_corrupt_video.mp4", "video", "error"),
]

RECORD_FETCH = """() => { window.__posts = []; const f = window.fetch;
  window.fetch = function (url, opt) { try { if (String(url).indexOf('/api/analyse') >= 0) { const keys = [];
      if (opt && opt.body && opt.body.keys) { for (const k of opt.body.keys()) keys.push(k); }
      window.__posts.push({url: String(url), method: (opt && opt.method) || 'GET', keys: keys}); } } catch (e) {} return f.apply(this, arguments); }; }"""

PAGE_PROBE = """() => {
  const vis = (n) => !!(n.offsetWidth || n.offsetHeight || n.getClientRects().length);
  const cards = Array.from(document.querySelectorAll('#evidence article.card')).map(c => {
    const faceBar = Array.from(c.querySelectorAll('.sbar')).filter(b => !b.closest('details'));
    const techBar = Array.from(c.querySelectorAll('details.tech-box .sbar'));
    return {module: c.dataset.module, group: c.dataset.group, tier: c.dataset.tier, verdict: c.dataset.verdict,
      label: (c.querySelector('.verdict-text') || {}).textContent || '', classes: c.className,
      face_bar_count: faceBar.length, face_bar_label: faceBar.length ? faceBar[0].querySelector('.sbar-track').getAttribute('aria-label') : null,
      tech_bar_count: techBar.length, tech_bar_label: techBar.length ? techBar[0].querySelector('.sbar-track').getAttribute('aria-label') : null,
      uncertainty: (c.querySelector('[data-uncertainty]') || {}).textContent || null,
      has_curve: !!c.querySelector('svg.curve'), has_band: !!c.querySelector('svg.curve .band'),
      mock_badge: !!c.querySelector('.badge.mock'), exp_badge: !!c.querySelector('.badge.exp'),
      face_text: Array.from(c.children).filter(k => k.tagName !== 'DETAILS' && !(k.classList.contains('more'))).map(k => k.innerText).join(' | ')};
  });
  const legend = Array.from(document.querySelectorAll('#outcome .legend-item')).map(li => ({group: li.dataset.group,
      label: li.querySelector('.legend-label').textContent, n: Number(li.querySelector('.legend-n').textContent)}));
  const svg = document.querySelector('#outcome svg.donut');
  return {cards, legend, donut_label: svg ? svg.getAttribute('aria-label') : null, donut_role: svg ? svg.getAttribute('role') : null,
    donut_total: svg ? (svg.querySelector('.donut-n') || {}).textContent : null, donut_segments: document.querySelectorAll('#outcome .dseg').length,
    outcome_caption: (document.querySelector('#outcome .outcome-cap') || {}).textContent || null,
    overflow: document.documentElement.scrollWidth > window.innerWidth + 1, scroll_width: document.documentElement.scrollWidth, inner_width: window.innerWidth,
    report_text: document.querySelector('#report') ? document.querySelector('#report').innerText : '',
    verdict_labels_outside_cards: document.querySelectorAll('.verdict-label:not(article.card .verdict-label)').length,
    summary_text: (document.querySelector('#summary') || {}).innerText || '',
    banner_text: (document.querySelector('#banner') || {}).innerText || '', banner_visible: vis(document.querySelector('#banner'))};
}"""


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def text_checks(text: str) -> dict:
    """No percent sign next to a number; every sentence that mentions a probability or a percent chance says it is NOT one."""
    pct = re.findall(r".{0,30}\d\s*%.{0,10}", text)
    neg = re.compile(r"\b(not|no|never|nor|cannot|without)\b|does not|is not", re.I)
    prob = [s for s in sentences(text) if re.search(r"probab|percent|chance|likelihood|confidence", s, re.I)]
    unnegated = [s for s in prob if not neg.search(s)]
    overall = [s for s in sentences(text) if re.search(r"overall (score|verdict|probability)|combined (score|verdict)", s, re.I) and not neg.search(s)]
    return {"percent_sign_with_number": pct, "probability_sentences": prob, "probability_sentences_without_negation": unnegated,
            "overall_wording_without_negation": overall}


def expected_groups(report: dict) -> dict:
    out: dict[str, int] = {}
    for m in report["modules"]:
        g = m["explanation"]["group"]
        out[g] = out.get(g, 0) + 1
    return out


def check_report(probe: dict, report: dict, problems: list[str], tag: str) -> dict:
    info: dict = {}
    # --- API JSON
    info["forbidden_report_keys_present"] = [k for k in schema.FORBIDDEN_REPORT_KEYS if k in report]
    info["validate_report"] = schema.validate_report(report)
    info["calibration_is_probability_values"] = sorted({str(m["assessment"]["calibration"]["is_probability"]) for m in report["modules"]})
    if info["forbidden_report_keys_present"]:
        problems.append(f"{tag}: forbidden keys {info['forbidden_report_keys_present']}")
    if info["validate_report"]:
        problems.append(f"{tag}: validate_report {info['validate_report'][:2]}")
    if info["calibration_is_probability_values"] != ["False"]:
        problems.append(f"{tag}: is_probability not always false")
    # --- donut counts versus the report
    want = expected_groups(report)
    got = {x["group"]: x["n"] for x in probe["legend"]}
    info["chart_counts_page"], info["chart_counts_report"] = got, want
    if got != want:
        problems.append(f"{tag}: chart counts {got} differ from module groups {want}")
    if str(probe["donut_total"]) != str(len(report["modules"])):
        problems.append(f"{tag}: donut centre {probe['donut_total']} != {len(report['modules'])} modules")
    if not probe["outcome_caption"] or "not a probability, a score or an overall verdict" not in probe["outcome_caption"]:
        problems.append(f"{tag}: chart caption missing or changed: {probe['outcome_caption']!r}")
    if probe["donut_role"] != "img" or not probe["donut_label"]:
        problems.append(f"{tag}: donut has no accessible text equivalent")
    # --- text safety
    tc = text_checks(probe["report_text"])
    info["text_checks"] = tc
    if tc["percent_sign_with_number"]:
        problems.append(f"{tag}: percent sign next to a number: {tc['percent_sign_with_number'][:2]}")
    if tc["overall_wording_without_negation"]:
        problems.append(f"{tag}: overall wording: {tc['overall_wording_without_negation'][:2]}")
    if probe["verdict_labels_outside_cards"]:
        problems.append(f"{tag}: report-level verdict label found")
    # --- per-card rules
    by_mod = {c["module"]: c for c in probe["cards"]}
    mods = {m["module_id"]: m for m in report["modules"]}
    info["cards"] = []
    if len(probe["cards"]) != len(report["modules"]):
        problems.append(f"{tag}: {len(probe['cards'])} cards for {len(report['modules'])} modules")
    for mid, m in mods.items():
        c, a = by_mod.get(mid), m["assessment"]
        if c is None:
            problems.append(f"{tag}: no card for {mid}")
            continue
        row = {"module": mid, "status": m["status"], "finding": m["finding"], "group": c["group"], "label": c["label"], "tier": c["tier"],
               "uncertainty": c["uncertainty"], "face_bar": c["face_bar_label"], "tech_bar": c["tech_bar_label"], "curve": c["has_curve"]}
        info["cards"].append(row)
        if c["label"] != a["verdict_label"]:
            problems.append(f"{tag}: {mid} card label {c['label']!r} != assessment {a['verdict_label']!r}")
        if mid == "face_video":
            flagged = m["status"] == "completed" and m["finding"] == "indicates_manipulation"
            if flagged and c["label"] != FACE_FLAG:
                problems.append(f"{tag}: flagged face video reads {c['label']!r}")
            if not flagged and c["label"] == FACE_FLAG:
                problems.append(f"{tag}: unflagged face video reads {FACE_FLAG}")
        if a["tier"] == "experimental":
            if c["label"] != EXPERIMENTAL_LABEL:
                problems.append(f"{tag}: experimental card {mid} reads {c['label']!r}")
            if "v-flag" in c["classes"] or c["group"] == "attention":
                problems.append(f"{tag}: experimental card {mid} styled or grouped as a flag")
            if c["face_bar_count"]:
                problems.append(f"{tag}: experimental card {mid} shows a score bar on its face")
            if re.search(r"possible manipulation|likely|fake", c["face_text"], re.I):
                problems.append(f"{tag}: experimental card {mid} face text claims a finding: {c['face_text'][:120]!r}")
            if c["tech_bar_count"] and "no finding" not in (c["tech_bar_label"] or ""):
                problems.append(f"{tag}: experimental bar label lacks 'no finding'")
        if a["tier"] in ("detector", "limited") and a["raw"]["score"] is not None and a["raw"]["score_kind"] == "model_output":
            if c["face_bar_count"] != 1 or "not a probability" not in (c["face_bar_label"] or ""):
                problems.append(f"{tag}: {mid} detector bar missing or unlabelled: {c['face_bar_label']!r}")
        if a["raw"]["score_kind"] not in ("model_output", "ranking_score") and (c["face_bar_count"] or c["tech_bar_count"]):
            problems.append(f"{tag}: {mid} shows a score bar although its score kind is {a['raw']['score_kind']!r}")
        if m["source"] == "mock" and not c["mock_badge"]:
            problems.append(f"{tag}: mock card {mid} lacks MOCK badge")
        for ev in m.get("evidence", []):
            if ev["kind"] == "offset_curve":
                if not (c["has_curve"] and c["has_band"]):
                    problems.append(f"{tag}: {mid} offset curve or band not drawn")
                info["band"] = ev.get("band")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8012")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="demo input file names to run (default: all cases)")
    a = ap.parse_args()
    out = Path(a.out)
    shots = out / "screens"
    shots.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    problems: list[str] = []
    runs: list[dict] = []
    misc: dict = {}
    console_all: dict[str, list[str]] = {}
    t_start = time.perf_counter()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        misc["browser"] = f"msedge {browser.version}"

        def new_page(w: int, h: int, label: str):
            pg = browser.new_page(viewport={"width": w, "height": h})
            errs: list[str] = []
            console_all[label] = errs
            pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
            pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text) if m.type in ("error", "warning") else None)
            return pg

        def open_page(pg, query: str = "?poll_interval_ms=400"):
            pg.goto(a.base + "/" + query)
            pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'", timeout=30000)

        for vtag, w, h in VIEWPORTS:
            # ---------------------------------------------------------------- landing + selector
            pg = new_page(w, h, f"landing_{vtag}")
            open_page(pg)
            pg.wait_for_function("() => document.querySelectorAll('#scenario option').length > 1")
            land = {"viewport": vtag, "title": pg.title(), "pill": pg.inner_text("#mode-pill"), "accept_text": pg.inner_text("#accept-text"),
                    "nav": pg.locator("nav .nav-link").all_inner_texts(), "selector_options": pg.locator("#type-picker .seg-opt").all_inner_texts(),
                    "overflow": pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1"),
                    "report_visible": pg.is_visible("#report"), "accept_attr_auto": pg.get_attribute("#analyse-file", "accept")}
            if "512" not in land["accept_text"]:
                problems.append(f"landing_{vtag}: 512 MB limit not shown: {land['accept_text']!r}")
            pg.screenshot(path=str(shots / f"landing_{vtag}.png"), full_page=True)
            # selector filters the picker and sets expectations; mismatch is explained and blocks only the button
            pg.check("input[name=content-type][value=video]")
            land["accept_attr_video"] = pg.get_attribute("#analyse-file", "accept")
            land["type_note_video"] = pg.inner_text("#type-note")
            pg.set_input_files("#analyse-file", str(INPUTS / "5_face_image_frame_ffpp_val.png"))
            land["mismatch_text"] = pg.inner_text("#type-mismatch") if pg.is_visible("#type-mismatch") else None
            land["analyse_disabled_on_mismatch"] = pg.is_disabled("#analyse-btn")
            pg.screenshot(path=str(shots / f"selector_mismatch_{vtag}.png"), full_page=True)
            if not land["analyse_disabled_on_mismatch"] or not land["mismatch_text"]:
                problems.append(f"landing_{vtag}: selector mismatch not explained or not blocked")
            pg.check("input[name=content-type][value=image]")
            land["analyse_enabled_after_matching_choice"] = not pg.is_disabled("#analyse-btn")
            land["will_run_image"] = pg.inner_text("#will-run")
            pg.screenshot(path=str(shots / f"selector_image_{vtag}.png"), full_page=True)
            if not land["analyse_enabled_after_matching_choice"]:
                problems.append(f"landing_{vtag}: analyse still blocked after matching choice")
            runs.append({"kind": "landing", **land})
            pg.close()

            # ---------------------------------------------------------------- uploads
            for fname, choice, expect in CASES:
                f = INPUTS / fname
                if a.only and fname not in a.only:
                    continue
                if not f.exists():
                    problems.append(f"missing input {fname}")
                    continue
                tag = f"{vtag}:{fname}"
                pg = new_page(w, h, tag)
                posts: list[dict] = []
                jobs: list[dict] = []

                def on_resp(resp, jobs=jobs):
                    if "/api/jobs/" in resp.url:
                        try:
                            j = resp.json()
                            if j.get("state") in ("done", "failed"):
                                jobs.append(j)
                        except Exception:  # noqa: BLE001
                            pass
                pg.add_init_script("(" + RECORD_FETCH + ")()")
                pg.on("response", on_resp)
                open_page(pg)
                if choice != "auto":
                    pg.check(f"input[name=content-type][value={choice}]")
                pg.set_input_files("#analyse-file", str(f))
                row = {"viewport": vtag, "file": fname, "selector_choice": choice, "analyse_enabled": not pg.is_disabled("#analyse-btn"),
                       "file_card_text": pg.inner_text("#file-card")[:200], "will_run": pg.inner_text("#will-run") if pg.is_visible("#will-run") else None,
                       "preview_elements": pg.locator("#preview img, #preview video, #preview audio").count()}
                if vtag == "1366":
                    pg.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_0_chosen.png"), full_page=True)
                t0 = time.perf_counter()
                pg.click("#analyse-btn")
                seen_progress = []
                try:
                    pg.wait_for_function("() => !document.querySelector('#stage-progress').hidden || !document.querySelector('#report').hidden", timeout=15000)
                    if pg.is_visible("#stage-progress"):
                        seen_progress.append(pg.inner_text("#progress-status"))
                        if vtag == "1366" and fname.startswith("3_"):
                            pg.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_1_progress.png"))
                except Exception:  # noqa: BLE001
                    pass
                pg.wait_for_function("() => !document.querySelector('#report').hidden || !document.querySelector('#progress-error').hidden", timeout=300000)
                row["seconds"] = round(time.perf_counter() - t0, 1)
                posts = pg.evaluate("() => window.__posts || []")
                row["request_form_fields"] = posts
                if not posts or any(x["keys"] != ["file"] or x["method"] != "POST" for x in posts):
                    problems.append(f"{tag}: upload request is not exactly one multipart field 'file': {posts}")
                row["progress_status_seen"] = seen_progress
                if expect == "report":
                    if not pg.is_visible("#report"):
                        problems.append(f"{tag}: expected a report, got {pg.inner_text('#progress-error')[:120]!r}")
                    else:
                        pg.wait_for_timeout(300)
                        probe = pg.evaluate(PAGE_PROBE)
                        rep = jobs[-1]["report"] if jobs and jobs[-1].get("state") == "done" else None
                        row["report_seen"] = rep is not None
                        row["overflow"] = probe["overflow"]
                        if probe["overflow"]:
                            problems.append(f"{tag}: horizontal overflow {probe['scroll_width']} > {probe['inner_width']}")
                        row["banner"] = probe["banner_text"][:80]
                        row["summary"] = probe["summary_text"][:200]
                        if rep is not None:
                            row["checks"] = check_report(probe, rep, problems, tag)
                        else:
                            problems.append(f"{tag}: no report JSON captured")
                        # expand / collapse
                        exp = {}
                        card = pg.locator("#evidence article.card:not(.quiet)").first
                        for summ in ("How this was reached", "Technical details"):
                            s = card.locator(f"summary:text-is('{summ}')").first
                            if s.count():
                                s.scroll_into_view_if_needed()
                                s.click()
                                opened = s.evaluate("e => e.parentElement.open")
                                if vtag == "1366" and fname.startswith("1_") and summ == "How this was reached":
                                    card.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_card_expanded.png"))
                                s.click()
                                closed = not s.evaluate("e => e.parentElement.open")
                                exp[summ] = {"opened": opened, "closed_again": closed}
                                if not (opened and closed):
                                    problems.append(f"{tag}: {summ} did not expand/collapse")
                        row["expand_collapse"] = exp
                        # navigation away and back keeps the report
                        pg.click("#nav-about")
                        pg.wait_for_selector("#about-checks .about-t", timeout=5000)
                        pg.click("#nav-analyse")
                        row["report_kept_after_navigation"] = pg.is_visible("#report")
                        if not row["report_kept_after_navigation"]:
                            problems.append(f"{tag}: report lost after navigation")
                        pg.evaluate("() => window.scrollTo(0, 0)")
                        pg.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_report.png"), full_page=True)
                        if vtag == "1366" and fname.startswith("1_"):
                            pg.locator("#outcome").screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_donut.png"))
                            pg.emulate_media(color_scheme="dark")
                            pg.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_report_dark.png"), full_page=True)
                            pg.emulate_media(color_scheme="light")
                        if vtag == "1366" and fname.startswith("3_"):
                            pg.locator("article[data-module=lip_sync]").screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_lipsync_card.png"))
                            pg.locator("article[data-module=generated_video] summary:text-is('Technical details')").click()
                            pg.locator("article[data-module=generated_video]").screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_experimental_card_tech_open.png"))
                else:
                    err_visible = pg.is_visible("#progress-error")
                    row["error_text"] = pg.inner_text("#progress-error") if err_visible else None
                    row["error_verdict"] = pg.get_attribute("#progress-error [data-error-verdict]", "data-error-verdict") if err_visible else None
                    row["report_visible"] = pg.is_visible("#report")
                    want_code = "UNSUPPORTED_MEDIA_TYPE" if fname.startswith("7_") else "UNREADABLE_FILE"
                    if not err_visible or want_code not in (row["error_text"] or ""):
                        problems.append(f"{tag}: expected {want_code}, got {row['error_text']!r}")
                    if row["report_visible"]:
                        problems.append(f"{tag}: a report is shown next to a failed analysis")
                    row["overflow"] = pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
                    if row["overflow"]:
                        problems.append(f"{tag}: horizontal overflow on error page")
                    pg.screenshot(path=str(shots / f"{Path(fname).stem}_{vtag}_error.png"), full_page=True)
                    # recovery: the button returns to the upload stage
                    pg.click("#progress-back")
                    row["back_to_upload"] = pg.is_visible("#stage-upload")
                    if not row["back_to_upload"]:
                        problems.append(f"{tag}: could not return to the upload stage")
                runs.append({"kind": "upload", **row})
                print(f"{tag}: {expect} in {row['seconds']} s, console errors: {len(console_all[tag])}")
                pg.close()

            # ---------------------------------------------------------------- Evaluation Lab (read only) and About
            pg = new_page(w, h, f"lab_about_{vtag}")
            open_page(pg)
            lab: dict = {"viewport": vtag}
            pg.click("#nav-lab")
            pg.wait_for_selector("#lab-files .pick, #lab-empty, #lab-root .error-box", timeout=30000)
            lab["lab_visible"] = pg.is_visible("#view-lab")
            lab["lab_hero"] = pg.inner_text("#lab-root .lab-hero")[:160]
            lab["lab_has_catalogue"] = pg.locator("#lab-files .pick").count()
            if lab["lab_has_catalogue"]:
                dev = pg.locator("#lab-blocks .pick:not([data-split^='test'])").first
                if dev.count():
                    dev.click()
                    pg.wait_for_selector("#lab-form", timeout=5000)
                    lab["lab_form_shown"] = True
            lab["lab_overflow"] = pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
            pg.screenshot(path=str(shots / f"evaluation_lab_{vtag}.png"), full_page=True)
            if lab["lab_overflow"]:
                problems.append(f"lab_{vtag}: horizontal overflow")
            pg.click("#nav-about")
            pg.wait_for_selector("#about-checks .about-t")
            lab["about_rows"] = pg.locator("#about-checks tbody tr").count()
            lab["about_overflow"] = pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
            pg.screenshot(path=str(shots / f"about_{vtag}.png"), full_page=True)
            if lab["about_overflow"]:
                problems.append(f"about_{vtag}: horizontal overflow")
            pg.click("#nav-analyse")
            lab["back_on_analyse"] = pg.is_visible("#stage-upload")
            if not lab["back_on_analyse"]:
                problems.append(f"lab_{vtag}: could not come back to Analyse")
            # direct link
            pg.goto(a.base + "/#lab")
            pg.wait_for_selector("#view-lab:not([hidden])")
            lab["direct_link_lab"] = True
            runs.append({"kind": "lab_about", **lab})
            pg.close()

            # ---------------------------------------------------------------- progress rendering with a stubbed API (no analysis is run)
            pg = new_page(w, h, f"progress_stub_{vtag}")
            states = [{"state": "queued", "progress": {"module_id": None, "completed": 0, "total": 8}},
                      {"state": "running", "progress": {"module_id": "face_video", "completed": 0, "total": 8}},
                      {"state": "running", "progress": {"module_id": "lip_sync", "completed": 3, "total": 8}},
                      {"state": "running", "progress": {"module_id": "generated_video", "completed": 5, "total": 8}}]
            counter = {"n": 0}

            def jobs_route(route):
                i = min(counter["n"], len(states) - 1)
                counter["n"] += 1
                route.fulfill(status=200, content_type="application/json", body=json.dumps({**states[i], "report": None, "error": None}))
            pg.route("**/api/analyse", lambda route: route.fulfill(status=202, content_type="application/json", body='{"job_id": "stub"}'))
            pg.route("**/api/jobs/stub", jobs_route)
            open_page(pg, "?poll_interval_ms=300")
            pg.set_input_files("#analyse-file", str(INPUTS / "2_original_video_ffpp_val.mp4"))
            pg.click("#analyse-btn")
            pg.wait_for_function("() => document.querySelectorAll('#steps li').length >= 3", timeout=15000)
            prog = {"viewport": vtag, "status": pg.inner_text("#progress-status"), "steps": pg.locator("#steps li").all_inner_texts(),
                    "meter_now": pg.get_attribute("#progress-meter", "aria-valuenow"), "meter_max": pg.get_attribute("#progress-meter", "aria-valuemax"),
                    "overflow": pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")}
            pg.screenshot(path=str(shots / f"progress_stub_{vtag}.png"), full_page=True)
            if prog["overflow"]:
                problems.append(f"progress_stub_{vtag}: horizontal overflow")
            runs.append({"kind": "progress_stub", **prog})
            pg.close()

            # ---------------------------------------------------------------- MOCK scenario
            pg = new_page(w, h, f"mock_{vtag}")
            open_page(pg)
            mock: dict = {"viewport": vtag}
            pg.wait_for_function("() => document.querySelectorAll('#scenario option').length > 1")
            names = pg.eval_on_selector_all("#scenario option", "os => os.map(o => o.value).filter(Boolean)")
            mock["scenarios"] = names
            for name in ("conflicting_modules", "video_manipulation_indicated"):
                if name not in names:
                    continue
                if pg.is_hidden("#scenario"):      # the scenario list lives on the upload stage: go back to it first
                    pg.click("#again-btn")
                pg.select_option("#scenario", name)
                pg.wait_for_selector("#report:not([hidden])")
                pg.wait_for_timeout(300)
                probe = pg.evaluate(PAGE_PROBE)
                rep = pg.evaluate(f"() => fetch('/api/fixtures/{name}').then(r => r.json())")
                info = check_report(probe, rep, problems, f"{vtag}:mock:{name}")
                mock[name] = {"banner": probe["banner_text"], "banner_visible": probe["banner_visible"], "overflow": probe["overflow"],
                              "checks": {k: info[k] for k in ("chart_counts_page", "chart_counts_report", "forbidden_report_keys_present")}}
                if "MOCK DATA" not in probe["banner_text"]:
                    problems.append(f"{vtag}:mock:{name}: no MOCK banner")
                if probe["overflow"]:
                    problems.append(f"{vtag}:mock:{name}: horizontal overflow")
                pg.screenshot(path=str(shots / f"mock_{name}_{vtag}.png"), full_page=True)
            pg.click("#again-btn")
            mock["again_returns_to_upload"] = pg.is_visible("#stage-upload") and not pg.is_visible("#report")
            if not mock["again_returns_to_upload"]:
                problems.append(f"{vtag}: 'Analyse another file' did not return to upload")
            runs.append({"kind": "mock", **mock})
            pg.close()
        browser.close()

    for k, v in console_all.items():
        if v:
            problems.append(f"console/page errors in {k}: {v[:3]}")
    summary = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "base": a.base, "browser": misc.get("browser"),
               "seconds": round(time.perf_counter() - t_start, 1), "problems": problems, "console": console_all, "runs": runs}
    (out / "ui_v2_run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("problems:", len(problems))
    for p in problems:
        print(" -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
