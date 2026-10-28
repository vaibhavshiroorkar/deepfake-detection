"""Browser run of the redesigned dashboard on the demonstration inputs, with the real detectors (live mode).

Starts the live dashboard server in-process on a free loopback port, drives Chromium through the upload flow for every file in
data/_demo_inputs/ (built by scripts/platform/make_demo_inputs.py from non-sealed sources only), and records, per file:
the job outcome, every module's status, finding, display group and headline, and whether what the page shows matches the report
JSON (schema 2.1 checker, key figures re-derived from the stored scores). Screenshots: desktop full page, plus a phone-width page
and an expanded-evidence page for selected inputs. Software behaviour only; nothing here is a measurement of detection accuracy.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/ui_demo_run.py --out docs/test_evidence/<date>_redesign
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "data" / "_demo_inputs"


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "changed_or_untracked_entries": len(dirty)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="file names to run (default: all demo inputs)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "screens").mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    from dfd.platform import explain, schema
    from dfd.platform.dashboard.serve import create_app

    files = sorted(p for p in INPUTS.iterdir() if p.is_file()) if INPUTS.is_dir() else []
    if a.only:
        files = [p for p in files if p.name in a.only]
    if not files:
        print(f"no demo inputs in {INPUTS}; run scripts/platform/make_demo_inputs.py first")
        return 2
    srv = make_server("127.0.0.1", 0, create_app(live=True), threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    rows, t_all = [], time.perf_counter()
    with sync_playwright() as pw:
        browser, used = None, None
        for kw in ({}, {"channel": "msedge"}, {"channel": "chrome"}):      # bundled browser, then an installed Edge or Chrome (as the tests do)
            try:
                browser, used = pw.chromium.launch(**kw), kw.get("channel", "playwright-chromium")
                break
            except Exception:  # noqa: BLE001
                continue
        if browser is None:
            print("no usable browser for playwright")
            return 2
        print("browser:", used, browser.version)
        for f in files:
            pg = browser.new_page(viewport={"width": 1280, "height": 900})
            errors, reports = [], []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

            def grab(resp):
                if "/api/jobs/" in resp.url:
                    try:
                        j = resp.json()
                        if j.get("state") == "done":
                            reports.append(j["report"])
                    except Exception:  # noqa: BLE001
                        pass
            pg.on("response", grab)
            pg.goto(base + "/?poll_interval_ms=400")
            pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
            row = {"file": f.name, "report_visible_at_load": pg.is_visible("#report")}
            pg.set_input_files("#analyse-file", str(f))
            row["preview_elements"] = pg.locator("#preview img, #preview video, #preview audio").count()
            t0 = time.perf_counter()
            pg.click("#analyse-btn")
            pg.wait_for_function("() => { const t = document.querySelector('#live-status').textContent;"
                                 " return t.includes('Done') || t.includes('Failed') || t.includes('Refused') || t.includes('Timed out'); }",
                                 timeout=900000)
            row["seconds"] = round(time.perf_counter() - t0, 1)
            row["status_line"] = pg.inner_text("#live-status")
            row["report_visible"] = pg.is_visible("#report")
            row["page_errors"] = errors
            if reports:
                r = reports[-1]
                row["schema_version"] = r["schema_version"]
                row["validate_report"] = schema.validate_report(r)
                row["summary_headline"] = r["summary"]["headline"]
                row["media_facts"] = [x["text"] for x in r["summary"]["media_facts"]]
                mods = []
                for m in r["modules"]:
                    e = m["explanation"]
                    card = pg.locator(f"article.card[data-module='{m['module_id']}']")
                    shown_kf = card.locator(".keyfig").inner_text() if card.locator(".keyfig").count() else None
                    want_kf = None
                    s = m.get("score") or {}
                    if e["key_figure"] and isinstance(s.get("threshold"), (int, float)):
                        want_kf = f"{e['key_figure']['label']}: {explain.fmt_score(s['value'], s['threshold'])} (decision line {explain.fmt(s['threshold'])})"
                    mods.append({"module_id": m["module_id"], "status": m["status"], "finding": m["finding"], "group": e["group"],
                                 "headline": e["headline"], "score_raw": s.get("value"), "threshold_raw": s.get("threshold"),
                                 "key_figure_shown": shown_kf, "key_figure_matches_json": (shown_kf == want_kf) if want_kf else (shown_kf is None or None),
                                 "card_group_attr": card.get_attribute("data-group"), "evidence": [x["kind"] for x in m.get("evidence", [])]})
                row["modules"] = mods
            stem = f.stem
            pg.screenshot(path=str(out / "screens" / f"{stem}_desktop.png"), full_page=True)
            if reports and f.suffix == ".mp4":
                pg.evaluate("() => document.querySelectorAll('#evidence details.why-box').forEach(d => { d.open = true; })")
                pg.screenshot(path=str(out / "screens" / f"{stem}_explanations_open.png"), full_page=True)
                pg.evaluate("() => document.querySelectorAll('#evidence details.why-box').forEach(d => { d.open = false; })")
            pg.set_viewport_size({"width": 390, "height": 844})
            row["horizontal_overflow_at_390px"] = pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
            pg.screenshot(path=str(out / "screens" / f"{stem}_phone.png"), full_page=True)
            rows.append(row)
            print(f"{f.name}: {row['status_line'][:90]!r} {row['seconds']} s, errors={len(errors)}, "
                  f"valid={row.get('validate_report', 'n/a') == []}")
            pg.close()
        pg = browser.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(base + "/")
        pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
        pg.screenshot(path=str(out / "screens" / "0_landing_desktop.png"), full_page=True)
        pg.emulate_media(color_scheme="dark")
        pg.screenshot(path=str(out / "screens" / "0_landing_dark.png"), full_page=True)
        pg.close()
        browser.close()
    srv.shutdown()
    meta = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "browser": used, "git": git_state(), "server": "dfd.platform.dashboard.serve.create_app(live=True), in-process, loopback", "total_s": round(time.perf_counter() - t_all, 1),
            "python": sys.version.split()[0]}
    (out / "ui_redesign_run.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1), encoding="utf-8")
    bad = [r["file"] for r in rows if r["page_errors"] or r.get("validate_report") not in (None, [])]
    print("rows:", len(rows), "with page errors or invalid reports:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
