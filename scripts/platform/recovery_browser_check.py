"""Recovery browser check 2026-10-28: viewport views after a real analysis, and the Evaluation Lab flow on the real API."""
import json, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server
from dfd.platform.dashboard.serve import create_app
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
INPUT = Path(sys.argv[2])
srv = make_server("127.0.0.1", 0, create_app(live=True), threaded=True)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_port}"
log = {}
with sync_playwright() as pw:
    b = pw.chromium.launch(channel="msedge")
    for w, h, tag in ((1366, 900, "1366"), (390, 844, "390")):
        pg = b.new_page(viewport={"width": w, "height": h}); errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(base + "/")
        pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
        pg.screenshot(path=str(OUT / f"{tag}_v_landing.png"))
        pg.set_input_files("#analyse-file", str(INPUT))
        pg.click("#analyse-btn")
        pg.wait_for_selector("#report:not([hidden])", timeout=180000)
        pg.wait_for_timeout(800)
        pg.screenshot(path=str(OUT / f"{tag}_v_after_analysis.png"))
        # where is the report top relative to the sticky header?
        geo = pg.evaluate("""() => { const h = document.querySelector('header.top').getBoundingClientRect();
            const m = document.querySelector('#meta').getBoundingClientRect();
            return {header_bottom: h.bottom, meta_top: m.top, scrollY: window.scrollY, header_pos: getComputedStyle(document.querySelector('header.top')).position}; }""")
        log[tag + "_geometry_after_analysis"] = geo
        for mid in ("face_video", "generated_video"):
            c = pg.locator(f"article.card[data-module='{mid}']")
            if c.count():
                c.scroll_into_view_if_needed(); pg.wait_for_timeout(200)
                pg.screenshot(path=str(OUT / f"{tag}_v_card_{mid}.png"))
                log[f"{tag}_{mid}_text"] = c.inner_text()[:600]
        # Evaluation Lab against the real local API (development slice only)
        pg.click("#nav-lab"); pg.wait_for_selector("#lab-files .pick")
        pg.screenshot(path=str(OUT / f"{tag}_v_lab_choose.png"))
        pg.locator("#lab-files .pick", has_text="I-4.csv").click()
        pg.locator("#lab-blocks .pick[data-split='dev_x']").first.click()
        pg.wait_for_selector("#lab-form"); pg.fill("#lab-nboot", "300"); pg.fill("#lab-label", "recovery browser check")
        pg.locator("#lab-config").screenshot(path=str(OUT / f"{tag}_lab_config.png"))
        pg.click("#lab-run-btn"); pg.wait_for_selector("#lab-metrics", timeout=120000); pg.wait_for_timeout(600)
        pg.screenshot(path=str(OUT / f"{tag}_lab_results_full.png"), full_page=True)
        log[tag + "_lab_metrics_text"] = pg.locator("#lab-metrics").inner_text()[:500]
        pg.locator("#lab-blocks .pick[data-split='test_x']").first.click(); pg.wait_for_selector("#lab-form")
        pg.locator("#lab-config").screenshot(path=str(OUT / f"{tag}_lab_sealed_config.png"))
        log[tag + "_lab_sealed_config_text"] = pg.locator("#lab-config").inner_text()[:700]
        log[tag + "_overflow"] = pg.evaluate("() => document.documentElement.scrollWidth > window.innerWidth + 1")
        log[tag + "_errors"] = errs
        pg.close()
    b.close()
srv.shutdown()
(OUT / "browser_check.json").write_text(json.dumps(log, indent=1))
print(json.dumps(log, indent=1))
