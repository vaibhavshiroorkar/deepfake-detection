"""Live end-to-end integration check of the face-bearing path (not an accuracy experiment; no accuracy claim is made).

Starts dfd.platform.dashboard.serve.create_app(live=True) in-process with the real frozen checkpoint, uploads each fixture through the real
HTTP API, polls the job and writes data/_results/platform/live_face_e2e.json. Scores are compared with the stored scores of the same sample
ids only as an adapter-path consistency check (tolerance 0.02, as in data/_results/freeze/parity.json).

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/live_face_e2e.py [--browser]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_face_fixtures as F  # noqa: E402

OUT = F.ROOT / "data/_results/platform"
TOL = 0.02
FORBIDDEN = [r"likely real", r"\bgenuine\b", r"\bsafe\b"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def start_server():
    from werkzeug.serving import make_server
    from dfd.platform.dashboard import serve
    app = serve.create_app(live=True, upload_dir=Path(tempfile.mkdtemp(prefix="dfd_e2e_up_")))
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def module_row(m: dict) -> dict:
    sc, mo = m.get("score") or {}, m.get("model") or {}
    return {"module_id": m["module_id"], "status": m["status"], "finding": m["finding"], "score_value": sc.get("value"),
            "threshold": sc.get("threshold"), "calibrated": sc.get("calibrated"), "model_version": mo.get("version"), "source": m["source"],
            "warnings": [w.get("code") for w in m.get("warnings", [])], "reason": m.get("reason"), "tested_scope": m.get("tested_scope"),
            "limitations": m.get("limitations")}


def run_api(base: str, fx: dict) -> dict:
    t0 = time.time()
    with open(fx["path"], "rb") as fh:
        r = requests.post(base + "/api/analyse", files={"file": (fx["path"].name, fh)}, timeout=120)
    row = {"fixture": fx["name"], "path": str(fx["path"]), "split": fx["split"], "sha256": sha256(fx["path"]),
           "size_bytes": fx["path"].stat().st_size, "http_status": r.status_code}
    if r.status_code != 202:
        return {**row, "state": "rejected", "error": r.json()}
    jid = r.json()["job_id"]
    j = {"state": "timeout", "error": None, "report": None}
    while time.time() - t0 < 900:
        j = requests.get(f"{base}/api/jobs/{jid}", timeout=30).json()
        if j["state"] in ("done", "failed"):
            break
        time.sleep(0.5)
    row.update(state=j["state"], wall_s=round(time.time() - t0, 2), error=j["error"])
    if j["report"]:
        row["source_summary"] = j["report"]["source_summary"]
        row["input_media_type"] = j["report"]["input"]["media_type"]
        row["modules"] = [module_row(m) for m in j["report"]["modules"]]
    return row


def parity(rows: list[dict], fixtures: list[dict]) -> list[dict]:
    stored, out = F.stored_scores(), []
    by = {r["fixture"]: r for r in rows}
    for fx in fixtures:
        for mod, key in fx["parity"]:
            m = {x["module_id"]: x for x in by.get(fx["name"], {}).get("modules", [])}.get(mod)
            got = m["score_value"] if m and m["status"] == "completed" else None
            src, ref = stored[key]
            d = None if got is None else round(abs(got - ref), 6)
            out.append({"fixture": fx["name"], "module": mod, "stored_key": key, "stored_from": src, "stored_score": ref, "live_score": got,
                        "abs_diff": d, "tolerance": TOL, "consistent": d is not None and d <= TOL})
    return out


def browser_check(base: str, fx: dict) -> dict:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1280, "height": 1600})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(base + "/?poll_interval_ms=500&poll_timeout_ms=900000")
        pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
        pg.set_input_files("#analyse-file", str(fx["path"]))
        pg.click("#analyse-btn")
        pg.wait_for_function("() => document.querySelector('#live-status').textContent.includes('Done')", timeout=900000)
        banner, body, stmt = pg.inner_text("#banner"), pg.inner_text("body"), pg.inner_text("#statement")
        card = pg.locator("article.card[data-module='face_video']")      # redesign (D-076): ids, version and source live under Technical details
        face_status = card.get_attribute("data-status") if card.count() else None
        if card.count():
            card.locator("details.tech-box > summary").click()
        face = card.inner_text() if card.count() else ""
        # the media_metadata limitation says "does not show it is genuine" (a negation); it is the only allowed use of the word
        low = body.lower().replace(stmt.lower(), "").replace("does not show it is genuine", "")
        res = {"banner": banner, "live_banner": "LIVE ANALYSIS" in banner, "mock_banner": "MOCK" in banner, "face_card_text": face, "face_status": face_status,
               "forbidden_hits": [p for p in FORBIDDEN if re.search(p, low)], "page_errors": errs}
        pg.screenshot(path=str(OUT / "live_face_dashboard.png"), full_page=True)
        b.close()
        return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fixtures, skipped = F.build(Path(tempfile.mkdtemp(prefix="dfd_e2e_fx_")))
    srv, base = start_server()
    rows = [run_api(base, fx) for fx in fixtures]
    res = {"purpose": "integration check of the live face-bearing path; not an accuracy experiment", "rows": rows,
           "skipped_fixtures": skipped, "adapter_path_consistency": parity(rows, fixtures)}
    if "--browser" in sys.argv:
        face = next((f for f in fixtures if f["name"] == "ffpp_fake_video"), fixtures[0])
        res["browser"] = {"fixture": face["name"], **browser_check(base, face)}
    (OUT / "live_face_e2e.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    srv.shutdown()
    print(json.dumps([{k: r.get(k) for k in ("fixture", "state", "wall_s")} for r in rows], indent=1))
    print(json.dumps(res["adapter_path_consistency"], indent=1))
    if "browser" in res:
        print({k: v for k, v in res["browser"].items() if k != "face_card_text"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
