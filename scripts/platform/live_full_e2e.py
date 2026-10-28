"""Live end-to-end integration check across the supported modalities and failure cases (NOT an accuracy experiment).

Starts dfd.platform.dashboard.serve.create_app(live=True) in-process with the default module set, uploads every fixture through the real HTTP API,
checks structural expectations (what must be reported, never what the scores should be), and drives the dashboard in a browser for one video.
Writes data/_results/platform/live_full_e2e.json. A pass here shows integration, not detection accuracy.

Run:  PYTHONPATH=src .venv/Scripts/python.exe scripts/platform/live_full_e2e.py [--browser]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_face_e2e as E  # noqa: E402
import live_face_fixtures as F  # noqa: E402


def ff(args, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", *args, str(out)], check=True)
    return out


def edge_fixtures(td: Path) -> list[dict]:
    fx = []

    def add(name, path):
        fx.append({"name": name, "path": Path(path), "split": "synthetic", "parity": []})
    add("synthetic_video_no_audio", ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p"], td / "noaudio.mp4"))
    add("synthetic_video_1s", ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"], td / "short.mp4"))
    add("synthetic_audio_only", ff(["-f", "lavfi", "-i", "sine=frequency=440:duration=3"], td / "tone.wav"))
    (td / "malformed.mp4").write_bytes(b"this is not a video" * 50)
    add("malformed_mp4", td / "malformed.mp4")
    (td / "notes.txt").write_text("plain text", encoding="utf-8")
    add("unsupported_txt", td / "notes.txt")
    return fx


def check(rows: list[dict]) -> list[dict]:
    by = {r["fixture"]: r for r in rows}
    out = []

    def mod(fixture, mid):
        return {m["module_id"]: m for m in by.get(fixture, {}).get("modules", [])}.get(mid)

    def expect(name, ok, detail=""):
        out.append({"check": name, "ok": bool(ok), "detail": detail})
    for name in ("ffpp_fake_video", "ffpp_real_video", "lavdf_face_audio_video", "synthetic_noface_video", "synthetic_video_no_audio"):
        g = mod(name, "generated_video")
        if g is None:
            expect(f"{name}: generated_video present", False, "module missing")
            continue
        expect(f"{name}: generated_video completed live", g["status"] == "completed" and g["source"] == "real", f"{g['status']}/{g['source']}")
        expect(f"{name}: finding is inconclusive (experimental head)", g["finding"] == "inconclusive", str(g["finding"]))
        expect(f"{name}: EXPERIMENTAL_UNVALIDATED warning", "EXPERIMENTAL_UNVALIDATED" in g["warnings"])
        expect(f"{name}: head identity E020-C2-seed0", g["model_version"] == "E020-C2-seed0", str(g["model_version"]))
        expect(f"{name}: score not calibrated", g["calibrated"] is False)
        expect(f"{name}: scope quotes the DF26 chance result", "DF26" in (g["tested_scope"] or "") and "chance" in (g["tested_scope"] or ""))
    g = mod("synthetic_video_1s", "generated_video")
    expect("1 s video: generated_video not_applicable", g is not None and g["status"] == "not_applicable", str(g and g["status"]))
    for name in ("ffpp_fake_frame_image", "asvspoof_fake_audio", "synthetic_audio_only"):
        g = mod(name, "generated_video")
        expect(f"{name}: generated_video not_applicable", g is not None and g["status"] == "not_applicable", str(g and g["status"]))
    na = mod("synthetic_video_no_audio", "speech")
    expect("no-audio video: speech not run", na is not None and na["status"] in ("not_applicable", "skipped"), str(na and na["status"]))
    nf = mod("synthetic_noface_video", "face_video")
    expect("no-face video: face_video does not claim a finding", nf is not None and nf["finding"] in (None, "inconclusive"), str(nf and (nf["status"], nf["finding"])))
    for name in ("malformed_mp4", "unsupported_txt"):
        r = by.get(name, {})
        completed = [m for m in r.get("modules", []) if m["status"] == "completed" and m["finding"] not in (None, "inconclusive", "no_credentials_found")]
        expect(f"{name}: no detector finding on junk input", not completed, f"http {r.get('http_status')} state {r.get('state')}")
    for r in rows:
        txt = json.dumps(r.get("modules", [])).lower().replace("does not show it is genuine", "")
        expect(f"{r['fixture']}: no forbidden wording", not any(w in txt for w in ("likely real", "genuine", "safe")))
        expect(f"{r['fixture']}: no mock result in a live run", "mock" not in (r.get("source_summary") or {}) or not (r.get("source_summary") or {}).get("mock"))
    return out


def browser_generated(base: str, fx: dict, shot: Path) -> dict:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1280, "height": 1800})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(base + "/?poll_interval_ms=500&poll_timeout_ms=900000")
        pg.wait_for_function("() => document.querySelector('#live-panel').dataset.live === 'true'")
        pg.set_input_files("#analyse-file", str(fx["path"]))
        pg.click("#analyse-btn")
        pg.wait_for_function("() => document.querySelector('#live-status').textContent.includes('Done')", timeout=900000)
        card = pg.locator("article.card[data-module='generated_video']")    # redesign (D-076): open the panels that hold the model text
        if card.count():
            for sel in ("details.why-box > summary", "details.tech-box > summary"):
                card.locator(sel).click()
        gv = card.inner_text() if card.count() else ""
        pg.screenshot(path=str(shot), full_page=True)
        b.close()
    return {"fixture": fx["name"], "generated_video_card": gv, "page_errors": errs,
            "shows_experimental_warning": "EXPERIMENTAL_UNVALIDATED" in gv or "Experimental" in gv,
            "shows_not_restrav": "NOT the ReStraV" in gv, "shows_df26": "DF26" in gv}


def main() -> int:
    E.OUT.mkdir(parents=True, exist_ok=True)
    td = Path(tempfile.mkdtemp(prefix="dfd_full_e2e_"))
    fixtures, skipped = F.build(td)
    fixtures += edge_fixtures(td)
    srv, base = E.start_server()
    t0 = time.time()
    rows = [E.run_api(base, fx) for fx in fixtures]
    checks = check(rows)
    res = {"purpose": "integration check of all modalities and failure cases; not an accuracy experiment", "rows": rows, "skipped_fixtures": skipped,
           "checks": checks, "checks_failed": [c for c in checks if not c["ok"]], "wall_s": round(time.time() - t0, 1)}
    if "--browser" in sys.argv:
        vid = next(f for f in fixtures if f["name"] == "ffpp_fake_video")
        res["browser"] = browser_generated(base, vid, E.OUT / "live_full_dashboard.png")
    (E.OUT / "live_full_e2e.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    srv.shutdown()
    print(json.dumps([{k: r.get(k) for k in ("fixture", "http_status", "state", "wall_s")} for r in rows], indent=1))
    print("checks:", len(checks), "failed:", len(res["checks_failed"]))
    for c in res["checks_failed"]:
        print("FAIL", c)
    if "browser" in res:
        print({k: v for k, v in res["browser"].items() if k != "generated_video_card"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
