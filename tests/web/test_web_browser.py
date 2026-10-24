"""Browser tests of the local web page in real Chromium (Playwright) against the mock adapter.

Skipped when Playwright or its Chromium is not installed (`uv pip install playwright` and
`PLAYWRIGHT_BROWSERS_PATH=D:\\cache\\playwright python -m playwright install chromium`).
"""
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and Path(r"D:\cache\playwright").is_dir():
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = r"D:\cache\playwright"

from werkzeug.serving import make_server  # noqa: E402

from dfd.infer.mock_adapter import MockAdapter, _unit  # noqa: E402
from dfd.web.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    app = create_app(MockAdapter(), upload_dir=tmp_path_factory.mktemp("up"))
    srv = make_server("127.0.0.1", 0, app, threaded=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}/"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright_sync.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # Chromium not installed
            pytest.skip(f"Chromium not available: {e}")
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    pg = browser.new_page(viewport={"width": 1100, "height": 800})
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.goto(server)
    yield pg
    pg.close()


def choose(page, path):
    """The real user path: click Choose file, answer the file chooser."""
    with page.expect_file_chooser() as fc:
        page.click("#choose")
    fc.value.set_files(str(path))


def analyse(page, path):
    choose(page, path)
    page.click("#go")
    page.wait_for_selector("#result:not([hidden])", timeout=10000)


def test_layout_zone_is_full_card_width_and_button_sits_under_it(page):
    card, zone, go = (page.locator(s).bounding_box() for s in ("#upload-card", "#drop", "#go"))
    assert zone["width"] >= card["width"] - 40                      # full width apart from the card padding
    assert go["y"] >= zone["y"] + zone["height"]                    # the button is under the zone
    assert abs(go["x"] - zone["x"]) < 2                             # and aligned to its left edge
    assert page.locator("#go").is_disabled()                        # disabled until a file is chosen
    assert page.locator("#nojs").is_hidden() and page.locator("#diag").is_hidden()
    assert page.locator("#mock-banner").is_visible()


def test_layout_on_a_narrow_screen(browser, server):
    pg = browser.new_page(viewport={"width": 380, "height": 700})
    pg.goto(server)
    card, zone = (pg.locator(s).bounding_box() for s in ("#upload-card", "#drop"))
    assert zone["width"] >= card["width"] - 40
    assert pg.evaluate("document.documentElement.scrollWidth") <= 380   # no sideways scrolling
    pg.close()


def test_choosing_a_file_enables_the_button(page, media):
    choose(page, media / "img.png")
    assert page.locator("#go").is_enabled()
    assert "img.png" in page.locator("#file-info").inner_text() and "img.png" in page.locator("#drop-text").inner_text()


def test_dropping_a_file_enables_the_button(page):
    page.evaluate("""() => {
        const dt = new DataTransfer();
        dt.items.add(new File([new Uint8Array([137, 80, 78, 71])], 'dropped.png', {type: 'image/png'}));
        document.getElementById('drop').dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
    }""")
    assert page.locator("#go").is_enabled() and "dropped.png" in page.locator("#file-info").inner_text()


@pytest.mark.parametrize("name,media_type", [("img.png", "image"), ("clip_audio.mp4", "video"), ("speech.wav", "audio")])
def test_upload_each_media_type_shows_progress_banner_and_result(page, media, name, media_type):
    analyse(page, media / name)
    assert page.locator("#result-mock-banner").is_visible() and page.locator("#mock-banner").is_visible()
    assert page.locator("#upload-fill").evaluate("e => e.style.width") == "100%"          # upload progress reached the end
    assert page.locator("#stage-list li.done").count() == 6                               # every stage ticked off
    assert page.locator("#verdict-label").inner_text() in ("Likely fake", "Likely real", "Undetermined")
    assert page.locator("#raw").text_content().count('"inference_source": "mock"') == 1
    assert media_type in page.locator("#details").text_content()
    assert page.locator("#meaning").inner_text().startswith("Calibrated probability")
    assert page.errors == []


def test_video_shows_separate_scores_and_audio_note(page, media):
    analyse(page, media / "clip_silent.mp4")
    rows = page.locator("#modalities tbody tr").all_inner_texts()
    assert any("visual" in r for r in rows) and any("video has no audio track" in r for r in rows)
    assert page.locator("#verdict-label").inner_text() == "Undetermined"


def test_warnings_are_listed(page, media, tmp_path):
    analyse(page, media / "tiny.png")
    assert "LOW_RESOLUTION" in page.locator("#warnings").inner_text()
    page.reload()
    analyse(page, media / "short.wav")
    assert "SHORT_AUDIO" in page.locator("#warnings").inner_text()
    # a file for which the mock simulates MULTIPLE_FACES (deterministic from its hash)
    import hashlib
    for seed in range(200):
        img = tmp_path / "multi.png"
        cv2.imwrite(str(img), np.random.default_rng(seed).integers(0, 255, (96, 128, 3), dtype=np.uint8))
        if _unit(hashlib.sha256(img.read_bytes()).hexdigest(), "faces") < 0.25:
            break
    page.reload()
    analyse(page, img)
    assert "MULTIPLE_FACES" in page.locator("#warnings").inner_text()


def test_an_unsupported_file_shows_the_error_and_no_score(page, media):
    analyse(page, media / "notes.txt")
    assert page.locator("#errors-card").is_visible() and "UNSUPPORTED_MEDIA_TYPE" in page.locator("#errors").inner_text()
    assert page.locator("#verdict-label").inner_text() == "No result" and page.locator("#verdict-score").inner_text() == ""
    assert page.locator("#result-mock-banner").is_visible()


def test_a_file_dropped_outside_the_zone_does_not_leave_the_page(page):
    url = page.url
    page.evaluate("""() => {
        const dt = new DataTransfer();
        dt.items.add(new File(['x'], 'x.png', {type: 'image/png'}));
        const ev = new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true});
        document.body.dispatchEvent(ev);
        window.__prevented = ev.defaultPrevented;
    }""")
    assert page.evaluate("window.__prevented") is True and page.url == url


def test_health_route(page, server):
    r = page.request.get(server + "api/health")
    body = r.json()
    assert r.ok and body["ok"] is True and body["adapter"] == "mock" and len(body["static_version"]) == 10
    assert r.headers["cache-control"] == "no-store"
