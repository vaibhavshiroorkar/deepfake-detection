"""The static frontend: every page exported under the Pages base path, calling the PC's API."""

import importlib.util
import re
from pathlib import Path

from fastapi.testclient import TestClient

from deepfake_detection.webapp import site

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_site.py"
API = "https://pc.example.ts.net"


def _export(tmp_path):
    spec = importlib.util.spec_from_file_location("export_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.export(tmp_path, api=API, base="/repo")
    return tmp_path


def test_every_page_is_exported_with_its_assets(tmp_path):
    out = _export(tmp_path)
    for route in site.PAGES:
        page = (
            out / route.strip("/") / "index.html"
            if route != "/"
            else out / "index.html"
        )
        assert page.is_file(), route
    assert (out / "static" / "site" / "site.js").is_file()
    assert (out / "static" / "site" / "site.css").is_file()
    assert (out / ".nojekyll").is_file()


def test_links_stay_inside_the_base_path_and_the_api_is_named(tmp_path):
    out = _export(tmp_path)
    html = (out / "check" / "index.html").read_text(encoding="utf-8")
    assert f'<meta name="ddf-api" content="{API}">' in html
    root_relative = re.findall(r'(?:href|src|action)="/(?!/|repo/)[^"]*"', html)
    assert root_relative == []
    assert 'href="/repo/check/"' in html
    assert 'src="/repo/static/site/site.js?v=30"' in html
    contact = (out / "contact" / "index.html").read_text(encoding="utf-8")
    assert f'action="{API}/contact"' in contact


def test_the_api_answers_the_pages_origin_and_no_other(monkeypatch):
    from deepfake_detection.webapp.server import create_app

    monkeypatch.setenv("DDF_CORS_ORIGINS", "https://owner.github.io")

    class _Registry:
        pass

    client = TestClient(create_app(_Registry()))
    allowed = client.options(
        "/api/check",
        headers={
            "Origin": "https://owner.github.io",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert (
        allowed.headers.get("access-control-allow-origin") == "https://owner.github.io"
    )
    other = client.options(
        "/api/check",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in other.headers
