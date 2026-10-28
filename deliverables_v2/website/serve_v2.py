"""Deliverables v2 front end: same backend, new static pages.

This wrapper builds the unchanged app from ``dfd.platform.dashboard.serve.create_app`` (same /api routes, same live mode, same upload
limit, same job queue, same Evaluation Lab backend) and replaces ONLY the routes that serve static pages. The original page files under
src/dfd/platform/dashboard/ are not read or modified. The old asset paths (/dashboard.js, /dashboard.css, /evallab.js, /evallab.css)
answer 404 here so that nothing of the old page can be mixed with the new one.

Run mock only:  PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py [port]            (default port 8012)
Run live:       PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py [port] --live
Binds 127.0.0.1 only. Ports in use by the project: 8001 frozen v1 app, 8002 original dashboard, 8012 this site.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "src") not in sys.path:           # so the command also works without PYTHONPATH=src
    sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, abort, make_response, send_from_directory  # noqa: E402

from dfd.platform.dashboard.serve import HOST, create_app            # noqa: E402

STATIC = HERE / "static"
DEFAULT_PORT = 8012
NEW_ASSETS = ("app.css", "app.js", "evallab.js")      # the only files served besides index.html (allowlist: no path traversal)
OLD_STATIC_ENDPOINTS = ("js", "css", "evallab_js", "evallab_css")      # endpoint names registered by create_app for the old assets


def _file(name: str):
    resp = make_response(send_from_directory(STATIC, name))
    resp.headers["Cache-Control"] = "no-store"      # a local tool: always serve the file on disk
    return resp


def build_app(live: bool = False, **kwargs) -> Flask:
    app = create_app(live=live, **kwargs)           # every /api route comes from here, unchanged
    app.view_functions["index"] = lambda: _file("index.html")
    for endpoint in OLD_STATIC_ENDPOINTS:
        if endpoint in app.view_functions:
            app.view_functions[endpoint] = lambda: abort(404)

    def asset(name: str):
        if name not in NEW_ASSETS:
            abort(404)
        return _file(name)

    app.add_url_rule("/assets/<name>", endpoint="v2_asset", view_func=asset, methods=["GET"])
    return app


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build_app(live="--live" in sys.argv).run(host=HOST, port=int(args[0]) if args else DEFAULT_PORT, debug=False, threaded=True)
