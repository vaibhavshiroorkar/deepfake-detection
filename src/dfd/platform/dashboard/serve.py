"""Local mock dashboard server (WS2). Serves the dashboard files and the scenario fixtures. Read-only, no upload.

Run: python -m dfd.platform.dashboard.serve [port]   (binds 127.0.0.1 only; default port 8002)
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, abort, jsonify, send_from_directory

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
HOST = "127.0.0.1"


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(HERE, "index.html")

    @app.get("/dashboard.js")
    def js():
        return send_from_directory(HERE, "dashboard.js")

    @app.get("/dashboard.css")
    def css():
        return send_from_directory(HERE, "dashboard.css")

    @app.get("/api/fixtures")
    def list_fixtures():
        return jsonify(fixture_names())

    @app.get("/api/fixtures/<name>")
    def fixture(name: str):
        if name not in fixture_names():      # allowlist from the directory listing: no path traversal
            abort(404)
        return send_from_directory(FIXTURES, name + ".json", mimetype="application/json")

    return app


if __name__ == "__main__":
    create_app().run(host=HOST, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8002, debug=False)
