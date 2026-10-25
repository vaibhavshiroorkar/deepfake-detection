"""Local dashboard server (WS2). Serves the dashboard files and the MOCK scenario fixtures. With ``live=True`` it also accepts one upload
and runs the real pipeline (``dfd.platform.pipeline``): POST /api/analyse, GET /api/jobs/<id>. The default is mock only:
no upload route exists unless live mode is requested. This server does not touch the frozen v1 web app (src/dfd/web, port 8001).

Run mock only:  python -m dfd.platform.dashboard.serve [port]            (binds 127.0.0.1 only; default port 8002)
Run live:       python -m dfd.platform.dashboard.serve [port] --live      (loads the frozen checkpoint on first use)
"""
from __future__ import annotations

import sys
import tempfile
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, abort, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures"
HOST = "127.0.0.1"


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


MAX_BYTES = 512 * 1024 * 1024
MAX_JOBS_KEPT = 50
MAX_QUEUED = 4


class JobStore:
    """In-memory jobs, one worker (the GPU is shared with the frozen web app). Uploaded files are deleted as soon as the pipeline returns."""

    def __init__(self, modules_factory: Optional[Callable[[], list]] = None, upload_dir: Optional[Path] = None):
        from dfd.platform import pipeline
        self._pipeline = pipeline
        self._factory = modules_factory or pipeline.default_modules
        self.upload_dir = Path(upload_dir) if upload_dir else Path(tempfile.gettempdir()) / "dfd_platform_uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.upload_dir.glob("*"):      # leftovers of a crashed earlier run
            stale.unlink(missing_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs: "OrderedDict[str, dict]" = OrderedDict()
        self.lock = threading.Lock()

    def pending(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["state"] in ("queued", "running"))

    def submit(self, path: Path, filename: str) -> str:
        jid = uuid.uuid4().hex
        job = {"state": "queued", "progress": {"module_id": None, "completed": 0, "total": 0}, "report": None, "error": None}
        with self.lock:
            self.jobs[jid] = job
            while len(self.jobs) > MAX_JOBS_KEPT:
                self.jobs.popitem(last=False)
        self.pool.submit(self._run, jid, path, filename)
        return jid

    def _run(self, jid: str, path: Path, filename: str) -> None:
        job = self.jobs.get(jid)
        if job is None:
            path.unlink(missing_ok=True)
            return
        job["state"] = "running"
        try:
            mods = self._factory()
            job["progress"]["total"] = len(mods)

            seen = [0]

            def progress(module_id: str) -> None:       # called by the coordinator just before each module starts
                job["progress"]["completed"], job["progress"]["module_id"] = seen[0], module_id
                seen[0] += 1

            report = self._pipeline.analyse_file(path, filename, modules=mods, request_id=jid, progress=progress)
            job["progress"]["completed"] = job["progress"]["total"]
            job["report"], job["state"] = report, "done"
        except Exception as e:  # noqa: BLE001 - the job reports the failure; nothing is raised into the pool
            code = getattr(e, "code", "PIPELINE_FAILED")
            msg = getattr(e, "message", None) or f"{type(e).__name__}: {str(e)[:200]}"
            job["error"], job["state"] = {"code": code, "message": msg}, "failed"
        finally:
            path.unlink(missing_ok=True)
            job["progress"]["module_id"] = None

    def get(self, jid: str) -> Optional[dict]:
        return self.jobs.get(jid)


def create_app(live: bool = False, modules_factory: Optional[Callable[[], list]] = None, upload_dir: Optional[Path] = None,
               max_bytes: int = MAX_BYTES) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes
    store = JobStore(modules_factory, upload_dir) if live else None
    app.config["JOB_STORE"] = store

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

    @app.get("/api/mode")
    def mode():
        return jsonify({"live": live, "mock_only": not live})

    if live:
        @app.post("/api/analyse")
        def analyse():
            f = request.files.get("file")
            if f is None or not f.filename:
                return jsonify({"error": {"code": "NO_FILE", "message": "no file was received"}}), 400
            if store.pending() >= MAX_QUEUED:
                return jsonify({"error": {"code": "QUEUE_FULL", "message": "too many analyses are waiting; try again shortly"}}), 429
            name = Path(f.filename).name
            fd, tmp = tempfile.mkstemp(dir=store.upload_dir, suffix=Path(name).suffix.lower())
            with open(fd, "wb") as out:
                f.save(out)
            return jsonify({"job_id": store.submit(Path(tmp), name)}), 202

        @app.get("/api/jobs/<jid>")
        def job(jid: str):
            j = store.get(jid)
            if j is None:
                return jsonify({"error": {"code": "NO_SUCH_JOB", "message": "unknown job id"}}), 404
            return jsonify(j)

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": {"code": "FILE_TOO_LARGE", "message": "the file exceeds the upload limit"}}), 413

    return app


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    create_app(live="--live" in sys.argv).run(host=HOST, port=int(args[0]) if args else 8002, debug=False, threaded=True)
