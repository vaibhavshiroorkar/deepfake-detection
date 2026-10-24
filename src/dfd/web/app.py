"""Local web page (D-031) served by Flask (D-043). One user, one machine, bound to 127.0.0.1.

    PYTHONPATH=src .venv/Scripts/python.exe -m dfd.web --adapter mock

Routes
    GET  /                 the page
    POST /api/analyse      multipart field `file`; returns the D-027 response (synchronous)
    POST /api/jobs         same upload, returns {job_id} at once; the file is scored in a worker thread
    GET  /api/jobs/<id>    {state: queued|running|done, stage, stages_done, response}
The page shows a MOCK banner whenever the adapter's inference_source is not `real` (D-029), and again on every result
whose inference_source is not `real`. Uploaded files are written to a temporary file and deleted after scoring.
"""
from __future__ import annotations

import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from dfd.infer.adapters import Adapter
from dfd.infer.library import analyse_file, failed_response

STATUS_FOR = {"NO_FILE": 400, "EMPTY_FILE": 400, "UNREADABLE_FILE": 400, "UNSUPPORTED_MEDIA_TYPE": 415,
              "FILE_TOO_LARGE": 413, "ADAPTER_FAILED": 500}
MAX_JOBS_KEPT = 50


def create_app(adapter: Adapter, upload_dir: str | Path | None = None, max_bytes: int = 512 * 1024 * 1024) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes
    upload_dir = Path(upload_dir) if upload_dir else Path(tempfile.gettempdir()) / "dfd_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=1)  # one scoring at a time; the GPU is shared with nothing else here
    app.extensions["dfd_jobs"] = jobs

    def source() -> str:
        return adapter.inference_source

    def reject(code: str, message: str, filename: str = ""):
        resp = failed_response(source(), filename, code, message)
        return jsonify(resp), STATUS_FOR[code]

    def save_upload():
        """Return (path, display_name) or a (response, status) error tuple."""
        f = request.files.get("file")
        if f is None or not f.filename:
            return None, reject("NO_FILE", "no file was received; choose an image, video or audio file")
        name = os.path.basename(f.filename.replace("\\", "/"))
        fd, tmp = tempfile.mkstemp(dir=upload_dir, suffix=Path(name).suffix.lower())
        os.close(fd)
        f.save(tmp)
        return (Path(tmp), name), None

    @app.errorhandler(413)
    def too_large(_e):
        return reject("FILE_TOO_LARGE", f"the file is larger than the limit of {max_bytes // (1024 * 1024)} MB")

    @app.get("/")
    def index():
        return render_template("index.html", source=source(), max_mb=max_bytes // (1024 * 1024))

    @app.post("/api/analyse")
    def analyse():
        got, err = save_upload()
        if err is not None:
            return err
        path, name = got
        try:
            resp = analyse_file(path, name, adapter)
        finally:
            path.unlink(missing_ok=True)
        code = resp["errors"][0]["code"] if resp["errors"] else None
        return jsonify(resp), (STATUS_FOR[code] if code else 200)

    def run_job(job_id: str, path: Path, name: str):
        job = jobs[job_id]
        job["state"] = "running"

        def progress(stage: str):
            with lock:
                if job["stage"] and job["stage"] not in job["stages_done"]:
                    job["stages_done"].append(job["stage"])
                job["stage"] = stage
        try:
            resp = analyse_file(path, name, adapter, progress=progress)
        except Exception as e:  # analyse_file handles adapter errors; this is a last resort
            resp = failed_response(source(), name, "ADAPTER_FAILED", f"{type(e).__name__}: {e}")
        finally:
            path.unlink(missing_ok=True)
        with lock:
            job["response"], job["stage"], job["state"] = resp, None, "done"

    @app.post("/api/jobs")
    def create_job():
        got, err = save_upload()
        if err is not None:
            return err
        path, name = got
        job_id = uuid.uuid4().hex
        with lock:
            jobs[job_id] = {"state": "queued", "stage": None, "stages_done": [], "response": None}
            for old in list(jobs)[:-MAX_JOBS_KEPT]:
                jobs.pop(old, None)
        pool.submit(run_job, job_id, path, name)
        return jsonify({"job_id": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify({"error": "unknown job"}), 404
            return jsonify(dict(job))

    return app
