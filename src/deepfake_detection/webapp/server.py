"""HTTP routes for the user app, the admin page and the report.

The admin routes are off unless `DDF_ADMIN_TOKEN` is set, and then every admin
request must carry that token. The server binds to the local host by default,
so the token guards against other users of a shared machine, not the internet.
"""

from __future__ import annotations

import functools
import os
import secrets
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import content, results, settings
from .engines import MODELS, Registry

STATIC = Path(__file__).parent / "static"
TEMPLATES = Path(__file__).parent / "templates"
ADMIN_TOKEN_ENV = "DDF_ADMIN_TOKEN"  # noqa: S105 (the variable name, not a token)
_CHUNK = 1 << 20


def _model_cards(enabled: tuple[str, ...]) -> list[dict]:
    cards = []
    for model in MODELS:
        if model.id not in enabled:
            continue
        metrics = results.model_metrics(model.reports)
        cards.append(
            {
                "id": model.id,
                "label": model.label,
                "summary": model.summary,
                "media": list(model.media),
                "metrics": metrics,
            }
        )
    return cards


@functools.cache
def _research() -> dict:
    """Everything the research tabs and the report read. Cached: it only
    changes when a new evaluation is written, and a restart picks that up."""
    return {
        "table": results.metric_table(),
        "roc": results.roc_curves(),
        "seeds": results.seed_spread(),
        "ensemble": results.ensemble_summary(),
        "math": content.MATH,
        "targets": content.TARGETS,
        "prior_work": content.PRIOR_WORK,
        "contributions": content.CONTRIBUTIONS,
        "failures": content.FAILURES,
        "limits": content.LIMITS,
    }


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="Deepfake check", docs_url=None, redoc_url=None)
    app.state.registry = registry or Registry()
    # One clip at a time on the GPU. Two concurrent forward passes through six
    # models would double the memory for no gain in a single-user app.
    inference_lock = threading.Lock()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def admin_guard(token: str | None) -> None:
        expected = os.environ.get(ADMIN_TOKEN_ENV)
        if not expected:
            raise HTTPException(404, "The admin page is off. Set DDF_ADMIN_TOKEN.")
        if token is None or not secrets.compare_digest(token, expected):
            raise HTTPException(401, "Wrong or missing admin token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page() -> FileResponse:
        return FileResponse(STATIC / "admin.html")

    @app.get("/api/config")
    def config() -> dict:
        current = settings.load()
        return {
            "settings": current.as_dict(),
            "models": _model_cards(current.enabled_models),
            "operating_points": results.operating_points(),
            "samples": [
                {key: value for key, value in sample.items() if key != "path"}
                for sample in results.samples().values()
            ],
            "admin_enabled": bool(os.environ.get(ADMIN_TOKEN_ENV)),
        }

    @app.get("/api/research")
    def research() -> dict:
        return _research()

    @app.get("/api/samples/{sample_id}/video")
    def sample_video(sample_id: str) -> Response:
        sample = results.samples().get(sample_id)
        if sample is None:
            raise HTTPException(404, "No such sample on this machine")
        from deepfake_detection.dashboard.lib.media import playable_video_bytes

        payload, _ = playable_video_bytes(sample["path"])
        return Response(payload, media_type="video/mp4")

    def run_analysis(
        path: Path, model: str | None, threshold: float | None, withhold: bool | None
    ) -> dict:
        current = settings.load()
        chosen = model or current.default_model
        if chosen not in current.enabled_models:
            raise HTTPException(400, f"Model {chosen!r} is not enabled")
        # None lets the registry pick the cut-off for the media kind. A locked
        # threshold ignores whatever the user sent.
        cut = None if current.lock_threshold else threshold
        withhold_unfamiliar = current.withhold_unfamiliar
        if withhold is not None and current.allow_unfamiliar_override:
            withhold_unfamiliar = withhold
        try:
            with inference_lock:
                analysis = app.state.registry.analyze(
                    path,
                    model=chosen,
                    threshold=cut,
                    default_threshold=current.default_threshold,
                    withhold_unfamiliar=withhold_unfamiliar,
                )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return asdict(analysis)

    @app.post("/api/analyze")
    def analyze(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        threshold: Annotated[float | None, Form()] = None,
        withhold_unfamiliar: Annotated[bool | None, Form()] = None,
    ) -> dict:
        limit = settings.load().max_upload_mb << 20
        suffix = Path(file.filename or "upload").suffix.lower()
        from deepfake_detection.dashboard.lib import media_kind

        # UPLOAD_SUFFIXES holds extensions without the dot, the form
        # st.file_uploader takes.
        if suffix.lstrip(".") not in media_kind.UPLOAD_SUFFIXES:
            raise HTTPException(415, f"Unsupported file type {suffix or '(none)'}")
        handle, name = tempfile.mkstemp(suffix=suffix, prefix="ddf-upload-")
        path = Path(name)
        try:
            written = 0
            with os.fdopen(handle, "wb") as out:
                while chunk := file.file.read(_CHUNK):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            413, f"The file is over the {limit >> 20} MB limit"
                        )
                    out.write(chunk)
            if written == 0:
                raise HTTPException(400, "The file is empty")
            return run_analysis(path, model, threshold, withhold_unfamiliar)
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/analyze-sample")
    def analyze_sample(
        sample: Annotated[str, Form()],
        model: Annotated[str | None, Form()] = None,
        threshold: Annotated[float | None, Form()] = None,
        withhold_unfamiliar: Annotated[bool | None, Form()] = None,
    ) -> dict:
        found = results.samples().get(sample)
        if found is None:
            raise HTTPException(404, "No such sample on this machine")
        return run_analysis(found["path"], model, threshold, withhold_unfamiliar)

    @app.get("/api/admin/settings")
    def admin_settings(
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict:
        admin_guard(x_admin_token)
        return {
            "settings": settings.load().as_dict(),
            "models": [{"id": m.id, "label": m.label} for m in MODELS],
            "engines": app.state.registry.status(),
        }

    @app.put("/api/admin/settings")
    async def update_settings(
        request: Request,
        x_admin_token: Annotated[str | None, Header()] = None,
    ) -> dict:
        admin_guard(x_admin_token)
        try:
            saved = settings.save(settings.from_mapping(await request.json()))
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        return {"settings": saved.as_dict()}

    @app.get("/report", response_class=HTMLResponse)
    def report() -> str:
        from .report import render_report

        return render_report(_research())

    @app.get("/report.pdf")
    def report_pdf() -> FileResponse:
        from .report import PDF_PATH

        if not PDF_PATH.is_file():
            raise HTTPException(
                404,
                "No PDF built yet. Run `uv run python scripts/build_report.py`.",
            )
        return FileResponse(
            PDF_PATH, media_type="application/pdf", filename="deepfake-report.pdf"
        )

    return app
