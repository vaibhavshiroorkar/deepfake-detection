"""Flask blueprint of the Evaluation Lab: /api/evallab/*."""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from dfd.eval.scorefile import FINAL_SPLITS

from . import export as _export
from .common import FULL_GROUP, N_BOOT_DEFAULT, LabError, jsonable
from .compare import compare_runs
from .lab import EvalLab

STATUS = {"UNKNOWN_SCORE_FILE": 404, "NO_SUCH_RUN": 404, "UNKNOWN_DATASET": 404, "UNKNOWN_GROUP": 404}
GROUP_RUN_KEYS = ("dataset", "group", "confirm_full", "n_boot")


def _err(code: str, message: str, status: int | None = None):
    return jsonify({"error": {"code": code, "message": message}}), status or STATUS.get(code, 400)


def _summary(rec: dict) -> dict:
    cfg = rec["config"]
    auc = (rec.get("metrics") or {}).get("auc")
    return {"id": rec["id"], "created": rec["created"], "state": rec["state"], "progress": rec["progress"],
            "label": cfg.get("label", ""), "purpose": rec["purpose"], "role": rec["role"],
            "iteration": cfg["iteration"], "modality": cfg["modality"], "dataset": cfg["dataset"], "split": cfg["split"],
            "run_ids": cfg["run_ids"], "score_file": rec["score_file"]["path"], "eval_group": cfg.get("eval_group"),
            "auc": auc["estimate"] if auc else None, "error": rec.get("error")}


def create_blueprint(lab: EvalLab | None = None, **options) -> Blueprint:
    lab = lab or EvalLab(**options)
    bp = Blueprint("evallab", __name__, url_prefix="/api/evallab")
    bp.lab = lab

    @bp.errorhandler(LabError)
    def lab_error(e: LabError):
        return _err(e.code, e.message)

    @bp.get("/catalogue")
    def catalogue():
        return jsonify(jsonable(lab.catalogue.scan()))

    @bp.post("/runs")
    def post_run():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _err("INVALID_CONFIG", "the request body must be a JSON object")
        rec = lab.runner.submit(body)
        return jsonify(rec), 202

    @bp.get("/groups")
    def groups():
        return jsonify(jsonable(lab.groups.listing()))

    @bp.get("/recommended")
    def recommended():
        """Logged results, read from their result files and displayed as logged: nothing is run or recomputed."""
        return jsonify(jsonable(lab.showcase.listing()))

    @bp.post("/group-runs")
    def post_group_run():
        """Guided run: dataset and group only. The configuration (runs, fixed threshold and its provenance, split,
        purpose) comes from the descriptor, so the client cannot change the threshold."""
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _err("INVALID_REQUEST", "the request body must be a JSON object")
        unknown = sorted(set(body) - set(GROUP_RUN_KEYS))
        if unknown:
            return _err("INVALID_REQUEST", f"unknown keys: {unknown}")
        ds, gid = body.get("dataset"), body.get("group")
        if not isinstance(ds, str) or not isinstance(gid, str):
            return _err("INVALID_REQUEST", "dataset and group are required strings")
        ent, _ = lab.groups.resolve(ds, gid)
        if gid == FULL_GROUP and body.get("confirm_full") is not True:
            return _err("FULL_NOT_CONFIRMED", "the whole eligible pool is an explicit action: send confirm_full true")
        d = ent["desc"]
        raw = {"score_file_id": ent["score_file_id"], "iteration": d["iteration"], "modality": d["modality"],
               "dataset": ds, "split": d["split"], "run_ids": list(d["run_ids"]),
               "purpose": "final_reanalysis" if d["split"] in FINAL_SPLITS else "post_freeze_evaluation",
               "eval_group": gid, "label": f"{d['experiment']} {ds} {gid}",
               "n_boot": body.get("n_boot", N_BOOT_DEFAULT), "seed": int(d.get("bootstrap_seed", 20261031))}
        if d["threshold"].get("source", "fixed") == "fixed":
            raw.update(threshold_source="fixed", threshold=float(d["threshold"]["value"]),
                       threshold_provenance=d["threshold"]["provenance"])
        else:                                          # per-seed threshold fitted on val rows of the same runs, as logged
            raw["threshold_source"] = "val_fit"
        rec = lab.runner.submit(raw)
        return jsonify(rec), 202

    @bp.get("/runs")
    def list_runs():
        return jsonify({"runs": [_summary(r) for r in lab.store.list()]})

    @bp.get("/runs/<rid>")
    def get_run(rid: str):
        rec = lab.store.get(rid)
        if rec is None:
            return _err("NO_SUCH_RUN", "unknown run id")
        return jsonify(rec)

    @bp.get("/runs/<rid>/export")
    def export_run(rid: str):
        rec = lab.store.get(rid)
        if rec is None:
            return _err("NO_SUCH_RUN", "unknown run id")
        fmt = request.args.get("format", "json")
        if fmt == "json":
            body, mime, ext = _export.to_json(rec), "application/json", "json"
        elif fmt == "md":
            body, mime, ext = _export.to_markdown(rec), "text/markdown", "md"
        else:
            return _err("INVALID_FORMAT", "format must be json or md")
        return Response(body, mimetype=mime, headers={"Content-Disposition": f'attachment; filename="evallab-{rid}.{ext}"'})

    @bp.post("/compare")
    def compare():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _err("INVALID_REQUEST", "the request body must be a JSON object")
        recs = []
        for key in ("run_a", "run_b"):
            rid = body.get(key)
            if not isinstance(rid, str):
                return _err("INVALID_REQUEST", f"{key} is required (a lab run id)")
            rec = lab.store.get(rid)
            if rec is None:
                return _err("NO_SUCH_RUN", f"{key}: unknown run id")
            recs.append(rec)
        metric = body.get("metric", "auc")
        n_boot = body.get("n_boot")
        if n_boot is not None and (isinstance(n_boot, bool) or not isinstance(n_boot, int) or not 100 <= n_boot <= 5000):
            return _err("INVALID_REQUEST", "n_boot must be an integer between 100 and 5000")
        return jsonify(jsonable(compare_runs(lab.runner, recs[0], recs[1], metric, n_boot)))

    return bp


def register(app, lab: EvalLab | None = None, **options) -> EvalLab:
    bp = create_blueprint(lab, **options)
    app.register_blueprint(bp)
    return bp.lab
