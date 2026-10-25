"""The central analysis coordinator (WS2).

Runs every registered module through ``applicability`` then ``run``, one at a time (the GPU is shared), with a
timeout per module. Nothing a module does can break the report: an exception, a timeout or a result that fails the
contract becomes a ``failed`` or ``timed_out`` entry with its reason. Every module appears in the report exactly once.

A timed-out module cannot be killed from outside in Python: its daemon thread is abandoned and its late result is
discarded. Modules that load models should therefore check ``ctx`` cancellation in a later version; this is recorded
as a known limitation, not hidden.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from . import schema
from .module import APPLICABLE, UNSUPPORTED, Context, DetectorModule, Probe
from .schema import ModuleResult

_SECTION_ORDER = schema.SECTIONS


def _stub(mod, status: str, reason: str, ms: float, errors: Optional[list] = None) -> ModuleResult:
    return ModuleResult(module_id=mod.module_id, version=mod.version, section=mod.section, source=getattr(mod, "source", "real"), status=status,
                        reason=reason, errors=errors or [], timing_ms=round(ms, 3))


def _call_with_timeout(fn: Callable, timeout_s: float):
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - reported, never raised past the coordinator
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return "timeout", None
    if "error" in box:
        return "error", box["error"]
    return "ok", box["value"]


class Coordinator:
    def __init__(self, modules: list, clock: Callable[[], float] = time.perf_counter):
        ids = [m.module_id for m in modules]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate module_id in the registry")
        self.modules: list[DetectorModule] = list(modules)
        self._clock = clock

    def _run_one(self, mod, ctx: Context) -> ModuleResult:
        t0 = self._clock()
        ms = lambda: (self._clock() - t0) * 1000.0  # noqa: E731
        state, val = _call_with_timeout(lambda: mod.applicability(ctx.probe), min(mod.timeout_s, 5.0))
        if state != "ok" or val is None or getattr(val, "kind", None) not in (APPLICABLE, "not_applicable", UNSUPPORTED):
            why = "timed out" if state == "timeout" else (type(val).__name__ if state == "error" else "invalid return value")
            return _stub(mod, "failed", f"applicability check failed: {why}", ms(),
                         [{"code": "APPLICABILITY_FAILED", "message": why}])
        if val.kind != APPLICABLE:
            return _stub(mod, val.kind, val.reason or "no reason given by the module", ms())
        state, val = _call_with_timeout(lambda: mod.run(ctx), mod.timeout_s)
        if state == "timeout":
            return _stub(mod, "timed_out", f"no result within {mod.timeout_s:g} s",
                         ms(), [{"code": "MODULE_TIMEOUT", "message": f"limit {mod.timeout_s:g} s"}])
        if state == "error":
            return _stub(mod, "failed", f"module raised {type(val).__name__}", ms(),
                         [{"code": "MODULE_EXCEPTION", "message": f"{type(val).__name__}: {str(val)[:200]}"}])
        res: ModuleResult = val
        d = res.to_dict() if isinstance(res, ModuleResult) else None
        problems = ["not a ModuleResult"] if d is None else schema.validate_module_result(d)
        if d is not None and (res.module_id != mod.module_id or res.section != mod.section):
            problems.append("module_id or section differs from the registered module")
        if problems:
            return _stub(mod, "failed", "module returned a result that breaks the contract", ms(),
                         [{"code": "INVALID_RESULT", "message": "; ".join(problems)[:300]}])
        if not res.timing_ms:
            res.timing_ms = round(ms(), 3)
        return res

    def analyse(self, probe: Probe, request_id: str, progress: Optional[Callable[[str], None]] = None) -> dict:
        t0 = self._clock()
        ctx = Context(probe=probe, progress=progress)
        results = []
        for mod in self.modules:
            if progress:
                progress(mod.module_id)
            results.append(self._run_one(mod, ctx))
        report = build_report(probe, request_id, results)
        report["status"]["processing_time_ms"] = round((self._clock() - t0) * 1000.0, 3)
        return report


def find_conflicts(results: list[ModuleResult]) -> list[dict]:
    """Authenticity modules that disagree. Different tested scopes mean this is a note, not a contradiction."""
    done = [r for r in results if r.status == "completed" and r.section == "authenticity"]
    hit = [r.module_id for r in done if r.finding == "indicates_manipulation"]
    clear = [r.module_id for r in done if r.finding == "no_indication_within_tested_scope"]
    if hit and clear:
        return [{"modules": hit + clear, "note": "Some modules found indications of manipulation and others found none within "
                 "their tested scope. The modules cover different kinds of manipulation, so this is not necessarily a contradiction."}]
    return []


def build_report(probe: Probe, request_id: str, results: list[ModuleResult]) -> dict:
    cov: dict = {s: [] for s in schema.STATUSES}
    for r in results:
        entry = {"module_id": r.module_id}
        if r.status != "completed":
            entry["reason"] = r.reason
        cov[r.status].append(entry)
    srcs = {r.source for r in results}
    n_ok = len(cov["completed"])
    n_bad = sum(len(cov[s]) for s in ("failed", "timed_out", "unsupported"))
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "request_id": request_id,
        "input": {"filename": probe.filename, "media_type": probe.media_type, "mime_type": probe.mime_type,
                  "size_bytes": probe.size_bytes, "sha256": probe.sha256},
        "probe": probe.to_dict(),
        "modules": [r.to_dict() for r in results],
        "sections": {s: [r.module_id for r in results if r.section == s] for s in _SECTION_ORDER},
        "coverage": cov,
        "conflicts": find_conflicts(results),
        "source_summary": "none" if not srcs else (next(iter(srcs)) if len(srcs) == 1 else "mixed"),
        "status": {"state": "failed" if n_ok == 0 else ("partial" if n_bad else "ok"), "processing_time_ms": 0.0},
        "statement": schema.STATEMENT,
    }
