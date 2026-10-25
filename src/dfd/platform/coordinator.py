"""The central analysis coordinator (WS2).

Runs every registered module through ``applicability`` then ``run``, one at a time (the GPU is shared), with a
timeout per module. Nothing a module does can break the report: an exception, a timeout or a result that fails the
contract becomes a ``failed`` or ``timed_out`` entry with its reason. Every module appears in the report exactly once.

Reliability model (local research prototype):

* A timed-out module's thread cannot be killed from Python. The coordinator sets ``ctx.cancel_event`` (modules should
  poll ``ctx.cancelled``), discards the thread's late result or exception, and gives the module a cache view whose
  writes are ignored once cancelled, so a late thread cannot change the report or the shared cache.
* Abandoned threads are tracked. A module with ``requires_gpu = True`` is reported ``skipped`` (not started) while an
  abandoned GPU module is still alive, so one stuck GPU job is never doubled up. Other modules still run.
* ``analyse(total_timeout_s=...)`` bounds wall time: modules not started by then are ``skipped`` with a reason, and
  a running module's wait is capped by the time left.
* Progress callbacks that raise are ignored. KeyboardInterrupt/SystemExit raised inside a module thread are recorded
  as ``failed``.
* One Coordinator may serve concurrent ``analyse()`` calls; module instances are then shared, so modules must be
  thread-safe. The coordinator does not serialise calls.
"""
from __future__ import annotations

import threading
import time
from collections.abc import MutableMapping
from typing import Callable, Optional

from . import schema
from .module import APPLICABLE, UNSUPPORTED, Context, DetectorModule, Probe
from .schema import ModuleResult

_SECTION_ORDER = schema.SECTIONS
ABANDONED_REASON = "a previous module is still running after its timeout"


def _stub(mod, status: str, reason: str, ms: float, errors: Optional[list] = None) -> ModuleResult:
    return ModuleResult(module_id=mod.module_id, version=mod.version, section=mod.section, source=getattr(mod, "source", "real"), status=status,
                        reason=reason, errors=errors or [], timing_ms=round(ms, 3))


class _CacheView(MutableMapping):
    """A module's view of the shared cache: reads pass through, writes are dropped once the module is cancelled."""

    def __init__(self, base: MutableMapping, cancel: threading.Event, lock: threading.Lock):
        self._base, self._cancel, self._lock = base, cancel, lock

    def __getitem__(self, k):
        return self._base[k]

    def __setitem__(self, k, v):
        with self._lock:
            if not self._cancel.is_set():
                self._base[k] = v

    def __delitem__(self, k):
        with self._lock:
            if not self._cancel.is_set():
                del self._base[k]

    def __iter__(self):
        return iter(list(self._base))

    def __len__(self):
        return len(self._base)


def _call_with_timeout(fn: Callable, timeout_s: float):
    """Returns (state, value, thread); state is ok, error or timeout. The thread is returned so it can be tracked."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - reported, never raised past the coordinator
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(max(timeout_s, 0.0))
    if t.is_alive():
        return "timeout", None, t
    if "error" in box:
        return "error", box["error"], t
    return "ok", box["value"], t


class Coordinator:
    def __init__(self, modules: list, clock: Callable[[], float] = time.perf_counter):
        ids = [m.module_id for m in modules]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate module_id in the registry")
        self.modules: list[DetectorModule] = list(modules)
        self._clock = clock
        self._lock = threading.Lock()
        self._abandoned: list[tuple[object, threading.Thread]] = []   # (module, thread) of timed-out work

    # -- abandoned-work accounting ------------------------------------------------------------------------
    def _prune(self) -> list:
        with self._lock:
            self._abandoned = [(m, t) for m, t in self._abandoned if t.is_alive()]
            return list(self._abandoned)

    def abandoned_alive(self) -> list[str]:
        """module_ids of timed-out modules whose threads are still running."""
        return [m.module_id for m, _ in self._prune()]

    def _gpu_abandoned(self) -> bool:
        return any(getattr(m, "requires_gpu", False) for m, _ in self._prune())

    def _abandon(self, mod, thread, cancel: threading.Event) -> None:
        with self._lock:
            cancel.set()
            self._abandoned.append((mod, thread))

    def _run_one(self, mod, ctx: Context, deadline: Optional[float] = None) -> ModuleResult:
        t0 = self._clock()
        ms = lambda: (self._clock() - t0) * 1000.0  # noqa: E731
        left = lambda: float("inf") if deadline is None else deadline - self._clock()  # noqa: E731
        if left() <= 0:
            return _stub(mod, "skipped", "total time limit reached before this module started", 0.0)
        if getattr(mod, "requires_gpu", False) and self._gpu_abandoned():
            return _stub(mod, "skipped", ABANDONED_REASON, 0.0)
        state, val, th = _call_with_timeout(lambda: mod.applicability(ctx.probe), min(mod.timeout_s, 5.0, left()))
        if state == "timeout":
            self._abandon(mod, th, ctx.cancel_event)
        if state != "ok" or val is None or getattr(val, "kind", None) not in (APPLICABLE, "not_applicable", UNSUPPORTED):
            why = "timed out" if state == "timeout" else (type(val).__name__ if state == "error" else "invalid return value")
            return _stub(mod, "failed", f"applicability check failed: {why}", ms(),
                         [{"code": "APPLICABILITY_FAILED", "message": why}])
        if val.kind != APPLICABLE:
            return _stub(mod, val.kind, val.reason or "no reason given by the module", ms())
        limit = min(mod.timeout_s, left())
        state, val, th = _call_with_timeout(lambda: mod.run(ctx), limit)
        if state == "timeout":
            self._abandon(mod, th, ctx.cancel_event)
            why = f"no result within {mod.timeout_s:g} s" if limit >= mod.timeout_s else "total time limit reached while running"
            return _stub(mod, "timed_out", why, ms(), [{"code": "MODULE_TIMEOUT", "message": f"limit {limit:g} s"}])
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

    def analyse(self, probe: Probe, request_id: str, progress: Optional[Callable[[str], None]] = None,
                total_timeout_s: Optional[float] = None) -> dict:
        t0 = self._clock()
        deadline = None if total_timeout_s is None else t0 + total_timeout_s

        def safe_progress(msg: str) -> None:
            try:
                progress(msg)
            except Exception:  # noqa: BLE001 - a broken observer must not break the analysis
                pass

        shared: dict = {}
        cache_lock = threading.Lock()
        results = []
        for mod in self.modules:
            if progress:
                safe_progress(mod.module_id)
            cancel = threading.Event()
            ctx = Context(probe=probe, cache=_CacheView(shared, cancel, cache_lock),
                          progress=safe_progress if progress else None, cancel_event=cancel)
            try:
                r = self._run_one(mod, ctx, deadline)
            except Exception as e:  # noqa: BLE001 - last line of defence: the module still appears exactly once
                r = _stub(mod, "failed", f"coordinator error {type(e).__name__}", 0.0,
                          [{"code": "COORDINATOR_ERROR", "message": str(e)[:200]}])
            results.append(r)
        degraded = any(r.status == "skipped" for r in results)
        report = build_report(probe, request_id, results, degraded=degraded)
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


def build_report(probe: Probe, request_id: str, results: list[ModuleResult], degraded: bool = False) -> dict:
    """``degraded`` marks a run where the coordinator itself skipped modules, so the state is partial, not ok."""
    cov: dict = {s: [] for s in schema.STATUSES}
    for r in results:
        entry = {"module_id": r.module_id}
        if r.status != "completed":
            entry["reason"] = r.reason
        cov[r.status].append(entry)
    srcs = {r.source for r in results}
    n_ok = len(cov["completed"])
    n_bad = sum(len(cov[s]) for s in ("failed", "timed_out", "unsupported")) + (len(cov["skipped"]) if degraded else 0)
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
