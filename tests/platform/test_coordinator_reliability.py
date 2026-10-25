"""Coordinator reliability: cancellation, late-result safety, abandoned GPU work, total timeout, observers, stress."""
import threading
import time

from dfd.platform.coordinator import ABANDONED_REASON, Coordinator
from dfd.platform.module import Applicability
from dfd.platform.schema import ModuleResult, validate_report
from tests.platform.test_platform_core import PROBE, REAL_MODEL, Fake, ok_result


class Mod(Fake):
    def __init__(self, module_id, run, timeout_s=1.0, gpu=False):
        super().__init__(module_id, run)
        self.timeout_s = timeout_s
        self.requires_gpu = gpu


def by_id(rep):
    return {m["module_id"]: m for m in rep["modules"]}


def check_once(rep, mods):
    assert sorted(m["module_id"] for m in rep["modules"]) == sorted(m.module_id for m in mods)
    assert validate_report(rep) == []


def test_cancel_event_is_set_on_timeout_and_visible_to_module():
    seen = {}
    done = threading.Event()

    def slow(ctx):
        while not ctx.cancelled:
            time.sleep(0.01)
        seen["cancelled"] = True
        done.set()
        return ok_result("a")

    m = Mod("a", slow, timeout_s=0.1)
    rep = Coordinator([m]).analyse(PROBE, "r")
    assert by_id(rep)["a"]["status"] == "timed_out"
    assert done.wait(2) and seen["cancelled"]
    check_once(rep, [m])


def test_late_result_exception_and_cache_write_cannot_change_anything():
    go = threading.Event()
    late = threading.Event()

    def writer(ctx):
        ctx.cache["k"] = "before"
        go.wait(2)
        ctx.cache["k"] = "late"
        ctx.cache["new"] = 1
        late.set()
        return ok_result("w")

    def thrower(ctx):
        go.wait(2)
        raise RuntimeError("late")

    def reader(ctx):
        return ok_result("r", reason=None, limitations=[str(ctx.cache.get("k")), str(sorted(ctx.cache))])

    mods = [Mod("w", writer, timeout_s=0.1), Mod("t", thrower, timeout_s=0.1), Mod("r", reader)]
    c = Coordinator(mods)
    rep = c.analyse(PROBE, "r")
    snapshot = repr(rep)
    go.set()
    assert late.wait(2)
    time.sleep(0.1)
    assert repr(rep) == snapshot
    d = by_id(rep)
    assert d["w"]["status"] == "timed_out" and d["t"]["status"] == "timed_out" and d["r"]["status"] == "completed"
    assert d["r"]["limitations"][0] == "before"      # the pre-timeout write is kept, the late write is not
    assert "new" not in d["r"]["limitations"][1]
    check_once(rep, mods)


def test_abandoned_gpu_module_blocks_next_gpu_module_only():
    release = threading.Event()
    started = []

    def stuck(ctx):
        release.wait(5)
        return ok_result("g1")

    def gpu2(ctx):
        started.append("g2")
        return ok_result("g2")

    def cpu(ctx):
        return ok_result("c")

    mods = [Mod("g1", stuck, timeout_s=0.1, gpu=True), Mod("g2", gpu2, gpu=True), Mod("c", cpu)]
    c = Coordinator(mods)
    rep = c.analyse(PROBE, "r")
    d = by_id(rep)
    assert d["g1"]["status"] == "timed_out"
    assert d["g2"]["status"] == "skipped" and d["g2"]["reason"] == ABANDONED_REASON and started == []
    assert d["c"]["status"] == "completed"
    assert [e["module_id"] for e in rep["coverage"]["skipped"]] == ["g2"]
    assert rep["status"]["state"] == "partial"
    assert c.abandoned_alive() == ["g1"]
    check_once(rep, mods)
    release.set()
    for _ in range(100):
        if not c.abandoned_alive():
            break
        time.sleep(0.02)
    assert c.abandoned_alive() == []


def test_gpu_module_runs_again_once_abandoned_thread_finished():
    release = threading.Event()
    n = {"calls": 0}

    def run(ctx):
        n["calls"] += 1
        if n["calls"] == 1:
            release.wait(5)
        return ok_result("g")

    m = Mod("g", run, timeout_s=0.1, gpu=True)
    c = Coordinator([m])
    assert by_id(c.analyse(PROBE, "a"))["g"]["status"] == "timed_out"
    assert by_id(c.analyse(PROBE, "b"))["g"]["status"] == "skipped"
    release.set()
    for _ in range(100):
        if not c.abandoned_alive():
            break
        time.sleep(0.02)
    assert by_id(c.analyse(PROBE, "c"))["g"]["status"] == "completed"


def test_total_timeout_skips_unstarted_modules_and_caps_running_one():
    def slow(ctx):
        time.sleep(0.6)
        return ok_result("a")

    mods = [Mod("a", slow, timeout_s=30), Mod("b", lambda ctx: ok_result("b")), Mod("c", lambda ctx: ok_result("c"))]
    t0 = time.perf_counter()
    rep = Coordinator(mods).analyse(PROBE, "r", total_timeout_s=0.2)
    assert time.perf_counter() - t0 < 0.55
    d = by_id(rep)
    assert d["a"]["status"] == "timed_out"
    assert d["b"]["status"] == d["c"]["status"] == "skipped"
    assert "total time limit" in d["b"]["reason"]
    assert rep["status"]["state"] == "failed"      # nothing completed
    check_once(rep, mods)


def test_total_timeout_none_and_generous_are_unchanged():
    mods = [Mod("a", lambda ctx: ok_result("a")), Mod("b", lambda ctx: ok_result("b"))]
    for kw in ({}, {"total_timeout_s": 60}):
        rep = Coordinator(mods).analyse(PROBE, "r", **kw)
        assert rep["status"]["state"] == "ok"
        check_once(rep, mods)


def test_progress_callback_exceptions_do_not_break_analysis():
    def bad(msg):
        raise ValueError("observer broke")

    def run(ctx):
        ctx.progress("inside")        # modules calling progress are protected too
        return ok_result("a")

    mods = [Mod("a", run), Mod("b", lambda ctx: ok_result("b"))]
    rep = Coordinator(mods).analyse(PROBE, "r", bad)
    assert rep["status"]["state"] == "ok"
    check_once(rep, mods)


def test_keyboardinterrupt_and_systemexit_in_module_are_failed():
    def ki(ctx):
        raise KeyboardInterrupt()

    def se(ctx):
        raise SystemExit(3)

    mods = [Mod("k", ki), Mod("s", se), Mod("ok", lambda ctx: ok_result("ok"))]
    rep = Coordinator(mods).analyse(PROBE, "r")
    d = by_id(rep)
    assert d["k"]["status"] == "failed" and "KeyboardInterrupt" in d["k"]["reason"]
    assert d["s"]["status"] == "failed" and "SystemExit" in d["s"]["reason"]
    assert d["ok"]["status"] == "completed"
    check_once(rep, mods)


def test_applicability_timeout_is_abandoned_and_failed():
    ev = threading.Event()

    class SlowApp(Mod):
        def applicability(self, probe):
            ev.wait(5)
            return Applicability.yes()

    m = SlowApp("a", lambda ctx: ok_result("a"), timeout_s=0.1)
    m.requires_gpu = True
    c = Coordinator([m])
    rep = c.analyse(PROBE, "r")
    assert by_id(rep)["a"]["status"] == "failed"
    assert c.abandoned_alive() == ["a"]
    ev.set()


def test_concurrent_analyse_calls_share_modules_and_keep_reports_separate():
    lock = threading.Lock()
    calls = []

    def run(ctx):
        time.sleep(0.02)
        with lock:                       # the module itself is thread-safe: the coordinator does not serialise calls
            calls.append(ctx.probe.sha256)
        ctx.cache["x"] = ctx.probe.sha256
        return ok_result("a", limitations=[ctx.cache["x"]])

    mods = [Mod("a", run), Mod("b", lambda ctx: ok_result("b"))]
    c = Coordinator(mods)
    out = {}

    def go(i):
        import dataclasses
        p = dataclasses.replace(PROBE, sha256=str(i) * 64)
        out[i] = c.analyse(p, f"r{i}")

    ts = [threading.Thread(target=go, args=(i,)) for i in (1, 2)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    for i in (1, 2):
        assert out[i]["request_id"] == f"r{i}"
        assert by_id(out[i])["a"]["limitations"] == [str(i) * 64]    # no cache leakage between analyses
        check_once(out[i], mods)
    assert sorted(calls) == ["1" * 64, "2" * 64]


def test_stress_50_modules_mixed_outcomes():
    mods = []
    expect = {}
    for i in range(50):
        mid = f"m{i}"
        k = i % 5
        if k == 0:
            mods.append(Mod(mid, lambda ctx, mid=mid: ok_result(mid))); expect[mid] = "completed"
        elif k == 1:
            def boom(ctx):
                raise RuntimeError("x")
            mods.append(Mod(mid, boom)); expect[mid] = "failed"
        elif k == 2:
            mods.append(Mod(mid, lambda ctx: (time.sleep(0.3), None)[1], timeout_s=0.02)); expect[mid] = "timed_out"
        elif k == 3:
            mods.append(Mod(mid, lambda ctx: "garbage")); expect[mid] = "failed"
        else:
            m = Mod(mid, lambda ctx: ok_result(mid))
            m._app = Applicability.no("nothing to analyse"); mods.append(m); expect[mid] = "not_applicable"
    rep = Coordinator(mods).analyse(PROBE, "r")
    d = by_id(rep)
    assert {k: v["status"] for k, v in d.items()} == expect
    check_once(rep, mods)
    time.sleep(0.4)      # let abandoned threads finish; nothing may change afterwards
