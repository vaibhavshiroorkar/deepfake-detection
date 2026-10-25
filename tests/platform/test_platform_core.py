"""WS2 contract tests: schema v2 checker, coordinator failure paths, applicability reasons, scenario mocks."""
import copy
import json

import pytest

from dfd.platform import mocks, schema
from dfd.platform.coordinator import Coordinator, find_conflicts
from dfd.platform.module import Applicability, Context, Probe, probe_from_media
from dfd.platform.schema import ModuleResult, validate_module_result, validate_report

PROBE = Probe(path="p", filename="a.mp4", media_type="video", mime_type="video/mp4", size_bytes=1, sha256="0" * 64,
              duration_s=3.0, has_video=True, has_audio=True)
REAL_MODEL = {"name": "m", "version": "1", "weights_sha256": "ab", "licence": "MIT", "evaluation": "E-014"}


class Fake:
    section = "authenticity"
    version = "t1"
    timeout_s = 1.0
    source = "real"

    def __init__(self, module_id, run=None, app=None):
        self.module_id = module_id
        self._run = run
        self._app = app or Applicability.yes()

    def applicability(self, probe):
        if isinstance(self._app, Exception):
            raise self._app
        return self._app

    def run(self, ctx):
        return self._run(ctx)


def ok_result(mid="m1", **kw):
    d = dict(module_id=mid, version="t1", section="authenticity", source="real", status="completed",
             finding="no_indication_within_tested_scope", model=REAL_MODEL, tested_scope="s", limitations=["l"])
    d.update(kw)
    return ModuleResult(**d)


def analyse(*mods):
    return Coordinator(list(mods)).analyse(PROBE, "rid")


# ---- schema -------------------------------------------------------------------------------------------------

def test_every_scenario_validates_and_is_mock():
    for name in mocks.SCENARIOS:
        r = mocks.build_scenario(name)
        assert validate_report(r) == [], name
        assert r["source_summary"] == "mock"
        assert all(m["source"] == "mock" and m["model"] is None for m in r["modules"])


def test_committed_fixtures_are_current():
    for name in mocks.SCENARIOS:
        on_disk = json.loads((mocks.FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert on_disk == mocks.build_scenario(name), f"run python -m dfd.platform.mocks ({name})"


def test_no_genuine_value_anywhere():
    words = {w for f in schema.FINDINGS.values() for w in f} | set(schema.STATUSES)
    assert not [w for w in words if any(x in w for x in ("genuine", "authentic", "real", "safe"))]


def test_report_rejects_global_score():
    r = mocks.build_scenario("video_nothing_found")
    for k in schema.FORBIDDEN_REPORT_KEYS:
        bad = copy.deepcopy(r)
        bad[k] = 0.5
        assert any(k in v for v in validate_report(bad))


def test_mock_with_model_or_calibrated_score_rejected():
    d = ok_result(source="mock", model=REAL_MODEL).to_dict()
    assert validate_module_result(d)
    d = ok_result(source="mock", model=None, score={"value": 0.5, "meaning": "x", "calibrated": True, "threshold": 0.5}).to_dict()
    assert validate_module_result(d)


def test_calibrated_score_needs_evaluation():
    s = {"value": 0.5, "meaning": "x", "calibrated": True, "threshold": 0.5}
    assert validate_module_result(ok_result(score=s).to_dict()) == []
    assert validate_module_result(ok_result(score=s, model={**REAL_MODEL, "evaluation": schema.NOT_EVALUATED}).to_dict())


def test_wrong_finding_for_section_and_missing_scope_rejected():
    assert validate_module_result(ok_result(finding="consistent").to_dict())
    assert validate_module_result(ok_result(tested_scope="").to_dict())
    assert validate_module_result(ok_result(limitations=[]).to_dict())
    assert validate_module_result(ok_result(finding="genuine").to_dict())


def test_non_completed_needs_reason_and_no_finding():
    d = ModuleResult("x", "1", "authenticity", "real", "not_applicable").to_dict()
    assert validate_module_result(d)
    d = ModuleResult("x", "1", "authenticity", "real", "unsupported", finding="inconclusive", reason="r").to_dict()
    assert validate_module_result(d)


def test_report_consistency_checks():
    r = mocks.build_scenario("module_failure")
    bad = copy.deepcopy(r)
    bad["coverage"]["completed"] = []
    assert validate_report(bad)
    bad = copy.deepcopy(r)
    bad["sections"]["authenticity"] = []
    assert validate_report(bad)
    bad = copy.deepcopy(r)
    bad["statement"] = "all good"
    assert validate_report(bad)


# ---- coordinator --------------------------------------------------------------------------------------------

def test_completed_module_listed_with_coverage():
    r = analyse(Fake("a", lambda c: ok_result("a")))
    assert validate_report(r) == []
    assert r["status"]["state"] == "ok" and [c["module_id"] for c in r["coverage"]["completed"]] == ["a"]


def test_not_applicable_and_unsupported_carry_reasons():
    r = analyse(Fake("a", app=Applicability.no("no audio stream")), Fake("b", app=Applicability.unsupported("odd container")),
                Fake("c", lambda c: ok_result("c")))
    assert validate_report(r) == []
    assert r["coverage"]["not_applicable"] == [{"module_id": "a", "reason": "no audio stream"}]
    assert r["coverage"]["unsupported"] == [{"module_id": "b", "reason": "odd container"}]
    assert r["status"]["state"] == "partial"


def test_exception_becomes_failed_entry_and_others_still_run():
    def boom(ctx):
        raise ValueError("x" * 500)
    r = analyse(Fake("a", boom), Fake("b", lambda c: ok_result("b")))
    assert validate_report(r) == []
    a = next(m for m in r["modules"] if m["module_id"] == "a")
    assert a["status"] == "failed" and a["errors"][0]["code"] == "MODULE_EXCEPTION" and len(a["errors"][0]["message"]) < 260
    assert r["status"]["state"] == "partial"


def test_timeout():
    import time
    m = Fake("slow", lambda c: time.sleep(0.5))
    m.timeout_s = 0.05
    r = analyse(m, Fake("b", lambda c: ok_result("b")))
    assert validate_report(r) == []
    s = next(x for x in r["modules"] if x["module_id"] == "slow")
    assert s["status"] == "timed_out" and s["errors"][0]["code"] == "MODULE_TIMEOUT"


def test_applicability_failure_is_reported():
    r = analyse(Fake("a", app=RuntimeError("bad")), Fake("b", lambda c: ok_result("b")))
    a = r["modules"][0]
    assert a["status"] == "failed" and a["errors"][0]["code"] == "APPLICABILITY_FAILED"
    m = Fake("a", lambda c: ok_result("a"))
    m.applicability = lambda probe: "yes"                                  # invalid return value
    r2 = analyse(m)
    assert r2["modules"][0]["status"] == "failed" and r2["modules"][0]["errors"][0]["code"] == "APPLICABILITY_FAILED"


@pytest.mark.parametrize("make", [
    lambda: ok_result("a", finding="genuine"),
    lambda: ok_result("other"),                       # wrong module_id
    lambda: ok_result("a", section="cross_modal", finding="consistent"),   # wrong section
    lambda: "not a result",
])
def test_invalid_module_result_becomes_failed(make):
    r = analyse(Fake("a", lambda c: make()))
    assert validate_report(r) == []
    assert r["modules"][0]["status"] == "failed" and r["modules"][0]["errors"][0]["code"] == "INVALID_RESULT"


def test_all_modules_not_applicable_is_not_ok():
    r = analyse(Fake("a", app=Applicability.no("x")))
    assert r["status"]["state"] == "failed"      # nothing was analysed


def test_duplicate_ids_refused():
    with pytest.raises(ValueError):
        Coordinator([Fake("a"), Fake("a")])


def test_conflict_noted_for_disagreeing_modules():
    a = ok_result("a", finding="indicates_manipulation")
    b = ok_result("b")
    assert find_conflicts([a, b])[0]["modules"] == ["a", "b"]
    assert find_conflicts([a, a]) == [] and find_conflicts([b]) == []


def test_mock_failure_keeps_mock_source():
    r = mocks.build_scenario("module_failure")
    assert r["source_summary"] == "mock"
    assert next(m for m in r["modules"] if m["module_id"] == "face_video")["status"] == "failed"


def test_timeout_scenario_statuses():
    r = mocks.build_scenario("module_timeout")
    assert r["modules"][0]["status"] == "timed_out" and r["modules"][1]["status"] == "completed"


def test_probe_from_media_duck_typed():
    class M:
        path, filename, media_type, mime_type, size_bytes, sha256 = "p", "a.wav", "audio", "audio/wav", 3, "0" * 64
        duration_s, has_audio, width, height = 2.0, True, None, None
    p = probe_from_media(M())
    assert p.has_video is False and p.has_audio is True and "path" not in p.to_dict()


def test_v1_schema_untouched():
    from dfd.infer import schema as v1
    assert v1.SCHEMA_VERSION == "1" and v1.DECISIONS == ("fake", "real", "undetermined")
