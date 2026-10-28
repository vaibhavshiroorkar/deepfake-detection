"""Verdict layer (D-087): categories, threshold boundaries, inconclusive and failure states, experimental exclusion, API consistency.

Pure functions on synthetic module results (no model, no GPU, no media). Every number here is synthetic test data.
"""
from __future__ import annotations

import copy
import json
import math
import re

import pytest

from dfd.platform import explain, schema, verdict
from dfd.platform.dashboard.serve import create_app

THR = 0.40


def mod(mid="face_video", finding="indicates_manipulation", score=0.9, thr=THR, status="completed", scope="", warnings=None, section="authenticity",
        source="real"):
    m = {"module_id": mid, "version": "t", "section": section, "source": source, "status": status, "finding": finding if status == "completed" else None,
         "score": None if status != "completed" else {"value": score, "meaning": "m", "calibrated": True, "threshold": thr},
         "tested_scope": scope, "warnings": warnings or [], "limitations": [], "errors": [], "observations": [], "model": None}
    return m


def img(path, score, thr=THR):
    f = "indicates_manipulation" if score >= thr else "no_indication_within_tested_scope"
    return mod("image_detector", f, score, thr, scope=f"path={path}: test")


# ------------------------------------------------------------------------------------------------ thresholds
@pytest.mark.parametrize("score,code", [(0.4 - 1e-9, "no_strong_evidence"), (0.4, "possible_face_manipulation"), (0.4 + 1e-9, "possible_face_manipulation"), (0.0, "no_strong_evidence"),
                                        (1.0, "possible_face_manipulation")])
def test_video_threshold_boundary_is_inclusive_like_the_module_rule(score, code):
    f = "indicates_manipulation" if score >= THR else "no_indication_within_tested_scope"
    a = verdict.assess_module(mod("face_video", f, score))
    assert a["verdict"] == code and a["tier"] == "detector"
    assert a["raw"] == {"score": score, "score_kind": "model_output", "threshold": THR}


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_missing_or_invalid_scores_never_give_a_verdict(bad):
    a = verdict.assess_module(mod("face_video", "indicates_manipulation", bad))
    assert a["verdict"] == "inconclusive" and a["reasons"][0] in ("score_invalid", "score_out_of_range")
    assert a["calibration"]["is_probability"] is False and not a["evidence"]


def test_score_missing_entirely_and_missing_threshold_are_inconclusive():
    m = mod("face_video", "indicates_manipulation")
    m["score"] = None
    assert verdict.assess_module(m)["reasons"] == ["score_missing"]
    assert verdict.assess_module(mod("face_video", "indicates_manipulation", 0.9, None))["reasons"] == ["threshold_missing"]


def test_a_finding_that_disagrees_with_score_and_threshold_is_refused():
    a = verdict.assess_module(mod("face_video", "no_indication_within_tested_scope", 0.9))
    assert a["verdict"] == "inconclusive" and a["reasons"] == ["finding_disagrees_with_threshold"]


def test_module_inconclusive_stays_inconclusive():
    assert verdict.assess_module(mod("speech", "inconclusive", 0.5))["verdict"] == "inconclusive"


# ------------------------------------------------------------------------------------------------ per path profiles
def test_negative_whole_image_score_is_not_reassuring():
    a = verdict.assess_module(img("whole_image", 0.01))
    assert a["verdict"] == "inconclusive" and "below_threshold_not_reassuring" in a["reasons"]
    ids = [e["id"] for e in a["evidence"]]
    assert "E028_ti_fnr" in ids


def test_positive_still_image_scores_are_weak_evidence_only():
    assert verdict.assess_module(img("face_crop", 0.9))["verdict"] == "possible_manipulation"
    assert verdict.assess_module(img("whole_image", 0.9))["verdict"] == "possible_ai_generation"
    assert verdict.assess_module(img("face_crop", 0.01))["verdict"] == "inconclusive"


def test_speech_flag_is_weak_and_clear_is_moderate_uncertainty():
    assert verdict.assess_module(mod("speech", "indicates_manipulation", 0.9))["verdict"] == "possible_manipulation"
    a = verdict.assess_module(mod("speech", "no_indication_within_tested_scope", 0.1))
    assert a["verdict"] == "no_strong_evidence" and a["uncertainty"]["level"] == "moderate"


def test_lip_sync_is_limited_evidence():
    m = mod("lip_sync", "potential_mismatch", None, section="cross_modal")
    m["score"] = None
    a = verdict.assess_module(m)
    assert a["verdict"] == "possible_mismatch" and a["tier"] == "limited"
    m["finding"] = "consistent"
    assert verdict.assess_module(m)["verdict"] == "no_mismatch"


# ------------------------------------------------------------------------------------------------ states
def test_not_applicable_unsupported_and_failed_states():
    assert verdict.assess_module(mod(status="not_applicable"))["verdict"] == "not_applicable"
    assert verdict.assess_module(mod(status="unsupported"))["verdict"] == "unsupported_input"
    for st in ("failed", "timed_out", "skipped"):
        assert verdict.assess_module(mod(status=st))["verdict"] == "analysis_failed"


def test_provenance_and_information_modules():
    p = mod("content_credentials", "credentials_valid", None, section="provenance")
    p["score"] = None
    assert verdict.assess_module(p)["verdict"] == "credentials_verified"
    p["finding"] = "no_credentials_found"
    a = verdict.assess_module(p)
    assert a["verdict"] == "no_credentials" and a["tier"] == "provenance"
    assert verdict.assess_module(mod("media_metadata", None, None, section="provenance"))["verdict"] == "information"


# ------------------------------------------------------------------------------------------------ the gate
@pytest.mark.parametrize("mid", ["generated_video", "generated_image"])
def test_experimental_modules_only_give_a_signal_whatever_the_score(mid):
    for s in (0.0, 0.5, 0.99999):
        m = mod(mid, "inconclusive", s, None, warnings=[{"code": "EXPERIMENTAL_UNVALIDATED", "message": "x"}])
        a = verdict.assess_module(m)
        assert a["verdict"] == "experimental_signal" and a["tier"] == "experimental"
        assert a["raw"]["score_kind"] == "ranking_score" and a["calibration"]["is_probability"] is False


def test_no_profile_can_reach_likely_ai_generated_or_a_probability():
    for (mid, path), prof in verdict._PROFILES.items():
        assert "likely_ai_generated" not in (prof.get("positive"), prof.get("negative"))
    for s in (0.0, 0.4, 0.99):
        for path in ("face_crop", "whole_image"):
            a = verdict.assess_module(img(path, s))
            assert a["verdict"] != "likely_ai_generated" and a["calibration"]["is_probability"] is False


# ------------------------------------------------------------------------------------------------ wording and consistency
FORBIDDEN = [r"\bauthentic\w*", r"\bgenuine\b", r"\bproves?\b", r"fake probability", "%"]


def test_assessments_have_no_forbidden_wording_and_never_a_probability():
    cases = [img("whole_image", 0.9), img("whole_image", 0.01), img("face_crop", 0.9), mod("face_video", "indicates_manipulation", 0.9),
             mod("face_video", "no_indication_within_tested_scope", 0.1), mod("speech", "indicates_manipulation", 0.9),
             mod("generated_video", "inconclusive", 0.99, None, warnings=[{"code": "EXPERIMENTAL_UNVALIDATED", "message": "x"}])]
    for m in cases:
        a = verdict.assess_module(m)
        blob = json.dumps(a).lower()
        assert not any(re.search(p, blob) for p in FORBIDDEN), (m["module_id"], blob)
        assert a["calibration"]["is_probability"] is False
        if a["raw"]["score"] is not None:
            assert "70 percent" in a["calibration"]["statement"]


def test_attach_adds_module_assessments_only_and_the_checker_catches_tampering():
    import pathlib
    rep = json.load(open(pathlib.Path(explain.__file__).parent / "fixtures" / "video_manipulation_indicated.json"))
    rep = explain.attach(rep)
    assert explain.check_report_explanations(rep) == []
    assert "assessment" not in rep and not any(k in rep for k in schema.FORBIDDEN_REPORT_KEYS)    # no fused verdict (D-087)
    assert schema.validate_report(rep) == []
    fv = next(m for m in rep["modules"] if m["module_id"] == "face_video")
    assert fv["assessment"]["verdict"] == "possible_face_manipulation" and fv["assessment"]["mock"] is True
    bad = copy.deepcopy(rep)
    bad["assessment"] = {"verdict": "possible_face_manipulation"}
    assert any("must not carry an assessment" in e for e in explain.check_report_explanations(bad))
    bad = copy.deepcopy(rep)
    m0 = next(m for m in bad["modules"] if m["module_id"] == "face_video")
    m0["assessment"]["verdict"] = "no_strong_evidence"
    assert any("assessment does not match" in e for e in explain.check_report_explanations(bad))
    bad = copy.deepcopy(rep)
    m0 = next(m for m in bad["modules"] if m["module_id"] == "face_video")
    m0["assessment"]["why"] = "This proves the video is authentic."
    errs = explain.check_report_explanations(bad)
    assert any("forbidden wording" in e for e in errs)


def test_every_fixture_gets_a_vocabulary_verdict_and_stays_valid():
    import pathlib
    vocab = verdict.vocabulary()["verdicts"]
    for p in sorted((pathlib.Path(explain.__file__).parent / "fixtures").glob("*.json")):
        rep = explain.attach(json.load(open(p)))
        assert "assessment" not in rep, p.name
        assert all(m["assessment"]["verdict"] in vocab for m in rep["modules"])
        assert explain.check_report_explanations(rep) == [], p.name


def test_api_error_carries_the_same_category_the_page_will_show():
    for code, expected in (("UNSUPPORTED_MEDIA_TYPE", "unsupported_input"), ("UNREADABLE_FILE", "unsupported_input"), ("PIPELINE_FAILED", "analysis_failed")):
        assert verdict.assess_error(code)["verdict"] == expected
    app = create_app(live=False)
    v = app.test_client().get("/api/vocabulary").get_json()
    assert v["assessment"]["verdicts"]["experimental_signal"]["label"].startswith("Experimental signal observed")
    assert math.isclose(1.0, 1.0)


# ------------------------------------------------------------------------------------------------ face-video wording (D-087, 2026-10-28)
def test_face_video_flag_reads_possible_manipulation_with_high_uncertainty_and_no_strong_claim():
    a = verdict.assess_module(mod("face_video", "indicates_manipulation", 0.9))
    assert a["verdict_label"] == "Possible manipulation"
    assert a["uncertainty"]["level"] == "high"
    text = (a["why"] + " " + a["meaning"]).lower()
    assert "not a determination" in text and ("other sealed video sets" in text or "differed between test sets" in text)
    assert "likely" not in a["verdict_label"].lower() and "few false alarms" not in text
    assert a["calibration"]["is_probability"] is False and a["tier"] == "detector"


def test_the_retired_face_video_label_is_gone_from_the_vocabulary_and_the_fixtures():
    import pathlib
    assert not any("likely manipulated" in v[0].lower() for v in verdict.VERDICTS.values())
    assert "likely_manipulated" not in verdict.VERDICTS
    for p in (pathlib.Path(explain.__file__).parent / "fixtures").glob("*.json"):
        assert "Likely manipulated" not in p.read_text(encoding="utf-8"), p.name


def test_threshold_still_decides_and_experimental_modules_still_never_reach_a_detector_label():
    flagged = verdict.assess_module(mod("face_video", "indicates_manipulation", THR))
    clear = verdict.assess_module(mod("face_video", "no_indication_within_tested_scope", THR - 1e-9))
    assert (flagged["verdict"], clear["verdict"]) == ("possible_face_manipulation", "no_strong_evidence")
    exp = verdict.assess_module(mod("generated_video", "inconclusive", 0.99, None, warnings=[{"code": "EXPERIMENTAL_X", "message": "x"}]))
    assert exp["verdict"] == "experimental_signal" and exp["tier"] == "experimental"
