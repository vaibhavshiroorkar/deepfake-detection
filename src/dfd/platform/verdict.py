"""Verdict layer (D-087): turns one module result into a categorical, evidence-grounded assessment. Module-specific only.

What it is. A pure function of the stored module results (no model, no I/O). Every verdict names the rule that produced it and the
stored evidence behind that rule (``verdict_evidence.EVIDENCE``, checked against the result files by a test). It never produces a
probability, never says that media is authentic, and never lets an experimental module decide.

What it is not. There is no report-level verdict and no fused score (schema.FORBIDDEN_REPORT_KEYS stays as it is). A score is shown as a
"ranking score" or "model output" and is never read as a percentage of certainty: ``calibration.is_probability`` is always false,
because every calibration in this project was fitted on in-domain validation data and the measured shifts (E-015, E-028) break it.

Consistency with the module result. The module's finding was produced by ``score >= threshold``. This layer re-derives that
comparison; if the finding, the score and the threshold disagree, or the score is missing, not finite or outside [0, 1], the
verdict is ``inconclusive`` with the reason recorded, never a guess.

Product gate. A module is an experimental head when its warning code starts with EXPERIMENTAL; it can only give
``experimental_signal``. ``likely_ai_generated`` exists in the vocabulary but no profile can reach it (no detector has passed a
validation that would justify the word); a test pins this. ``APPROVED_HEADS`` is not read or changed here.
"""
from __future__ import annotations

import math
from typing import Optional

from .verdict_evidence import text_of

ASSESSMENT_VERSION = "v1"

# code -> (label, one-line meaning). The order is the order of the vocabulary shown to the user.
VERDICTS = {
    "possible_face_manipulation": ("Possible manipulation", "The face-video check scored the video above its decision line. This is a possible sign only: its false-alarm rate differed between test sets and it is not a determination."),
    "possible_manipulation": ("Possible manipulation (weak evidence)", "A check found signs of manipulation, but it raised false alarms on unmanipulated material in tests."),
    "possible_ai_generation": ("Possible AI-generation (weak evidence)", "The generated-image check scored the image above its line, but it also flags photographs and misses most modern generated ones."),
    "likely_ai_generated": ("Likely AI-generated", "Reserved: no check of this project has the evidence to give this verdict."),
    "no_strong_evidence": ("No strong evidence of manipulation detected", "The check found nothing within its tested scope. This does not show that the media is unmanipulated."),
    "possible_mismatch": ("Possible audio-visual mismatch (weak evidence)", "The audio and the mouth movements were not in step by the measured rule, which is weak evidence."),
    "no_mismatch": ("No audio-visual mismatch found", "The measured offset lay in the band seen on unmanipulated clips."),
    "experimental_signal": ("Experimental signal observed, no reliable verdict available", "A research prototype produced a number; it has not been validated as a detector."),
    "inconclusive": ("Inconclusive", "The result does not support a verdict."),
    "credentials_verified": ("Content credentials verified", "A signed manifest is present and valid; this concerns origin, not whether the content is true."),
    "credentials_invalid": ("Content credentials invalid", "A manifest is present but did not verify."),
    "no_credentials": ("No content credentials found", "A missing credential says nothing about the content."),
    "information": ("Information only", "Facts about the file; not evidence either way."),
    "not_applicable": ("Did not apply", "This check does not run on this kind of file."),
    "unsupported_input": ("Unsupported input", "The file type or content cannot be analysed."),
    "analysis_failed": ("Analysis failed", "The check could not complete; nothing can be said from it."),
}
UNCERTAINTY_LEVELS = ("moderate", "high", "not_assessed")
NOT_ESTABLISHED = ("This reading is not a probability and not proof. A reading of no strong evidence, an inconclusive reading or a "
                   "failed check does not show that the media is unmanipulated.")
PROBABILITY_STATEMENT = ("The scores behind this reading are ranking scores of research models. They were calibrated on in-domain "
                         "validation data only, so a score of 0.7 does not mean a 70 percent chance, and no probability is given.")

# (module_id, path) -> profile. ``path`` is the part after ``path=`` of the module's tested_scope; None when the module has one path.
# positive: finding that the module flagged; negative: finding that it did not. ``unc_*``: uncertainty level of that verdict.
# ``evidence``: ids of verdict_evidence.EVIDENCE that justify the profile. ``tier`` decides whether the module can drive the reading.
_PROFILES = {
    ("face_video", None): dict(
        tier="detector", positive="possible_face_manipulation", negative="no_strong_evidence", unc_pos="high", unc_neg="high",
        evidence=["E015_visual_specificity", "E015_visual_sensitivity"],
        pos_why="The frozen video head flagged few unmanipulated videos in one sealed test (FakeAVCeleb), but its false-alarm rate was higher on other sealed video sets and its results differ between datasets, so a flag is a possible sign and not a determination.",
        neg_why="The same head missed about half of the fake videos in a sealed test, so a missing flag is weak."),
    ("image_detector", "face_crop"): dict(
        tier="limited", positive="possible_manipulation", negative="inconclusive", unc_pos="high", unc_neg="high",
        evidence=["E028_tv_still_fpr", "E028_tv_still_fnr"],
        pos_why="The face-swap model was built for video frames and flagged camera photographs of faces in a benchmark.",
        neg_why="The face-swap model has not been validated on still photographs, so a low score does not support a verdict."),
    ("image_detector", "whole_image"): dict(
        tier="limited", positive="possible_ai_generation", negative="inconclusive", unc_pos="high", unc_neg="high",
        evidence=["E028_ti_fpr", "E028_ti_fnr"],
        pos_why="The generated-image model flags some photographs (more among PNG files) and detects only a minority of modern generated images.",
        neg_why="The generated-image model misses most images from modern generators, so a low score does not support a verdict."),
    ("speech", None): dict(
        tier="limited", positive="possible_manipulation", negative="no_strong_evidence", unc_pos="high", unc_neg="moderate",
        evidence=["E015_audio_specificity", "E015_audio_sensitivity", "E030_speech_eer_itw"],
        pos_why="The speech model flagged a third of the unmanipulated audio in a sealed test, so a flag alone is weak evidence.",
        neg_why="The speech model caught most fake audio in its sealed test, but its error rate is 15.8 percent on speakers recorded in the wild."),
    ("lip_sync", None): dict(
        tier="limited", positive="possible_mismatch", negative="no_mismatch", unc_pos="high", unc_neg="high",
        evidence=[], pos_why="The rule was fitted on unmanipulated clips and its band covers about one such clip in ten at the edges; it is weak evidence, not a detector.",
        neg_why="Fake clips with a small offset also lie inside the band."),
    ("generated_image", None): dict(tier="experimental", evidence=["E028_ti_face_fnr", "E029_candidate_image_fpr"]),
    ("generated_video", None): dict(tier="experimental", evidence=["E026_installed_video_head_fpr", "E027_candidate_video_head_fpr"]),
}
_INFO = {"media_metadata"}
_PROVENANCE = {"content_credentials", "c2pa"}


EXPERIMENTAL_MODULES = ("generated_image", "generated_video")


def is_experimental(m: dict) -> bool:
    """One definition for the verdict layer and the explanations: an EXPERIMENTAL warning, or a module that is experimental by design."""
    return m.get("module_id") in EXPERIMENTAL_MODULES or any(
        isinstance(w, dict) and str(w.get("code", "")).startswith("EXPERIMENTAL") for w in m.get("warnings") or [])


def _path_of(m: dict) -> Optional[str]:
    scope = str(m.get("tested_scope") or "")
    if scope.startswith("path="):
        return scope[5:].split(":")[0].split(" ")[0].split("_with_face")[0].strip()
    return None


def _profile(m: dict) -> Optional[dict]:
    mid = m.get("module_id")
    return _PROFILES.get((mid, _path_of(m))) or _PROFILES.get((mid, None))


def _score_problem(m: dict) -> Optional[str]:
    """None when score, threshold and finding are mutually consistent; otherwise the reason (a stable code)."""
    s = m.get("score")
    if not isinstance(s, dict):
        return "score_missing"
    v, t = s.get("value"), s.get("threshold")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return "score_invalid"
    if not 0.0 <= v <= 1.0:
        return "score_out_of_range"
    if t is None:
        return "threshold_missing"
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        return "threshold_invalid"
    expected = "indicates_manipulation" if v >= t else "no_indication_within_tested_scope"
    if m.get("finding") != expected:
        return "finding_disagrees_with_threshold"
    return None


def _out(code: str, *, tier: str, basis: str, unc: str, unc_text: str, reasons=(), evidence=(), why: str = "", m: Optional[dict] = None,
         score_kind: str = "none") -> dict:
    s = (m or {}).get("score") if isinstance((m or {}).get("score"), dict) else {}
    v = s.get("value") if isinstance(s.get("value"), (int, float)) and not isinstance(s.get("value"), bool) and math.isfinite(s.get("value")) else None
    return {
        "version": ASSESSMENT_VERSION, "verdict": code, "verdict_label": VERDICTS[code][0], "meaning": VERDICTS[code][1], "tier": tier,
        "basis": basis, "why": why,
        "raw": {"score": v, "score_kind": score_kind, "threshold": s.get("threshold") if v is not None else None},
        "calibration": {"is_probability": False, "statement": PROBABILITY_STATEMENT if v is not None else None},
        "uncertainty": {"level": unc, "text": unc_text}, "reasons": list(reasons),
        "evidence": [e for e in (text_of(i) for i in evidence) if e], "not_established": NOT_ESTABLISHED,
    }


def assess_module(m: dict) -> dict:
    """Assessment of one module result (see the module docstring for the rules). A mock module gets the same rules, flagged ``mock``."""
    a = _assess(m)
    if m.get("source") == "mock":
        a["mock"] = True
        a["why"] = ("Invented scenario data, no model ran. " + a["why"]).strip()
    return a


def _assess(m: dict) -> dict:
    mid, st = m.get("module_id"), m.get("status")
    if st == "not_applicable":
        return _out("not_applicable", tier="none", basis="none", unc="not_assessed", unc_text="", reasons=["not_applicable"], m=m)
    if st == "unsupported":
        return _out("unsupported_input", tier="none", basis="none", unc="not_assessed", unc_text="No analysis was possible.", reasons=["unsupported"], m=m)
    if st != "completed":
        return _out("analysis_failed", tier="none", basis="none", unc="not_assessed", unc_text="No result was produced.", reasons=[str(st)], m=m)
    if mid in _INFO:
        return _out("information", tier="context", basis="none", unc="not_assessed", unc_text="", m=m)
    if mid in _PROVENANCE:
        f = m.get("finding")
        code = {"credentials_valid": "credentials_verified", "credentials_invalid": "credentials_invalid", "no_credentials_found": "no_credentials"}.get(f, "inconclusive")
        return _out(code, tier="provenance", basis="provenance", unc="not_assessed",
                    unc_text="Credentials describe where a file came from, not whether what it shows is true.", m=m)
    prof = _profile(m)
    if prof is None:
        return _out("inconclusive", tier="none", basis="none", unc="high", unc_text="This check has no verdict rule.", reasons=["no_profile"], m=m)
    if is_experimental(m) or prof["tier"] == "experimental":
        return _out("experimental_signal", tier="experimental", basis="none", unc="high", score_kind="ranking_score", m=m,
                    unc_text="A research prototype: its score is a ranking score and it has flagged unmanipulated material in tests. It does not decide anything.",
                    reasons=["experimental_module"], evidence=prof.get("evidence", []),
                    why="Experimental modules cannot give a verdict; they are shown as signals only.")
    section = m.get("section")
    if section == "cross_modal":
        f = m.get("finding")
        if f == "potential_mismatch":
            code, unc = prof["positive"], prof["unc_pos"]
        elif f == "consistent":
            code, unc = prof["negative"], prof["unc_neg"]
        else:
            return _out("inconclusive", tier=prof["tier"], basis="heuristic", unc="high", unc_text="The measurement was not confident enough.",
                        reasons=["module_inconclusive"], m=m)
        why = prof["pos_why"] if f == "potential_mismatch" else prof["neg_why"]
        return _out(code, tier=prof["tier"], basis="heuristic", unc=unc, unc_text=why, evidence=prof["evidence"], why=why, m=m, score_kind="none")
    if m.get("finding") == "inconclusive":
        return _out("inconclusive", tier=prof["tier"], basis="threshold_rule", unc="high", unc_text="The module itself reported no finding.",
                    reasons=["module_inconclusive"], m=m, score_kind="model_output")
    problem = _score_problem(m)
    if problem == "threshold_missing" and m.get("source") == "mock":
        problem = None            # scripted scenario data carries no decision line; real modules always do
    if problem:
        return _out("inconclusive", tier=prof["tier"], basis="threshold_rule", unc="high",
                    unc_text="The score, the decision line and the finding do not agree or are not valid, so no verdict is given.",
                    reasons=[problem], m=m, score_kind="model_output")
    positive = m["finding"] == "indicates_manipulation"
    code = prof["positive"] if positive else prof["negative"]
    unc = prof["unc_pos"] if positive else prof["unc_neg"]
    why = prof["pos_why"] if positive else prof["neg_why"]
    return _out(code, tier=prof["tier"], basis="threshold_rule", unc=unc, unc_text=why, evidence=prof["evidence"], why=why, m=m,
                score_kind="model_output", reasons=["below_threshold_not_reassuring"] if (not positive and code == "inconclusive") else [])


def assess_error(code: str) -> dict:
    """Assessment for a job that failed before a report existed (API errors carry it, so the UI never invents a label)."""
    unsupported = code in ("UNSUPPORTED_MEDIA_TYPE", "UNREADABLE_FILE", "NO_FILE")
    v = "unsupported_input" if unsupported else "analysis_failed"
    return {"version": ASSESSMENT_VERSION, "verdict": v, "verdict_label": VERDICTS[v][0], "meaning": VERDICTS[v][1], "summary": "",
            "uncertainty": {"level": "not_assessed", "text": ""}, "reasons": [str(code)], "not_established": NOT_ESTABLISHED,
            "calibration": {"is_probability": False, "statement": None}}


def vocabulary() -> dict:
    return {"verdicts": {k: {"label": v[0], "meaning": v[1]} for k, v in VERDICTS.items()}, "uncertainty_levels": list(UNCERTAINTY_LEVELS),
            "not_established": NOT_ESTABLISHED, "probability_statement": PROBABILITY_STATEMENT, "version": ASSESSMENT_VERSION}
