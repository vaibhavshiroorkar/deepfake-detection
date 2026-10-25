"""Report schema v2 and its contract checker (WS2).

Differences from schema v1 (``dfd.infer.schema``): the unit of output is a module result, not one fused score. Every
module states its status, finding, model, tested scope and limitations. There is no global fake score and no value that
means "genuine": the vocabularies below contain none, and ``validate_report`` rejects fused-score keys.
``source`` is per module: "mock" results carry no model block and no calibrated score (D-029 extended).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "2"
SECTIONS = ("authenticity", "cross_modal", "provenance")
SOURCES = ("real", "mock", "external")
STATUSES = ("completed", "not_applicable", "unsupported", "failed", "timed_out", "skipped")
FINDINGS = {
    "authenticity": ("indicates_manipulation", "no_indication_within_tested_scope", "inconclusive"),
    "cross_modal": ("consistent", "potential_mismatch", "inconclusive"),
    "provenance": ("credentials_valid", "credentials_invalid", "no_credentials_found", "inconclusive"),
}
ERROR_CODES = ("MODULE_EXCEPTION", "MODULE_TIMEOUT", "INVALID_RESULT", "APPLICABILITY_FAILED")
OBSERVATION_KINDS = ("file_metadata", "heuristic", "marker_scan", "verification")
MODEL_KEYS = ("name", "version", "weights_sha256", "licence", "evaluation")
NOT_EVALUATED = "not evaluated"
STATEMENT = ("This report lists what each module found within its tested scope. A module that found nothing, did not run, "
             "was unsupported or was inconclusive does not show that the media is authentic. No single probability that "
             "the media is fake is given.")
FORBIDDEN_REPORT_KEYS = ("fused", "overall_score", "overall_probability", "probability_fake", "verdict")


@dataclass
class ModuleResult:
    module_id: str
    version: str
    section: str
    source: str
    status: str
    finding: Optional[str] = None
    score: Optional[dict] = None           # {value, meaning, calibrated, threshold}
    reason: Optional[str] = None           # required unless completed
    units_analysed: Optional[int] = None
    segments: list = field(default_factory=list)   # [{start_s, end_s, note}]
    model: Optional[dict] = None           # MODEL_KEYS; None for mock and for non-model checks
    tested_scope: str = ""
    limitations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)   # [{code, message}]
    errors: list = field(default_factory=list)     # [{code, message}]
    observations: list = field(default_factory=list)   # [{kind, code, text, value}]: metadata, heuristic flags, marker scans, verifications
    timing_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _num01(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0


def validate_module_result(m: dict) -> list[str]:
    bad: list[str] = []
    if not isinstance(m, dict):
        return ["module result is not an object"]
    for k in ModuleResult.__dataclass_fields__:
        if k not in m:
            bad.append(f"missing key {k}")
    if bad:
        return bad
    mid = m["module_id"] or "?"
    if not m["module_id"] or not m["version"]:
        bad.append("module_id and version must be non-empty")
    if m["section"] not in SECTIONS:
        bad.append(f"{mid}: section must be one of {SECTIONS}")
    if m["source"] not in SOURCES:
        bad.append(f"{mid}: source must be one of {SOURCES}")
    if m["status"] not in STATUSES:
        bad.append(f"{mid}: status must be one of {STATUSES}")
        return bad
    if m["source"] == "mock":
        if m["model"] is not None:
            bad.append(f"{mid}: mock result must have model null")
        if isinstance(m["score"], dict) and m["score"].get("calibrated"):
            bad.append(f"{mid}: mock result must not carry a calibrated score")
    elif m["model"] is not None and (not isinstance(m["model"], dict) or any(k not in m["model"] for k in MODEL_KEYS)):
        bad.append(f"{mid}: model block must carry {MODEL_KEYS}")
    for lst in ("limitations", "warnings", "errors", "segments", "observations"):
        if not isinstance(m[lst], list):
            bad.append(f"{mid}: {lst} must be a list")
            return bad
    for w in m["warnings"] + m["errors"]:
        if not isinstance(w, dict) or not w.get("code") or not w.get("message"):
            bad.append(f"{mid}: bad warning or error entry {w}")
    for e in m["errors"]:
        if isinstance(e, dict) and e.get("code") not in ERROR_CODES:
            bad.append(f"{mid}: unknown error code {e.get('code')}")
    for o in m["observations"]:
        if not isinstance(o, dict) or o.get("kind") not in OBSERVATION_KINDS or not o.get("code") or not o.get("text"):
            bad.append(f"{mid}: bad observation {o}")
        elif o["kind"] == "verification" and m["source"] == "mock":
            bad.append(f"{mid}: a mock result cannot carry a verification observation")
    if not isinstance(m["timing_ms"], (int, float)) or isinstance(m["timing_ms"], bool) or m["timing_ms"] < 0:
        bad.append(f"{mid}: timing_ms must be a non-negative number")
    if m["status"] != "completed":
        if m["finding"] is not None or m["score"] is not None:
            bad.append(f"{mid}: finding and score must be null unless completed")
        if not m["reason"]:
            bad.append(f"{mid}: a reason is required unless completed")
        if m["status"] in ("failed", "timed_out") and not m["errors"]:
            bad.append(f"{mid}: failed and timed_out results need an error entry")
        return bad
    if m["section"] in FINDINGS and m["finding"] not in FINDINGS[m["section"]]:
        bad.append(f"{mid}: finding {m['finding']!r} not allowed in section {m['section']}")
    if not m["tested_scope"]:
        bad.append(f"{mid}: completed result needs tested_scope")
    if not m["limitations"]:
        bad.append(f"{mid}: completed result needs at least one limitation")
    if m["errors"]:
        bad.append(f"{mid}: errors present on a completed result")
    s = m["score"]
    if s is not None:
        if not isinstance(s, dict) or any(k not in s for k in ("value", "meaning", "calibrated", "threshold")):
            bad.append(f"{mid}: score must have value, meaning, calibrated, threshold")
        else:
            if not _num01(s["value"]):
                bad.append(f"{mid}: score.value must be in [0, 1]")
            if not s["meaning"]:
                bad.append(f"{mid}: score.meaning must say what the number is")
            if s["calibrated"]:
                ev = (m["model"] or {}).get("evaluation", NOT_EVALUATED)
                if m["source"] != "real" or ev == NOT_EVALUATED:
                    bad.append(f"{mid}: calibrated score needs source real and an evaluation reference")
    if m["source"] == "real" and m["section"] != "provenance" and m["model"] is None:
        bad.append(f"{mid}: real model-based result needs a model block")
    return bad


def validate_report(r: dict) -> list[str]:
    bad: list[str] = []
    if not isinstance(r, dict):
        return ["report is not an object"]
    for k in ("schema_version", "request_id", "input", "probe", "modules", "sections", "coverage", "conflicts",
              "source_summary", "status", "statement"):
        if k not in r:
            bad.append(f"missing key {k}")
    if bad:
        return bad
    for k in FORBIDDEN_REPORT_KEYS:
        if k in r:
            bad.append(f"report must not carry {k} (no global score or verdict)")
    if r["schema_version"] != SCHEMA_VERSION:
        bad.append("schema_version must be '2'")
    if r["statement"] != STATEMENT:
        bad.append("statement differs from the fixed text")
    ids = []
    for m in r["modules"]:
        bad += validate_module_result(m)
        ids.append(m.get("module_id"))
    if len(set(ids)) != len(ids):
        bad.append("duplicate module_id")
    by_id = {m.get("module_id"): m for m in r["modules"]}
    secs = r["sections"]
    if set(secs) != set(SECTIONS):
        bad.append(f"sections must be exactly {SECTIONS}")
    else:
        listed = [i for s in SECTIONS for i in secs[s]]
        if sorted(listed) != sorted(ids):
            bad.append("each module must appear in exactly one section")
        for s in SECTIONS:
            for i in secs[s]:
                if i in by_id and by_id[i]["section"] != s:
                    bad.append(f"{i} listed under the wrong section")
    cov = r["coverage"]
    expect: dict[str, list] = {}
    for m in r["modules"]:
        expect.setdefault(m["status"], []).append(m["module_id"])
    for st in STATUSES:
        if sorted(i["module_id"] for i in cov.get(st, [])) != sorted(expect.get(st, [])):
            bad.append(f"coverage.{st} does not match the module list")
        if st != "completed":
            for i in cov.get(st, []):
                if not i.get("reason"):
                    bad.append(f"coverage entry {i.get('module_id')} lacks a reason")
    srcs = {m["source"] for m in r["modules"]}
    want = "none" if not srcs else (next(iter(srcs)) if len(srcs) == 1 else "mixed")
    if r["source_summary"] != want:
        bad.append(f"source_summary must be {want!r}")
    n_ok = len(expect.get("completed", []))
    n_bad = sum(len(expect.get(s, [])) for s in ("failed", "timed_out", "unsupported"))
    want_state = "failed" if n_ok == 0 else ("partial" if n_bad else "ok")   # nothing analysed is never "ok"
    if not isinstance(r["status"], dict) or r["status"].get("state") != want_state:
        bad.append(f"status.state must be {want_state!r}")
    for c in r["conflicts"]:
        if not (isinstance(c, dict) and c.get("modules") and c.get("note")):
            bad.append(f"bad conflict entry {c}")
    return bad
