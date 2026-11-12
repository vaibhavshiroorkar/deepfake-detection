"""Response structure of the inference library (D-027, schema version 1) and its contract checker.

``validate_response`` returns a list of contract violations (empty when the response obeys the contract). The same
checker is used in the tests for every adapter, so that a mock result can never pass as real (D-029) and a failed
request always has ``result`` null and at least one error.
"""
from __future__ import annotations

SCHEMA_VERSION = "1"
SCORE_MEANING = ("Calibrated probability that this content is fake, from a model fitted on the training data of this project; "
                 "valid only for the kinds of content in that data; it is not proof.")
MEDIA_TYPES = ("image", "video", "audio")
MODALITIES = ("visual", "audio", "image")
STAGES = ("validate", "decode", "detect", "score", "calibrate", "fuse")
STATES = ("ok", "partial", "failed")
DECISIONS = ("fake", "real", "undetermined")
SOURCES = ("real", "mock")
MODEL_KEYS = ("checkpoint_id", "config_hash", "preprocess_id", "git_commit", "iteration", "protocol", "calibration", "threshold")
ERROR_CODES = ("NO_FILE", "EMPTY_FILE", "FILE_TOO_LARGE", "UNSUPPORTED_MEDIA_TYPE", "UNREADABLE_FILE", "ADAPTER_FAILED")
WARNING_CODES = ("NO_FACE_DETECTED", "MULTIPLE_FACES", "SHORT_AUDIO", "LOW_RESOLUTION", "AUDIO_CALIBRATION_EXPLORATORY")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_response(r: dict) -> list[str]:
    bad: list[str] = []
    if not isinstance(r, dict):
        return ["response is not an object"]
    for k in ("schema_version", "request_id", "input", "inference_source", "model", "result", "score_meaning", "status", "warnings", "errors"):
        if k not in r:
            bad.append(f"missing key {k}")
    if bad:
        return bad
    if r["schema_version"] != SCHEMA_VERSION:
        bad.append("schema_version must be '1'")
    if r["score_meaning"] != SCORE_MEANING:
        bad.append("score_meaning text differs from the fixed text")
    if r["inference_source"] not in SOURCES:
        bad.append("inference_source must be 'real' or 'mock'")
    if r["inference_source"] == "mock" and r["model"] is not None:
        bad.append("mock response must have model null")
    inp = r["input"]
    for k in ("filename", "media_type", "mime_type", "size_bytes", "sha256", "duration_s", "frames_used"):
        if not isinstance(inp, dict) or k not in inp:
            bad.append(f"input missing {k}")
    st = r["status"]
    if not isinstance(st, dict) or st.get("state") not in STATES:
        bad.append("status.state must be ok, partial or failed")
        return bad
    if not _is_num(st.get("processing_time_ms")):
        bad.append("status.processing_time_ms must be a number")
    for s in st.get("stages", []):
        if s.get("name") not in STAGES or not _is_num(s.get("ms")):
            bad.append(f"bad stage entry {s}")
    for w in r["warnings"]:
        if w.get("code") not in WARNING_CODES or not w.get("message"):
            bad.append(f"bad warning {w}")
    for e in r["errors"]:
        if e.get("code") not in ERROR_CODES or not e.get("message"):
            bad.append(f"bad error {e}")
    if st["state"] == "failed":
        if r["result"] is not None:
            bad.append("failed response must have result null")
        if not r["errors"]:
            bad.append("failed response must list at least one error")
        return bad
    if r["errors"]:
        bad.append("errors present but state is not failed")
    if r["inference_source"] == "real":
        if not isinstance(r["model"], dict) or any(k not in r["model"] for k in MODEL_KEYS):
            bad.append("real response must carry all model keys")
    res = r["result"]
    if not isinstance(res, dict) or "fused" not in res or "per_modality" not in res:
        bad.append("result must have fused and per_modality")
        return bad
    f = res["fused"]
    if f.get("decision") not in DECISIONS:
        bad.append("fused.decision invalid")
    if f.get("score") is None:
        if f.get("decision") != "undetermined":
            bad.append("fused.score null requires decision undetermined")
    elif not (_is_num(f["score"]) and 0.0 <= f["score"] <= 1.0):
        bad.append("fused.score must be in [0, 1]")
    pm = res["per_modality"]
    if not pm:
        bad.append("per_modality is empty")
    for name, m in pm.items():
        if name not in MODALITIES:
            bad.append(f"unknown modality {name}")
        if "available" not in m or "score" not in m or "n_units" not in m:
            bad.append(f"modality {name} missing available, score or n_units")
            continue
        if m["available"]:
            if not (_is_num(m["score"]) and 0.0 <= m["score"] <= 1.0):
                bad.append(f"modality {name} available but score invalid")
        else:
            if m["score"] is not None or not m.get("reason"):
                bad.append(f"modality {name} unavailable needs score null and a reason")
    return bad
