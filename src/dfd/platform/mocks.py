"""Scenario mocks for dashboard development (WS2). Everything here is MOCK: no model ran, no number is a measurement.

Each scenario is a list of scripted modules run through the real coordinator, so the fixtures have exactly the shape
the live system will produce. Every module has ``source = "mock"``, no model block, and a tested_scope and
limitations that say so. ``python -m dfd.platform.mocks`` rewrites ``fixtures/*.json``; a test checks they are current.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .coordinator import Coordinator
from .module import Applicability, Context, Probe
from .schema import ModuleResult

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MOCK_NOTE = "MOCK: invented by a scenario script; no model ran."
_SCORE_MEANING = "MOCK number with no meaning; it is not a measurement."


@dataclass
class ScriptedModule:
    """A mock module whose behaviour is fixed by the scenario. ``outcome`` is a finding, or one of
    not_applicable, unsupported, raise, sleep (the timeout path), or invalid (a result that breaks the contract)."""
    module_id: str
    section: str
    outcome: str
    reason: Optional[str] = None
    score: Optional[float] = None
    segments: Optional[list] = None
    warnings: Optional[list] = None
    needs: Optional[str] = None            # video, audio or both: simple applicability from the probe
    version: str = "mock-0"
    timeout_s: float = 5.0
    source: str = "mock"
    sleep_s: float = 0.0

    def applicability(self, probe: Probe) -> Applicability:
        if self.outcome == "unsupported":
            return Applicability.unsupported(self.reason or "MOCK: input kind not handled")
        if self.outcome == "not_applicable":
            return Applicability.no(self.reason or "MOCK: input lacks what this module needs")
        if self.needs in ("video", "both") and not probe.has_video:
            return Applicability.no("no video stream")
        if self.needs in ("audio", "both") and not probe.has_audio:
            return Applicability.no("no audio stream")
        return Applicability.yes()

    def run(self, ctx: Context) -> ModuleResult:
        if self.outcome == "raise":
            raise RuntimeError("MOCK: scripted failure")
        if self.outcome == "sleep":
            time.sleep(self.sleep_s)
        base = dict(module_id=self.module_id, version=self.version, section=self.section, source="mock")
        if self.outcome == "invalid":
            return ModuleResult(**base, status="completed", finding="genuine", tested_scope="x", limitations=["x"])
        return ModuleResult(
            **base, status="completed", finding=self.outcome if self.outcome != "sleep" else "inconclusive",
            score=None if self.score is None else {"value": self.score, "meaning": _SCORE_MEANING, "calibrated": False, "threshold": None},
            units_analysed=16, segments=self.segments or [], warnings=self.warnings or [],
            tested_scope=MOCK_NOTE, limitations=[MOCK_NOTE, "Scenario data for dashboard development only."], timing_ms=1.0)


def _m(mid, section, outcome, **kw):
    return ScriptedModule(mid, section, outcome, **kw)


_NO = "no_indication_within_tested_scope"
_HIT = "indicates_manipulation"

# scenario name -> (probe overrides, modules)
SCENARIOS: dict[str, tuple[dict, Callable[[], list]]] = {
    "video_nothing_found": ({}, lambda: [
        _m("face_video", "authenticity", _NO, score=0.12, needs="video"), _m("speech", "authenticity", _NO, score=0.20, needs="audio"),
        _m("generated_video", "authenticity", "inconclusive", needs="video"), _m("lip_sync", "cross_modal", "consistent", needs="both"),
        _m("c2pa", "provenance", "no_credentials_found")]),
    "video_manipulation_indicated": ({}, lambda: [
        _m("face_video", "authenticity", _HIT, score=0.97, needs="video", segments=[{"start_s": 1.0, "end_s": 3.5, "note": "MOCK segment"}]),
        _m("speech", "authenticity", _NO, score=0.30, needs="audio"), _m("lip_sync", "cross_modal", "potential_mismatch", needs="both"),
        _m("c2pa", "provenance", "no_credentials_found")]),
    "conflicting_modules": ({}, lambda: [
        _m("face_video", "authenticity", _HIT, score=0.91, needs="video"), _m("generated_video", "authenticity", _NO, score=0.15, needs="video"),
        _m("speech", "authenticity", "inconclusive", needs="audio")]),
    "no_face": ({"n_faces": 0}, lambda: [
        _m("face_video", "authenticity", "not_applicable", reason="no face found in the sampled frames"),
        _m("generated_video", "authenticity", "inconclusive", needs="video"), _m("speech", "authenticity", _NO, score=0.25, needs="audio"),
        _m("lip_sync", "cross_modal", "not_applicable", reason="no face found, so no mouth region to compare with the audio")]),
    "multiple_speakers": ({"n_faces": 3}, lambda: [
        _m("face_video", "authenticity", "inconclusive", needs="video",
           warnings=[{"code": "MULTIPLE_FACES", "message": "MOCK: 3 faces; only the largest is scored"}]),
        _m("speech", "authenticity", "inconclusive", needs="audio",
           warnings=[{"code": "MULTIPLE_SPEAKERS", "message": "MOCK: overlapping speakers lower reliability"}]),
        _m("lip_sync", "cross_modal", "inconclusive", needs="both",
           warnings=[{"code": "SPEAKER_NOT_ATTRIBUTED", "message": "MOCK: cannot tell which face is speaking"}])]),
    "no_audio": ({"has_audio": False, "speech_detected": False}, lambda: [
        _m("face_video", "authenticity", _NO, score=0.2, needs="video"), _m("speech", "authenticity", "inconclusive", needs="audio"),
        _m("lip_sync", "cross_modal", "inconclusive", needs="both"), _m("c2pa", "provenance", "no_credentials_found")]),
    "module_failure": ({}, lambda: [
        _m("face_video", "authenticity", "raise"), _m("speech", "authenticity", _NO, score=0.2, needs="audio"),
        _m("c2pa", "provenance", "no_credentials_found")]),
    "module_timeout": ({}, lambda: [
        _m("generated_video", "authenticity", "sleep", sleep_s=0.4, timeout_s=0.05), _m("speech", "authenticity", _NO, score=0.2, needs="audio")]),
    "unsupported_input": ({"media_type": "audio", "has_video": False}, lambda: [
        _m("face_video", "authenticity", "not_applicable", reason="no video stream"),
        _m("c2pa", "provenance", "unsupported", reason="MOCK: this container is not a format the credential reader handles"),
        _m("speech", "authenticity", _NO, score=0.3, needs="audio")]),
    "provenance_inconclusive": ({}, lambda: [
        _m("face_video", "authenticity", _NO, score=0.1, needs="video"),
        _m("c2pa", "provenance", "inconclusive")]),
}


def _probe(overrides: dict) -> Probe:
    kw = dict(path="MOCK", filename="mock_clip.mp4", media_type="video", mime_type="video/mp4", size_bytes=1_000_000,
              sha256="0" * 64, duration_s=8.0, has_video=True, has_audio=True, width=640, height=360, container="mov,mp4",
              n_faces=1, speech_detected=True)
    kw.update(overrides)
    return Probe(**kw)


def build_scenario(name: str) -> dict:
    overrides, make = SCENARIOS[name]
    report = Coordinator(make(), clock=lambda: 0.0).analyse(_probe(overrides), request_id=f"mock-{name}")
    for m in report["modules"]:
        m["timing_ms"] = 0.0
    return report


def write_fixtures(out: Path = FIXTURE_DIR) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SCENARIOS:
        p = out / f"{name}.json"
        p.write_text(json.dumps(build_scenario(name), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_fixtures():
        print(p)
