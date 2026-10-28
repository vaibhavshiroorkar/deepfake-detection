"""Evidence-grounded explanations (schema 2.1, D3): every number from a field, wording follows status, experimental and mock stay
labelled, failures get no finding, missing values are left out, and the checker catches fabrications. No model, no GPU, no media.

Module results come from the real wrappers driven by test doubles (fake adapter, lip-sync and generated-video doubles as in their own
suites), so the explanations are tested on the shapes the live system produces. Every number here is synthetic test data.
"""
from __future__ import annotations

import base64
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.platform import explain, schema
from dfd.platform.coordinator import Coordinator, build_report
from dfd.platform.genvideo import canonical
from dfd.platform.genvideo.features import PICK
from dfd.platform.genvideo.model import save_head
from dfd.platform.module import Applicability, Context, Probe
from dfd.platform.modules import AdapterProvider, build_default_modules
from dfd.platform.modules import lip_sync as ls
from dfd.platform.modules.generated_video import GeneratedVideoModule
from dfd.platform.provenance import metadata as M
from dfd.platform.provenance.modules import ContentCredentialsModule, MediaMetadataModule
from dfd.platform.schema import ModuleResult
from dfd.platform.syncnet.scoring import SyncResult

FIX = Path(__file__).resolve().parents[2] / "src" / "dfd" / "platform" / "fixtures"
# SYNTHETIC checkpoint metadata (values made up, as in tests/platform_modules); thresholds deliberately unround.
META = {"checkpoint_id": "synthetic-test-checkpoint", "files": {},
        "tasks": {"T-V": {"threshold": {"value": 0.6138830927259571}}, "T-I": {"threshold": {"value": 0.52}},
                  "T-A": {"threshold": {"value": 0.6954}}}}
AUDIO_WARN = {"code": "AUDIO_CALIBRATION_EXPLORATORY", "message": "the audio calibration is exploratory", "modality": "audio"}
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


class FakeAdapter:
    inference_source = "real"
    meta = META

    def __init__(self, per=None, warnings=None):
        self.per, self.warnings = per or {}, warnings or []

    def analyse(self, media, progress=None):
        return AdapterResult(self.per, self.warnings, {}, 0.5, {})


def probe(mt="video", has_audio=True, **kw):
    name = {"image": "x.png", "video": "x.mp4", "audio": "x.wav"}[mt]
    d = dict(path="/tmp/" + name, filename=name, media_type=mt, mime_type="x/y", size_bytes=10, sha256="a" * 64, duration_s=5.0,
             has_video=mt in ("image", "video"), has_audio=has_audio if mt != "image" else False, width=640, height=360)
    d.update(kw)
    return Probe(**d)


def wrappers(per, warnings=()):
    return build_default_modules(AdapterProvider(adapter=FakeAdapter(per, list(warnings))))


def speechlike(seconds=4.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * 120 * h * t) / h for h in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, None) ** 2 + 0.05
    return (0.25 * x * env / np.abs(x * env).max() + 0.004 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)


def tone(seconds=4.0, sr=16000):
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)


class FakeLip(ls.LipSyncModule):
    def __init__(self, wav=None, faces=None, result=None, **kw):
        super().__init__(weights_path="nonexistent.model", device="cpu", **kw)
        self.wav = speechlike() if wav is None else wav
        self.faces = [[SimpleNamespace(x=100, y=80, w=60, h=60, score=0.9)] for _ in range(100)] if faces is None else faces
        self.result = result or SyncResult(0, 4.0, 5.0, 80, 15, np.zeros(31))

    def weights_sha256(self):
        return "a" * 64

    def _weights_available(self):
        return True

    def _load_audio(self, path):
        return self.wav

    def _detect_faces(self, path):
        return self.faces

    def _make_crops(self, path, boxes):
        return np.zeros((len(boxes), 224, 224, 3), np.uint8)

    def _score(self, crops, wav):
        return self.result


def curve_result(best_offset, confidence=5.0, vshift=15):
    """A SyncResult whose curve has its minimum exactly at ``best_offset`` (mean_dist[i] belongs to offset vshift - i)."""
    md = np.array([10.0 + abs((vshift - i) - best_offset) for i in range(2 * vshift + 1)])
    return SyncResult(best_offset, float(md.min()), confidence, 80, vshift, md)


E020_LIKE = {   # wording modelled on the installed head's metadata; numbers are those of E-020 as recorded in that file
    "head_id": "TEST-EXPERIMENTAL", "feature": "cls_mean", "protocol": canonical.PROTOCOL_ID, "validated": False,
    "evaluation": "E-020, experimental, NOT validated: DeepAction leave-one-generator-out macro AUC 0.8281 [0.7908, 0.8662] (development); "
                  "DF26 held-out AUC 0.4957 [0.4755, 0.5168] on all 2691 clips",
    "tested_scope": "Tested only as E-020; chance level on DF26.", "limitations": ["EXPERIMENTAL: the head did not generalise."],
    "score_meaning": "Uncalibrated ranking score in [0, 1]; not a probability."}


class ClsRunner:
    revision = "fake-rev"

    def __init__(self, d0):
        self.d0 = d0

    def features(self, frames):
        f = np.zeros(384, dtype=np.float32)
        f[0] = self.d0
        return {"cls_mean": f}


def genvideo(tmp_path, d0=0.4, frames=None):
    w = np.zeros((384, 1)); w[0, 0] = 1.0
    head = tmp_path / "head.npz"
    save_head(head, np.zeros(384), np.ones(384), [w], [np.zeros(1)], 0.9623, dict(E020_LIKE))
    fr = frames if frames is not None else np.zeros((14, 224, 224, 3), np.uint8)
    return GeneratedVideoModule(head, ClsRunner(d0), canonicalise=lambda s, d: None, read_frames=lambda p: fr)


class Fixed:
    """A module returning a fixed result (for provenance results built from the real modules' helpers)."""
    timeout_s = 5.0
    source = "real"

    def __init__(self, res: ModuleResult):
        self.res, self.module_id, self.version, self.section = res, res.module_id, res.version, res.section

    def applicability(self, p):
        return Applicability.yes()

    def run(self, ctx):
        return copy.deepcopy(self.res)


def run(mods, p):
    r = Coordinator(mods).analyse(p, "test")
    assert schema.validate_report(r) == [], schema.validate_report(r)
    return r


def mod(r, mid):
    return next(m for m in r["modules"] if m["module_id"] == mid)


def all_text(e):
    return " ".join([e["headline"], e.get("does_not_mean", "")] + [s["text"] for s in e["sentences"]])


# ---------------------------------------------------------------------------------------------------------------- structure
def test_build_report_always_attaches_explanations_and_summary():
    r = run(wrappers({"visual": ModalityScore(True, 0.2, 12)}), probe(has_audio=False))
    assert r["schema_version"] == "2.1" and isinstance(r["summary"], dict)
    assert all(isinstance(m["explanation"], dict) and m["explanation"]["sentences"] for m in r["modules"])


def test_schema_fields_are_optional_and_version_bound():
    r = run(wrappers({"visual": ModalityScore(True, 0.2, 12)}), probe(has_audio=False))
    bare = copy.deepcopy(r)
    del bare["summary"]
    for m in bare["modules"]:
        del m["explanation"]
    assert schema.validate_report(bare) == []                       # D3: optional in 2.1
    v2 = copy.deepcopy(r)
    v2["schema_version"] = "2"
    assert any("version 2 report cannot carry" in p for p in schema.validate_report(v2))
    del v2["summary"]
    for m in v2["modules"]:
        del m["explanation"]
    assert schema.validate_report(v2) == []                         # stored version 2 reports stay valid


def test_every_fixture_is_current_and_valid_and_marked_mock():
    from dfd.platform.mocks import SCENARIOS, build_scenario
    for name in SCENARIOS:
        on_disk = json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(build_scenario(name))), f"{name}: run python -m dfd.platform.mocks"
        assert schema.validate_report(on_disk) == [], name
        for m in on_disk["modules"]:
            e = m["explanation"]
            assert e["headline"].startswith("MOCK") and e["sentences"][0]["kind"] == "mock", (name, m["module_id"])
            assert e["key_figure"] is None


def test_explaining_never_changes_the_module_result():
    mods_ = wrappers({"visual": ModalityScore(True, 0.97, 16)})
    p = probe(has_audio=False)
    raw = Coordinator(mods_).analyse(p, "t")
    for m in raw["modules"]:
        before = {k: v for k, v in m.items() if k != "explanation"}
        again = copy.deepcopy(before)
        explain.explain_module(again, raw)
        assert again == before


# ---------------------------------------------------------------------------------------------------------------- numbers
@pytest.mark.parametrize("score", [0.97231, 0.2, 0.6102, 0.6170, 0.61391])
def test_face_score_and_threshold_shown_as_stored_after_declared_rounding(score):
    r = run(wrappers({"visual": ModalityScore(True, score, 16)}), probe(has_audio=False))
    m = mod(r, "face_video")
    thr = META["tasks"]["T-V"]["threshold"]["value"]
    assert m["score"]["threshold"] == thr                            # stored value untouched
    shown = explain.fmt_score(score, thr)
    assert float(shown) != float(explain.fmt(thr)) or len(shown) > 4          # the side of the line stays visible
    assert round(float(shown), len(shown) - 2) == round(score, len(shown) - 2)
    text = all_text(m["explanation"])
    assert shown in text and "0.61 " not in text.replace("line is 0.61.", "")
    assert m["explanation"]["key_figure"]["text"] == f"{shown} (decision line 0.61)"
    hit = score >= thr
    assert m["finding"] == ("indicates_manipulation" if hit else "no_indication_within_tested_scope")
    assert ("at or above" in text) == hit and ("below the decision line" in text) == (not hit)
    assert "0.6138830927259571" not in json.dumps(m["explanation"])  # never the raw float in the readable text


def test_every_number_in_every_explanation_comes_from_a_cited_field(tmp_path):
    reports = [
        run(wrappers({"visual": ModalityScore(True, 0.83, 9), "audio": ModalityScore(True, 0.15311, 1)}, [AUDIO_WARN]), probe()),
        run(wrappers({"image": ModalityScore(True, 0.999983, 1, None, "face_crop")}), probe("image")),
        run([FakeLip(result=curve_result(1, 4.46))], probe()),
        run([FakeLip(result=curve_result(-9, 6.1))], probe()),
        run([genvideo(tmp_path, 1e6)], probe(has_audio=False)),
    ]
    for r in reports:
        for m in r["modules"]:
            for s in m["explanation"]["sentences"]:
                allowed = set(explain.NUMBER_CONSTANTS)
                for src in s["sources"]:
                    allowed |= explain._numbers_in(explain.resolve(m, src, r))
                assert set(NUM.findall(s["text"])) <= allowed, (m["module_id"], s)


# ---------------------------------------------------------------------------------------------------------------- wording per status
def test_no_indication_never_means_unmanipulated():
    r = run(wrappers({"visual": ModalityScore(True, 0.05, 16)}), probe(has_audio=False))
    e = mod(r, "face_video")["explanation"]
    assert e["group"] == "nothing_found" and "No indication" in e["headline"]
    assert "does not establish" in e["does_not_mean"].lower()
    assert not re.search(r"authentic|genuine|\bis real\b|\bsafe\b", all_text(e), re.I)


@pytest.mark.parametrize("status_module", ["failed", "timed_out", "not_applicable", "unsupported"])
def test_modules_that_did_not_complete_get_no_finding_and_no_score(status_module):
    reason = {"failed": "module raised RuntimeError", "timed_out": "no result within 1 s", "not_applicable": "the video has no audio stream",
              "unsupported": "the SyncNet weights file is not installed"}[status_module]
    errors = [{"code": "MODULE_EXCEPTION" if status_module == "failed" else "MODULE_TIMEOUT", "message": "x"}] \
        if status_module in ("failed", "timed_out") else []
    res = ModuleResult("lip_sync", "v", "cross_modal", "real", status_module, reason=reason, errors=errors)
    m = res.to_dict()
    e = explain.explain_module(m, {"modules": [m]})
    m["explanation"] = e
    assert explain.check_explanation(m, {"modules": [m]}) == []
    kinds = {s["kind"] for s in e["sentences"]}
    assert not kinds & {"model_output", "interpretation"} and e["key_figure"] is None and e["finding_label"] is None
    assert reason in all_text(e)
    assert e["group"] == {"failed": "did_not_run", "timed_out": "did_not_run"}.get(status_module, status_module)


def test_partial_report_keeps_other_results_and_counts_the_failure():
    class Boom:
        module_id, version, section, timeout_s, source = "speech", "v", "authenticity", 5.0, "real"

        def applicability(self, p):
            return Applicability.yes()

        def run(self, ctx):
            raise RuntimeError("decoder exploded")
    mods_ = [m for m in wrappers({"visual": ModalityScore(True, 0.3, 16)}) if m.module_id == "face_video"] + [Boom()]
    r = run(mods_, probe())
    assert r["status"]["state"] == "partial"
    assert mod(r, "face_video")["explanation"]["group"] == "nothing_found"
    assert mod(r, "speech")["explanation"]["headline"].startswith("Did not complete")
    assert r["summary"]["counts"]["did_not_run"] == 1 and "1 did not complete" in r["summary"]["headline"]


def test_experimental_generated_video_keeps_its_caveats_and_is_never_a_key_figure(tmp_path):
    r = run([genvideo(tmp_path, 1e6)], probe(has_audio=False))
    m = mod(r, "generated_video")
    e = m["explanation"]
    assert m["finding"] == "inconclusive" and m["score"]["threshold"] is None
    assert e["group"] == "experimental" and "Experimental" in e["headline"] and e["key_figure"] is None
    text = all_text(e)
    assert f"ranking score of {explain.fmt(m['score']['value'])}" in text
    assert "not a probability" in text and "0.4957" in text and "cannot establish whether the video was AI-generated" in text
    assert "Real videos also receive high scores" in text and "338 of 440" in text          # E-026, cited from the module's limitation
    assert any("E-026" in x for x in m["limitations"])
    assert "experimental" in r["summary"]["headline"]
    assert explain.fmt(m["score"]["value"]) not in r["summary"]["headline"]


def test_image_face_crop_path_states_the_generated_image_gap():
    r = run(wrappers({"image": ModalityScore(True, 0.12, 1, None, "face_crop")}), probe("image"))
    text = all_text(mod(r, "image_detector")["explanation"])
    assert "face-swap model built for video frames" in text and "generated-image check did not run" in text
    assert r["summary"]["media_facts"][0]["value"] is True


def test_image_whole_image_path():
    r = run(wrappers({"image": ModalityScore(True, 0.9, 1, None, "whole_image")}), probe("image"))
    e = mod(r, "image_detector")["explanation"]
    assert "fully generated" in e["headline"] and "ProGAN" in all_text(e)
    assert r["summary"]["media_facts"][0]["value"] is False


# ---------------------------------------------------------------------------------------------------------------- speech, lip-sync
def test_speech_on_a_tone_keeps_its_finding_but_explains_the_caveat():
    """UX-5 / D9 at the explanation layer: the speech result is untouched; its explanation cites the lip-sync speech screen."""
    speech = [m for m in wrappers({"audio": ModalityScore(True, 0.0, 1)}, [AUDIO_WARN]) if m.module_id == "speech"]
    r = run(speech + [FakeLip(wav=tone())], probe())
    s, lip = mod(r, "speech"), mod(r, "lip_sync")
    assert lip["warnings"][0]["code"] == "AUDIO_NOT_SPEECH_LIKE"
    assert s["finding"] == "no_indication_within_tested_scope" and s["score"]["value"] == 0.0   # detector decision unchanged
    text = all_text(s["explanation"])
    assert "should not be relied on" in text and "does not check whether the audio contains speech" in text
    assert r["summary"]["media_facts"][1]["value"] is False


def test_speech_fact_is_not_inferred_from_an_audio_stream():
    r = run([m for m in wrappers({"audio": ModalityScore(True, 0.2, 1)}, [AUDIO_WARN]) if m.module_id == "speech"], probe("audio"))
    fact = r["summary"]["media_facts"][1]
    assert fact["value"] is None and "does not mean it contains speech" in fact["text"]
    faces = r["summary"]["media_facts"][0]
    assert faces["value"] is None and "audio file" in faces["text"]


def test_lip_sync_curve_is_the_scored_curve_and_decisions_are_unchanged():
    for off, want in ((1, "consistent"), (-6, "consistent"), (2, "consistent"), (3, "potential_mismatch"), (-9, "potential_mismatch")):
        res = FakeLip(result=curve_result(off)).run(Context(probe()))
        assert res.finding == want
        ev = res.evidence[0]
        assert ev["kind"] == "offset_curve" and ev["band"] == list(ls.OFFSET_BAND_FRAMES) == [-6, 2]
        assert ev["offsets"] == list(range(-15, 16)) and len(ev["mean_dist"]) == 31
        assert ev["offsets"][int(np.argmin(ev["mean_dist"]))] == off == ev["best_offset"]
        assert schema.validate_module_result(res.to_dict()) == []
    assert ls.OFFSET_BAND_FRAMES == (-6, 2) and ls.MIN_CONFIDENCE == 3.0


def test_lip_sync_explanations_follow_the_finding():
    ok = mod(run([FakeLip(result=curve_result(1, 4.46))], probe()), "lip_sync")["explanation"]
    assert "No audio-visual timing mismatch" in ok["headline"] and "+1 frames" in all_text(ok) and "-6 to +2" in all_text(ok)
    assert "passes this check" in all_text(ok)
    bad = mod(run([FakeLip(result=curve_result(-9))], probe()), "lip_sync")["explanation"]
    assert bad["group"] == "attention" and "not proof of manipulation" in all_text(bad)
    low = mod(run([FakeLip(result=curve_result(0, confidence=1.2))], probe()), "lip_sync")["explanation"]
    assert low["group"] == "inconclusive" and "could not reach a conclusion" in low["headline"] and "1.20" in all_text(low)


def test_lip_sync_without_evidence_omits_the_band_sentence():
    """Missing values are left out, never filled with defaults (an older result without the curve)."""
    m = FakeLip(result=curve_result(1)).run(Context(probe())).to_dict()
    m["evidence"] = []
    e = explain.explain_module(m, {"modules": [m]})
    text = all_text(e)
    assert "+1 frames" in text and "band" not in text and "minimum for a timing result" not in text


def test_lip_sync_no_face_records_that_the_speech_screen_passed():
    r = run([FakeLip(faces=[[] for _ in range(100)])], probe())
    m = mod(r, "lip_sync")
    assert m["status"] == "not_applicable" and m["observations"][0]["code"] == "speech_like_screen"
    assert r["summary"]["media_facts"][1]["value"] is True


# ---------------------------------------------------------------------------------------------------------------- generated video frames
def test_frame_strip_is_the_eight_frames_the_encoder_received(tmp_path):
    frames = np.stack([np.full((224, 224, 3), 10 * i, np.uint8) for i in range(14)])
    res = genvideo(tmp_path, 0.4, frames).run(Context(probe(has_audio=False)))
    ev = res.evidence[0]
    assert ev["kind"] == "frames" and len(ev["frames"]) == len(PICK) == ev["n_frames"] == 8 and ev["n_decoded"] == res.units_analysed == 14
    assert "saw only those 8 frames" in all_text(explain.explain_module(res.to_dict()))
    assert ev["times_s"] == [round(int(i) / canonical.FPS, 2) for i in PICK]
    for uri, i in zip(ev["frames"], PICK):
        img = cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (112, 112, 3) and abs(int(img.mean()) - 10 * int(i)) <= 2      # frame order and content preserved
    plain = GeneratedVideoModule(genvideo(tmp_path, 0.4, frames).head_path, ClsRunner(0.4), canonicalise=lambda s, d: None,
                                 read_frames=lambda p: frames).run(Context(probe(has_audio=False)))
    assert plain.score == res.score and plain.finding == res.finding == "inconclusive"


# ---------------------------------------------------------------------------------------------------------------- provenance
def test_metadata_and_credentials_are_explained_as_descriptions_not_findings(monkeypatch):
    info = {"format": {"format_name": "mov,mp4", "duration": "5.0", "tags": {"encoder": "Lavf61.7.100"}},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "r_frame_rate": "25/1"}]}
    monkeypatch.setattr(M, "ffprobe", lambda p: info)
    md = MediaMetadataModule().run(Context(probe()))
    cc = ContentCredentialsModule()._build("no_credentials_found", {"scanned_bytes": 10, "found": [], "both_markers": False, "truncated": False},
                                           [{"kind": "marker_scan", "code": "C2PA_MARKER_ABSENT", "text": "no manifest", "value": []}],
                                           ["Absence of credentials proves nothing: credentials are optional.", "SynthID and other invisible watermarks are not checked."],
                                           {"status": "absent", "library": {}})
    r = run([Fixed(md), Fixed(cc)], probe())
    e_md, e_cc = mod(r, "media_metadata")["explanation"], mod(r, "content_credentials")["explanation"]
    assert e_md["group"] == "informational" and "not evidence either way" in e_md["headline"] and "1280x720" in all_text(e_md)
    assert e_cc["group"] == "nothing_found" and "says nothing about the content" in all_text(e_cc)
    assert "SynthID" in all_text(e_cc)


# ---------------------------------------------------------------------------------------------------------------- the checker
def _face_report():
    return run(wrappers({"visual": ModalityScore(True, 0.97, 16)}), probe(has_audio=False))


@pytest.mark.parametrize("tamper,expect", [
    (lambda e: e["sentences"].__setitem__(1, {**e["sentences"][1], "text": "The model gave a score of 0.42."}), "number 0.42"),
    (lambda e: e.__setitem__("headline", "The video is authentic"), "forbidden"),
    (lambda e: e["sentences"].append({"kind": "observation", "text": "A blending seam was seen.", "sources": ["evidence[kind=heatmap]"]}), "missing"),
    (lambda e: e["sentences"].append({"kind": "unknown", "text": "x", "sources": ["status"]}), "needs kind"),
    (lambda e: e.__setitem__("group", "nothing_found"), "does not match"),
    (lambda e: e["sentences"].append({"kind": "observation", "text": "Nothing.", "sources": ["status"], "quoted": True}), "marked quoted"),
])
def test_checker_catches_fabricated_or_inconsistent_explanations(tamper, expect):
    r = _face_report()
    tamper(mod(r, "face_video")["explanation"])
    problems = schema.validate_report(r)
    assert any(expect in p for p in problems), problems


def test_checker_rejects_a_summary_that_disagrees_with_the_modules():
    r = _face_report()
    r["summary"]["counts"]["attention"] = 0
    assert "summary does not match the module results" in schema.validate_report(r)


def test_checker_rejects_experimental_key_figures(tmp_path):
    r = run([genvideo(tmp_path, 1e6)], probe(has_audio=False))
    mod(r, "generated_video")["explanation"]["key_figure"] = {"label": "Score", "text": "1.00", "sources": ["score.value"]}
    assert any("experimental score must not be a key figure" in p for p in schema.validate_report(r))


def test_a_template_bug_falls_back_without_losing_the_report(monkeypatch):
    monkeypatch.setitem(explain._TEMPLATES, "face_video", lambda m, r: 1 / 0)
    r = run(wrappers({"visual": ModalityScore(True, 0.97, 16)}), probe(has_audio=False))
    e = mod(r, "face_video")["explanation"]
    assert e["template_id"] == f"fallback.{explain.TEMPLATE_VERSION}" and "No explanation could be produced" in e["headline"]
    assert mod(r, "face_video")["finding"] == "indicates_manipulation"


def test_summary_has_no_overall_score_or_verdict():
    r = run(wrappers({"visual": ModalityScore(True, 0.97, 16), "audio": ModalityScore(True, 0.1, 1)}, [AUDIO_WARN]), probe())
    s = r["summary"]
    assert set(s) == {"headline", "counts", "groups", "media_facts", "disagreements", "no_overall_score"}
    assert not re.search(r"\d\.\d", json.dumps({k: v for k, v in s.items() if k != "media_facts"}))
    assert s["disagreements"] == [{"modules": r["conflicts"][0]["modules"], "text": r["conflicts"][0]["note"]}]


def test_vocabulary_covers_every_status_and_finding():
    v = explain.vocabulary()
    assert set(v["statuses"]) == set(schema.STATUSES)
    assert set(v["findings"]) == {f for fs in schema.FINDINGS.values() for f in fs}
    assert v["statement"] == schema.STATEMENT and v["schema_versions"] == ["2", "2.1"]


# ---------------------------------------------------------------------------------------------------------------- v2: processing and decision
def test_every_sentence_kind_has_exactly_one_heading_on_the_dashboard():
    parts = explain.vocabulary()["explanation_parts"]
    kinds = [k for p in parts for k in p["kinds"]]
    assert sorted(kinds) == sorted(explain.KINDS) and len(set(kinds)) == len(kinds)
    assert [p["label"] for p in parts][1:6] == ["What we did", "What we examined", "What we observed", "Why this was reported", "What it means"]


def test_face_processing_cites_the_recorded_frame_count_and_is_left_out_without_it():
    r = run(wrappers({"visual": ModalityScore(True, 0.2, 11)}), probe(has_audio=False))
    m = mod(r, "face_video")
    sents = m["explanation"]["sentences"]
    assert "decision" in [s["kind"] for s in sents]
    assert not any("frames at even intervals" in s["text"] for s in sents)        # the probe path is not a real file: no record, no sentence
    m2 = copy.deepcopy(m)
    m2["observations"] = [{"kind": "processing", "code": "frames_sampled", "text": "frames sampled", "value": 16}]
    m2["explanation"] = explain.explain_module(m2, r)
    s = next(x for x in m2["explanation"]["sentences"] if "frames at even intervals" in x["text"])
    assert "It took 16 frames" in s["text"] and "11 of them contained a face" in s["text"] and s["kind"] == "processing"
    assert explain.check_explanation(m2, r) == []
    s["text"] = s["text"].replace("16 frames", "15 frames")
    assert any("number 15" in p for p in explain.check_explanation(m2, r))


def test_processing_or_decision_is_refused_for_a_check_that_did_not_run():
    r = run(wrappers({"visual": ModalityScore(False, None, 0, "no face detected")}), probe(has_audio=False))
    m = mod(r, "face_video")
    assert m["status"] == "not_applicable" and explain.check_explanation(m, r) == []
    for kind in ("processing", "decision"):
        mm = copy.deepcopy(m)
        mm["explanation"]["sentences"].append({"kind": kind, "text": "Frames were scored.", "sources": ["status"]})
        assert any("did not complete" in p for p in explain.check_explanation(mm, r)), kind


def test_generated_video_says_it_averages_frames_and_ignores_motion(tmp_path):
    frames = np.stack([np.full((224, 224, 3), 10 * i, np.uint8) for i in range(14)])
    res = genvideo(tmp_path, 0.4, frames).run(Context(probe(has_audio=False))).to_dict()
    res["model"]["name"] = "Appearance baseline: logistic regression on the mean DINOv2-small CLS embedding of 8 frames"
    e = explain.explain_module(res)
    proc = [s["text"] for s in e["sentences"] if s["kind"] == "processing"]
    assert any("averaged into one vector" in t and "not movement" in t for t in proc)
    assert any(s["kind"] == "decision" and "makes no finding" in s["text"] for s in e["sentences"])
    res["explanation"] = e
    assert explain.check_explanation(res) == []


def test_metadata_chip_says_information_only():
    r = run(wrappers({}), probe(has_audio=False))
    for m in r["modules"]:
        if explain.group_of(m) == "informational":
            assert m["explanation"]["finding_label"] == "Information only"


def test_applies_to_table_matches_every_module_applicability():
    from dfd.platform.pipeline import default_modules
    from dfd.platform.module import NOT_APPLICABLE
    mods = {m.module_id: m for m in default_modules()}
    assert set(explain.APPLIES_TO) == set(mods)
    for mid, (kinds, _needs) in explain.APPLIES_TO.items():
        for mt in ("image", "video", "audio"):
            a = mods[mid].applicability(probe(mt, has_audio=True))
            assert (a.kind != NOT_APPLICABLE) == (mt in kinds), (mid, mt, a)
