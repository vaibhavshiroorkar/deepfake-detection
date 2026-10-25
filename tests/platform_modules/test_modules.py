"""Unit tests of the three wrapper modules with a fake adapter (CPU only, no model is loaded)."""
import json
import re

import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.infer.real_adapter import AUDIO_CALIBRATION_WARNING
from dfd.platform import schema
from dfd.platform.coordinator import Coordinator
from dfd.platform.module import Context, Probe
from dfd.platform.modules import AdapterProvider, SpeechModule, build_default_modules
from dfd.platform.modules.base import DEFAULT_CHECKPOINT, LARGEST_FACE_LIMITATION

META = json.loads((DEFAULT_CHECKPOINT / "metadata.json").read_text(encoding="utf-8"))
T_V, T_I, T_A = (META["tasks"][t]["threshold"]["value"] for t in ("T-V", "T-I", "T-A"))
FORBIDDEN = re.compile(r"\b(real|genuine|authentic\w*)\b", re.I)


class FakeAdapter:
    inference_source = "real"
    meta = META

    def __init__(self, per=None, warnings=None, boom=False):
        self.per, self.warnings, self.boom, self.calls = per or {}, warnings or [], boom, []

    def analyse(self, media, progress=None):
        self.calls.append(media)
        if self.boom:
            raise RuntimeError("decoder exploded")
        return AdapterResult(self.per, self.warnings, {}, 0.5, {})


def probe(mt, sha="a" * 64, has_audio=None, has_video=None):
    name = {"image": "x.png", "video": "x.mp4", "audio": "x.wav"}[mt]
    return Probe(path="/tmp/" + name, filename=name, media_type=mt, mime_type="x/y", size_bytes=10, sha256=sha, duration_s=1.0,
                 has_video=has_video if has_video is not None else mt in ("image", "video"), has_audio=has_audio)


def texts(d):
    out = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and key not in ("source", "sha256", "module_id", "section"):
            out.append(x)
    walk(d)
    return out


def mods(adapter):
    return build_default_modules(AdapterProvider(adapter=adapter))


def test_default_factory_and_timeouts():
    ms = mods(FakeAdapter())
    assert [m.module_id for m in ms] == ["face_video", "image_detector", "speech"]
    assert all(m.timeout_s == 120 and m.section == "authenticity" and m.source == "real" for m in ms)
    assert ms[0].version == META["checkpoint_id"]


@pytest.mark.parametrize("mt,audio,want", [
    ("image", False, ["not_applicable", "applicable", "not_applicable"]),
    ("video", True, ["applicable", "not_applicable", "applicable"]),
    ("video", False, ["applicable", "not_applicable", "not_applicable"]),
    ("video", None, ["applicable", "not_applicable", "applicable"]),
    ("audio", True, ["not_applicable", "not_applicable", "applicable"])])
def test_applicability(mt, audio, want):
    p = probe(mt, has_audio=audio)
    apps = [m.applicability(p) for m in mods(FakeAdapter())]
    assert [a.kind for a in apps] == want
    assert all(a.kind == "applicable" or a.reason for a in apps)


def test_video_without_video_stream_not_applicable():
    assert mods(FakeAdapter())[0].applicability(probe("video", has_video=False)).kind == "not_applicable"


def test_no_face_is_not_applicable_never_a_finding():
    ad = FakeAdapter({"visual": ModalityScore(False, None, 0, "no face detected"), "audio": ModalityScore(False, None, None, "video has no audio track")},
                     [{"code": "NO_FACE_DETECTED", "message": "no face was found in the sampled frames; the visual score is not available.", "modality": "visual"}])
    r = mods(ad)[0].run(Context(probe("video", has_audio=False)))
    assert r.status == "not_applicable" and r.finding is None and r.score is None and "no face" in r.reason
    assert r.warnings[0]["code"] == "NO_FACE_DETECTED"
    assert schema.validate_module_result(r.to_dict()) == []


def test_multi_face_limitation_and_warning():
    ad = FakeAdapter({"visual": ModalityScore(True, 0.9, 16)},
                     [{"code": "MULTIPLE_FACES", "message": "some frames show several faces; the largest face is scored.", "modality": "visual"}])
    r = mods(ad)[0].run(Context(probe("video", has_audio=False)))
    assert r.status == "completed" and LARGEST_FACE_LIMITATION in r.limitations and r.warnings[0]["code"] == "MULTIPLE_FACES"
    r1 = mods(FakeAdapter({"visual": ModalityScore(True, 0.9, 16)}))[0].run(Context(probe("video", has_audio=False)))
    assert LARGEST_FACE_LIMITATION not in r1.limitations


@pytest.mark.parametrize("score,finding", [(T_V, "indicates_manipulation"), (T_V + 1e-6, "indicates_manipulation"),
                                          (T_V - 1e-6, "no_indication_within_tested_scope"), (0.0, "no_indication_within_tested_scope")])
def test_threshold_mapping_video(score, finding):
    r = mods(FakeAdapter({"visual": ModalityScore(True, score, 16)}))[0].run(Context(probe("video", has_audio=False)))
    assert r.finding == finding and r.score["threshold"] == T_V and r.score["calibrated"] is True
    assert "not an overall verdict" in r.score["meaning"] and r.model["weights_sha256"] == META["files"]["weights_T-V.pt"]
    assert schema.validate_module_result(r.to_dict()) == []


def test_image_paths():
    r = mods(FakeAdapter({"image": ModalityScore(True, 0.7, 1, None, "face_crop")}))[1].run(Context(probe("image")))
    assert r.score["threshold"] == T_V and r.finding == "indicates_manipulation" and r.score["calibrated"] is False
    assert "video-trained model scored a still image" in " ".join(r.limitations) and "path=face_crop" in r.tested_scope
    assert "Not evaluated on still face-swapped photos" in r.model["evaluation"]
    r2 = mods(FakeAdapter({"image": ModalityScore(True, T_I - 0.01, 1, None, "whole_image")}))[1].run(Context(probe("image")))
    assert r2.score["threshold"] == T_I and r2.finding == "no_indication_within_tested_scope" and r2.score["calibrated"] is True
    assert "path=whole_image" in r2.tested_scope and r2.model["weights_sha256"] == META["files"]["head_T-I.npz"]
    r3 = mods(FakeAdapter({"image": ModalityScore(False, None, None, "no face found and this checkpoint has no whole-image model")}))[1].run(Context(probe("image")))
    assert r3.status == "not_applicable"
    for x in (r, r2, r3):
        assert schema.validate_module_result(x.to_dict()) == []


@pytest.mark.parametrize("score,finding", [(T_A, "indicates_manipulation"), (T_A - 1e-6, "no_indication_within_tested_scope")])
def test_speech(score, finding):
    ad = FakeAdapter({"audio": ModalityScore(True, score, 1)}, [dict(AUDIO_CALIBRATION_WARNING)])
    r = mods(ad)[2].run(Context(probe("audio", has_audio=True)))
    assert r.finding == finding and r.score["calibrated"] is False and r.score["threshold"] == T_A
    lim = " ".join(r.limitations)
    assert "64,600" in lim and "exploratory" in lim and "0.6462" in lim
    assert r.warnings[0]["code"] == "AUDIO_CALIBRATION_EXPLORATORY"
    assert "head_T-A_AU-8.npz" in r.model["weights_sha256"]
    assert schema.validate_module_result(r.to_dict()) == []


def test_speech_unavailable_not_applicable():
    r = mods(FakeAdapter({"audio": ModalityScore(False, None, None, "no decodable audio")}))[2].run(Context(probe("audio", has_audio=True)))
    assert r.status == "not_applicable" and r.reason == "no decodable audio"


def test_one_analysis_per_file_and_cache_key():
    ad = FakeAdapter({"visual": ModalityScore(True, 0.9, 16), "audio": ModalityScore(True, 0.8, 1)})
    fv, _, sp = build_default_modules(AdapterProvider(adapter=ad))
    ctx = Context(probe("video", has_audio=True))
    fv.run(ctx)
    sp.run(ctx)
    assert len(ad.calls) == 1 and ("dfd_adapter_result:" + "a" * 64) in ctx.cache
    sp.run(Context(probe("video", has_audio=True)))     # a new context does not reuse the cache
    assert len(ad.calls) == 2
    assert ad.calls[0].has_audio is True and str(ad.calls[0].path).endswith("x.mp4")


def test_adapter_exception_propagates():
    with pytest.raises(RuntimeError):
        mods(FakeAdapter(boom=True))[0].run(Context(probe("video", has_audio=False)))


def test_missing_checkpoint_raises_clear_error(tmp_path):
    from dfd.infer.checkpoint import NoCheckpointError
    ms = build_default_modules(AdapterProvider(tmp_path / "none"))
    assert ms[0].version == "checkpoint-unavailable"
    with pytest.raises(NoCheckpointError):
        ms[0].run(Context(probe("video", has_audio=False)))


def test_no_forbidden_words_in_any_field():
    ad = FakeAdapter({"visual": ModalityScore(True, 0.1, 16), "audio": ModalityScore(True, 0.9, 1), "image": ModalityScore(True, 0.1, 1, None, "face_crop")},
                     [dict(AUDIO_CALIBRATION_WARNING), {"code": "MULTIPLE_FACES", "message": "several faces; the largest face is scored.", "modality": "visual"}])
    fv, im, sp = mods(ad)
    for m, p in ((fv, probe("video", has_audio=True)), (sp, probe("video", has_audio=True)), (im, probe("image"))):
        bad = [t for t in texts(m.run(Context(p)).to_dict()) if FORBIDDEN.search(t)]
        assert not bad, bad


def test_coordinator_end_to_end_with_one_failing_module():
    ad = FakeAdapter({"visual": ModalityScore(True, 0.95, 16), "audio": ModalityScore(True, 0.2, 1)}, [dict(AUDIO_CALIBRATION_WARNING)])
    fv, im, sp = mods(ad)

    class Boom(SpeechModule):
        module_id = "speech_boom"

        def run(self, ctx):
            raise RuntimeError("boom")

    rep = Coordinator([fv, im, sp, Boom(fv.provider)]).analyse(probe("video", has_audio=True), "req-1")
    assert schema.validate_report(rep) == []
    by = {m["module_id"]: m for m in rep["modules"]}
    assert by["face_video"]["status"] == "completed" and by["face_video"]["finding"] == "indicates_manipulation"
    assert by["speech"]["status"] == "completed" and by["speech"]["finding"] == "no_indication_within_tested_scope"
    assert by["image_detector"]["status"] == "not_applicable"
    assert by["speech_boom"]["status"] == "failed" and by["speech_boom"]["errors"][0]["code"] == "MODULE_EXCEPTION"
    assert rep["status"]["state"] == "partial" and len(ad.calls) == 1 and rep["conflicts"]
    assert not [t for t in texts(rep["modules"]) if FORBIDDEN.search(t)]


def test_coordinator_image_and_audio_only():
    ad = FakeAdapter({"image": ModalityScore(True, 0.9, 1, None, "face_crop"), "audio": ModalityScore(True, 0.9, 1)})
    for p, done in ((probe("image"), "image_detector"), (probe("audio", has_audio=True), "speech")):
        rep = Coordinator(mods(ad)).analyse(p, "r")
        assert schema.validate_report(rep) == []
        assert [m["module_id"] for m in rep["modules"] if m["status"] == "completed"] == [done]
