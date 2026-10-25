"""WS6 end-to-end tests: real files through inspect, real coordinator, provenance modules and the planned-module placeholders.
The detector modules use a fake adapter here (the real adapter is covered by the opt-in parity test in tests/platform_modules)."""
import pytest

from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.infer.library import InputError
from dfd.platform.modules import build_default_modules
from dfd.platform.modules.base import AdapterProvider
from dfd.platform.pipeline import analyse_file, planned_modules
from dfd.platform.provenance.modules import build_provenance_modules
from dfd.platform.schema import validate_report


class NoFaceAdapter:
    meta = {"checkpoint_id": "fake-ckpt", "files": {}, "tasks": {"T-V": {"threshold": {"value": 0.6}}, "T-I": {"threshold": {"value": 0.5}},
                                                                    "T-A": {"threshold": {"value": 0.5}}}}

    def analyse(self, media, progress=None):
        pm = {}
        if media.media_type == "video":
            pm["visual"] = ModalityScore(False, reason="no face found")
        if media.has_audio:
            pm["audio"] = ModalityScore(True, 0.2, 1)
        if media.media_type == "image":
            pm["image"] = ModalityScore(True, 0.3, 1, path="whole_image")
        return AdapterResult(pm, [{"code": "NO_FACE_DETECTED", "modality": "visual", "message": "no face"}] if media.media_type == "video" else [],
                             {"checkpoint_id": "fake-ckpt"}, 0.5, {})


def modules():
    return build_default_modules(AdapterProvider(adapter=NoFaceAdapter())) + planned_modules() + build_provenance_modules()


def by_id(r):
    return {m["module_id"]: m for m in r["modules"]}


def test_video_with_audio_is_routed_and_every_module_is_accounted_for(media):
    r = analyse_file(media / "clip_audio.mp4", modules=modules())
    assert validate_report(r) == []
    m = by_id(r)
    assert m["face_video"]["status"] == "not_applicable"            # no face is never a finding
    assert m["image_detector"]["status"] == "not_applicable"
    assert "generated_video" not in m                                  # not in this test's registry (fake adapter); see test_generated_video_registered_and_unsupported_without_head
    assert m["media_metadata"]["status"] == "completed" and m["content_credentials"]["status"] == "completed"
    assert r["status"]["state"] == "ok"                                 # nothing failed or unsupported in this registry
    assert r["source_summary"] == "real"
    assert all(x["finding"] != "no_indication_within_tested_scope" for x in r["modules"] if x["module_id"] in ("face_video", "image_detector"))


def test_silent_video_has_no_speech_result(media):
    m = by_id(analyse_file(media / "clip_silent.mp4", modules=modules()))
    assert m["speech"]["status"] == "not_applicable"


def test_audio_file_routes_to_speech_only(media):
    r = analyse_file(media / "speech.wav", modules=modules())
    assert validate_report(r) == []
    m = by_id(r)
    assert m["speech"]["status"] == "completed" and m["face_video"]["status"] == "not_applicable"


def test_image_routes_to_image_detector(media):
    r = analyse_file(media / "img.png", modules=modules())
    assert validate_report(r) == []
    assert by_id(r)["image_detector"]["status"] == "completed"
    assert by_id(r)["speech"]["status"] == "not_applicable"


def test_one_module_failing_does_not_stop_the_rest(media):
    class Boom(NoFaceAdapter):
        def analyse(self, media, progress=None):
            raise RuntimeError("boom")
    mods = build_default_modules(AdapterProvider(adapter=Boom())) + build_provenance_modules()
    r = analyse_file(media / "clip_audio.mp4", modules=mods)
    assert validate_report(r) == []
    m = by_id(r)
    assert m["speech"]["status"] == "failed" and m["media_metadata"]["status"] == "completed"


@pytest.mark.parametrize("name,code", [("corrupt.png", "UNREADABLE_FILE"), ("empty.png", "EMPTY_FILE"), ("notes.txt", "UNSUPPORTED_MEDIA_TYPE")])
def test_unusable_input_is_refused_before_any_module_runs(media, name, code):
    with pytest.raises(InputError) as e:
        analyse_file(media / name, modules=modules())
    assert e.value.code == code


def test_planned_modules_and_default_registry():
    from dfd.platform.pipeline import default_modules
    assert planned_modules() == []
    ids = [m.module_id for m in default_modules()]
    assert ids == ["face_video", "image_detector", "speech", "lip_sync", "generated_video", "media_metadata", "content_credentials"]


def test_generated_video_registered_and_unsupported_without_head(media, tmp_path):
    from dfd.platform.modules.generated_video import GeneratedVideoModule
    mods = modules() + [GeneratedVideoModule(tmp_path / "missing_head.npz")]
    r = analyse_file(media / "clip_audio.mp4", modules=mods)
    assert validate_report(r) == []
    g = by_id(r)["generated_video"]
    assert g["status"] == "unsupported" and "no trained generated-video detector" in g["reason"] and g["finding"] is None
