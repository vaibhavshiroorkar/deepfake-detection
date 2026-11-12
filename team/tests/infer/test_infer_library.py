import json

import pytest

from dfd.infer import library as L
from dfd.infer import mock_adapter, real_adapter
from dfd.infer.adapters import AdapterResult, ModalityScore
from dfd.infer.factory import make_adapter
from dfd.infer.mock_adapter import MockAdapter
from dfd.infer.real_adapter import NoCheckpointError, RealAdapter
from dfd.infer.schema import MODEL_KEYS, SCORE_MEANING, validate_response

MOCK = MockAdapter()


def run(media, name, adapter=MOCK):
    return L.analyse_file(media / name, name, adapter)


@pytest.mark.parametrize("name,mtype", [("img.png", "image"), ("speech.wav", "audio"), ("clip_audio.mp4", "video"), ("clip_silent.mp4", "video")])
def test_mock_response_obeys_contract(media, name, mtype):
    r = run(media, name)
    assert validate_response(r) == []
    assert r["inference_source"] == "mock" and r["model"] is None
    assert r["input"]["media_type"] == mtype and r["status"]["state"] in ("ok", "partial")
    assert r["score_meaning"] == SCORE_MEANING
    assert [s["name"] for s in r["status"]["stages"]] == ["validate", "decode", "detect", "score", "calibrate", "fuse"]


def test_mock_is_deterministic_from_the_hash(media):
    a, b = run(media, "img.png"), run(media, "img.png")
    assert a["result"] == b["result"] and a["input"]["sha256"] == b["input"]["sha256"]
    assert run(media, "tiny.png")["result"] != a["result"]


def test_video_scores_are_separate_and_fused_is_undetermined(media):
    r = run(media, "clip_audio.mp4")["result"]
    assert r["fused"]["decision"] == "undetermined" and r["fused"]["score"] is None and r["fused"]["reason"]
    assert r["per_modality"]["visual"]["available"] and r["per_modality"]["audio"]["available"]


def test_video_without_audio_track_reports_it_not_as_failure(media):
    r = run(media, "clip_silent.mp4")
    audio = r["result"]["per_modality"]["audio"]
    assert audio["available"] is False and audio["score"] is None and "no audio" in audio["reason"]
    assert r["status"]["state"] == "ok"


def test_image_and_audio_fused_equal_their_only_modality(media):
    r = run(media, "img.png")["result"]
    assert r["fused"]["score"] == r["per_modality"]["image"]["score"]
    assert r["fused"]["decision"] in ("fake", "real")
    r = run(media, "speech.wav")["result"]
    assert r["fused"]["score"] == r["per_modality"]["audio"]["score"]


def test_input_warnings(media):
    assert "LOW_RESOLUTION" in [w["code"] for w in run(media, "tiny.png")["warnings"]]
    assert "LOW_RESOLUTION" not in [w["code"] for w in run(media, "img.png")["warnings"]]
    assert "SHORT_AUDIO" in [w["code"] for w in run(media, "short.wav")["warnings"]]
    assert "SHORT_AUDIO" not in [w["code"] for w in run(media, "speech.wav")["warnings"]]


@pytest.mark.parametrize("name,code", [("corrupt.png", "UNREADABLE_FILE"), ("corrupt.wav", "UNREADABLE_FILE"), ("corrupt.mp4", "UNREADABLE_FILE"),
                                       ("empty.png", "EMPTY_FILE"), ("notes.txt", "UNSUPPORTED_MEDIA_TYPE"), ("missing.png", "NO_FILE")])
def test_bad_inputs_give_a_failed_response_without_a_result(media, name, code):
    r = run(media, name)
    assert validate_response(r) == []
    assert r["status"]["state"] == "failed" and r["result"] is None and r["errors"][0]["code"] == code
    assert r["inference_source"] == "mock"


def test_mock_module_and_source_are_unmistakable():
    assert "mock" in mock_adapter.__name__ and MockAdapter.inference_source == "mock"
    assert real_adapter.INFERENCE_SOURCE == "real" and "mock" not in real_adapter.__name__


def test_real_adapter_fails_loudly_without_a_checkpoint(tmp_path):
    with pytest.raises(NoCheckpointError, match="no checkpoint"):
        RealAdapter(None)
    with pytest.raises(NoCheckpointError):
        RealAdapter(tmp_path)
    with pytest.raises(NoCheckpointError):
        make_adapter("real")
    with pytest.raises(ValueError):
        make_adapter("other")


def test_real_adapter_refuses_an_invalid_checkpoint(tmp_path):
    from dfd.infer.checkpoint import CheckpointError
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(CheckpointError, match="lacks"):
        RealAdapter(tmp_path)


class FakeRealAdapter:
    """Test double with source real, to check the contract for a real-shaped response; it is not a model."""
    inference_source = "real"

    def __init__(self, model):
        self.model = model

    def analyse(self, media, progress=None):
        return AdapterResult({"image": ModalityScore(True, 0.9, 1)}, [], self.model, 0.5, {"detect": 1, "score": 1, "calibrate": 1})


def test_real_shaped_response_needs_all_model_keys(media):
    full = {k: "x" for k in MODEL_KEYS}
    assert validate_response(run(media, "img.png", FakeRealAdapter(full))) == []
    partial = dict(full)
    partial.pop("calibration")
    assert any("model keys" in v for v in validate_response(run(media, "img.png", FakeRealAdapter(partial))))
    assert any("model keys" in v for v in validate_response(run(media, "img.png", FakeRealAdapter(None))))


def _copy(r):
    return json.loads(json.dumps(r))


def test_contract_checker_rejects_broken_responses(media):
    good = run(media, "img.png")
    r = _copy(good)
    r["model"] = {"checkpoint_id": "x"}
    assert any("mock response" in v for v in validate_response(r))
    r = _copy(good)
    r["inference_source"] = "REAL"
    assert any("inference_source" in v for v in validate_response(r))
    r = _copy(good)
    r["score_meaning"] = "trust me"
    assert any("score_meaning" in v for v in validate_response(r))
    r = _copy(good)
    r["result"]["fused"]["score"] = 1.5
    assert any("[0, 1]" in v for v in validate_response(r))
    failed = run(media, "notes.txt")
    failed["result"] = good["result"]
    assert any("result null" in v for v in validate_response(failed))
    failed2 = run(media, "notes.txt")
    failed2["errors"] = []
    assert any("at least one error" in v for v in validate_response(failed2))


def test_command_line_prints_the_response(media, capsys):
    assert L.main([str(media / "img.png"), "--adapter", "mock"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inference_source"] == "mock" and validate_response(out) == []
    with pytest.raises(NoCheckpointError):
        L.main([str(media / "img.png"), "--adapter", "real"])
