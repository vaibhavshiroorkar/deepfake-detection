import json
import shutil

import numpy as np
import pytest

from dfd.data import loaders
from dfd.infer import library as L
from dfd.infer.checkpoint import (CheckpointError, Head, NoCheckpointError, PreprocessMismatchError, load_checkpoint, save_checkpoint)
from dfd.infer.real_adapter import RealAdapter
from dfd.infer.schema import MODEL_KEYS, validate_response
from dfd.prep import config as P
from dfd.prep import face as F
from dfd.prep.pipeline import Params


class OneFace:
    def detect(self, frame):
        h, w = frame.shape[:2]
        return [F.Face(w * 0.2, h * 0.2, w * 0.5, h * 0.5, 0.9)]


class TwoFaces:
    def detect(self, frame):
        h, w = frame.shape[:2]
        return [F.Face(0, 0, w * 0.3, h * 0.3, 0.9), F.Face(w * 0.5, h * 0.5, w * 0.4, h * 0.4, 0.8)]


class NoFace:
    def detect(self, frame):
        return []


def fake_encoder(cid):
    """Per-channel mean of the image as a 3-dimensional feature (a test double, not a model)."""
    def enc(batch):
        return batch.float().mean(dim=(2, 3)) / 255.0
    return enc


def make_ckpt(folder, tasks=("T-V", "T-I"), temperature=1.0, mutate=None):
    meta = {"checkpoint_id": "test-ckpt", "iteration": "I-0", "git_commit": "abc1234", "protocol": "PROTOCOL test", "created": "2026-10-25",
            "seed": 0, "library_versions": {"torch": "x"}, "manifest_ids": {"ffpp": "m1"}, "split_ids": {"ffpp": "s1"}, "tasks": {}}
    heads = {}
    for t in tasks:
        cfg = {"T-V": P.video_config(), "T-I": P.image_config(), "T-A": P.audio_config()}[t]
        meta["tasks"][t] = {"encoder": {"cid": "FAKE", "name": "fake", "revision": "r0", "dim": 3, "input_size": 224},
                            "preprocess_id": P.IN_USE[t](), "preprocess_config": cfg,
                            "calibration": {"method": "temperature", "temperature": temperature, "fitted_on": "val"},
                            "threshold": {"value": 0.5, "how": "test"}, "train": {"dataset": "toy", "splits": ["train"]}}
        heads[t] = Head(np.zeros(3, np.float32), np.ones(3, np.float32), np.array([0, 0, 12], np.float32), -6.0)  # brighter blue means more fake
    if mutate:
        mutate(meta)
    save_checkpoint(folder, meta, heads)
    return folder


def fake_audio_encoder(cid):
    """Three summary statistics of the waveform as a 3-dimensional feature (a test double, not a model)."""
    def enc(wave):
        w = np.asarray(wave, np.float32)
        return np.stack([np.abs(w).mean(1), w.std(1), np.abs(w).max(1)], axis=1) * 10
    return enc


def adapter(folder, det=OneFace):
    return RealAdapter(folder, encoder_factory=fake_encoder, detector_factory=det, audio_encoder_factory=fake_audio_encoder)


def run(media, name, a):
    return L.analyse_file(media / name, name, a)


def test_valid_checkpoint_loads_and_adapter_scores_an_image(media, tmp_path):
    a = adapter(make_ckpt(tmp_path / "ck"))
    r = run(media, "img.png", a)
    assert validate_response(r) == []
    assert r["inference_source"] == "real" and all(k in r["model"] for k in MODEL_KEYS)
    m = r["result"]["per_modality"]["image"]
    assert m["available"] and 0 < m["score"] < 1 and m["path"] == "face_crop"
    assert r["result"]["fused"]["score"] == m["score"] and r["result"]["fused"]["decision"] in ("fake", "real")
    assert run(media, "img.png", a)["result"] == r["result"]          # deterministic


def test_image_without_a_face_uses_the_whole_image_path(media, tmp_path):
    r = run(media, "img.png", adapter(make_ckpt(tmp_path / "ck"), NoFace))
    assert validate_response(r) == [] and r["result"]["per_modality"]["image"]["path"] == "whole_image"


def test_image_without_face_and_without_whole_image_model_is_unavailable(media, tmp_path):
    r = run(media, "img.png", adapter(make_ckpt(tmp_path / "ck", tasks=("T-V",)), NoFace))
    m = r["result"]["per_modality"]["image"]
    assert validate_response(r) == [] and m["available"] is False and m["score"] is None
    assert r["result"]["fused"]["decision"] == "undetermined" and r["status"]["state"] == "partial"


def test_temperature_flattens_the_probability(media, tmp_path):
    p1 = run(media, "img.png", adapter(make_ckpt(tmp_path / "a", temperature=1.0)))["result"]["fused"]["score"]
    p2 = run(media, "img.png", adapter(make_ckpt(tmp_path / "b", temperature=4.0)))["result"]["fused"]["score"]
    assert abs(p2 - 0.5) < abs(p1 - 0.5)


def test_video_visual_score_and_separate_audio(media, tmp_path):
    r = run(media, "clip_audio.mp4", adapter(make_ckpt(tmp_path / "ck", tasks=("T-V", "T-A"))))
    per = r["result"]["per_modality"]
    assert validate_response(r) == [] and per["visual"]["available"] and per["visual"]["n_units"] == 10
    assert per["audio"]["available"] and 0 < per["audio"]["score"] < 1 and per["audio"]["n_units"] == 1
    assert r["result"]["fused"]["decision"] == "undetermined" and r["status"]["state"] == "ok"      # separate scores, no fused claim (O-009)
    assert set(r["model"]["tasks"]) == {"T-V", "T-A"}


def test_video_warnings_for_several_faces_and_for_no_face(media, tmp_path):
    ck = make_ckpt(tmp_path / "ck")
    r = run(media, "clip_silent.mp4", adapter(ck, TwoFaces))
    assert "MULTIPLE_FACES" in [w["code"] for w in r["warnings"]] and r["result"]["per_modality"]["visual"]["available"]
    r = run(media, "clip_silent.mp4", adapter(ck, NoFace))
    assert "NO_FACE_DETECTED" in [w["code"] for w in r["warnings"]]
    v = r["result"]["per_modality"]["visual"]
    assert validate_response(r) == [] and v["available"] is False and v["score"] is None and r["status"]["state"] == "partial"


def test_audio_file_is_scored_with_the_audio_head(media, tmp_path):
    a = adapter(make_ckpt(tmp_path / "ck", tasks=("T-V", "T-I", "T-A")))
    r = run(media, "speech.wav", a)
    m = r["result"]["per_modality"]["audio"]
    assert validate_response(r) == [] and m["available"] and 0 < m["score"] < 1
    assert r["result"]["fused"]["score"] == m["score"] and r["result"]["fused"]["decision"] in ("fake", "real")
    assert r["model"]["calibration"]["method"] == "temperature" and r["model"]["preprocess_id"] == P.audio_preprocess_id()
    assert run(media, "speech.wav", a)["result"] == r["result"]                          # deterministic
    short = run(media, "short.wav", a)                                                    # 0.5 s, repeat-padded to the window
    assert validate_response(short) == [] and short["result"]["per_modality"]["audio"]["available"]
    assert "SHORT_AUDIO" in [w["code"] for w in short["warnings"]]


def test_audio_scores_carry_the_exploratory_calibration_warning(media, tmp_path):
    a = adapter(make_ckpt(tmp_path / "ck", tasks=("T-V", "T-I", "T-A")))
    for name in ("speech.wav", "clip_audio.mp4"):                                         # an audio file and the audio track of a video
        r = run(media, name, a)
        w = [x for x in r["warnings"] if x["code"] == "AUDIO_CALIBRATION_EXPLORATORY"]
        assert validate_response(r) == [] and len(w) == 1 and w[0]["modality"] == "audio"
    img = run(media, "img.png", a)                                                        # no audio score, no such warning
    assert "AUDIO_CALIBRATION_EXPLORATORY" not in [x["code"] for x in img["warnings"]]


def test_audio_without_an_audio_model_is_unavailable_not_faked(media, tmp_path):
    r = run(media, "speech.wav", adapter(make_ckpt(tmp_path / "ck")))                    # checkpoint has T-V and T-I only
    m = r["result"]["per_modality"]["audio"]
    assert validate_response(r) == [] and m["available"] is False and m["score"] is None and "no audio model" in m["reason"]
    assert r["result"]["fused"]["score"] is None and r["result"]["fused"]["decision"] == "undetermined"


def test_no_checkpoint_means_no_adapter(tmp_path):
    with pytest.raises(NoCheckpointError):
        RealAdapter(None)
    with pytest.raises(NoCheckpointError):
        RealAdapter(tmp_path / "empty")


def test_missing_metadata_field_is_refused(tmp_path):
    ck = make_ckpt(tmp_path / "ck")
    meta = json.loads((ck / "metadata.json").read_text())
    del meta["git_commit"]
    del meta["tasks"]["T-V"]["calibration"]["temperature"]
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="git_commit"):
        load_checkpoint(ck)
    meta["git_commit"] = "x"
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="calibration.temperature"):
        load_checkpoint(ck)


def test_invalid_json_and_unsupported_version_are_refused(tmp_path):
    ck = make_ckpt(tmp_path / "ck")
    (ck / "metadata.json").write_text("{not json")
    with pytest.raises(CheckpointError, match="not valid JSON"):
        load_checkpoint(ck)
    meta = {"metadata_version": "9"}
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError):
        load_checkpoint(ck)


def test_changed_head_file_is_refused(tmp_path):
    ck = make_ckpt(tmp_path / "ck")
    np.savez(ck / "head_T-V.npz", mean=np.zeros(3), std=np.ones(3), w=np.ones(3), b=np.array(0.0))
    with pytest.raises(CheckpointError, match="hash differs"):
        load_checkpoint(ck)


def test_preprocess_id_mismatch_is_refused(tmp_path):
    def other(meta):
        meta["tasks"]["T-V"]["preprocess_id"] = "000000000000"
    ck = make_ckpt(tmp_path / "ck", mutate=other)
    with pytest.raises(PreprocessMismatchError, match="preprocess_id"):
        RealAdapter(ck, encoder_factory=fake_encoder, detector_factory=OneFace)
    # a changed configuration in use gives a different id, so an old checkpoint is refused
    assert P.video_preprocess_id(Params(n_frames=8)) != P.video_preprocess_id()


def test_head_width_must_match_the_encoder(tmp_path):
    def wide(meta):
        meta["tasks"]["T-V"]["encoder"]["dim"] = 5
    with pytest.raises(CheckpointError, match="head width"):
        load_checkpoint(make_ckpt(tmp_path / "ck", mutate=wide))


def test_video_preprocess_id_equals_the_id_of_a_crop_folder(tmp_path):
    p = Params()
    folder = tmp_path / "crops"
    folder.mkdir()
    ident = P.video_config(p)
    (folder / "params.json").write_text(json.dumps(ident, sort_keys=True), encoding="utf-8")   # what dfd.prep.pipeline writes
    assert loaders.preprocess_id(folder) == P.video_preprocess_id(p)


class OwnRuleEncoder:
    """An encoder double with its own image rule: `processor` (PIL -> normalised tensor) and `encode_pixels`, as in dfd.sweep.encoders."""
    seen = []

    def processor(self, pil):
        import torch
        OwnRuleEncoder.seen.append(pil.size)               # the full-resolution image reaches the processor, not a pre-resized copy
        return torch.from_numpy(np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0)

    def encode_pixels(self, px):
        return px.mean(dim=(2, 3))

    def __call__(self, batch):
        return batch.float().mean(dim=(2, 3)) / 255.0


def test_whole_image_path_applies_the_encoders_own_rule_to_the_full_image(media, tmp_path):
    OwnRuleEncoder.seen.clear()
    a = RealAdapter(make_ckpt(tmp_path / "ck"), encoder_factory=lambda cid: OwnRuleEncoder(), detector_factory=NoFace)
    r = run(media, "img.png", a)
    assert validate_response(r) == [] and r["result"]["per_modality"]["image"]["path"] == "whole_image"
    assert OwnRuleEncoder.seen == [(128, 96)]
