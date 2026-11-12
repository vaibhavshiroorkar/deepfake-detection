"""Version 2 checkpoints: a fine-tuned T-V encoder, an ensemble T-A and the recorded preprocessing (E-005, E-007, E-009). CPU only, test doubles."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from dfd.infer import library as L
from dfd.infer import real_adapter
from dfd.infer.checkpoint import (CheckpointError, Ensemble, FineTunedModel, Head, PreprocessMismatchError, load_checkpoint, save_checkpoint,
                                  save_checkpoint_v2)
from dfd.infer.real_adapter import RealAdapter
from dfd.infer.schema import validate_response
from dfd.prep import audio as A
from dfd.prep import config as P
from dfd.prep import face as F

from test_infer_real_adapter import NoFace, OneFace, fake_audio_encoder, fake_encoder, make_ckpt  # noqa: F401


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def task_meta(t, model, temperature=1.0, cfg=None):
    cfg = cfg or {"T-V": P.video_config(), "T-I": P.image_config(), "T-A": P.audio_config()}[t]
    return {"encoder": {"cid": "FAKE", "name": "fake", "revision": "r0", "dim": 3, "input_size": 224},
            "preprocess_id": P.config_hash(cfg), "preprocess_config": cfg,
            "calibration": {"method": "temperature", "temperature": temperature, "fitted_on": "val"},
            "threshold": {"value": 0.5, "how": "test"}, "train": {"dataset": "toy", "splits": ["train"]}, "model": model}


HEAD_A = Head(np.zeros(3, np.float32), np.ones(3, np.float32), np.array([0.5, 0.0, 0.0], np.float32), -1.0)
HEAD_B = Head(np.zeros(3, np.float32), np.ones(3, np.float32), np.array([0.0, 0.8, 0.2], np.float32), -0.5)
HEAD_BLUE = Head(np.zeros(3, np.float32), np.ones(3, np.float32), np.array([0, 0, 12], np.float32), -6.0)


def make_v2(folder, temperature=1.0, tasks=("T-V", "T-A"), mutate=None, audio_cfg=None):
    meta = {"checkpoint_id": "test-v2", "iteration": "I-5", "git_commit": "abc1234", "protocol": "PROTOCOL test", "created": "2026-10-25", "seed": 0,
            "library_versions": {"torch": "x"}, "manifest_ids": ["m1"], "split_ids": ["s1"], "tasks": {}}
    art = {}
    if "T-V" in tasks:
        meta["tasks"]["T-V"] = task_meta("T-V", {"kind": "finetuned_encoder", "weights_file": "weights_T-V.pt"}, temperature)
        torch.save({"params": {}, "head": {"weight": torch.tensor([[0.0, 0.0, 12.0]]), "bias": torch.tensor([-6.0])}}, folder.parent / f"{folder.name}_w.pt")
        art["weights_T-V.pt"] = folder.parent / f"{folder.name}_w.pt"
    if "T-I" in tasks:
        meta["tasks"]["T-I"] = task_meta("T-I", {"kind": "linear_probe"}, temperature)
        meta["tasks"]["T-I"]["head_file"] = "head_T-I.npz"
        art["head_T-I.npz"] = HEAD_BLUE
    if "T-A" in tasks:
        members = [{"cid": "M1", "name": "m1", "revision": "r1", "dim": 3, "input_size": 64600, "head_file": "head_T-A_M1.npz"},
                   {"cid": "M2", "name": "m2", "revision": "r2", "dim": 3, "input_size": 64600, "head_file": "head_T-A_M2.npz"}]
        meta["tasks"]["T-A"] = task_meta("T-A", {"kind": "ensemble_logit_mean", "members": members}, temperature, audio_cfg)
        art["head_T-A_M1.npz"], art["head_T-A_M2.npz"] = HEAD_A, HEAD_B
    if mutate:
        mutate(meta, art)
    save_checkpoint_v2(folder, meta, art)
    return folder


class Recorder:
    """An audio encoder double that records the windows it receives and returns three summary statistics."""
    def __init__(self):
        self.windows = []

    def __call__(self, cid):
        def enc(wave):
            w = np.asarray(wave, np.float32)
            self.windows.append(w[0].copy())
            return np.stack([np.abs(w).mean(1), w.std(1), np.abs(w).max(1)], axis=1) * 10
        return enc


def adapter(folder, det=OneFace, audio=None):
    return RealAdapter(folder, encoder_factory=fake_encoder, detector_factory=det, audio_encoder_factory=audio or fake_audio_encoder)


def run(media, name, a):
    return L.analyse_file(media / name, name, a)


def test_v2_loads_with_the_expected_scorer_types_and_scores_a_video(media, tmp_path):
    ck = make_v2(tmp_path / "ck")
    meta, heads = load_checkpoint(ck)
    assert meta["metadata_version"] == "2" and isinstance(heads["T-V"], FineTunedModel) and isinstance(heads["T-A"], Ensemble) and len(heads["T-A"].members) == 2
    r = run(media, "clip_audio.mp4", adapter(ck))
    per = r["result"]["per_modality"]
    assert validate_response(r) == [] and per["visual"]["available"] and per["audio"]["available"]
    assert r["model"]["tasks"]["T-V"]["model_kind"] == "finetuned_encoder" and r["model"]["tasks"]["T-A"]["model_kind"] == "ensemble_logit_mean"
    assert r["result"]["fused"]["decision"] == "undetermined"                  # no audio-visual fusion rule exists
    assert any(w["code"] == "AUDIO_CALIBRATION_EXPLORATORY" for w in r["warnings"])


def test_finetuned_t_v_path_equals_the_linear_probe_with_the_same_head(media, tmp_path):
    v2 = adapter(make_v2(tmp_path / "v2", tasks=("T-V",)))
    v1 = adapter(make_ckpt(tmp_path / "v1", tasks=("T-V",)))                   # linear head [0, 0, 12], bias -6 on the same fake features
    s2 = run(media, "clip_silent.mp4", v2)["result"]["per_modality"]["visual"]
    s1 = run(media, "clip_silent.mp4", v1)["result"]["per_modality"]["visual"]
    assert s2["score"] == pytest.approx(s1["score"], abs=1e-5) and s2["n_units"] == s1["n_units"]
    img2 = run(media, "img.png", v2)["result"]["per_modality"]["image"]
    img1 = run(media, "img.png", v1)["result"]["per_modality"]["image"]
    assert img2["path"] == "face_crop" and img2["score"] == pytest.approx(img1["score"], abs=1e-5)


def test_finetuned_t_v_temperature_is_applied(media, tmp_path):
    a = run(media, "clip_silent.mp4", adapter(make_v2(tmp_path / "a", tasks=("T-V",), temperature=1.0)))["result"]["per_modality"]["visual"]["score"]
    b = run(media, "clip_silent.mp4", adapter(make_v2(tmp_path / "b", tasks=("T-V",), temperature=4.0)))["result"]["per_modality"]["visual"]["score"]
    assert abs(b - 0.5) < abs(a - 0.5)


def expected_window(path, trim):
    wav = A.load_audio(path)
    return A.window_first_repeat(A.trim_and_peak_normalise(wav) if trim else wav)


def stats(window):
    w = window[None]
    return np.stack([np.abs(w).mean(1), w.std(1), np.abs(w).max(1)], axis=1) * 10


def test_ensemble_score_is_the_mean_of_the_members_logits(media, tmp_path):
    r = run(media, "speech.wav", adapter(make_v2(tmp_path / "ck", tasks=("T-A",))))
    f = stats(expected_window(media / "speech.wav", trim=True))
    z1, z2 = float(HEAD_A.logit_rows(f)[0]), float(HEAD_B.logit_rows(f)[0])
    want = sigmoid((z1 + z2) / 2)
    got = r["result"]["per_modality"]["audio"]["score"]
    assert got == pytest.approx(want, abs=2e-6)
    assert min(sigmoid(z1), sigmoid(z2)) - 1e-6 <= got <= max(sigmoid(z1), sigmoid(z2)) + 1e-6


def test_audio_is_trimmed_and_normalised_before_the_window_is_cut(tmp_path):
    sig = (np.sin(np.arange(16000) * 0.05) * 0.2).astype(np.float32)
    wav = np.concatenate([np.zeros(24000, np.float32), sig, np.zeros(8000, np.float32)])        # 1.5 s silence, 1 s tone, 0.5 s silence
    path = tmp_path / "lead.wav"
    sf.write(path, wav, 16000)
    rec = Recorder()
    run(tmp_path, "lead.wav", adapter(make_v2(tmp_path / "ck", tasks=("T-A",)), audio=rec))
    w = rec.windows[0]
    assert w.size == 64600 and abs(np.abs(w).max() - 1.0) < 1e-5
    first_loud = int(np.nonzero(np.abs(w) > 0.005)[0][0])
    assert first_loud < 5, "leading silence was not removed before the window was cut"
    assert np.allclose(w, expected_window(path, trim=True))                                    # trim and normalise first, then the repeat-padded window
    assert not np.allclose(w, expected_window(path, trim=False))


def test_a_version_1_audio_checkpoint_is_still_scored_with_the_first_window_it_was_trained_with(tmp_path):
    legacy = P.audio_config(trim_normalise=False)

    def old(meta):
        meta["tasks"]["T-A"]["preprocess_config"] = legacy
        meta["tasks"]["T-A"]["preprocess_id"] = P.audio_preprocess_id(trim_normalise=False)
    ck = make_ckpt(tmp_path / "v1", tasks=("T-A",), mutate=old)
    meta, _ = load_checkpoint(ck)
    assert meta["metadata_version"] == "1" and meta["tasks"]["T-A"]["preprocess_id"] != P.audio_preprocess_id()
    wav = np.concatenate([np.zeros(24000, np.float32), (np.sin(np.arange(16000) * 0.05) * 0.2).astype(np.float32)])
    sf.write(tmp_path / "lead.wav", wav, 16000)
    rec = Recorder()
    r = run(tmp_path, "lead.wav", adapter(ck, audio=rec))
    assert validate_response(r) == [] and rec.windows[0][0] == 0.0 and np.abs(rec.windows[0]).max() < 0.25     # first window, not normalised


def test_unknown_audio_preprocessing_is_refused(tmp_path):
    other = P.audio_config(window_samples=32000)
    with pytest.raises(PreprocessMismatchError, match="preprocess_id"):
        load_checkpoint(make_v2(tmp_path / "ck", tasks=("T-A",), audio_cfg=other))              # id is the hash of its config but the code cannot apply it
    wrong_id = make_ckpt(tmp_path / "w", tasks=("T-A",), mutate=lambda m: m["tasks"]["T-A"].update(preprocess_id="123456789abc"))
    with pytest.raises(PreprocessMismatchError):
        load_checkpoint(wrong_id)
    meta = json.loads((wrong_id / "metadata.json").read_text())
    meta["tasks"]["T-A"]["preprocess_id"] = P.audio_preprocess_id(trim_normalise=False)          # an accepted id that is not the hash of the recorded config
    (wrong_id / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(PreprocessMismatchError, match="hash of the recorded"):
        load_checkpoint(wrong_id)


def test_missing_and_changed_files_are_refused(tmp_path):
    ck = make_v2(tmp_path / "a")
    (ck / "weights_T-V.pt").unlink()
    with pytest.raises(CheckpointError, match="weights_T-V.pt"):
        load_checkpoint(ck)
    ck = make_v2(tmp_path / "b")
    with open(ck / "weights_T-V.pt", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(CheckpointError, match="hash differs"):
        load_checkpoint(ck)
    ck = make_v2(tmp_path / "c")
    np.savez(ck / "head_T-A_M2.npz", mean=np.zeros(3), std=np.ones(3), w=np.ones(3), b=np.array(0.0))
    with pytest.raises(CheckpointError, match="head_T-A_M2.npz"):
        load_checkpoint(ck)


def test_every_file_named_by_a_model_block_must_be_listed(tmp_path):
    ck = make_v2(tmp_path / "ck")
    meta = json.loads((ck / "metadata.json").read_text())
    del meta["files"]["head_T-A_M1.npz"]
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="not in the `files` map"):
        load_checkpoint(ck)
    meta.pop("files")
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="files"):
        load_checkpoint(ck)


def test_a_fusion_rule_in_the_metadata_is_refused_until_it_exists(tmp_path):
    ck = make_v2(tmp_path / "ck")
    meta = json.loads((ck / "metadata.json").read_text())
    meta["fusion"] = {"kind": "logistic_regression", "weights": [0.1, 0.2, 0.3]}
    (ck / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(CheckpointError, match="fusion"):
        load_checkpoint(ck)
    assert real_adapter.AUDIO_VISUAL_FUSION is None


def test_unknown_kind_and_missing_member_fields_are_refused(tmp_path):
    def bad_kind(meta, art):
        meta["tasks"]["T-V"]["model"] = {"kind": "mystery"}
    with pytest.raises(CheckpointError, match="unknown model kind"):
        load_checkpoint(make_v2(tmp_path / "a", mutate=bad_kind))

    def no_members(meta, art):
        meta["tasks"]["T-A"]["model"] = {"kind": "ensemble_logit_mean", "members": []}
    with pytest.raises(CheckpointError, match="model.members"):
        load_checkpoint(make_v2(tmp_path / "b", mutate=no_members))


def test_member_head_width_must_match_its_encoder(tmp_path):
    def wide(meta, art):
        meta["tasks"]["T-A"]["model"]["members"][0]["dim"] = 5
    with pytest.raises(CheckpointError, match="head width"):
        load_checkpoint(make_v2(tmp_path / "ck", mutate=wide))


def test_the_config_hash_follows_the_files(tmp_path):
    a = load_checkpoint(make_v2(tmp_path / "a"))[0]
    b = load_checkpoint(make_v2(tmp_path / "b", temperature=2.0))[0]
    from dfd.infer.checkpoint import config_hash_of
    assert config_hash_of(a) == config_hash_of(load_checkpoint(tmp_path / "a")[0]) and isinstance(config_hash_of(b), str)


def test_version_1_checkpoint_roundtrip_still_works(media, tmp_path):
    ck = make_ckpt(tmp_path / "v1", tasks=("T-V", "T-I", "T-A"))
    meta, heads = load_checkpoint(ck)
    assert meta["metadata_version"] == "1" and all(isinstance(h, Head) for h in heads.values())
    r = run(media, "clip_audio.mp4", adapter(ck))
    assert validate_response(r) == [] and r["result"]["per_modality"]["visual"]["available"] and r["result"]["per_modality"]["audio"]["available"]
    assert r["model"]["tasks"]["T-V"]["model_kind"] == "linear_probe"
