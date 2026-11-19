"""The website's v2 candidate models: weight mapping, the batch adapter and heatmaps from a map function."""

from __future__ import annotations

import numpy as np
import torch

from deepfake_detection.webapp import candidates
from deepfake_detection.webapp.explain import explain_face


def test_fairseq_names_map_onto_transformers_wav2vec2():
    key = candidates._hf_wav2vec2_key
    assert (
        key("encoder.layers.3.self_attn.q_proj.weight")
        == "encoder.layers.3.attention.q_proj.weight"
    )
    assert (
        key("encoder.layers.3.self_attn_layer_norm.bias")
        == "encoder.layers.3.layer_norm.bias"
    )
    assert (
        key("encoder.layers.0.fc1.weight")
        == "encoder.layers.0.feed_forward.intermediate_dense.weight"
    )
    assert (
        key("feature_extractor.conv_layers.0.0.weight")
        == "feature_extractor.conv_layers.0.conv.weight"
    )
    assert (
        key("feature_extractor.conv_layers.2.2.1.bias")
        == "feature_extractor.conv_layers.2.layer_norm.bias"
    )
    assert key("post_extract_proj.weight") == "feature_projection.projection.weight"
    assert key("layer_norm.weight") == "feature_projection.layer_norm.weight"
    assert key("encoder.layer_norm.weight") == "encoder.layer_norm.weight"
    assert (
        key("encoder.pos_conv.0.weight_g")
        == "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    )
    assert key("quantizer.weight_proj.weight") is None
    assert key("final_proj.bias") is None


def test_effort_residual_is_folded_into_the_weight():
    u, s, v = torch.ones(4, 1), torch.tensor([2.0]), torch.ones(1, 4)
    state = {
        "backbone.encoder.layers.0.self_attn.q_proj.weight_main": torch.zeros(4, 4),
        "backbone.encoder.layers.0.self_attn.q_proj.U_residual": u,
        "backbone.encoder.layers.0.self_attn.q_proj.S_residual": s,
        "backbone.encoder.layers.0.self_attn.q_proj.V_residual": v,
        "backbone.encoder.layers.0.self_attn.q_proj.U_r": torch.zeros(4, 3),
        "backbone.encoder.layers.0.self_attn.q_proj.bias": torch.ones(4),
        "head.weight": torch.zeros(2, 4),
    }
    folded = candidates.fold_effort_state(state)
    assert set(folded) == {
        "encoder.layers.0.self_attn.q_proj.weight",
        "encoder.layers.0.self_attn.q_proj.bias",
    }
    assert torch.equal(
        folded["encoder.layers.0.self_attn.q_proj.weight"], torch.full((4, 4), 2.0)
    )


class _Batch:
    def __init__(self, values, valid):
        self.values = values
        self.valid_samples = torch.tensor([valid])


def test_voice_adapter_passes_only_the_valid_samples():
    seen = {}

    def model(wave):
        seen["n"] = wave.numel()
        return 0.5

    assert (
        candidates.OnBatch(model, "voice")(_Batch(torch.zeros(1, 64000), 48000)) == 0.5
    )
    assert seen["n"] == 48000


def test_face_adapter_passes_the_crops_of_the_one_clip():
    seen = {}

    def model(crops):
        seen["shape"] = tuple(crops.shape)
        return 0.25

    candidates.OnBatch(model, "face")(_Batch(torch.zeros(1, 16, 3, 8, 8), 0))
    assert seen["shape"] == (16, 3, 8, 8)


def test_heatmaps_come_from_the_given_map_function_when_there_is_one():
    frames = torch.zeros(1, 4, 3, 16, 16)
    cams = np.zeros((4, 16, 16), dtype=np.float32)
    cams[2, :8, :] = 1.0  # all evidence in frame 2, top half

    def maps(_frames):
        return cams, cams.sum(axis=(1, 2))

    out = explain_face(
        None, frames, [0.0, 1.0, 2.0, 3.0], offset=10.0, top=1, maps=maps
    )
    assert out["frames"][0]["time"] == 12.0


def test_a_live_model_file_with_another_hash_is_refused(tmp_path):
    import pytest

    (tmp_path / "model.bin").write_bytes(b"changed")
    choice = {"face": {"loader": "clipln", "file": "model.bin", "sha256": "0" * 64}}
    with pytest.raises(RuntimeError, match="not the pinned"):
        candidates.website_models(choice, tmp_path, "cpu")


def test_the_study_models_stay_live_without_a_choice_or_when_asked(
    tmp_path, monkeypatch
):
    from deepfake_detection.webapp import site

    monkeypatch.delenv("DDF_WEB_MODELS")
    monkeypatch.setattr(site, "WEB_MODELS", tmp_path / "web-models.json")
    assert site.web_models() is None
    (tmp_path / "web-models.json").write_text('{"calibration": {}}', encoding="utf-8")
    assert site.web_models() == {"calibration": {}}
    monkeypatch.setenv("DDF_WEB_MODELS", "p001")
    assert site.web_models() is None


def test_the_generated_video_probe_scores_the_mean_embedding_through_its_linear_head(
    tmp_path,
):
    import json

    params = {
        "mean": [1.0, 0.0],
        "scale": [2.0, 1.0],
        "coef": [1.0, -1.0],
        "intercept": 0.5,
    }
    (tmp_path / "gen.json").write_text(json.dumps(params), encoding="utf-8")
    probe = candidates.GenProbe(
        tmp_path / "gen.json", "cpu", embed=lambda crops: np.array([3.0, 1.0])
    )
    # standardized (3-1)/2 = 1 and (1-0)/1 = 1, so the logit is 1 - 1 + 0.5 = 0.5
    assert probe(torch.zeros(4, 3, 8, 8)) == float(torch.sigmoid(torch.tensor(0.5)))


def test_a_generated_video_check_is_loaded_as_a_live_check(tmp_path, monkeypatch):
    (tmp_path / "gen.json").write_text("{}", encoding="utf-8")
    digest = candidates.sha256(tmp_path / "gen.json")
    monkeypatch.setitem(
        candidates.LOADERS, "genprobe", lambda path, device: torch.nn.Identity()
    )
    choice = {"gen": {"loader": "genprobe", "file": "gen.json", "sha256": digest}}
    live = candidates.website_models(choice, tmp_path, "cpu")
    assert (
        set(live) == {"gen"} and live["gen"].check == "face"
    )  # it reads the face crops
