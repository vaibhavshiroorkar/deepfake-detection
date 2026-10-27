"""Regression checks for input, architecture and provenance contracts."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402


class Recorder(nn.Module):
    def forward(self, values, **kwargs):
        self.values, self.kwargs = values, kwargs
        return SimpleNamespace(logits=values.mean(1), embedding=values[:, :2])


def test_audio_stream_preserves_cached_normalization():
    from deepfake_detection.fusion.stream_export import StreamSpec, _forward

    model = Recorder()
    waveform = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    _forward(
        StreamSpec("audio", model, "hash", "audio"),
        SimpleNamespace(audio_view=waveform, audio_valid_samples=2),
        "cpu",
    )
    assert np.array_equal(model.values.numpy()[0], waveform)


def test_checkpoint_mask_mode_controls_forward():
    from deepfake_detection.training.binary import _forward

    model = Recorder()
    batch = SimpleNamespace(valid_samples=torch.tensor([2]))
    values = torch.ones(1, 4)
    model.masking_mode = "legacy_none"
    _forward(model, batch, values)
    assert model.kwargs == {}
    model.masking_mode = "valid_samples"
    _forward(model, batch, values)
    assert torch.equal(model.kwargs["valid_samples"], batch.valid_samples)
    with pytest.raises(ValueError, match="valid"):
        _forward(model, SimpleNamespace(valid_samples=None), values)


def test_stream_loading_restores_full_architecture(monkeypatch):
    import deepfake_detection.streams.cross_modal_stream as streams
    from deepfake_detection.fusion.stream_loading import build_model

    monkeypatch.setattr(streams, "build_lipsync_stream", lambda **kw: kw)
    config = {
        "common_dim": 32,
        "attention_heads": 8,
        "video_backbone": "resnet18",
        "audio_model": "custom-audio",
        "temporal_position_mode": "sinusoidal",
        "query_steps": 20,
    }
    model, kind = build_model("stream-lipsync", {"config": {"model": config}})
    assert kind == "lipsync"
    for key, value in config.items():
        assert model[key] == value
    legacy, _ = build_model("stream-lipsync", {"config": {"model": {}}})
    assert legacy["temporal_position_mode"] == "none"


@pytest.mark.parametrize("probabilities", ["\n", "0.3\n"])
def test_abstention_report_survives_missing_scored_classes(tmp_path, probabilities):
    from deepfake_detection.cli.evaluate import _evaluate

    path = tmp_path / "predictions.csv"
    path.write_text(
        "label,probability,source,method\n0,,a,real\n1,"
        + probabilities.strip()
        + ",b,fake\n"
    )
    output = tmp_path / "report.json"
    args = argparse.Namespace(
        predictions=path, output=output, threshold=0.5, bootstrap_samples=10, seed=17
    )
    assert _evaluate(args) == 0
    report = json.loads(output.read_text())
    assert report["overall"]["metrics"] is None
    assert report["confidence_intervals"] is None


def test_mouth_order_can_change_new_stream_embedding():
    from deepfake_detection.streams.config import StreamConfig
    from deepfake_detection.streams.cross_modal_stream import CrossModalStream

    class Pair(nn.Module):
        width = 8

        def forward(self, video, audio):
            return video, audio

    torch.manual_seed(17)
    model = CrossModalStream(
        config=StreamConfig(common_dim=8), pair_encoder=Pair(), attention_heads=2
    ).eval()
    video, audio = torch.randn(1, 12, 8), torch.randn(1, 12, 8)
    first = model(video=video, audio=audio).embedding
    for permutation in (
        video.flip(1),
        video.roll(3, 1),
        video[:, [2, 0, 7, 1, 9, 4, 6, 8, 11, 10, 3, 5]],
    ):
        other = model(video=permutation, audio=audio).embedding
        assert not torch.allclose(first, other, atol=1e-6)


def test_fusion_rejects_replaced_stream_checkpoint():
    from deepfake_detection.fusion.deep_loading import LoadedFusion

    fusion = LoadedFusion(
        model=nn.Identity(),
        stream_dims={"audio": 2},
        split_hash="s",
        preprocessing_hash="p",
        stream_checkpoint_hashes={"audio": "trained"},
    )
    fusion.validate_checkpoints({"audio": "trained"})
    with pytest.raises(ValueError, match="checkpoint"):
        fusion.validate_checkpoints({"audio": "replacement"})


def test_deep_fusion_validates_cross_row_provenance(monkeypatch, tmp_path):
    import deepfake_detection.cli.fusion as cli

    rows = [
        SimpleNamespace(
            partition_role="holdout",
            dataset="FakeAVCeleb",
            split_hash=split,
            preprocessing_hash="p",
        )
        for split in ("first", "second")
    ]
    monkeypatch.setattr(cli.FeatureStore, "assemble", lambda *a, **kw: rows)
    monkeypatch.setattr(
        cli, "_deep_fusion", lambda *a: pytest.fail("dispatched mixed provenance")
    )
    args = argparse.Namespace(
        branches=["audio"], feature_store=tmp_path, drop_incomplete=False, model="deep"
    )
    with pytest.raises(ValueError, match="split hashes"):
        cli._train_fusion(args)


def test_stream_export_rejects_selection_overlap_before_reading_cache(tmp_path):
    from deepfake_detection.fusion.stream_export import (
        StreamSpec,
        export_stream_features,
    )

    model = Recorder()
    model.trained_source_identities = ("train",)
    model.selection_source_identities = ("holdout",)
    with pytest.raises(ValueError, match="selection"):
        export_stream_features(
            records=[SimpleNamespace(source="holdout")],
            cache_index={},
            cache_store=None,
            feature_store=None,
            streams=[StreamSpec("audio", model, "hash", "audio")],
            split_hash="s",
            preprocessing_hash="p",
            partition_role="oof",
            run_id="test",
            device="cpu",
        )
