import pytest
import torch

from dfd.av.avhubert import (
    AVHubert,
    _rename,
    audio_features,
)


def _released_layout(model: AVHubert) -> dict[str, torch.Tensor]:
    """A state dict keyed as the fairseq release keys it, from this module."""
    out = {}
    for key, value in model.state_dict().items():
        if key.startswith("pos_conv.conv."):
            suffix = {
                "pos_conv.conv.bias": "bias",
                "pos_conv.conv.parametrizations.weight.original0": "weight_g",
                "pos_conv.conv.parametrizations.weight.original1": "weight_v",
            }[key]
            out[f"encoder.pos_conv.0.{suffix}"] = value
        elif key.startswith("encoder_layer_norm."):
            out[key.replace("encoder_layer_norm.", "encoder.layer_norm.")] = value
        elif key.startswith("layers."):
            if "in_proj" in key:
                prefix, part = key.rsplit(".in_proj_", 1)
                for name, chunk in zip("qkv", value.chunk(3), strict=True):
                    out[f"encoder.{prefix}.{name}_proj.{part}"] = chunk
            else:
                out[f"encoder.{key}"] = value
        else:
            out[key] = value
    # Pretraining-only tensors the release also carries.
    out["mask_emb"] = torch.zeros(104)
    out["final_proj.weight"] = torch.zeros(256, 768)
    return out


def test_released_keys_load_strictly_and_reproduce_the_model():
    torch.manual_seed(0)
    source = AVHubert(layers=1).eval()
    target = AVHubert(layers=1).eval()
    target.load_state_dict(_rename(_released_layout(source)), strict=True)
    audio = torch.randn(1, 104, 6)
    with torch.no_grad():
        assert torch.allclose(source(audio=audio), target(audio=audio))


def test_either_modality_alone_gives_tokens_at_the_video_rate():
    model = AVHubert(layers=1).eval()
    lips = torch.randn(1, 1, 5, 88, 88)
    audio = torch.randn(1, 104, 5)
    with torch.no_grad():
        assert model(video=lips).shape == (1, 5, 768)
        assert model(audio=audio).shape == (1, 5, 768)
        assert model(video=lips, audio=audio).shape == (1, 5, 768)
    with pytest.raises(ValueError):
        model()


def test_two_seconds_of_audio_become_fifty_steps():
    assert audio_features(torch.randn(2, 32_000)).shape == (2, 104, 50)


def test_audio_features_ignore_the_waveform_gain():
    # The cache stores z-normalised audio, so this is what makes it usable. The
    # last step is excluded: it holds zero padding, which a gain does not
    # shift, exactly as in the released loader.
    wave = torch.randn(1, 16_000, dtype=torch.float64)
    plain, louder = audio_features(wave), audio_features(7.5 * wave)
    assert torch.allclose(plain[..., :-1], louder[..., :-1], atol=1e-6)


def test_audio_features_match_python_speech_features():
    psf = pytest.importorskip("python_speech_features")
    import numpy as np



def test_large_shape_runs_and_keeps_its_width():
    model = AVHubert(width=1024, layers=1, heads=16, ffn=4096).eval()
    with torch.no_grad():
        assert model(audio=torch.randn(1, 104, 5)).shape == (1, 5, 1024)


def test_output_layer_stops_early_and_skips_the_final_norm():
    torch.manual_seed(0)
    model = AVHubert(layers=2).eval()
    audio = torch.randn(1, 104, 5)
    with torch.no_grad():
        first = model(audio=audio, output_layer=1)
        full = model(audio=audio)
    assert not torch.allclose(first, full)
    # The final LayerNorm leaves each step with zero mean; layer 1 output does not.
    assert full.mean(-1).abs().max() < 1e-4
    assert first.mean(-1).abs().max() > 1e-4
