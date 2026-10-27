import pytest
import torch

from deepfake_detection.streams.avhubert import (
    AVHubert,
    _rename,
    audio_features,
    lip_frames,
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

    wave = np.random.default_rng(0).standard_normal(8_000)
    reference = psf.logfbank(wave, samplerate=16_000)
    reference = np.concatenate([reference, np.zeros((-len(reference) % 4, 26))])
    reference = torch.from_numpy(reference.reshape(-1, 104))
    reference = torch.nn.functional.layer_norm(reference, (104,)).T
    ours = audio_features(torch.from_numpy(wave)[None])[0]
    assert torch.allclose(ours, reference, atol=1e-6)


def test_lip_frames_are_grey_88_pixel_squares():
    mouth = torch.zeros(2, 4, 3, 112, 112)
    lips = lip_frames(mouth)
    assert lips.shape == (2, 1, 4, 88, 88)
    with pytest.raises(ValueError):
        lip_frames(torch.zeros(2, 4, 1, 112, 112))


def test_lip_crop_reads_the_region_it_names():
    mouth = torch.full((1, 1, 3, 100, 100), -3.0)
    mouth[..., 70:, :] = 3.0
    # A crop centred low in the view sees only the bright lower band.
    low = lip_frames(mouth, crop=0.2, center_y=0.9)
    high = lip_frames(mouth, crop=0.2, center_y=0.2)
    assert low.mean() > high.mean()
