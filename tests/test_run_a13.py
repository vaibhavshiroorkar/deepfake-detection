"""run_a13.py: the frame statistics of P-001 amendment A13."""

import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_a13.py"
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def _module():
    spec = importlib.util.spec_from_file_location("run_a13", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _view(gray):
    """A cached visual view (ImageNet-normalized RGB) whose luminance is `gray`."""
    frames = np.repeat(gray[:, None], 3, axis=1)
    return ((frames - MEAN) / STD).astype(np.float32)


def test_luminance_undoes_the_imagenet_normalization():
    module = _module()
    gray = np.random.default_rng(0).uniform(0, 1, (2, 32, 32))
    assert np.allclose(module.luminance(_view(gray)), gray, atol=1e-5)


def test_blur_lowers_sharpness_and_high_frequency_share():
    module = _module()
    sharp = np.random.default_rng(1).uniform(0, 1, (4, 64, 64))
    blurred = np.stack([gaussian_filter(f, 2) for f in sharp])
    a, b = module.clip_statistics(_view(sharp)), module.clip_statistics(_view(blurred))
    assert a["sharpness"] > b["sharpness"]
    assert a["high_frequency"] > b["high_frequency"]


def test_eight_pixel_blocks_are_blocky_and_noise_is_not():
    module = _module()
    rng = np.random.default_rng(2)
    blocks = np.kron(rng.uniform(0, 1, (2, 8, 8)), np.ones((8, 8)))
    noise = rng.uniform(0, 1, (2, 64, 64))
    assert module.clip_statistics(_view(blocks))["blockiness"] > 5
    assert abs(module.clip_statistics(_view(noise))["blockiness"] - 1) < 0.1


def test_static_frames_have_no_temporal_change():
    module = _module()
    frame = np.random.default_rng(3).uniform(0, 1, (1, 32, 32))
    stats = module.clip_statistics(_view(np.repeat(frame, 4, axis=0)))
    assert stats["temporal"] < 1e-6
    assert abs(stats["brightness"] - frame.mean()) < 1e-5
